"""Who is blocking whom, who is idle in a transaction, and where the connections went.

``pg_blocking_pids()`` gives one edge of the graph at a time. The useful artefact
is the whole chain: at 3am the question is never "is pid 4711 waiting" but "which
single session, if I terminate it, unblocks the other forty" — and that session is
usually not the one anybody is looking at, because it is not waiting for anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pgtk.db import Database
from pgtk.findings import Finding, Severity

ACTIVITY_SQL = """
SELECT a.pid,
       a.usename,
       a.datname,
       coalesce(a.application_name, '') AS application_name,
       coalesce(host(a.client_addr), 'local') AS client,
       coalesce(a.state, '') AS state,
       coalesce(a.wait_event_type, '') AS wait_event_type,
       coalesce(a.wait_event, '') AS wait_event,
       a.xact_start,
       a.state_change,
       {query_id} AS query_id,
       left(coalesce(a.query, ''), 500) AS query,
       pg_blocking_pids(a.pid) AS blocked_by
FROM pg_stat_activity a
WHERE a.pid <> pg_backend_pid()
  AND a.datname IS NOT DISTINCT FROM current_database()
"""

UNGRANTED_LOCKS_SQL = """
SELECT l.pid,
       l.locktype,
       l.mode,
       coalesce(c.relname, l.locktype) AS object
FROM pg_locks l
LEFT JOIN pg_class c ON c.oid = l.relation
WHERE NOT l.granted
"""

CONNECTION_LIMITS_SQL = """
SELECT (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_connections,
       (SELECT setting::int FROM pg_settings
         WHERE name = 'superuser_reserved_connections') AS reserved,
       (SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'client backend')::int AS used
"""

DISTRIBUTION_SQL = """
SELECT coalesce(datname, '-') AS datname,
       coalesce(usename, '-') AS usename,
       coalesce(nullif(application_name, ''), '-') AS application_name,
       coalesce(state, '-') AS state,
       count(*)::int AS sessions,
       max(extract(epoch FROM clock_timestamp() - state_change))::float8 AS max_state_age_s
FROM pg_stat_activity
WHERE backend_type = 'client backend'
GROUP BY 1, 2, 3, 4
ORDER BY sessions DESC, 1, 2, 3, 4
"""


@dataclass(frozen=True)
class Session:
    pid: int
    user: str
    database: str | None
    application: str
    client: str
    state: str
    wait_event_type: str
    wait_event: str
    xact_start: datetime | None
    state_change: datetime | None
    query_id: int | None
    query: str
    blocked_by: tuple[int, ...]

    def age(self, now: datetime, since: datetime | None) -> timedelta:
        return timedelta(0) if since is None else now - since

    @property
    def one_line(self) -> str:
        collapsed = " ".join(self.query.split())
        return collapsed[:120] + ("…" if len(collapsed) > 120 else "")


@dataclass
class BlockingNode:
    session: Session
    waiting_for: str
    children: list[BlockingNode] = field(default_factory=list)

    def pids(self) -> set[int]:
        """Distinct sessions in this subtree.

        A session can be reported as blocked by several others at once, in which
        case it hangs under each of them. Counting the nodes would then claim the
        root blocks more sessions than exist.
        """
        found = {self.session.pid}
        for child in self.children:
            found |= child.pids()
        return found

    def size(self) -> int:
        return len(self.pids())

    def depth(self) -> int:
        return 1 + max((child.depth() for child in self.children), default=0)


def load_sessions(db: Database) -> list[Session]:
    sql = ACTIVITY_SQL.format(
        query_id="a.query_id" if db.caps.has("pg_stat_activity.query_id") else "NULL::bigint"
    )
    sessions: list[Session] = []
    for row in db.rows(sql):
        sessions.append(
            Session(
                pid=int(row["pid"]),
                user=str(row["usename"] or "-"),
                database=row["datname"],
                application=str(row["application_name"]),
                client=str(row["client"]),
                state=str(row["state"]),
                wait_event_type=str(row["wait_event_type"]),
                wait_event=str(row["wait_event"]),
                xact_start=row["xact_start"],
                state_change=row["state_change"],
                query_id=row["query_id"],
                query=str(row["query"]),
                blocked_by=tuple(int(pid) for pid in (row["blocked_by"] or [])),
            )
        )
    return sessions


def build_blocking_forest(
    sessions: list[Session], lock_targets: dict[int, str] | None = None
) -> list[BlockingNode]:
    """Turn the ``blocked_by`` edges into trees rooted at whoever is not waiting.

    ``pg_blocking_pids`` can momentarily report a cycle (the deadlock detector has
    not fired yet, or the snapshot caught two edges from different instants), so
    the walk carries a visited set. A tool that hangs while rendering a lock tree
    is worse than one that prints a shallow one.
    """
    targets = lock_targets or {}
    by_pid = {session.pid: session for session in sessions}
    blocked = {session.pid: session.blocked_by for session in sessions if session.blocked_by}
    involved = set(blocked) | {pid for pids in blocked.values() for pid in pids}
    roots = [pid for pid in involved if pid in by_pid and not blocked.get(pid)]

    children_of: dict[int, list[int]] = {}
    for pid, blockers in blocked.items():
        for blocker in blockers:
            children_of.setdefault(blocker, []).append(pid)

    def build(pid: int, seen: frozenset[int]) -> BlockingNode:
        node = BlockingNode(session=by_pid[pid], waiting_for=targets.get(pid, ""))
        for child in sorted(children_of.get(pid, [])):
            if child in seen or child not in by_pid:
                continue
            node.children.append(build(child, seen | {child}))
        return node

    return [build(pid, frozenset({pid})) for pid in sorted(roots)]


def ungranted_lock_targets(db: Database) -> dict[int, str]:
    targets: dict[int, str] = {}
    for row in db.rows(UNGRANTED_LOCKS_SQL):
        targets.setdefault(int(row["pid"]), f"{row['mode']} on {row['object']}")
    return targets


def connection_distribution(db: Database) -> list[dict[str, Any]]:
    return db.rows(DISTRIBUTION_SQL)


def analyze_locks(
    db: Database,
    *,
    idle_in_transaction: timedelta = timedelta(minutes=5),
    long_wait: timedelta = timedelta(seconds=30),
    connection_pressure: float = 0.8,
) -> tuple[list[Finding], list[BlockingNode]]:
    sessions = load_sessions(db)
    forest = build_blocking_forest(sessions, ungranted_lock_targets(db))
    now = datetime.now(UTC)
    findings: list[Finding] = []

    for root in forest:
        waiters = root.size() - 1
        if waiters == 0:
            continue
        session = root.session
        idle = session.state.startswith("idle in transaction")
        held = session.age(now, session.xact_start)
        findings.append(
            Finding(
                check="locks.blocking_root",
                severity=Severity.CRITICAL if idle or waiters > 2 else Severity.WARNING,
                subject=f"pid {session.pid} ({session.user}@{session.application or '-'})",
                summary=(
                    f"blocks {waiters} session(s), chain depth {root.depth()}; "
                    f"state '{session.state}', transaction open for "
                    f"{_seconds(held)}"
                    + ("; it is not running a query" if idle else "")
                ),
                facts={
                    "pid": session.pid,
                    "blocked_sessions": waiters,
                    "chain_depth": root.depth(),
                    "state": session.state,
                    "xact_age_s": round(held.total_seconds(), 1),
                    "query": session.one_line,
                    "query_id": session.query_id,
                    "blocked_pids": [child.session.pid for child in root.children],
                },
                remediation=(
                    f"-- read the query above before doing this\n"
                    f"SELECT pg_cancel_backend({session.pid});"
                    + (
                        f"\n-- an idle transaction has nothing to cancel; only "
                        f"pg_terminate_backend({session.pid}) ends it"
                        if idle
                        else ""
                    )
                ),
            )
        )

    for session in sessions:
        if not session.state.startswith("idle in transaction"):
            continue
        age = session.age(now, session.state_change)
        if age < idle_in_transaction:
            continue
        aborted = session.state == "idle in transaction (aborted)"
        findings.append(
            Finding(
                check="locks.idle_in_transaction",
                severity=Severity.WARNING if age > 4 * idle_in_transaction else Severity.NOTICE,
                subject=f"pid {session.pid} ({session.user}@{session.application or '-'})",
                summary=(
                    f"idle in transaction for {_seconds(age)}"
                    + (" after an error" if aborted else "")
                    + "; holds locks and the xmin horizon"
                ),
                facts={
                    "pid": session.pid,
                    "state": session.state,
                    "idle_s": round(age.total_seconds(), 1),
                    "client": session.client,
                    "application": session.application,
                    "last_query": session.one_line,
                },
                remediation=(
                    "-- the fix is in the application: commit or roll back before going idle.\n"
                    "ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';  "
                    "-- server-side backstop"
                ),
            )
        )

    for session in sessions:
        if not session.blocked_by:
            continue
        waited = session.age(now, session.state_change)
        if waited < long_wait:
            continue
        findings.append(
            Finding(
                check="locks.long_wait",
                severity=Severity.WARNING,
                subject=f"pid {session.pid} ({session.user}@{session.application or '-'})",
                summary=(
                    f"waiting {_seconds(waited)} on "
                    f"{session.wait_event_type}:{session.wait_event}, blocked by "
                    f"{', '.join(str(pid) for pid in session.blocked_by)}"
                ),
                facts={
                    "pid": session.pid,
                    "waited_s": round(waited.total_seconds(), 1),
                    "wait_event_type": session.wait_event_type,
                    "wait_event": session.wait_event,
                    "blocked_by": list(session.blocked_by),
                    "query": session.one_line,
                },
                remediation="-- act on the root of the chain, not on the waiter",
            )
        )

    limits = db.one(CONNECTION_LIMITS_SQL)
    used = int(limits["used"])
    maximum = int(limits["max_connections"])
    if maximum and used >= maximum * connection_pressure:
        findings.append(
            Finding(
                check="locks.connection_pressure",
                severity=Severity.WARNING if used < maximum else Severity.CRITICAL,
                subject="cluster",
                summary=f"{used} of {maximum} connections in use, {limits['reserved']} reserved",
                facts={"used": used, "max_connections": maximum, "reserved": limits["reserved"]},
                remediation=(
                    "-- raising max_connections raises memory use per backend; a pooler in "
                    "transaction mode is the answer more often than a bigger number"
                ),
            )
        )
    return findings, forest


def _seconds(delta: timedelta) -> str:
    total = delta.total_seconds()
    if total < 90:
        return f"{total:.0f}s"
    if total < 5400:
        return f"{total / 60:.0f}m"
    return f"{total / 3600:.1f}h"

"""Vacuum, freezing and wraparound.

Three separate questions get answered here, and they fail in that order of
urgency:

1. how close is any relation to the wraparound horizon;
2. is autovacuum keeping up with dead tuples;
3. is something holding the xmin horizon open, which makes (2) unfixable by
   running VACUUM harder.

(3) is last in the file and first in most real incidents: a forgotten
``idle in transaction`` session, an orphaned prepared transaction or an inactive
replication slot means vacuum runs, reports success, and removes nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pgtk.db import Database, qualified
from pgtk.findings import Finding, Severity, human_bytes

RELATION_AGE_SQL = """
SELECT ns.nspname AS schemaname,
       c.relname AS relname,
       c.relkind,
       age(c.relfrozenxid)::bigint AS xid_age,
       mxid_age(c.relminmxid)::bigint AS mxid_age,
       pg_total_relation_size(c.oid)::bigint AS total_bytes,
       greatest(c.reltuples, 0)::float8 AS reltuples,
       coalesce(s.n_dead_tup, 0)::bigint AS n_dead_tup,
       coalesce(s.n_live_tup, 0)::bigint AS n_live_tup,
       {n_ins_since_vacuum} AS n_ins_since_vacuum,
       s.last_vacuum,
       s.last_autovacuum,
       array_to_string(coalesce(c.reloptions, '{{}}'), ' ') AS reloptions
FROM pg_class c
JOIN pg_namespace ns ON ns.oid = c.relnamespace
LEFT JOIN pg_stat_all_tables s ON s.relid = c.oid
WHERE c.relkind IN ('r', 'm', 't')
ORDER BY age(c.relfrozenxid) DESC
"""

DATABASE_AGE_SQL = """
SELECT datname,
       age(datfrozenxid)::bigint AS xid_age,
       mxid_age(datminmxid)::bigint AS mxid_age
FROM pg_database
WHERE datallowconn
ORDER BY age(datfrozenxid) DESC
"""

# Exactly the settings the checks below read. Fetching more would put values in
# the report that nothing compares anything against.
AUTOVACUUM_SETTINGS = (
    "autovacuum",
    "autovacuum_freeze_max_age",
    "autovacuum_multixact_freeze_max_age",
    "autovacuum_vacuum_threshold",
    "autovacuum_vacuum_scale_factor",
    "vacuum_failsafe_age",
)

SETTINGS_SQL = "SELECT name, setting FROM pg_settings WHERE name = ANY(%s)"

XMIN_HOLDERS_SQL = """
SELECT 'backend' AS source,
       coalesce(a.application_name, '') || ' pid=' || a.pid AS holder,
       age(a.backend_xmin)::bigint AS xid_age,
       a.state AS detail
FROM pg_stat_activity a
WHERE a.backend_xmin IS NOT NULL
UNION ALL
SELECT 'prepared_xact', p.gid, age(p.transaction)::bigint, p.owner
FROM pg_prepared_xacts p
UNION ALL
SELECT 'replication_slot',
       r.slot_name,
       greatest(age(r.xmin), age(r.catalog_xmin))::bigint,
       CASE WHEN r.active THEN 'active' ELSE 'inactive' END
FROM pg_replication_slots r
WHERE r.xmin IS NOT NULL OR r.catalog_xmin IS NOT NULL
ORDER BY 3 DESC NULLS LAST
"""

IN_PROGRESS_SQL = """
SELECT p.pid,
       p.datname,
       p.relid::regclass::text AS relation,
       p.phase,
       p.heap_blks_total::bigint AS heap_blks_total,
       p.heap_blks_scanned::bigint AS heap_blks_scanned,
       a.backend_type
FROM pg_stat_progress_vacuum p
JOIN pg_stat_activity a ON a.pid = p.pid
"""

# reloptions reach us as "a=1 b=2" from array_to_string, but the same text is
# printed by psql as "{a=1,b=2}"; both separators are accepted.
_RELOPTION = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s,}]+)")


@dataclass(frozen=True)
class VacuumSettings:
    autovacuum_on: bool
    freeze_max_age: int
    multixact_freeze_max_age: int
    vacuum_threshold: int
    vacuum_scale_factor: float
    failsafe_age: int

    @classmethod
    def load(cls, db: Database) -> VacuumSettings:
        raw = {
            row["name"]: row["setting"]
            for row in db.rows(SETTINGS_SQL, [list(AUTOVACUUM_SETTINGS)])
        }
        return cls(
            autovacuum_on=raw.get("autovacuum") == "on",
            freeze_max_age=int(raw.get("autovacuum_freeze_max_age", 200_000_000)),
            multixact_freeze_max_age=int(
                raw.get("autovacuum_multixact_freeze_max_age", 400_000_000)
            ),
            vacuum_threshold=int(raw.get("autovacuum_vacuum_threshold", 50)),
            vacuum_scale_factor=float(raw.get("autovacuum_vacuum_scale_factor", 0.2)),
            failsafe_age=int(raw.get("vacuum_failsafe_age", 1_600_000_000)),
        )


def parse_reloptions(text: str) -> dict[str, str]:
    """``{autovacuum_enabled=false,fillfactor=70}`` as it comes back from the catalog."""
    return {m.group("key"): m.group("value") for m in _RELOPTION.finditer(text.strip("{} "))}


def dead_tuple_threshold(reltuples: float, options: dict[str, str], s: VacuumSettings) -> float:
    """The number autovacuum itself compares ``n_dead_tup`` against.

    Reimplemented rather than approximated, because per-table reloptions are
    exactly the case where a global-settings approximation says "autovacuum
    should have run" about a table where somebody deliberately turned the knob.
    """
    base = float(options.get("autovacuum_vacuum_threshold", s.vacuum_threshold))
    factor = float(options.get("autovacuum_vacuum_scale_factor", s.vacuum_scale_factor))
    return base + factor * reltuples


def _age_severity(age: int, limit: int) -> Severity:
    share = age / limit if limit else 0.0
    if share >= 0.9:
        return Severity.CRITICAL
    if share >= 0.5:
        return Severity.WARNING
    return Severity.NOTICE


def _xmin_severity(age: int) -> Severity:
    if age >= 150_000_000:
        return Severity.CRITICAL
    if age >= 50_000_000:
        return Severity.WARNING
    return Severity.NOTICE


def analyze_vacuum(
    db: Database,
    *,
    wraparound_report_share: float = 0.5,
    stale_autovacuum: timedelta = timedelta(days=1),
    min_dead_tuples: int = 1000,
    xmin_report_age: int = 1_000_000,
) -> tuple[list[Finding], VacuumSettings]:
    settings = VacuumSettings.load(db)
    findings: list[Finding] = []

    if not settings.autovacuum_on:
        findings.append(
            Finding(
                check="vacuum.disabled",
                severity=Severity.CRITICAL,
                subject="cluster",
                summary="autovacuum is off cluster-wide; wraparound is now a manual duty",
                facts={"autovacuum": "off"},
                remediation="ALTER SYSTEM SET autovacuum = on; SELECT pg_reload_conf();",
            )
        )

    for row in db.rows(DATABASE_AGE_SQL):
        age = int(row["xid_age"])
        if age < settings.freeze_max_age * wraparound_report_share:
            continue
        findings.append(
            Finding(
                check="vacuum.database_wraparound",
                severity=_age_severity(age, settings.freeze_max_age),
                subject=f"database {row['datname']}",
                summary=(
                    f"datfrozenxid age {age:,} "
                    f"({age / settings.freeze_max_age * 100:.0f}% of "
                    f"autovacuum_freeze_max_age)"
                ),
                facts={
                    "xid_age": age,
                    "mxid_age": int(row["mxid_age"]),
                    "freeze_max_age": settings.freeze_max_age,
                    "failsafe_age": settings.failsafe_age,
                },
                remediation=(
                    "-- the whole database is only as young as its oldest relation; "
                    "find it in the vacuum.wraparound findings below"
                ),
            )
        )

    sql = RELATION_AGE_SQL.format(
        n_ins_since_vacuum=(
            "coalesce(s.n_ins_since_vacuum, 0)::bigint"
            if db.caps.has("pg_stat_all_tables.n_ins_since_vacuum")
            else "NULL::bigint"
        )
    )
    now = datetime.now(UTC)
    for row in db.rows(sql):
        findings.extend(
            _relation_findings(
                row,
                settings,
                now=now,
                stale_autovacuum=stale_autovacuum,
                min_dead_tuples=min_dead_tuples,
                wraparound_report_share=wraparound_report_share,
            )
        )

    for row in db.rows(XMIN_HOLDERS_SQL):
        age_value = row["xid_age"]
        if age_value is None or int(age_value) < xmin_report_age:
            continue
        age = int(age_value)
        findings.append(
            Finding(
                check="vacuum.xmin_horizon",
                severity=_xmin_severity(age),
                subject=f"{row['source']}: {row['holder']}",
                summary=(
                    f"holds the xmin horizon {age:,} transactions back; "
                    "vacuum cannot remove anything newer than this"
                ),
                facts={"source": row["source"], "detail": row["detail"], "xid_age": age},
                remediation=_xmin_remediation(str(row["source"]), str(row["holder"])),
            )
        )
    return findings, settings


def _xmin_remediation(source: str, holder: str) -> str:
    if source == "prepared_xact":
        return (
            f"ROLLBACK PREPARED '{holder}';  -- an orphaned 2PC transaction blocks vacuum "
            "cluster-wide until it is resolved"
        )
    if source == "replication_slot":
        return (
            "-- an inactive slot pins the horizon forever:\n"
            "SELECT pg_drop_replication_slot('<slot>');  -- only once you know the consumer "
            "is gone for good"
        )
    return (
        "-- find the session in pgtk locks; terminating it is the last resort, "
        "understanding why it is open is the first"
    )


def _relation_findings(
    row: dict[str, Any],
    settings: VacuumSettings,
    *,
    now: datetime,
    stale_autovacuum: timedelta,
    min_dead_tuples: int,
    wraparound_report_share: float,
) -> list[Finding]:
    findings: list[Finding] = []
    schema = str(row["schemaname"])
    name = str(row["relname"])
    subject = f"{schema}.{name}"
    xid_age = int(row["xid_age"])
    options = parse_reloptions(str(row["reloptions"]))

    if xid_age >= settings.freeze_max_age * wraparound_report_share:
        findings.append(
            Finding(
                check="vacuum.wraparound",
                severity=_age_severity(xid_age, settings.freeze_max_age),
                subject=subject,
                summary=(
                    f"relfrozenxid age {xid_age:,} "
                    f"({xid_age / settings.freeze_max_age * 100:.0f}% of "
                    f"autovacuum_freeze_max_age), {human_bytes(int(row['total_bytes']))}"
                ),
                facts={
                    "xid_age": xid_age,
                    "freeze_max_age": settings.freeze_max_age,
                    "last_autovacuum": _iso(row["last_autovacuum"]),
                    "relkind": row["relkind"],
                },
                remediation=(
                    f"VACUUM (FREEZE, VERBOSE) {qualified(schema, name)};"
                    "  -- an anti-wraparound autovacuum cannot be cancelled, "
                    "so do this on your own schedule"
                ),
            )
        )

    mxid_age = int(row["mxid_age"])
    if mxid_age >= settings.multixact_freeze_max_age * wraparound_report_share:
        findings.append(
            Finding(
                check="vacuum.multixact_wraparound",
                severity=_age_severity(mxid_age, settings.multixact_freeze_max_age),
                subject=subject,
                summary=(
                    f"relminmxid age {mxid_age:,} "
                    f"({mxid_age / settings.multixact_freeze_max_age * 100:.0f}% of "
                    "autovacuum_multixact_freeze_max_age)"
                ),
                facts={"mxid_age": mxid_age, "limit": settings.multixact_freeze_max_age},
                remediation=f"VACUUM (FREEZE) {qualified(schema, name)};",
            )
        )

    if row["relkind"] == "t" or schema in ("pg_catalog", "information_schema"):
        return findings

    dead = int(row["n_dead_tup"])
    threshold = dead_tuple_threshold(float(row["reltuples"]), options, settings)
    if dead >= max(min_dead_tuples, threshold):
        last_auto = row["last_autovacuum"]
        last_manual = row["last_vacuum"]
        latest = max(
            (value for value in (last_auto, last_manual) if value is not None),
            default=None,
        )
        stale = latest is None or (now - latest) > stale_autovacuum
        disabled = options.get("autovacuum_enabled") == "false"
        check = "vacuum.skipped" if (stale or disabled) else "vacuum.backlog"
        severity = Severity.WARNING if check == "vacuum.skipped" else Severity.NOTICE
        if disabled:
            severity = Severity.CRITICAL
        findings.append(
            Finding(
                check=check,
                severity=severity,
                subject=subject,
                summary=(
                    f"{dead:,} dead tuples against a threshold of {threshold:,.0f}"
                    + (
                        "; autovacuum_enabled=false on this table"
                        if disabled
                        else (
                            f"; last vacuumed {_ago(now, latest)}"
                            if latest is not None
                            else "; never vacuumed"
                        )
                    )
                ),
                facts={
                    "n_dead_tup": dead,
                    "n_live_tup": int(row["n_live_tup"]),
                    "threshold": round(threshold, 0),
                    "last_autovacuum": _iso(last_auto),
                    "last_vacuum": _iso(last_manual),
                    "n_ins_since_vacuum": row["n_ins_since_vacuum"],
                    "reloptions": options or None,
                    "total_bytes": int(row["total_bytes"]),
                },
                remediation=(
                    f"ALTER TABLE {qualified(schema, name)} SET (autovacuum_enabled = true);"
                    if disabled
                    else (
                        f"VACUUM (ANALYZE) {qualified(schema, name)};\n"
                        "-- if dead tuples do not drop afterwards, the xmin horizon is held "
                        "open — see vacuum.xmin_horizon"
                    )
                ),
            )
        )
    return findings


def vacuum_in_progress(db: Database) -> list[dict[str, Any]]:
    return db.rows(IN_PROGRESS_SQL)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _ago(now: datetime, then: datetime) -> str:
    delta = now - then
    if delta < timedelta(minutes=1):
        return "less than a minute ago"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)} min ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)} h ago"
    return f"{delta.days} days ago"

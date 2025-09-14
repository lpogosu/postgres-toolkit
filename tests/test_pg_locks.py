"""Blocking chains built by real sessions fighting over a real row."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from datetime import timedelta

import psycopg
import pytest

from pgtk.db import Database
from pgtk.locks import analyze_locks, connection_distribution, load_sessions
from pgtk.render import make_console, render_blocking_forest
from tests.conftest import PgInstance

pytestmark = pytest.mark.pg

ROW_ID = 1


class Chain:
    """One session holding a row lock and two more queueing behind it."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.holder = psycopg.connect(dsn, application_name="nightly-batch")
        self.waiters: list[psycopg.Connection[tuple[object, ...]]] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.holder.execute("UPDATE orders SET status = 'held' WHERE id = %s", (ROW_ID,))
        for position, name in enumerate(("checkout-api", "reporting"), start=1):
            self._queue(name)
            # The second waiter must arrive after the first is already waiting,
            # otherwise both queue directly behind the holder and the chain is
            # two levels deep instead of three.
            self._wait_until_blocked(position)

    def _queue(self, application: str) -> None:
        def run() -> None:
            conn = psycopg.connect(self.dsn, application_name=application)
            self.waiters.append(conn)
            # The statement is expected to be released by the holder's rollback,
            # but a cancelled or terminated waiter must not take the thread down.
            with contextlib.suppress(psycopg.Error):
                conn.execute("UPDATE orders SET status = %s WHERE id = %s", (application, ROW_ID))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.threads.append(thread)

    def _wait_until_blocked(self, expected: int) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with psycopg.connect(self.dsn, autocommit=True) as probe:
                blocked = probe.execute(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE cardinality(pg_blocking_pids(pid)) > 0"
                ).fetchone()
            if blocked is not None and int(blocked[0]) >= expected:
                return
            time.sleep(0.1)
        raise AssertionError(f"only saw fewer than {expected} blocked sessions")

    def release(self) -> None:
        self.holder.rollback()
        self.holder.close()
        for thread in self.threads:
            thread.join(timeout=20)
        for conn in self.waiters:
            conn.rollback()
            conn.close()


@pytest.fixture
def chain(seeded: PgInstance) -> Iterator[Chain]:
    built = Chain(seeded.dsn)
    built.start()
    try:
        yield built
    finally:
        built.release()


def test_the_chain_is_one_tree_rooted_at_the_session_holding_the_row(
    db: Database, chain: Chain
) -> None:
    _, forest = analyze_locks(db)
    assert len(forest) == 1
    root = forest[0]
    assert root.session.application == "nightly-batch"
    assert root.size() == 3
    assert root.depth() == 3


def test_the_root_is_reported_as_blocking_and_as_idle_rather_than_working(
    db: Database, chain: Chain
) -> None:
    findings, _ = analyze_locks(db, idle_in_transaction=timedelta(seconds=0))
    root = next(f for f in findings if f.check == "locks.blocking_root")
    assert root.facts["blocked_sessions"] == 2
    assert root.facts["state"] == "idle in transaction"
    assert "it is not running a query" in root.summary
    assert "pg_terminate_backend" in (root.remediation or "")


def test_the_waiters_are_told_to_act_on_the_root_instead(db: Database, chain: Chain) -> None:
    findings, _ = analyze_locks(db, long_wait=timedelta(seconds=0))
    waits = [f for f in findings if f.check == "locks.long_wait"]
    assert len(waits) == 2
    assert all("root of the chain" in (f.remediation or "") for f in waits)
    assert {f.facts["wait_event_type"] for f in waits} == {"Lock"}


def test_only_the_holder_is_a_root_even_though_a_waiter_also_blocks_someone(
    db: Database, chain: Chain
) -> None:
    findings, _ = analyze_locks(db)
    roots = [f for f in findings if f.check == "locks.blocking_root"]
    assert len(roots) == 1


def test_an_idle_transaction_younger_than_the_window_is_not_reported(
    db: Database, chain: Chain
) -> None:
    findings, _ = analyze_locks(db, idle_in_transaction=timedelta(hours=1))
    assert [f for f in findings if f.check == "locks.idle_in_transaction"] == []


def test_the_lock_the_waiter_is_queued_for_is_named_in_the_tree(
    db: Database, chain: Chain
) -> None:
    _, forest = analyze_locks(db)
    first_waiter = forest[0].children[0]
    assert "on" in first_waiter.waiting_for
    assert first_waiter.waiting_for.split()[0].endswith("Lock")


def test_the_rendered_tree_shows_every_session_in_the_chain(
    db: Database, chain: Chain
) -> None:
    console = make_console(force_plain=True)
    with console.capture() as capture:
        _, forest = analyze_locks(db)
        render_blocking_forest(console, forest)
    output = capture.get()
    assert "nightly-batch" in output
    assert "checkout-api" in output
    assert "reporting" in output
    assert output.count("waits") == 2


def test_a_quiet_server_produces_an_empty_forest(db: Database) -> None:
    _, forest = analyze_locks(db)
    assert forest == []


def test_the_tool_does_not_report_itself(db: Database) -> None:
    pids = {session.pid for session in load_sessions(db)}
    own = db.one("SELECT pg_backend_pid() AS pid")["pid"]
    assert own not in pids


def test_connections_are_grouped_by_who_opened_them(db: Database, chain: Chain) -> None:
    rows = connection_distribution(db)
    applications = {row["application_name"]: row for row in rows}
    assert "nightly-batch" in applications
    assert applications["nightly-batch"]["state"] == "idle in transaction"
    assert sum(int(row["sessions"]) for row in rows) >= 4


def test_connection_pressure_is_silent_on_an_idle_server(db: Database) -> None:
    findings, _ = analyze_locks(db)
    assert [f for f in findings if f.check == "locks.connection_pressure"] == []


def test_a_low_pressure_threshold_reports_the_real_connection_count(db: Database) -> None:
    findings, _ = analyze_locks(db, connection_pressure=0.0)
    pressure = next(f for f in findings if f.check == "locks.connection_pressure")
    assert pressure.facts["max_connections"] == 100
    assert 0 < int(pressure.facts["used"]) <= 100

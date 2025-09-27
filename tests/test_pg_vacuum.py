"""Vacuum, freezing and the xmin horizon against a live server.

One thing here is deliberately not tested against a real fixture: a relation
actually close to wraparound. Getting ``age(relfrozenxid)`` to 180 million means
committing 180 million transactions, which on this hardware is hours — the
measured rate on the fixture container is about 1 700 transactions per second,
because each one is a round trip. So the *query* and the finding it produces run
against real catalog values with the reporting threshold lowered, and the
severity ladder is unit-tested next door in ``test_units.py``.
"""

from __future__ import annotations

import contextlib
import threading
import time
from datetime import timedelta
from typing import Any

import psycopg
import pytest

from demo.pathology import Conn
from pgtk.db import Database
from pgtk.findings import Finding, Severity
from pgtk.vacuum import VacuumSettings, analyze_vacuum, vacuum_in_progress
from tests.conftest import PgInstance

pytestmark = pytest.mark.pg

# The phases pg_stat_progress_vacuum documents; the set has been stable since 12.
VACUUM_PHASES = frozenset(
    {
        "initializing",
        "scanning heap",
        "vacuuming indexes",
        "vacuuming heap",
        "cleaning up indexes",
        "truncating heap",
        "performing final cleanup",
    }
)


@pytest.fixture
def findings(db: Database) -> list[Finding]:
    return analyze_vacuum(db)[0]


def subjects(findings: list[Finding], check: str) -> set[str]:
    return {f.subject for f in findings if f.check == check}


def test_the_table_with_autovacuum_switched_off_is_reported_as_skipped(
    findings: list[Finding],
) -> None:
    # Other seeded tables drift in and out of this check depending on whether
    # autovacuum has reached them yet; audit_log cannot, because autovacuum is
    # switched off on it. That is the one this test is about.
    skipped = {f.subject: f for f in findings if f.check == "vacuum.skipped"}
    audit = skipped["public.audit_log"]
    assert audit.severity is Severity.CRITICAL
    assert audit.facts["reloptions"] == {"autovacuum_enabled": "false"}
    assert "SET (autovacuum_enabled = true)" in (audit.remediation or "")
    assert all(f.severity is Severity.WARNING for s, f in skipped.items() if s != audit.subject)


def test_the_backlog_is_measured_against_the_tables_own_threshold(
    findings: list[Finding],
) -> None:
    audit = next(f for f in findings if f.subject == "public.audit_log")
    assert audit.facts["n_dead_tup"] > audit.facts["threshold"]
    # 150 000 rows, default scale factor 0.2, default threshold 50.
    assert audit.facts["threshold"] == pytest.approx(30_050, rel=0.05)


def test_an_insert_only_table_produces_no_backlog_finding(findings: list[Finding]) -> None:
    assert "public.ledger" not in subjects(findings, "vacuum.skipped")
    assert "public.ledger" not in subjects(findings, "vacuum.backlog")


def test_the_settings_the_checks_use_come_from_the_server(db: Database) -> None:
    settings = VacuumSettings.load(db)
    assert settings.autovacuum_on
    assert settings.freeze_max_age == 200_000_000
    assert settings.vacuum_scale_factor == pytest.approx(0.2)


def test_wraparound_reads_real_relfrozenxid_ages_when_the_threshold_is_lowered(
    db: Database,
) -> None:
    findings, _ = analyze_vacuum(db, wraparound_report_share=1e-9)
    wraparound = {f.subject: f for f in findings if f.check == "vacuum.wraparound"}
    assert "public.orders" in wraparound
    assert wraparound["public.orders"].facts["xid_age"] > 0
    assert "VACUUM (FREEZE, VERBOSE) public.orders" in (
        wraparound["public.orders"].remediation or ""
    )


def test_wraparound_covers_the_catalog_and_toast_relations_too(db: Database) -> None:
    """The relation that wraps first is often not one anybody created."""
    findings, _ = analyze_vacuum(db, wraparound_report_share=1e-9)
    reported = subjects(findings, "vacuum.wraparound")
    assert any(subject.startswith("pg_catalog.") for subject in reported)
    assert any(subject.startswith("pg_toast.") for subject in reported)


def test_the_database_level_finding_appears_beside_the_relation_level_ones(
    db: Database,
) -> None:
    findings, _ = analyze_vacuum(db, wraparound_report_share=1e-9)
    assert any(f.check == "vacuum.database_wraparound" for f in findings)


def test_at_the_default_threshold_a_fresh_cluster_is_quiet_about_wraparound(
    findings: list[Finding],
) -> None:
    assert subjects(findings, "vacuum.wraparound") == set()
    assert subjects(findings, "vacuum.database_wraparound") == set()


def test_an_open_transaction_shows_up_as_an_xmin_horizon_holder(
    db: Database, seeded: PgInstance
) -> None:
    with psycopg.connect(seeded.dsn) as holder:
        holder.execute("SELECT count(*) FROM orders")  # opens a snapshot and keeps it
        findings, _ = analyze_vacuum(db, xmin_report_age=0)
        holders = [f for f in findings if f.check == "vacuum.xmin_horizon"]
        assert any(f.facts["source"] == "backend" for f in holders)
        holder.rollback()


def test_a_throttled_vacuum_is_visible_while_it_runs(db: Database, seeded: PgInstance) -> None:
    """pg_stat_progress_vacuum is empty almost always, so the vacuum is slowed down.

    The cost limit makes VACUUM sleep after a handful of pages, which is long
    enough for the progress query to catch it in a named phase.
    """
    started = threading.Event()

    def slow_vacuum() -> None:
        with psycopg.connect(seeded.dsn, autocommit=True) as runner:
            runner.execute("SET vacuum_cost_delay = 50")
            runner.execute("SET vacuum_cost_limit = 10")
            started.set()
            with contextlib.suppress(psycopg.errors.QueryCanceled):
                runner.execute("VACUUM (FREEZE, DISABLE_PAGE_SKIPPING) invoices")

    worker = threading.Thread(target=slow_vacuum, daemon=True)
    worker.start()
    assert started.wait(10)
    try:
        deadline = time.monotonic() + 20
        observed: list[dict[str, Any]] = []
        while time.monotonic() < deadline and not observed:
            observed = [row for row in vacuum_in_progress(db) if row["relation"] == "invoices"]
            if not observed:
                time.sleep(0.1)
        assert observed, "a throttled VACUUM should be visible in pg_stat_progress_vacuum"
        assert observed[0]["phase"] in VACUUM_PHASES
        assert observed[0]["pid"] != db.one("SELECT pg_backend_pid() AS pid")["pid"]
    finally:
        db.rows(
            "SELECT pg_cancel_backend(pid) FROM pg_stat_progress_vacuum "
            "WHERE relid = 'invoices'::regclass"
        )
        worker.join(timeout=30)


def test_lowering_the_stale_window_moves_a_table_from_backlog_to_skipped(
    db: Database, writable: Conn
) -> None:
    writable.execute("CREATE TABLE IF NOT EXISTS churn (id bigserial PRIMARY KEY, v int)")
    writable.execute("TRUNCATE churn")
    writable.execute("INSERT INTO churn (v) SELECT g FROM generate_series(1, 20000) AS g")
    writable.execute("ANALYZE churn")
    writable.execute("VACUUM churn")
    writable.execute("DELETE FROM churn WHERE id % 2 = 0")
    writable.execute("SELECT pg_stat_force_next_flush()")
    try:
        recent, _ = analyze_vacuum(db, stale_autovacuum=timedelta(days=1))
        stale, _ = analyze_vacuum(db, stale_autovacuum=timedelta(seconds=0))
        assert "public.churn" in subjects(recent, "vacuum.backlog")
        assert "public.churn" in subjects(stale, "vacuum.skipped")
    finally:
        writable.execute("DROP TABLE churn")

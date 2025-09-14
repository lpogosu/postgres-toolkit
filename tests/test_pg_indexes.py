"""Index hygiene against a live catalog.

The seeded schema contains one of each pathology plus indexes that are healthy in
a way that would trip a sloppier rule: a primary key nobody scans, an index that
is a prefix of another but enforces uniqueness, and a wide index that is used.
"""

from __future__ import annotations

import psycopg
import pytest

from demo.pathology import Conn
from pgtk.db import Database, connect
from pgtk.findings import Finding, Severity
from pgtk.indexes import analyze_indexes, load_indexes
from tests.conftest import PG_DATABASE, PgInstance

pytestmark = pytest.mark.pg


def subjects(findings: list[Finding], check: str) -> set[str]:
    return {f.subject for f in findings if f.check == check}


@pytest.fixture
def findings(db: Database) -> list[Finding]:
    return analyze_indexes(db)


def test_the_two_identical_indexes_are_reported_as_one_duplicate_pair(
    findings: list[Finding],
) -> None:
    duplicates = subjects(findings, "index.duplicate")
    assert duplicates == {"public.orders_placed_at_idx"}
    finding = next(f for f in findings if f.check == "index.duplicate")
    assert finding.facts["duplicate_of"] == "public.orders_placed_at_copy_idx"


def test_the_single_column_index_is_reported_as_a_prefix_of_the_two_column_one(
    findings: list[Finding],
) -> None:
    assert subjects(findings, "index.redundant") == {"public.orders_customer_idx"}
    finding = next(f for f in findings if f.check == "index.redundant")
    assert finding.facts["covered_by"] == "public.orders_customer_status_idx"


def test_the_failed_concurrent_build_is_reported_as_invalid_and_critical(
    findings: list[Finding],
) -> None:
    invalid = [f for f in findings if f.check == "index.invalid"]
    assert [f.subject for f in invalid] == ["public.coupons_code_uniq"]
    assert invalid[0].severity is Severity.CRITICAL
    assert "DROP INDEX CONCURRENTLY public.coupons_code_uniq" in (invalid[0].remediation or "")


def test_an_index_that_has_been_scanned_is_not_called_unused(findings: list[Finding]) -> None:
    unused = subjects(findings, "index.unused")
    assert "public.orders_region_idx" in unused
    assert "public.orders_customer_status_idx" not in unused


def test_primary_keys_are_never_reported_as_unused_however_idle_they_are(
    db: Database, findings: list[Finding]
) -> None:
    never_scanned = {
        index.subject
        for index in load_indexes(db)
        if index.idx_scan == 0 and index.is_primary and index.size_bytes > 1024 * 1024
    }
    assert never_scanned, "the fixture should contain an unscanned primary key"
    assert never_scanned & subjects(findings, "index.unused") == set()


def test_the_size_floor_keeps_tiny_indexes_out_of_the_report(db: Database) -> None:
    large_only = subjects(analyze_indexes(db, min_unused_bytes=4 * 1024 * 1024), "index.unused")
    everything = subjects(analyze_indexes(db, min_unused_bytes=0), "index.unused")
    assert large_only < everything


def test_the_repeatedly_scanned_table_is_a_missing_index_candidate(
    findings: list[Finding],
) -> None:
    candidates = subjects(findings, "index.missing_candidate")
    assert candidates == {"public.sessions"}
    finding = next(f for f in findings if f.check == "index.missing_candidate")
    assert finding.facts["rows_per_seq_scan"] > 100_000
    assert finding.facts["idx_scan"] == 0


def test_the_candidate_advice_sends_the_reader_to_the_queries_not_to_a_guessed_column(
    findings: list[Finding],
) -> None:
    finding = next(f for f in findings if f.check == "index.missing_candidate")
    assert "pg_stat_statements" in (finding.remediation or "")
    assert "CREATE INDEX" not in (finding.remediation or "")


def test_without_a_statistics_reset_point_unused_is_only_a_notice(
    findings: list[Finding],
) -> None:
    unused = [f for f in findings if f.check == "index.unused"]
    assert unused
    assert all(f.severity is Severity.NOTICE for f in unused)
    assert all(f.facts["stats_reset"] is None for f in unused)


def test_resetting_statistics_makes_a_used_index_look_unused(
    seeded: PgInstance, writable: Conn
) -> None:
    """The reason unused-index advice needs a date attached.

    A separate database is used so the reset cannot leak into the other tests.
    """
    scratch = f"{PG_DATABASE}_reset"
    writable.execute(f"DROP DATABASE IF EXISTS {scratch}")
    writable.execute(f"CREATE DATABASE {scratch}")
    dsn = seeded.dsn.replace(f"dbname={PG_DATABASE}", f"dbname={scratch}")
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE TABLE t (id bigserial PRIMARY KEY, k bigint NOT NULL)")
            conn.execute("INSERT INTO t (k) SELECT g FROM generate_series(1, 50000) AS g")
            conn.execute("CREATE INDEX t_k_idx ON t (k)")
            conn.execute("ANALYZE t")
            conn.execute("SET enable_seqscan = off")
            for value in range(50):
                conn.execute("SELECT count(*) FROM t WHERE k = %s", (value,))
            conn.execute("SELECT pg_stat_force_next_flush()")

            with connect(dsn) as handle:
                before = subjects(analyze_indexes(handle, min_unused_bytes=0), "index.unused")
            assert "public.t_k_idx" not in before

            conn.execute("SELECT pg_stat_reset()")
            conn.execute("SELECT pg_stat_force_next_flush()")
            with connect(dsn) as handle:
                after = analyze_indexes(handle, min_unused_bytes=0)
        unused = [f for f in after if f.check == "index.unused"]
        assert "public.t_k_idx" in {f.subject for f in unused}
        reset_finding = next(f for f in unused if f.subject == "public.t_k_idx")
        assert reset_finding.severity is Severity.WARNING
        assert reset_finding.facts["stats_reset"] is not None
    finally:
        writable.execute(f"DROP DATABASE IF EXISTS {scratch} WITH (FORCE)")


def test_dropping_an_index_is_always_advised_concurrently(findings: list[Finding]) -> None:
    for finding in findings:
        sql = finding.remediation or ""
        if "DROP INDEX" in sql:
            assert "DROP INDEX CONCURRENTLY" in sql, finding.subject


def test_the_unused_advice_warns_about_replicas(findings: list[Finding]) -> None:
    unused = next(f for f in findings if f.check == "index.unused")
    assert "replica" in (unused.remediation or "")

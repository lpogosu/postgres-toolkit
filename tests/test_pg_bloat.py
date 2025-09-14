"""Bloat against a live server, including how far the estimate is from the truth."""

from __future__ import annotations

import psycopg
import pytest

from demo.pathology import Conn
from pgtk.bloat import (
    BloatRow,
    PgstattupleUnavailableError,
    analyze_bloat,
    estimate_index_bloat,
    estimate_table_bloat,
    measure_exact,
)
from pgtk.db import Database, connect
from tests.conftest import PgInstance

pytestmark = pytest.mark.pg

# The seeded tables have 70% of their rows deleted, so both methods must land
# near 70. The window is wide enough to survive an autovacuum arriving mid-run
# and narrow enough that a broken formula fails it.
EXPECTED_BLOAT = (55.0, 85.0)


def by_subject(rows: list[BloatRow]) -> dict[str, BloatRow]:
    return {row.subject: row for row in rows}


def test_the_deleted_heap_is_reported_at_roughly_the_share_that_was_deleted(
    db: Database,
) -> None:
    rows = by_subject(estimate_table_bloat(db))
    shipments = rows["public.shipments"]
    assert EXPECTED_BLOAT[0] <= shipments.bloat_pct <= EXPECTED_BLOAT[1]
    assert shipments.bloat_bytes > 8 * 1024 * 1024
    assert not shipments.stats_unusable


def test_a_freshly_written_table_is_not_reported_as_bloated(db: Database) -> None:
    rows = by_subject(estimate_table_bloat(db))
    assert rows["public.ledger"].bloat_pct < 15.0
    assert rows["public.events"].bloat_pct < 15.0


def test_a_vacuumed_btree_keeps_its_pages_and_the_estimator_notices(db: Database) -> None:
    rows = by_subject(estimate_index_bloat(db))
    index = rows["public.invoices_number_idx"]
    assert EXPECTED_BLOAT[0] <= index.bloat_pct <= EXPECTED_BLOAT[1]
    assert index.parent == "invoices"


def test_indexes_built_a_moment_ago_are_reported_as_dense(db: Database) -> None:
    rows = by_subject(estimate_index_bloat(db))
    for name in ("public.orders_customer_status_idx", "public.orders_region_idx"):
        assert rows[name].bloat_pct < 20.0, name


def test_the_estimate_agrees_with_pgstattuple_within_a_stated_margin(db: Database) -> None:
    """The number the README quotes. If this window widens, the README is wrong."""
    _, estimated = analyze_bloat(db)
    measured = by_subject(measure_exact(db, estimated))
    assert estimated, "the fixture should produce bloat to compare"
    for row in estimated:
        exact = measured[row.subject]
        assert abs(row.bloat_pct - exact.bloat_pct) <= 6.0, (
            f"{row.subject}: estimate {row.bloat_pct:.1f}% vs measured {exact.bloat_pct:.1f}%"
        )


def test_exact_mode_labels_where_each_number_came_from(db: Database) -> None:
    _, estimated = analyze_bloat(db)
    measured = measure_exact(db, estimated)
    assert {row.method for row in estimated} == {"estimate"}
    assert {row.method for row in measured} <= {"pgstattuple", "pgstatindex"}


def test_pgstatindex_reports_leaf_density_for_the_bloated_index(db: Database) -> None:
    _, estimated = analyze_bloat(db)
    index = next(row for row in measure_exact(db, estimated) if row.kind == "index")
    assert 0.0 < float(index.extra["avg_leaf_density"]) < 50.0


def test_raising_the_thresholds_silences_the_report(db: Database) -> None:
    findings, rows = analyze_bloat(db, min_bloat_bytes=10 * 1024**3)
    assert (findings, rows) == ([], [])


def test_a_schema_filter_reaches_the_catalog_query(db: Database) -> None:
    assert estimate_table_bloat(db, schemas=("nonexistent_schema",)) == []
    assert estimate_table_bloat(db, schemas=("public",))


def test_the_finding_prints_the_rewrite_and_names_the_lock_it_takes(db: Database) -> None:
    findings, _ = analyze_bloat(db)
    heavy = next(f for f in findings if f.subject == "public.shipments")
    assert "VACUUM FULL public.shipments" in (heavy.remediation or "")
    assert "ACCESS EXCLUSIVE" in (heavy.remediation or "")
    assert "pg_repack" in (heavy.remediation or "")


def test_the_index_finding_prints_a_concurrent_reindex(db: Database) -> None:
    findings, _ = analyze_bloat(db)
    index = next(f for f in findings if f.check == "bloat.index")
    assert "REINDEX INDEX CONCURRENTLY public.invoices_number_idx" in (index.remediation or "")


def test_the_connection_the_tool_uses_cannot_write(db: Database) -> None:
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        db.rows("CREATE TABLE should_not_exist (id int)")


def test_exact_mode_says_so_when_the_extension_is_missing(
    seeded: PgInstance, writable: Conn
) -> None:
    writable.execute("DROP EXTENSION pgstattuple")
    try:
        with (
            connect(seeded.dsn) as without_extension,
            pytest.raises(PgstattupleUnavailableError, match="pgstattuple"),
        ):
            measure_exact(without_extension, [])
    finally:
        writable.execute("CREATE EXTENSION pgstattuple")

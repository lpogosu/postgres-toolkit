"""The version seams, checked against every image the session runs.

Catalog columns move between releases, and the failure mode is bad: the query
raises ``UndefinedColumn`` on the one server the user has, or — worse — a column
that is silently absent turns into a check that never fires. Both are covered
here:

* every column in ``CATALOG_ADDITIONS`` is looked up in the live catalog and its
  presence compared with what ``Capabilities.has`` claims;
* the pre-16 fallback SQL is forced on a modern server, so the branch nobody runs
  locally is executed on every test run.
"""

from __future__ import annotations

import pytest

from pgtk.db import CATALOG_ADDITIONS, Capabilities, Database, ServerVersion
from pgtk.indexes import analyze_indexes, load_indexes
from pgtk.vacuum import analyze_vacuum
from tests.conftest import PgInstance

pytestmark = pytest.mark.pg

COLUMN_EXISTS_SQL = """
SELECT EXISTS (
    SELECT 1 FROM pg_attribute
    WHERE attrelid = to_regclass(%s)
      AND attname = %s
      AND attnum > 0
      AND NOT attisdropped
) AS present
"""


def test_the_version_the_tool_reports_is_the_version_of_the_image(
    db: Database, seeded: PgInstance
) -> None:
    assert db.version.major == seeded.major


@pytest.mark.parametrize("column", sorted(CATALOG_ADDITIONS))
def test_capability_claims_match_the_live_catalog(db: Database, column: str) -> None:
    relation, attribute = column.rsplit(".", 1)
    present = bool(db.one(COLUMN_EXISTS_SQL, [relation, attribute])["present"])
    assert present is db.caps.has(column), (
        f"{column} is {'present' if present else 'absent'} on PostgreSQL {db.version}, "
        f"but CATALOG_ADDITIONS says it arrived in {CATALOG_ADDITIONS[column]}"
    )


def test_pg_stat_bgwriter_lost_its_checkpoint_columns_in_17(db: Database) -> None:
    """A seam this toolkit does not cross, recorded because it is the classic one.

    Checkpoint counters moved from ``pg_stat_bgwriter`` to ``pg_stat_checkpointer``
    in PostgreSQL 17. Nothing here reads them; the assertion exists so that the
    claim in the README is checked rather than remembered.
    """
    present = bool(db.one(COLUMN_EXISTS_SQL, ["pg_stat_bgwriter", "checkpoints_timed"])["present"])
    assert present is (db.version.major < 17)


def force_version(db: Database, major: int) -> None:
    db.caps = Capabilities(ServerVersion(major, 0), db.caps.extensions)


def test_the_pre_16_index_query_still_runs_and_reports_no_last_scan_time(
    db: Database,
) -> None:
    modern = {index.name: index for index in load_indexes(db)}
    force_version(db, 15)
    try:
        legacy = {index.name: index for index in load_indexes(db)}
    finally:
        force_version(db, db.version.major)

    assert modern.keys() == legacy.keys()
    assert all(index.last_idx_scan is None for index in legacy.values())
    assert any(index.last_idx_scan is not None for index in modern.values())
    # The verdict itself does not depend on the column, only the evidence does.
    assert {name for name, i in modern.items() if i.idx_scan == 0} == {
        name for name, i in legacy.items() if i.idx_scan == 0
    }


def test_the_pre_16_fallback_drops_the_column_from_the_findings_rather_than_lying(
    db: Database,
) -> None:
    modern = analyze_indexes(db)
    force_version(db, 15)
    try:
        legacy = analyze_indexes(db)
    finally:
        force_version(db, db.version.major)

    unused_modern = next(f for f in modern if f.check == "index.unused")
    unused_legacy = next(f for f in legacy if f.check == "index.unused")
    assert "last_idx_scan" in unused_modern.facts
    assert "last_idx_scan" not in unused_legacy.facts
    assert unused_modern.subject == unused_legacy.subject


def test_the_pre_13_vacuum_query_runs_without_the_insert_counter(db: Database) -> None:
    modern, _ = analyze_vacuum(db)
    force_version(db, 12)
    try:
        legacy, _ = analyze_vacuum(db)
    finally:
        force_version(db, db.version.major)

    audit_modern = next(f for f in modern if f.subject == "public.audit_log")
    audit_legacy = next(f for f in legacy if f.subject == "public.audit_log")
    assert audit_modern.facts["n_ins_since_vacuum"] is not None
    assert audit_legacy.facts["n_ins_since_vacuum"] is None
    assert audit_modern.check == audit_legacy.check

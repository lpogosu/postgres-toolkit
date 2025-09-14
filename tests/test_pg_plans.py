"""Plans produced by the server itself, not by hand.

``test_plans.py`` pins the rules; this file checks that the shapes the rules
expect are the shapes PostgreSQL actually emits — that ``Heap Fetches`` is spelled
that way, that a disk sort still reports ``Sort Space Type``, and that a plan can
be read back from ``EXPLAIN`` without any adjustment.
"""

from __future__ import annotations

import pytest

from demo.pathology import (
    EXTERNAL_SORT_QUERY,
    HASH_SPILL_QUERY,
    HEAP_FETCH_QUERY,
    MISESTIMATING_QUERY,
    SEQ_SCAN_FILTER_QUERY,
    Conn,
    explain_json,
)
from pgtk.findings import Severity
from pgtk.plans import analyze_plan, parse_explain

pytestmark = pytest.mark.pg


def checks(payload: str) -> set[str]:
    return {finding.check for finding in analyze_plan(parse_explain(payload))}


def test_two_correlated_columns_produce_a_misestimate_the_reader_can_act_on(
    writable: Conn,
) -> None:
    payload = explain_json(writable, MISESTIMATING_QUERY)
    findings = analyze_plan(parse_explain(payload))
    misestimates = [f for f in findings if f.check == "plan.misestimate"]
    assert misestimates
    worst = max(misestimates, key=lambda f: float(f.facts["ratio"]))
    assert float(worst.facts["ratio"]) >= 10
    assert worst.facts["direction"] == "under"


def test_the_underestimate_turns_into_a_nested_loop_over_a_large_outer_side(
    writable: Conn,
) -> None:
    payload = explain_json(writable, MISESTIMATING_QUERY)
    findings = analyze_plan(parse_explain(payload))
    loop = next(f for f in findings if f.check == "plan.nested_loop")
    assert float(loop.facts["loops"]) >= 10_000
    assert loop.severity in (Severity.WARNING, Severity.CRITICAL)


def test_an_index_only_scan_on_an_unvacuumed_table_reports_full_heap_access(
    writable: Conn,
) -> None:
    payload = explain_json(writable, HEAP_FETCH_QUERY)
    findings = analyze_plan(parse_explain(payload))
    fetches = next(f for f in findings if f.check == "plan.heap_fetches")
    assert fetches.facts["heap_fetch_share"] == pytest.approx(1.0, abs=0.01)
    assert fetches.facts["relation"] == "ledger"


def test_a_sort_that_does_not_fit_in_work_mem_is_reported_with_its_size(
    writable: Conn,
) -> None:
    payload = explain_json(writable, EXTERNAL_SORT_QUERY, work_mem="64kB")
    findings = analyze_plan(parse_explain(payload))
    sort = next(f for f in findings if f.check == "plan.external_sort")
    assert "external" in str(sort.facts["sort_method"])
    assert int(sort.facts["sort_space_kb"]) > 1024


def test_the_same_sort_with_enough_memory_produces_no_finding(writable: Conn) -> None:
    payload = explain_json(writable, EXTERNAL_SORT_QUERY, work_mem="256MB")
    assert "plan.external_sort" not in checks(payload)


def test_a_hash_join_that_spills_reports_more_batches_than_planned(writable: Conn) -> None:
    payload = explain_json(writable, HASH_SPILL_QUERY, work_mem="64kB")
    findings = analyze_plan(parse_explain(payload))
    spill = next(f for f in findings if f.check == "plan.hash_spill")
    assert int(spill.facts["hash_batches"]) > 1


def test_a_sequential_scan_that_throws_away_everything_is_reported(writable: Conn) -> None:
    payload = explain_json(writable, SEQ_SCAN_FILTER_QUERY)
    findings = analyze_plan(parse_explain(payload))
    discard = next(f for f in findings if f.check == "plan.filter_discard")
    assert float(discard.facts["rows_removed"]) > 10_000
    assert discard.facts["rows_returned"] == 0.0


def test_a_plan_from_the_server_round_trips_without_adjustment(writable: Conn) -> None:
    document = parse_explain(explain_json(writable, HEAP_FETCH_QUERY))
    assert document.execution_ms is not None
    assert document.planning_ms is not None
    assert document.root.node_type == "Aggregate"


def test_a_trivial_query_produces_no_findings_at_all(writable: Conn) -> None:
    assert checks(explain_json(writable, "SELECT 1")) == set()

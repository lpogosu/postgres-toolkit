"""Plan reading, on hand-built JSON.

These are the cases that decide whether a finding is right or noise, so they are
written as plans rather than captured from a server: a captured plan cannot be
edited to sit exactly on a threshold.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pgtk.findings import Severity
from pgtk.plans import (
    PlanFormatError,
    PlanNotAnalyzedError,
    analyze_plan,
    misestimate_ratio,
    parse_explain,
)


def wrap(plan: dict[str, Any], **top: Any) -> str:
    document: dict[str, Any] = {"Plan": plan, "Planning Time": 0.4, "Execution Time": 12.0}
    document.update(top)
    return json.dumps([document])


def scan(**overrides: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "Node Type": "Seq Scan",
        "Relation Name": "orders",
        "Alias": "orders",
        "Plan Rows": 100,
        "Actual Rows": 100,
        "Actual Loops": 1,
    }
    node.update(overrides)
    return node


def checks(findings: list[Any]) -> set[str]:
    return {f.check for f in findings}


def test_a_one_element_array_a_bare_object_and_a_query_plan_wrapper_all_parse() -> None:
    plan = scan()
    bare = json.dumps({"Plan": plan})
    wrapped = json.dumps({"QUERY PLAN": [{"Plan": plan}]})
    for payload in (wrap(plan), bare, wrapped):
        assert parse_explain(payload).root.node_type == "Seq Scan"


def test_text_format_explains_itself_instead_of_raising_a_json_error() -> None:
    with pytest.raises(PlanFormatError, match="FORMAT JSON"):
        parse_explain("Seq Scan on orders  (cost=0.00..1.00 rows=1 width=4)")


def test_a_plan_without_analyze_is_rejected_by_name() -> None:
    plan = {"Node Type": "Seq Scan", "Plan Rows": 10}
    with pytest.raises(PlanNotAnalyzedError, match="ANALYZE"):
        parse_explain(json.dumps([{"Plan": plan}]))


def test_node_paths_identify_a_node_uniquely_in_a_branching_plan() -> None:
    doc = parse_explain(
        wrap(
            {
                "Node Type": "Hash Join",
                "Plan Rows": 1,
                "Actual Rows": 1,
                "Actual Loops": 1,
                "Plans": [scan(**{"Relation Name": "a"}), scan(**{"Relation Name": "b"})],
            }
        )
    )
    assert [node.path for node in doc.nodes()] == ["0", "0.0", "0.1"]


def test_misestimate_is_symmetric_and_never_below_one() -> None:
    assert misestimate_ratio(parse_explain(wrap(scan())).root) == 1.0
    over = parse_explain(wrap(scan(**{"Plan Rows": 1000, "Actual Rows": 10}))).root
    under = parse_explain(wrap(scan(**{"Plan Rows": 10, "Actual Rows": 1000}))).root
    assert misestimate_ratio(over) == misestimate_ratio(under) == 100.0


def test_a_large_ratio_on_three_rows_is_not_reported() -> None:
    """Ten rows where one was expected is a factor of ten and means nothing."""
    doc = parse_explain(wrap(scan(**{"Plan Rows": 1, "Actual Rows": 10})))
    assert analyze_plan(doc) == []


def test_a_hundredfold_misestimate_is_critical_and_a_tenfold_one_is_not() -> None:
    mild = parse_explain(wrap(scan(**{"Plan Rows": 100, "Actual Rows": 1500})))
    severe = parse_explain(wrap(scan(**{"Plan Rows": 100, "Actual Rows": 15000})))
    assert analyze_plan(mild)[0].severity is Severity.WARNING
    assert analyze_plan(severe)[0].severity is Severity.CRITICAL


def test_a_filtered_underestimate_suggests_extended_statistics_and_a_plain_one_does_not() -> None:
    filtered = parse_explain(
        wrap(scan(**{"Plan Rows": 20, "Actual Rows": 4000, "Filter": "(a = 1 AND b = 1)"}))
    )
    plain = parse_explain(wrap(scan(**{"Plan Rows": 20, "Actual Rows": 4000})))
    assert "CREATE STATISTICS" in (analyze_plan(filtered)[0].remediation or "")
    assert "CREATE STATISTICS" not in (analyze_plan(plain)[0].remediation or "")


def test_an_overestimate_never_suggests_extended_statistics() -> None:
    doc = parse_explain(
        wrap(scan(**{"Plan Rows": 40000, "Actual Rows": 100, "Filter": "(a = 1 AND b = 1)"}))
    )
    finding = analyze_plan(doc)[0]
    assert finding.facts["direction"] == "over"
    assert "CREATE STATISTICS" not in (finding.remediation or "")


def nested_loop(loops: int) -> str:
    return wrap(
        {
            "Node Type": "Nested Loop",
            "Plan Rows": 50,
            "Actual Rows": loops,
            "Actual Loops": 1,
            "Plans": [
                scan(**{"Plan Rows": 50, "Actual Rows": loops}),
                {
                    "Node Type": "Index Scan",
                    "Index Name": "orders_pkey",
                    "Plan Rows": 1,
                    "Actual Rows": 1,
                    "Actual Loops": loops,
                },
            ],
        }
    )


def test_a_nested_loop_is_judged_by_its_inner_loop_count_not_its_own_rows() -> None:
    assert "plan.nested_loop" not in checks(analyze_plan(parse_explain(nested_loop(500))))
    assert "plan.nested_loop" in checks(analyze_plan(parse_explain(nested_loop(50_000))))


def test_the_nested_loop_finding_points_at_the_outer_estimate_as_the_cause() -> None:
    finding = next(
        f for f in analyze_plan(parse_explain(nested_loop(50_000))) if f.check == "plan.nested_loop"
    )
    assert finding.facts["outer_estimate_ratio"] == 1000.0
    assert "disabling the join type" in (finding.remediation or "")


def test_a_sort_in_memory_is_silent_and_one_on_disk_is_not() -> None:
    memory = scan(
        **{
            "Node Type": "Sort",
            "Sort Method": "quicksort",
            "Sort Space Type": "Memory",
            "Sort Space Used": 900,
        }
    )
    disk = scan(
        **{
            "Node Type": "Sort",
            "Sort Method": "external merge",
            "Sort Space Type": "Disk",
            "Sort Space Used": 43_008,
        }
    )
    assert analyze_plan(parse_explain(wrap(memory))) == []
    finding = analyze_plan(parse_explain(wrap(disk)))[0]
    assert finding.check == "plan.external_sort"
    assert "42.0 MB" in finding.summary


def test_a_hash_that_used_more_batches_than_planned_outranks_one_that_did_not() -> None:
    as_planned = scan(**{"Node Type": "Hash", "Hash Batches": 4, "Original Hash Batches": 4})
    spilled = scan(**{"Node Type": "Hash", "Hash Batches": 16, "Original Hash Batches": 1})
    assert analyze_plan(parse_explain(wrap(as_planned)))[0].severity is Severity.NOTICE
    assert analyze_plan(parse_explain(wrap(spilled)))[0].severity is Severity.WARNING


def test_a_single_batch_hash_is_not_a_finding() -> None:
    assert analyze_plan(parse_explain(wrap(scan(**{"Node Type": "Hash", "Hash Batches": 1})))) == []


def test_heap_fetch_severity_follows_the_share_of_rows_not_the_count() -> None:
    def index_only(rows: int, fetches: int) -> Any:
        return parse_explain(
            wrap(
                {
                    "Node Type": "Index Only Scan",
                    "Index Name": "ledger_pkey",
                    "Relation Name": "ledger",
                    "Plan Rows": rows,
                    "Actual Rows": rows,
                    "Actual Loops": 1,
                    "Heap Fetches": fetches,
                }
            )
        )

    occasional = analyze_plan(index_only(100_000, 500))[0]
    total = analyze_plan(index_only(100_000, 100_000))[0]
    assert occasional.severity is Severity.NOTICE
    assert total.severity is Severity.WARNING
    assert "VACUUM (ANALYZE) ledger" in (total.remediation or "")


def test_a_healthy_index_only_scan_produces_nothing() -> None:
    doc = parse_explain(
        wrap(
            {
                "Node Type": "Index Only Scan",
                "Index Name": "ledger_pkey",
                "Plan Rows": 5000,
                "Actual Rows": 5000,
                "Actual Loops": 1,
                "Heap Fetches": 0,
            }
        )
    )
    assert analyze_plan(doc) == []


def test_a_filter_that_discards_almost_everything_is_reported_once_it_is_big_enough() -> None:
    small = scan(**{"Actual Rows": 1, "Rows Removed by Filter": 900, "Filter": "(x = 1)"})
    large = scan(**{"Actual Rows": 1, "Rows Removed by Filter": 120_000, "Filter": "(x = 1)"})
    assert "plan.filter_discard" not in checks(analyze_plan(parse_explain(wrap(small))))
    finding = next(
        f
        for f in analyze_plan(parse_explain(wrap(large)))
        if f.check == "plan.filter_discard"
    )
    assert finding.facts["rows_removed"] == 120_000


def test_a_selective_filter_that_keeps_most_rows_is_not_a_finding() -> None:
    node = scan(**{"Actual Rows": 100_000, "Rows Removed by Filter": 20_000, "Filter": "(x = 1)"})
    assert "plan.filter_discard" not in checks(analyze_plan(parse_explain(wrap(node))))


def test_planning_and_execution_time_survive_parsing() -> None:
    doc = parse_explain(wrap(scan(), **{"Planning Time": 1.25, "Execution Time": 800.5}))
    assert (doc.planning_ms, doc.execution_ms) == (1.25, 800.5)


def test_the_label_keeps_the_alias_only_when_it_differs_from_the_relation() -> None:
    same = parse_explain(wrap(scan(**{"Relation Name": "orders", "Alias": "orders"}))).root
    aliased = parse_explain(wrap(scan(**{"Relation Name": "orders", "Alias": "o"}))).root
    assert same.label == "Seq Scan on orders"
    assert aliased.label == "Seq Scan on orders (o)"

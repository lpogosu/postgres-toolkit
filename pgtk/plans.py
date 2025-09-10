"""Read ``EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`` and say what is wrong with it.

This is a static reader: it takes the JSON a human already produced and applies
the handful of rules that account for most of what goes wrong. It does not
connect to a database, which is why ``pgtk plan`` works on a plan pasted from an
incident channel at 3am when the database in question is on the other side of a
VPN.

What it deliberately cannot do is in the README; the short version is that a plan
tells you what happened once, not what happens usually.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pgtk.findings import Finding, Severity, human_bytes


class PlanFormatError(ValueError):
    pass


class PlanNotAnalyzedError(ValueError):
    pass


@dataclass(frozen=True)
class PlanNode:
    node_type: str
    path: str
    raw: dict[str, Any]
    children: tuple[PlanNode, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        target = self.raw.get("Index Name") or self.raw.get("Relation Name")
        alias = self.raw.get("Alias")
        if target and alias and alias != target:
            return f"{self.node_type} on {target} ({alias})"
        if target:
            return f"{self.node_type} on {target}"
        return self.node_type

    @property
    def subject(self) -> str:
        return f"#{self.path} {self.label}"

    @property
    def plan_rows(self) -> float:
        return float(self.raw.get("Plan Rows", 0))

    @property
    def actual_rows(self) -> float:
        """Rows produced *per loop*, which is how PostgreSQL reports it."""
        return float(self.raw.get("Actual Rows", 0))

    @property
    def loops(self) -> float:
        return float(self.raw.get("Actual Loops", 1))

    @property
    def total_rows(self) -> float:
        return self.actual_rows * self.loops

    def walk(self) -> Iterator[PlanNode]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class ExplainDocument:
    root: PlanNode
    planning_ms: float | None
    execution_ms: float | None

    def nodes(self) -> list[PlanNode]:
        return list(self.root.walk())


def _build(raw: dict[str, Any], path: str) -> PlanNode:
    children = tuple(
        _build(child, f"{path}.{i}") for i, child in enumerate(raw.get("Plans", []) or [])
    )
    return PlanNode(
        node_type=str(raw.get("Node Type", "unknown")),
        path=path,
        raw=raw,
        children=children,
    )


def parse_explain(payload: str) -> ExplainDocument:
    """Accept what ``psql`` and the drivers actually hand you.

    ``EXPLAIN (FORMAT JSON)`` returns a one-element array; some tools unwrap it,
    some wrap it again under ``QUERY PLAN``. All three shapes arrive here.
    """
    text = payload.strip()
    if not text:
        raise PlanFormatError("empty input")
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanFormatError(
            "input is not JSON — run EXPLAIN with (ANALYZE, BUFFERS, FORMAT JSON); "
            "the default text format is not machine readable"
        ) from exc

    while isinstance(data, list):
        if not data:
            raise PlanFormatError("JSON array is empty")
        data = data[0]
    if isinstance(data, dict) and "QUERY PLAN" in data:
        inner: Any = data["QUERY PLAN"]
        while isinstance(inner, list) and inner:
            inner = inner[0]
        data = inner
    if not isinstance(data, dict) or "Plan" not in data:
        raise PlanFormatError("no 'Plan' key found; this does not look like EXPLAIN FORMAT JSON")

    root = _build(dict(data["Plan"]), "0")
    if "Actual Rows" not in root.raw:
        raise PlanNotAnalyzedError(
            "the plan has estimates but no measurements — re-run with "
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON), which executes the query"
        )
    return ExplainDocument(
        root=root,
        planning_ms=_optional_float(data.get("Planning Time")),
        execution_ms=_optional_float(data.get("Execution Time")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def misestimate_ratio(node: PlanNode) -> float:
    """How far the planner was out, in either direction, as a factor >= 1."""
    estimated = max(node.plan_rows, 1.0)
    actual = max(node.actual_rows, 1.0)
    return max(estimated / actual, actual / estimated)


SCAN_NODES = frozenset({"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan"})


def _misestimate_findings(doc: ExplainDocument, factor: float, floor_rows: int) -> list[Finding]:
    findings: list[Finding] = []
    for node in doc.nodes():
        if node.total_rows < floor_rows and node.plan_rows < floor_rows:
            continue
        ratio = misestimate_ratio(node)
        if ratio < factor:
            continue
        under = node.actual_rows > node.plan_rows
        severity = Severity.CRITICAL if ratio >= 100 else Severity.WARNING
        direction = "under" if under else "over"
        remediation = (
            f"ANALYZE {node.raw['Relation Name']};"
            if node.raw.get("Relation Name")
            else "-- no single relation: the error is in a join or aggregate estimate"
        )
        if under and node.node_type in SCAN_NODES and node.raw.get("Filter"):
            remediation += (
                "\n-- a filter on several columns of one table underestimates when the "
                "columns correlate:\n"
                "-- CREATE STATISTICS ... (dependencies, ndistinct) ON col_a, col_b FROM tbl; "
                "then ANALYZE tbl;"
            )
        findings.append(
            Finding(
                check="plan.misestimate",
                severity=severity,
                subject=node.subject,
                summary=(
                    f"planner {direction}estimated by {ratio:.0f}x: "
                    f"expected {node.plan_rows:,.0f} rows per loop, got {node.actual_rows:,.0f}"
                    + (f" over {node.loops:,.0f} loops" if node.loops > 1 else "")
                ),
                facts={
                    "node_type": node.node_type,
                    "plan_rows": node.plan_rows,
                    "actual_rows": node.actual_rows,
                    "loops": node.loops,
                    "ratio": round(ratio, 1),
                    "direction": direction,
                    "filter": node.raw.get("Filter"),
                },
                remediation=remediation,
            )
        )
    return findings


def _nested_loop_findings(doc: ExplainDocument, loop_threshold: int) -> list[Finding]:
    findings: list[Finding] = []
    for node in doc.nodes():
        if node.node_type != "Nested Loop" or not node.children:
            continue
        inner = node.children[-1]
        if inner.loops < loop_threshold:
            continue
        outer = node.children[0]
        outer_misestimate = misestimate_ratio(outer)
        findings.append(
            Finding(
                check="plan.nested_loop",
                severity=(
                    Severity.CRITICAL if inner.loops >= 10 * loop_threshold else Severity.WARNING
                ),
                subject=node.subject,
                summary=(
                    f"inner side executed {inner.loops:,.0f} times "
                    f"({inner.label}); the loop count is the outer row count"
                ),
                facts={
                    "loops": inner.loops,
                    "inner": inner.label,
                    "outer": outer.label,
                    "outer_estimate_ratio": round(outer_misestimate, 1),
                    "node_total_ms": node.raw.get("Actual Total Time"),
                },
                remediation=(
                    "-- a nested loop is only wrong when the outer side is bigger than planned;\n"
                    f"-- here the outer estimate was off by {outer_misestimate:.0f}x, "
                    "so fix that first rather than disabling the join type."
                ),
            )
        )
    return findings


def _sort_findings(doc: ExplainDocument) -> list[Finding]:
    findings: list[Finding] = []
    for node in doc.nodes():
        method = str(node.raw.get("Sort Method", ""))
        space_type = str(node.raw.get("Sort Space Type", ""))
        if space_type.lower() != "disk" and "external" not in method.lower():
            continue
        used_kb = int(node.raw.get("Sort Space Used", 0))
        findings.append(
            Finding(
                check="plan.external_sort",
                severity=Severity.WARNING,
                subject=node.subject,
                summary=(
                    f"sort spilled to disk ({method or 'external'}), "
                    f"{human_bytes(used_kb * 1024)} written per loop"
                ),
                facts={
                    "sort_method": method,
                    "sort_space_kb": used_kb,
                    "sort_key": node.raw.get("Sort Key"),
                    "loops": node.loops,
                },
                remediation=(
                    f"-- this sort needed more than work_mem; "
                    f"{human_bytes(used_kb * 1024)} on disk per loop\n"
                    "SET work_mem = '<size>';  -- per sort, per backend: multiply before "
                    "raising it globally"
                ),
            )
        )
    return findings


def _hash_spill_findings(doc: ExplainDocument) -> list[Finding]:
    findings: list[Finding] = []
    for node in doc.nodes():
        batches = int(node.raw.get("Hash Batches", 1) or 1)
        original = int(node.raw.get("Original Hash Batches", batches) or batches)
        if batches <= 1:
            continue
        findings.append(
            Finding(
                check="plan.hash_spill",
                severity=Severity.WARNING if batches > original else Severity.NOTICE,
                subject=node.subject,
                summary=(
                    f"hash table used {batches} batches "
                    f"(planned {original}); the build side did not fit in work_mem"
                ),
                facts={
                    "hash_batches": batches,
                    "original_hash_batches": original,
                    "peak_memory_kb": node.raw.get("Peak Memory Usage"),
                },
                remediation=(
                    "-- batches above the planned number mean the build side was larger than "
                    "estimated;\n-- check the estimate on the hashed relation before raising "
                    "work_mem."
                ),
            )
        )
    return findings


def _heap_fetch_findings(doc: ExplainDocument) -> list[Finding]:
    findings: list[Finding] = []
    for node in doc.nodes():
        if node.node_type != "Index Only Scan":
            continue
        fetches = float(node.raw.get("Heap Fetches", 0))
        if fetches <= 0:
            continue
        rows = max(node.total_rows, 1.0)
        share = fetches / rows
        relation = node.raw.get("Relation Name", "")
        findings.append(
            Finding(
                check="plan.heap_fetches",
                severity=Severity.WARNING if share >= 0.1 else Severity.NOTICE,
                subject=node.subject,
                summary=(
                    f"index-only scan fell back to the heap for {fetches:,.0f} of "
                    f"{rows:,.0f} rows ({share * 100:.0f}%): the visibility map is stale"
                ),
                facts={
                    "heap_fetches": fetches,
                    "rows": rows,
                    "heap_fetch_share": round(share, 3),
                    "relation": relation,
                },
                remediation=(
                    f"VACUUM (ANALYZE) {relation};  -- sets all-visible bits so the scan can "
                    f"skip the heap\n"
                    "-- recurring: lower autovacuum_vacuum_scale_factor on this table"
                    if relation
                    else "-- vacuum the underlying table to refresh its visibility map"
                ),
            )
        )
    return findings


def _filter_findings(doc: ExplainDocument, discard_ratio: float, floor_rows: int) -> list[Finding]:
    findings: list[Finding] = []
    for node in doc.nodes():
        removed = float(node.raw.get("Rows Removed by Filter", 0))
        if removed < floor_rows:
            continue
        kept = max(node.actual_rows, 1.0)
        if removed / kept < discard_ratio:
            continue
        findings.append(
            Finding(
                check="plan.filter_discard",
                severity=Severity.NOTICE,
                subject=node.subject,
                summary=(
                    f"read {removed + node.actual_rows:,.0f} rows per loop to return "
                    f"{node.actual_rows:,.0f}; {removed:,.0f} discarded by the filter"
                ),
                facts={
                    "rows_removed": removed,
                    "rows_returned": node.actual_rows,
                    "filter": node.raw.get("Filter"),
                    "node_type": node.node_type,
                },
                remediation=(
                    "-- the predicate is evaluated after the rows are read; an index on the "
                    "filtered\n-- column(s) moves the work into the access method:\n"
                    f"-- Filter: {node.raw.get('Filter')}"
                ),
            )
        )
    return findings


def analyze_plan(
    doc: ExplainDocument,
    *,
    misestimate_factor: float = 10.0,
    misestimate_floor_rows: int = 100,
    nested_loop_threshold: int = 10_000,
    filter_discard_ratio: float = 10.0,
    filter_floor_rows: int = 10_000,
) -> list[Finding]:
    return [
        *_misestimate_findings(doc, misestimate_factor, misestimate_floor_rows),
        *_nested_loop_findings(doc, nested_loop_threshold),
        *_sort_findings(doc),
        *_hash_spill_findings(doc),
        *_heap_fetch_findings(doc),
        *_filter_findings(doc, filter_discard_ratio, filter_floor_rows),
    ]

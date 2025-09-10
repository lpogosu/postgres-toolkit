"""Index hygiene: unused, duplicate, redundant, invalid, and missing candidates.

The redundancy analysis is done in Python rather than SQL on purpose. Deciding
that index A is covered by index B is a comparison of two ordered key vectors
plus four side conditions, and expressing that as a self-join with array slicing
produces SQL nobody can review. Here it is a function with a table of cases in
the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pgtk.db import Database, qualified
from pgtk.findings import Finding, Severity, human_bytes

INDEX_INVENTORY_SQL = """
SELECT ns.nspname AS schemaname,
       tc.relname AS tblname,
       ic.relname AS idxname,
       i.indexrelid,
       i.indnkeyatts,
       i.indkey::text AS indkey,
       i.indclass::text AS indclass,
       i.indoption::text AS indoption,
       am.amname,
       i.indisunique,
       i.indisprimary,
       i.indisreplident,
       i.indisvalid,
       i.indisready,
       coalesce(pg_get_expr(i.indpred, i.indrelid), '') AS predicate,
       coalesce(pg_get_expr(i.indexprs, i.indrelid), '') AS expressions,
       pg_relation_size(i.indexrelid)::bigint AS index_bytes,
       pg_get_indexdef(i.indexrelid) AS definition,
       EXISTS (SELECT 1 FROM pg_constraint c WHERE c.conindid = i.indexrelid) AS backs_constraint,
       s.idx_scan::bigint AS idx_scan,
       {last_idx_scan} AS last_idx_scan
FROM pg_index i
JOIN pg_class ic ON ic.oid = i.indexrelid
JOIN pg_class tc ON tc.oid = i.indrelid
JOIN pg_namespace ns ON ns.oid = ic.relnamespace
JOIN pg_am am ON am.oid = ic.relam
LEFT JOIN pg_stat_all_indexes s ON s.indexrelid = i.indexrelid
WHERE ns.nspname NOT IN ('pg_catalog', 'information_schema')
  AND (cardinality(%s::text[]) = 0 OR ns.nspname = ANY(%s::text[]))
ORDER BY 1, 2, 3
"""

SEQ_SCAN_SQL = """
SELECT ns.nspname AS schemaname,
       c.relname AS tblname,
       coalesce(s.seq_scan, 0)::bigint AS seq_scan,
       coalesce(s.seq_tup_read, 0)::bigint AS seq_tup_read,
       coalesce(s.idx_scan, 0)::bigint AS idx_scan,
       coalesce(c.reltuples, 0)::float8 AS reltuples,
       pg_relation_size(c.oid)::bigint AS table_bytes,
       {last_seq_scan} AS last_seq_scan
FROM pg_stat_all_tables s
JOIN pg_class c ON c.oid = s.relid
JOIN pg_namespace ns ON ns.oid = c.relnamespace
WHERE c.relkind IN ('r', 'm')
  AND ns.nspname NOT IN ('pg_catalog', 'information_schema')
  AND (cardinality(%s::text[]) = 0 OR ns.nspname = ANY(%s::text[]))
"""

STATS_RESET_SQL = """
SELECT stats_reset FROM pg_stat_database WHERE datname = current_database()
"""


@dataclass(frozen=True)
class IndexInfo:
    schema: str
    table: str
    name: str
    amname: str
    key_columns: tuple[tuple[str, str, str], ...]
    all_columns: frozenset[str]
    predicate: str
    expressions: str
    is_unique: bool
    is_primary: bool
    is_replident: bool
    is_valid: bool
    is_ready: bool
    backs_constraint: bool
    size_bytes: int
    definition: str
    idx_scan: int
    last_idx_scan: datetime | None

    @property
    def subject(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def protected(self) -> bool:
        """Indexes that enforce something. Never advise dropping these."""
        return self.is_unique or self.is_primary or self.is_replident or self.backs_constraint

    def shape(self) -> tuple[str, str, str, str, tuple[tuple[str, str, str], ...]]:
        """What makes two indexes the same index."""
        return (self.table, self.amname, self.predicate, self.expressions, self.key_columns)


def _split(vector: str) -> list[str]:
    return [part for part in vector.strip().split() if part]


def load_indexes(db: Database, schemas: tuple[str, ...] = ()) -> list[IndexInfo]:
    sql = INDEX_INVENTORY_SQL.format(
        last_idx_scan=(
            "s.last_idx_scan"
            if db.caps.has("pg_stat_all_indexes.last_idx_scan")
            else "NULL::timestamptz"
        )
    )
    result: list[IndexInfo] = []
    for row in db.rows(sql, [list(schemas), list(schemas)]):
        keys = _split(row["indkey"])
        classes = _split(row["indclass"])
        options = _split(row["indoption"])
        nkeys = int(row["indnkeyatts"])
        key_columns = tuple(
            (
                keys[i],
                classes[i] if i < len(classes) else "",
                options[i] if i < len(options) else "",
            )
            for i in range(min(nkeys, len(keys)))
        )
        result.append(
            IndexInfo(
                schema=row["schemaname"],
                table=row["tblname"],
                name=row["idxname"],
                amname=row["amname"],
                key_columns=key_columns,
                all_columns=frozenset(keys),
                predicate=row["predicate"],
                expressions=row["expressions"],
                is_unique=bool(row["indisunique"]),
                is_primary=bool(row["indisprimary"]),
                is_replident=bool(row["indisreplident"]),
                is_valid=bool(row["indisvalid"]),
                is_ready=bool(row["indisready"]),
                backs_constraint=bool(row["backs_constraint"]),
                size_bytes=int(row["index_bytes"]),
                definition=row["definition"],
                idx_scan=int(row["idx_scan"] or 0),
                last_idx_scan=row["last_idx_scan"],
            )
        )
    return result


def find_duplicates(indexes: list[IndexInfo]) -> list[list[IndexInfo]]:
    """Groups of indexes that are byte-for-byte the same access path."""
    groups: dict[tuple[str, ...], list[IndexInfo]] = {}
    for index in indexes:
        if not index.is_valid:
            continue
        key = (index.schema, *(str(part) for part in index.shape()))
        groups.setdefault(key, []).append(index)
    # The keeper comes first. An index that enforces a constraint always wins;
    # otherwise the larger one, on the assumption that it is the one already in
    # the plans, and size ties break by name so the output is stable.
    return [
        sorted(group, key=lambda i: (not i.protected, -i.size_bytes, i.name))
        for group in groups.values()
        if len(group) > 1
    ]


def covers(wider: IndexInfo, narrower: IndexInfo) -> bool:
    """True when every query served by ``narrower`` is also served by ``wider``.

    The prefix rule is only valid when the leading key columns match exactly,
    including operator class and sort direction: a btree on ``(a DESC)`` cannot
    serve a plan that wants ``(a ASC)`` in a merge join. INCLUDE columns are
    folded into ``all_columns`` because losing one turns an index-only scan back
    into a heap access.
    """
    if wider is narrower:
        return False
    if (wider.schema, wider.table, wider.amname) != (
        narrower.schema,
        narrower.table,
        narrower.amname,
    ):
        return False
    if (wider.predicate, wider.expressions) != (narrower.predicate, narrower.expressions):
        return False
    if len(narrower.key_columns) >= len(wider.key_columns):
        return False
    if wider.key_columns[: len(narrower.key_columns)] != narrower.key_columns:
        return False
    return narrower.all_columns <= wider.all_columns


def find_redundant(indexes: list[IndexInfo]) -> list[tuple[IndexInfo, IndexInfo]]:
    """Pairs of (redundant index, index that covers it)."""
    pairs: list[tuple[IndexInfo, IndexInfo]] = []
    for narrower in indexes:
        if not narrower.is_valid or narrower.protected:
            continue
        wider = next((w for w in indexes if w.is_valid and covers(w, narrower)), None)
        if wider is not None:
            pairs.append((narrower, wider))
    return pairs


def _unused_findings(
    indexes: list[IndexInfo],
    stats_reset: datetime | None,
    min_size_bytes: int,
    version_reports_last_scan: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    for index in indexes:
        if not index.is_valid or index.protected or index.idx_scan > 0:
            continue
        if index.size_bytes < min_size_bytes:
            continue
        facts: dict[str, object] = {
            "index_bytes": index.size_bytes,
            "idx_scan": index.idx_scan,
            "stats_reset": stats_reset.isoformat() if stats_reset else None,
            "definition": index.definition,
        }
        if version_reports_last_scan:
            facts["last_idx_scan"] = (
                index.last_idx_scan.isoformat() if index.last_idx_scan else None
            )
        # Without a known statistics window the counter says "zero since some
        # unknown moment", which is not evidence of anything.
        severity = Severity.NOTICE if stats_reset is None else Severity.WARNING
        findings.append(
            Finding(
                check="index.unused",
                severity=severity,
                subject=index.subject,
                summary=(
                    f"never scanned in this database, {human_bytes(index.size_bytes)}"
                    + ("" if stats_reset else " — statistics have no recorded reset point")
                ),
                facts=facts,
                remediation=(
                    f"-- confirm on every replica first: pg_stat_all_indexes.idx_scan there is "
                    f"counted separately\n"
                    f"DROP INDEX CONCURRENTLY {qualified(index.schema, index.name)};"
                ),
            )
        )
    return findings


def analyze_indexes(
    db: Database,
    *,
    schemas: tuple[str, ...] = (),
    min_unused_bytes: int = 1024 * 1024,
    min_seq_scans: int = 50,
    min_rows_per_seq_scan: int = 1000,
    min_table_bytes: int = 8 * 1024 * 1024,
) -> list[Finding]:
    indexes = load_indexes(db, schemas)
    stats_reset = db.one(STATS_RESET_SQL)["stats_reset"]
    findings: list[Finding] = []

    for index in indexes:
        if index.is_valid:
            continue
        # An invalid index is dead weight that is still maintained on every
        # write, and a unique one does not enforce its constraint.
        findings.append(
            Finding(
                check="index.invalid",
                severity=Severity.CRITICAL if index.is_unique else Severity.WARNING,
                subject=index.subject,
                summary=(
                    f"invalid index, {human_bytes(index.size_bytes)}, still written to on "
                    f"every DML; a failed CREATE INDEX CONCURRENTLY leaves exactly this"
                ),
                facts={
                    "index_bytes": index.size_bytes,
                    "indisready": index.is_ready,
                    "indisunique": index.is_unique,
                    "definition": index.definition,
                },
                remediation=(
                    f"DROP INDEX CONCURRENTLY {qualified(index.schema, index.name)};\n"
                    f"-- then recreate it: {index.definition};"
                ),
            )
        )

    for group in find_duplicates(indexes):
        keeper, *copies = group
        for copy in copies:
            findings.append(
                Finding(
                    check="index.duplicate",
                    severity=Severity.WARNING,
                    subject=copy.subject,
                    summary=(
                        f"identical to {keeper.subject}; "
                        f"{human_bytes(copy.size_bytes)} of duplicated write amplification"
                    ),
                    facts={
                        "duplicate_of": keeper.subject,
                        "index_bytes": copy.size_bytes,
                        "definition": copy.definition,
                    },
                    remediation=(
                        f"-- keep {keeper.subject}"
                        + (
                            " (it enforces a constraint)"
                            if keeper.protected
                            else " (larger, so more likely the one in use)"
                        )
                        + f"\nDROP INDEX CONCURRENTLY {qualified(copy.schema, copy.name)};"
                    ),
                )
            )

    for narrower, wider in find_redundant(indexes):
        findings.append(
            Finding(
                check="index.redundant",
                severity=Severity.NOTICE,
                subject=narrower.subject,
                summary=(
                    f"leading columns are a prefix of {wider.subject}; "
                    f"{human_bytes(narrower.size_bytes)}"
                ),
                facts={
                    "covered_by": wider.subject,
                    "index_bytes": narrower.size_bytes,
                    "definition": narrower.definition,
                    "covering_definition": wider.definition,
                },
                remediation=(
                    f"-- a narrower index is still smaller and cheaper to scan; drop it only if "
                    f"{wider.subject} is not itself much wider\n"
                    f"DROP INDEX CONCURRENTLY {qualified(narrower.schema, narrower.name)};"
                ),
            )
        )

    findings.extend(
        _unused_findings(
            indexes,
            stats_reset,
            min_unused_bytes,
            db.caps.has("pg_stat_all_indexes.last_idx_scan"),
        )
    )

    findings.extend(
        _missing_index_findings(
            db,
            schemas=schemas,
            min_seq_scans=min_seq_scans,
            min_rows_per_seq_scan=min_rows_per_seq_scan,
            min_table_bytes=min_table_bytes,
        )
    )
    return findings


def _missing_index_findings(
    db: Database,
    *,
    schemas: tuple[str, ...],
    min_seq_scans: int,
    min_rows_per_seq_scan: int,
    min_table_bytes: int,
) -> list[Finding]:
    sql = SEQ_SCAN_SQL.format(
        last_seq_scan=(
            "s.last_seq_scan"
            if db.caps.has("pg_stat_all_tables.last_seq_scan")
            else "NULL::timestamptz"
        )
    )
    findings: list[Finding] = []
    for row in db.rows(sql, [list(schemas), list(schemas)]):
        seq_scan = int(row["seq_scan"])
        table_bytes = int(row["table_bytes"])
        if seq_scan < min_seq_scans or table_bytes < min_table_bytes:
            continue
        rows_per_scan = int(row["seq_tup_read"]) / seq_scan
        if rows_per_scan < min_rows_per_seq_scan:
            continue
        idx_scan = int(row["idx_scan"])
        if idx_scan > seq_scan:
            continue
        subject = f"{row['schemaname']}.{row['tblname']}"
        findings.append(
            Finding(
                check="index.missing_candidate",
                severity=Severity.NOTICE,
                subject=subject,
                summary=(
                    f"{seq_scan} sequential scans reading {rows_per_scan:,.0f} rows each "
                    f"on {human_bytes(table_bytes)}, against {idx_scan} index scans"
                ),
                facts={
                    "seq_scan": seq_scan,
                    "seq_tup_read": int(row["seq_tup_read"]),
                    "rows_per_seq_scan": round(rows_per_scan, 1),
                    "idx_scan": idx_scan,
                    "table_bytes": table_bytes,
                    "last_seq_scan": (
                        row["last_seq_scan"].isoformat() if row["last_seq_scan"] else None
                    ),
                },
                # The catalog knows a scan happened; it does not know what the
                # predicate was. Anything more specific here would be invention.
                remediation=(
                    "-- the catalog cannot name the column: find the predicates first\n"
                    "SELECT query, calls, mean_exec_time FROM pg_stat_statements\n"
                    f" WHERE query ILIKE '%{row['tblname']}%' ORDER BY total_exec_time DESC "
                    "LIMIT 10;"
                ),
            )
        )
    return findings

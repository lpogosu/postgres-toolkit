"""Bloat for heap relations and btree indexes.

Two methods, deliberately both present:

*estimate*
    Rebuild the size the relation *should* have from planner statistics
    (``pg_stats.avg_width``, ``null_frac``, ``reltuples``, fillfactor) and
    subtract it from ``relpages``. Costs a catalog scan, touches no data pages,
    and is safe to run on a busy primary. It is an estimate, and the README says
    by how much it was wrong on the demo fixture.

*pgstattuple*
    Ask the extension. It reads every page of the relation. Exact, and on a large
    table it is a full scan you have chosen to run on production.

The estimation SQL follows the well-known bloat-estimation approach: model the
per-tuple size from column statistics, divide the usable space per page by it,
and compare the resulting page count with reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pgtk.db import Database, qualified
from pgtk.findings import Finding, Severity, human_bytes

# Filtered by schema through a parameter so the catalog scan stays in the server.
TABLE_BLOAT_SQL = """
SELECT schemaname,
       tblname,
       (bs * tblpages)::bigint AS real_bytes,
       (bs * est_tblpages_ff)::bigint AS estimated_bytes,
       CASE WHEN tblpages > est_tblpages_ff
            THEN (bs * (tblpages - est_tblpages_ff))::bigint ELSE 0::bigint END AS bloat_bytes,
       CASE WHEN tblpages > est_tblpages_ff
            THEN 100.0 * (tblpages - est_tblpages_ff) / tblpages ELSE 0.0 END AS bloat_pct,
       fillfactor,
       reltuples,
       is_na
FROM (
    SELECT ceil(reltuples / nullif((bs - page_hdr) * fillfactor / (tpl_size * 100), 0))
             + ceil(toasttuples / 4) AS est_tblpages_ff,
           tblpages, fillfactor, bs, schemaname, tblname, reltuples, is_na
    FROM (
        SELECT (4 + tpl_hdr_size + tpl_data_size + (2 * ma)
                - CASE WHEN tpl_hdr_size %% ma = 0 THEN ma ELSE tpl_hdr_size %% ma END
                - CASE WHEN ceil(tpl_data_size)::int %% ma = 0
                       THEN ma ELSE ceil(tpl_data_size)::int %% ma END) AS tpl_size,
               heappages + toastpages AS tblpages,
               reltuples, toasttuples, bs, page_hdr, schemaname, tblname, fillfactor, is_na
        FROM (
            SELECT ns.nspname AS schemaname,
                   tbl.relname AS tblname,
                   tbl.reltuples,
                   tbl.relpages AS heappages,
                   coalesce(toast.relpages, 0) AS toastpages,
                   coalesce(toast.reltuples, 0) AS toasttuples,
                   coalesce(substring(array_to_string(tbl.reloptions, ' ')
                            FROM 'fillfactor=([0-9]+)')::smallint, 100) AS fillfactor,
                   current_setting('block_size')::numeric AS bs,
                   CASE WHEN version() ~ '64-bit|x86_64|ppc64|ia64|amd64|aarch64|arm64'
                        THEN 8 ELSE 4 END AS ma,
                   24 AS page_hdr,
                   23 + CASE WHEN max(coalesce(s.null_frac, 0)) > 0
                             THEN (7 + count(s.attname)) / 8 ELSE 0 END AS tpl_hdr_size,
                   sum((1 - coalesce(s.null_frac, 0)) * coalesce(s.avg_width, 0)) AS tpl_data_size,
                   bool_or(att.atttypid = 'pg_catalog.name'::regtype)
                     OR count(s.attname) <> count(att.attname) AS is_na
            FROM pg_attribute att
            JOIN pg_class tbl ON att.attrelid = tbl.oid
            JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
            LEFT JOIN pg_stats s ON s.schemaname = ns.nspname
                                AND s.tablename = tbl.relname
                                AND s.attname = att.attname
                                AND NOT s.inherited
            LEFT JOIN pg_class toast ON tbl.reltoastrelid = toast.oid
            WHERE NOT att.attisdropped
              AND att.attnum > 0
              AND tbl.relkind IN ('r', 'm')
              AND tbl.relpages > 0
              AND tbl.reltuples >= 0
              AND ns.nspname NOT IN ('pg_catalog', 'information_schema')
              AND ns.nspname !~ '^pg_toast'
              AND (cardinality(%s::text[]) = 0 OR ns.nspname = ANY(%s::text[]))
            GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ) AS base
    ) AS sized
) AS est
"""

INDEX_BLOAT_SQL = """
WITH idx AS (
    SELECT ns.nspname AS schemaname,
           ic.relname AS idxname,
           tc.relname AS tblname,
           ic.reltuples,
           ic.relpages,
           ic.oid AS idxoid,
           coalesce(substring(array_to_string(ic.reloptions, ' ')
                    FROM 'fillfactor=([0-9]+)')::smallint, 90) AS fillfactor
    FROM pg_index i
    JOIN pg_class ic ON ic.oid = i.indexrelid
    JOIN pg_class tc ON tc.oid = i.indrelid
    JOIN pg_namespace ns ON ns.oid = ic.relnamespace
    JOIN pg_am am ON am.oid = ic.relam
    WHERE am.amname = 'btree'
      AND i.indisvalid
      AND ic.relpages > 0
      AND ic.reltuples >= 0
      AND ns.nspname NOT IN ('pg_catalog', 'information_schema')
      AND (cardinality(%s::text[]) = 0 OR ns.nspname = ANY(%s::text[]))
), widths AS (
    SELECT idx.schemaname, idx.idxname, idx.tblname, idx.reltuples, idx.relpages,
           idx.idxoid, idx.fillfactor,
           current_setting('block_size')::numeric AS bs,
           8::numeric AS maxalign,
           24::numeric AS pagehdr,
           16::numeric AS pageopqdata,
           CASE WHEN max(coalesce(st.null_frac, 0)) = 0 THEN 2::numeric
                ELSE 6::numeric END AS index_tuple_hdr,
           sum((1 - coalesce(st.null_frac, 0)) * coalesce(st.avg_width, 1024))::numeric
             AS nulldatawidth,
           bool_or(st.attname IS NULL) AS is_na
    FROM idx
    JOIN pg_attribute ia ON ia.attrelid = idx.idxoid AND ia.attnum > 0 AND NOT ia.attisdropped
    LEFT JOIN pg_stats st ON st.schemaname = idx.schemaname
                         AND st.tablename = idx.tblname
                         AND st.attname = ia.attname
                         AND NOT st.inherited
    GROUP BY idx.schemaname, idx.idxname, idx.tblname, idx.reltuples, idx.relpages,
             idx.idxoid, idx.fillfactor
), aligned AS (
    SELECT widths.*,
           (index_tuple_hdr + maxalign
            - CASE WHEN index_tuple_hdr %% maxalign = 0
                   THEN maxalign ELSE index_tuple_hdr %% maxalign END
            + nulldatawidth + maxalign
            - CASE WHEN nulldatawidth::bigint %% maxalign = 0
                   THEN maxalign ELSE nulldatawidth::bigint %% maxalign END) AS nulldatahdrwidth
    FROM widths
), estimated AS (
    SELECT aligned.*,
           coalesce(1 + ceil(reltuples / nullif(
               floor((bs - pageopqdata - pagehdr) * fillfactor / (100 * (4 + nulldatahdrwidth))),
               0)), 0) AS est_pages_ff
    FROM aligned
)
SELECT schemaname, idxname, tblname, reltuples, fillfactor, is_na,
       (bs * relpages)::bigint AS real_bytes,
       (bs * est_pages_ff)::bigint AS estimated_bytes,
       CASE WHEN relpages > est_pages_ff
            THEN (bs * (relpages - est_pages_ff))::bigint ELSE 0::bigint END AS bloat_bytes,
       CASE WHEN relpages > est_pages_ff
            THEN 100.0 * (relpages - est_pages_ff) / relpages ELSE 0.0 END AS bloat_pct
FROM estimated
"""

EXACT_TABLE_SQL = """
SELECT s.table_len::bigint AS real_bytes,
       (s.dead_tuple_len + s.free_space)::bigint AS bloat_bytes,
       CASE WHEN s.table_len > 0
            THEN 100.0 * (s.dead_tuple_len + s.free_space) / s.table_len
            ELSE 0.0 END AS bloat_pct,
       s.dead_tuple_count::bigint AS dead_tuples
FROM pgstattuple(%s::regclass) AS s
"""

EXACT_INDEX_SQL = """
SELECT s.index_size::bigint AS real_bytes,
       s.avg_leaf_density::float8 AS avg_leaf_density,
       s.leaf_fragmentation::float8 AS leaf_fragmentation
FROM pgstatindex(%s::regclass) AS s
"""


@dataclass(frozen=True)
class BloatRow:
    kind: str
    schema: str
    name: str
    parent: str | None
    real_bytes: int
    bloat_bytes: int
    bloat_pct: float
    method: str
    stats_unusable: bool
    extra: dict[str, Any]

    @property
    def subject(self) -> str:
        return f"{self.schema}.{self.name}"


def _schema_params(schemas: tuple[str, ...]) -> list[object]:
    return [list(schemas), list(schemas)]


def estimate_table_bloat(db: Database, schemas: tuple[str, ...] = ()) -> list[BloatRow]:
    rows = db.rows(TABLE_BLOAT_SQL, _schema_params(schemas))
    return [
        BloatRow(
            kind="table",
            schema=row["schemaname"],
            name=row["tblname"],
            parent=None,
            real_bytes=int(row["real_bytes"]),
            bloat_bytes=int(row["bloat_bytes"]),
            bloat_pct=float(row["bloat_pct"]),
            method="estimate",
            stats_unusable=bool(row["is_na"]),
            extra={"fillfactor": int(row["fillfactor"]), "reltuples": float(row["reltuples"])},
        )
        for row in rows
    ]


def estimate_index_bloat(db: Database, schemas: tuple[str, ...] = ()) -> list[BloatRow]:
    rows = db.rows(INDEX_BLOAT_SQL, _schema_params(schemas))
    return [
        BloatRow(
            kind="index",
            schema=row["schemaname"],
            name=row["idxname"],
            parent=row["tblname"],
            real_bytes=int(row["real_bytes"]),
            bloat_bytes=int(row["bloat_bytes"]),
            bloat_pct=float(row["bloat_pct"]),
            method="estimate",
            stats_unusable=bool(row["is_na"]),
            extra={"fillfactor": int(row["fillfactor"])},
        )
        for row in rows
    ]


class PgstattupleUnavailableError(RuntimeError):
    pass


def measure_exact(db: Database, rows: list[BloatRow]) -> list[BloatRow]:
    """Replace estimates with pgstattuple/pgstatindex readings.

    Every call here is a full scan of the relation. The caller decides; this
    function does not silently upgrade anything.
    """
    if not db.caps.has_extension("pgstattuple"):
        raise PgstattupleUnavailableError(
            "pgstattuple is not installed in this database "
            "(CREATE EXTENSION pgstattuple; needs it to be present in the image)"
        )
    measured: list[BloatRow] = []
    for row in rows:
        ref = qualified(row.schema, row.name)
        if row.kind == "table":
            exact = db.one(EXACT_TABLE_SQL, [ref])
            measured.append(
                BloatRow(
                    kind=row.kind,
                    schema=row.schema,
                    name=row.name,
                    parent=row.parent,
                    real_bytes=int(exact["real_bytes"]),
                    bloat_bytes=int(exact["bloat_bytes"]),
                    bloat_pct=float(exact["bloat_pct"]),
                    method="pgstattuple",
                    stats_unusable=False,
                    extra={"dead_tuples": int(exact["dead_tuples"])},
                )
            )
        else:
            exact = db.one(EXACT_INDEX_SQL, [ref])
            density = float(exact["avg_leaf_density"])
            size = int(exact["real_bytes"])
            pct = max(0.0, 100.0 - density)
            measured.append(
                BloatRow(
                    kind=row.kind,
                    schema=row.schema,
                    name=row.name,
                    parent=row.parent,
                    real_bytes=size,
                    bloat_bytes=int(size * pct / 100.0),
                    bloat_pct=pct,
                    method="pgstatindex",
                    stats_unusable=False,
                    extra={
                        "avg_leaf_density": round(density, 2),
                        "leaf_fragmentation": round(float(exact["leaf_fragmentation"]), 2),
                    },
                )
            )
    return measured


def _severity(pct: float, bloat_bytes: int) -> Severity:
    # Percentage alone flags every small table that ever saw an UPDATE; bytes
    # alone flags huge tables that are perfectly healthy. The verdict needs both.
    if pct >= 50 and bloat_bytes >= 512 * 1024 * 1024:
        return Severity.CRITICAL
    if pct >= 30:
        return Severity.WARNING
    return Severity.NOTICE


def _table_remediation(row: BloatRow) -> str:
    ref = qualified(row.schema, row.name)
    if row.bloat_pct >= 50:
        return (
            f"-- rewrite required to return {human_bytes(row.bloat_bytes)} to the filesystem\n"
            f"-- pg_repack -t {ref}          -- online, needs 2x disk and a trigger\n"
            f"VACUUM FULL {ref};             -- ACCESS EXCLUSIVE for the whole rewrite"
        )
    return (
        f"VACUUM (ANALYZE) {ref};  -- makes the space reusable in place; "
        f"the file does not shrink"
    )


def analyze_bloat(
    db: Database,
    *,
    schemas: tuple[str, ...] = (),
    min_bloat_bytes: int = 8 * 1024 * 1024,
    min_bloat_pct: float = 20.0,
    exact: bool = False,
) -> tuple[list[Finding], list[BloatRow]]:
    """Return findings plus the rows behind them, so callers can print a table."""
    candidates = [
        row
        for row in estimate_table_bloat(db, schemas) + estimate_index_bloat(db, schemas)
        if row.bloat_bytes >= min_bloat_bytes and row.bloat_pct >= min_bloat_pct
    ]
    if exact:
        candidates = measure_exact(db, candidates)

    findings: list[Finding] = []
    for row in candidates:
        severity = _severity(row.bloat_pct, row.bloat_bytes)
        if row.stats_unusable:
            # A NAME column or a column with no analyzed statistics makes the
            # width model wrong in an unknown direction. Reporting the number
            # without saying so would be the dishonest option.
            severity = min(severity, Severity.NOTICE)
        if row.kind == "table":
            remediation = _table_remediation(row)
        else:
            remediation = (
                f"REINDEX INDEX CONCURRENTLY {qualified(row.schema, row.name)};"
                "  -- no ACCESS EXCLUSIVE, but needs a second copy on disk"
            )
        findings.append(
            Finding(
                check=f"bloat.{row.kind}",
                severity=severity,
                subject=row.subject,
                summary=(
                    f"{row.bloat_pct:.0f}% bloat, {human_bytes(row.bloat_bytes)} "
                    f"of {human_bytes(row.real_bytes)} ({row.method})"
                    + (" — statistics incomplete, treat as a hint" if row.stats_unusable else "")
                ),
                facts={
                    "method": row.method,
                    "real_bytes": row.real_bytes,
                    "bloat_bytes": row.bloat_bytes,
                    "bloat_pct": round(row.bloat_pct, 2),
                    "stats_unusable": row.stats_unusable,
                    **row.extra,
                },
                remediation=remediation,
            )
        )
    return findings, candidates

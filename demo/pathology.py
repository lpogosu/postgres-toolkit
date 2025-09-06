"""Build a database that is broken in named, specific ways.

Every function here creates exactly one problem and says which check is supposed
to notice it. The integration tests assert on those names, so a check that stops
firing fails a test instead of quietly producing an empty report — the failure
mode that matters most for a diagnostic tool.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

Conn = psycopg.Connection[Any]

# Sizes are the smallest that still clear the tool's default thresholds. Every
# extra row is time in CI and disk in a container, and the pathology is the same.
SHIPMENT_ROWS = 120_000
ORDER_ROWS = 200_000
EVENT_ROWS = 400_000
LEDGER_ROWS = 200_000
SESSION_ROWS = 120_000
AUDIT_ROWS = 150_000
INVOICE_ROWS = 500_000
TENANTS = 25
SEQ_SCAN_QUERIES = 60


def _exec(conn: Conn, sql: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql)


def bloated_table(conn: Conn) -> None:
    """bloat.table — 70% of the heap is dead space that VACUUM will not give back."""
    _exec(
        conn,
        """
        CREATE TABLE shipments (
            id          bigserial PRIMARY KEY,
            carrier     text NOT NULL,
            status      text NOT NULL,
            payload     text NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """,
    )
    _exec(
        conn,
        f"""
        INSERT INTO shipments (carrier, status, payload)
        SELECT 'carrier-' || (g % 7), 'delivered', repeat('x', 180)
        FROM generate_series(1, {SHIPMENT_ROWS}) AS g
        """,
    )
    _exec(conn, "ANALYZE shipments")
    # Deleting without a rewrite is the ordinary way a table ends up twice the
    # size it needs: the pages stay, the tuples do not.
    _exec(conn, "DELETE FROM shipments WHERE id % 10 < 7")
    _exec(conn, "ANALYZE shipments")


def bloated_index(conn: Conn) -> None:
    """bloat.index — a btree whose pages survived the rows they indexed.

    VACUUM is run explicitly here because that is what makes the pathology
    *visible*: it removes the dead index entries and updates ``reltuples``, while
    ``relpages`` stays where it was. An index that has never been vacuumed still
    counts its dead entries and looks healthy to the estimator.
    """
    _exec(
        conn,
        """
        CREATE TABLE invoices (
            id      bigserial PRIMARY KEY,
            number  text NOT NULL,
            amount  bigint NOT NULL
        )
        """,
    )
    _exec(
        conn,
        f"""
        INSERT INTO invoices (number, amount)
        SELECT 'INV-' || lpad(g::text, 12, '0'), g * 13
        FROM generate_series(1, {INVOICE_ROWS}) AS g
        """,
    )
    _exec(conn, "CREATE INDEX invoices_number_idx ON invoices (number)")
    _exec(conn, "DELETE FROM invoices WHERE id % 10 < 7")
    _exec(conn, "VACUUM invoices")
    _exec(conn, "ANALYZE invoices")


def index_zoo(conn: Conn) -> None:
    """index.duplicate, index.redundant, index.unused — and one index that is fine."""
    _exec(
        conn,
        """
        CREATE TABLE orders (
            id           bigserial PRIMARY KEY,
            customer_id  bigint NOT NULL,
            region       text NOT NULL,
            status       text NOT NULL,
            total_cents  bigint NOT NULL,
            placed_at    timestamptz NOT NULL
        )
        """,
    )
    _exec(
        conn,
        f"""
        INSERT INTO orders (customer_id, region, status, total_cents, placed_at)
        SELECT g % 5000,
               'region-' || (g % 12),
               CASE WHEN g % 17 = 0 THEN 'cancelled' ELSE 'shipped' END,
               (g * 37) % 500000,
               now() - make_interval(secs => g)
        FROM generate_series(1, {ORDER_ROWS}) AS g
        """,
    )
    _exec(conn, "CREATE INDEX orders_customer_idx ON orders (customer_id)")
    _exec(conn, "CREATE INDEX orders_customer_status_idx ON orders (customer_id, status)")
    _exec(conn, "CREATE INDEX orders_placed_at_idx ON orders (placed_at)")
    _exec(conn, "CREATE INDEX orders_placed_at_copy_idx ON orders (placed_at)")
    _exec(conn, "CREATE INDEX orders_region_idx ON orders (region)")
    _exec(conn, "ANALYZE orders")

    # Give the wide index and one of the placed_at pair a scan history, so
    # "unused" has to discriminate rather than list everything it can see.
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        for customer in range(200):
            cur.execute(
                "SELECT count(*) FROM orders WHERE customer_id = %s AND status = 'shipped'",
                (customer,),
            )
            cur.execute(
                "SELECT count(*) FROM orders WHERE placed_at > now() - interval '1 hour'"
            )
        cur.execute("RESET enable_seqscan")


def invalid_index(conn: Conn) -> None:
    """index.invalid — the leftover of a CREATE INDEX CONCURRENTLY that failed.

    This is how invalid indexes actually appear in production, so it is how they
    appear here: a unique build over data that is not unique. The exception is
    expected; the index it leaves behind is the point.
    """
    _exec(
        conn,
        """
        CREATE TABLE coupons (
            id   bigserial PRIMARY KEY,
            code text NOT NULL
        )
        """,
    )
    _exec(
        conn,
        """
        INSERT INTO coupons (code)
        SELECT 'CODE-' || (g % 500) FROM generate_series(1, 5000) AS g
        """,
    )
    try:
        _exec(conn, "CREATE UNIQUE INDEX CONCURRENTLY coupons_code_uniq ON coupons (code)")
    except psycopg.errors.UniqueViolation:
        return
    else:
        raise AssertionError("the unique build was supposed to fail on duplicate codes")


def correlated_columns(conn: Conn) -> None:
    """plan.misestimate and plan.nested_loop — two columns the planner thinks are independent."""
    _exec(
        conn,
        """
        CREATE TABLE events (
            id         bigserial PRIMARY KEY,
            tenant_id  int NOT NULL,
            bucket     int NOT NULL,
            body       text NOT NULL
        )
        """,
    )
    # bucket is a copy of tenant_id. The planner multiplies two 1/25
    # selectivities and lands 25x below the truth.
    _exec(
        conn,
        f"""
        INSERT INTO events (tenant_id, bucket, body)
        SELECT g % {TENANTS}, g % {TENANTS}, repeat('e', 40)
        FROM generate_series(1, {EVENT_ROWS}) AS g
        """,
    )
    _exec(conn, "CREATE INDEX events_tenant_idx ON events (tenant_id)")
    _exec(conn, "ANALYZE events")


def stale_visibility_map(conn: Conn) -> None:
    """plan.heap_fetches — an index-only scan that is not index-only.

    autovacuum is off on this table on purpose: since PostgreSQL 13 an
    insert-only table gets vacuumed by autovacuum_vacuum_insert_threshold, which
    would set the visibility map and remove the pathology.
    """
    _exec(
        conn,
        """
        CREATE TABLE ledger (
            id     bigint PRIMARY KEY,
            amount bigint NOT NULL
        ) WITH (autovacuum_enabled = false)
        """,
    )
    _exec(
        conn,
        f"INSERT INTO ledger SELECT g, g * 7 FROM generate_series(1, {LEDGER_ROWS}) AS g",
    )
    _exec(conn, "ANALYZE ledger")


def skipped_by_autovacuum(conn: Conn) -> None:
    """vacuum.skipped — dead tuples over the threshold on a table autovacuum will not touch."""
    _exec(
        conn,
        """
        CREATE TABLE audit_log (
            id      bigserial PRIMARY KEY,
            actor   text NOT NULL,
            action  text NOT NULL,
            at      timestamptz NOT NULL DEFAULT now()
        ) WITH (autovacuum_enabled = false)
        """,
    )
    _exec(
        conn,
        f"""
        INSERT INTO audit_log (actor, action)
        SELECT 'user-' || (g % 900), 'action-' || (g % 20)
        FROM generate_series(1, {AUDIT_ROWS}) AS g
        """,
    )
    _exec(conn, "ANALYZE audit_log")
    _exec(conn, "DELETE FROM audit_log WHERE id % 3 <> 0")


def sequential_scan_pressure(conn: Conn) -> None:
    """index.missing_candidate — a big table read end to end, repeatedly, with no index to help."""
    _exec(
        conn,
        """
        CREATE TABLE sessions (
            id         bigserial PRIMARY KEY,
            token      text NOT NULL,
            user_agent text NOT NULL,
            seen_at    timestamptz NOT NULL DEFAULT now()
        )
        """,
    )
    _exec(
        conn,
        f"""
        INSERT INTO sessions (token, user_agent)
        SELECT md5(g::text), 'agent-' || (g % 40)
        FROM generate_series(1, {SESSION_ROWS}) AS g
        """,
    )
    _exec(conn, "ANALYZE sessions")
    with conn.cursor() as cur:
        for i in range(SEQ_SCAN_QUERIES):
            cur.execute(
                "SELECT count(*) FROM sessions WHERE user_agent = %s", (f"missing-{i}",)
            )


def flush_statistics(conn: Conn) -> None:
    """Make the counters visible to the next reader without waiting for the throttle.

    Since PostgreSQL 15 backends accumulate statistics locally and flush them at
    most once a second; without this the seeding finishes before the counters it
    just produced are readable, and the tests become timing-dependent.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT current_setting('server_version_num')::int >= 150000")
        row = cur.fetchone()
        if row is not None and row[0]:
            cur.execute("SELECT pg_stat_force_next_flush()")


def seed_all(conn: Conn) -> None:
    bloated_table(conn)
    bloated_index(conn)
    index_zoo(conn)
    invalid_index(conn)
    correlated_columns(conn)
    stale_visibility_map(conn)
    skipped_by_autovacuum(conn)
    sequential_scan_pressure(conn)
    flush_statistics(conn)


MISESTIMATING_QUERY = """
SELECT count(*)
FROM events e
JOIN orders o ON o.id = e.id
WHERE e.tenant_id = 7 AND e.bucket = 7
"""

HEAP_FETCH_QUERY = """
SELECT count(*) FROM ledger WHERE id BETWEEN 1 AND 5000
"""

EXTERNAL_SORT_QUERY = """
SELECT count(*) FROM (SELECT id FROM orders ORDER BY region, placed_at) AS s
"""

# Neither join column is indexed, so the planner has to build a hash table; with
# a small work_mem it does not fit and the join runs in batches.
HASH_SPILL_QUERY = """
SELECT count(*)
FROM orders o
JOIN invoices i ON i.amount = o.total_cents
WHERE i.id < 100000
"""

SEQ_SCAN_FILTER_QUERY = """
SELECT count(*) FROM sessions WHERE user_agent = 'agent-that-never-existed'
"""


def explain_json(conn: Conn, query: str, *, work_mem: str | None = None) -> str:
    """Run EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) and return the raw JSON text."""
    with conn.cursor() as cur:
        if work_mem is not None:
            cur.execute("SELECT set_config('work_mem', %s, false)", (work_mem,))
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
        row = cur.fetchone()
        if work_mem is not None:
            cur.execute("RESET work_mem")
    if row is None:
        raise RuntimeError("EXPLAIN returned no rows")
    return json.dumps(row[0])

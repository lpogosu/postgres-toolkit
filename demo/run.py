"""Seed the pathologies, then run the real CLI against them.

The commands are invoked as subprocesses rather than imported, so what appears on
screen is exactly what a user gets — including the exit codes, which is the part
a scheduled job depends on.

The lock section is the reason this is a script and not a shell one-liner: the
blocking chain has to exist while ``pgtk locks`` runs, so the sessions are opened
here and held until the command returns.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg

from demo.pathology import (
    EXTERNAL_SORT_QUERY,
    HEAP_FETCH_QUERY,
    MISESTIMATING_QUERY,
    explain_json,
    seed_all,
)

DSN = os.environ.get(
    "PGTK_DSN",
    "host=127.0.0.1 port=55432 user=pgtk password=pgtk_demo_password dbname=pgtk_demo",
)
PLAN_DIR = Path(".demo-cache")


def banner(text: str) -> None:
    print(f"\n\033[1m{'=' * 78}\n{text}\n{'=' * 78}\033[0m", flush=True)


def pgtk(*args: str) -> int:
    printable = " ".join(("pgtk", *args))
    print(f"\n\033[36m$ {printable}\033[0m", flush=True)
    return subprocess.run(
        [sys.executable, "-m", "pgtk", "--dsn", DSN, *args], check=False
    ).returncode


def wait_for_server(timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(DSN, connect_timeout=3):
                return
        except psycopg.OperationalError as exc:
            last = exc
            time.sleep(1)
    raise SystemExit(f"no PostgreSQL at {DSN.split('password=')[0]}...: {last}")


def already_seeded(conn: psycopg.Connection[tuple[object, ...]]) -> bool:
    row = conn.execute("SELECT to_regclass('public.shipments') IS NOT NULL").fetchone()
    return bool(row and row[0])


@contextmanager
def blocking_chain() -> Iterator[None]:
    """Hold one row hostage from three sessions while the locks command runs."""
    holder = psycopg.connect(DSN, application_name="nightly-batch")
    waiters: list[psycopg.Connection[tuple[object, ...]]] = []
    threads: list[threading.Thread] = []
    holder.execute("UPDATE orders SET status = 'held' WHERE id = 1")

    def queue(application: str) -> None:
        def run() -> None:
            conn = psycopg.connect(DSN, application_name=application)
            waiters.append(conn)
            # The holder's rollback releases these, but a cancelled waiter
            # must not take its thread down with a traceback.
            with contextlib.suppress(psycopg.Error):
                conn.execute("UPDATE orders SET status = %s WHERE id = 1", (application,))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.append(thread)

    for application in ("checkout-api", "reporting"):
        queue(application)
        time.sleep(1.5)
    try:
        yield
    finally:
        holder.rollback()
        holder.close()
        for thread in threads:
            thread.join(timeout=15)
        for conn in waiters:
            conn.rollback()
            conn.close()


def capture_plans() -> list[tuple[str, Path]]:
    PLAN_DIR.mkdir(exist_ok=True)
    captured: list[tuple[str, Path]] = []
    with psycopg.connect(DSN, autocommit=True) as conn:
        for name, query, work_mem in (
            ("correlated-columns", MISESTIMATING_QUERY, None),
            ("stale-visibility-map", HEAP_FETCH_QUERY, None),
            ("sort-without-memory", EXTERNAL_SORT_QUERY, "64kB"),
        ):
            path = PLAN_DIR / f"{name}.json"
            path.write_text(explain_json(conn, query, work_mem=work_mem), encoding="utf-8")
            captured.append((name, path))
    return captured


def main() -> int:
    wait_for_server()
    with psycopg.connect(DSN, autocommit=True) as conn:
        if already_seeded(conn):
            print("database already seeded, reusing it")
        else:
            banner("seeding a database that is broken in eight named ways")
            started = time.monotonic()
            conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
            seed_all(conn)
            print(f"seeded in {time.monotonic() - started:.1f}s")

    banner("bloat — statistics-based estimate, no data pages read")
    pgtk("bloat")

    banner("bloat — the same relations measured with pgstattuple")
    pgtk("bloat", "--exact")

    banner("index hygiene")
    pgtk("indexes")

    banner("vacuum, freezing and the xmin horizon")
    pgtk("vacuum")

    banner("plans captured from this database")
    for name, path in capture_plans():
        print(f"\n--- {name} ---")
        pgtk("plan", str(path))

    banner("locks — three sessions queued behind one idle transaction")
    with blocking_chain():
        pgtk("locks", "--idle-minutes", "0")

    banner("everything at once, the way a scheduled job would run it")
    code = pgtk("--fail-on", "critical", "report")
    print(f"\nexit code: {code}  (--fail-on critical, and there are critical findings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

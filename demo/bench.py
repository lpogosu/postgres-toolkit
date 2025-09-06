"""What the two bloat methods cost and how far apart their answers are.

Prints the table quoted in the README. The comparison is only meaningful next to
the size of what was scanned, so that is printed too: the estimate reads the
catalog and its cost does not depend on relation size, while pgstattuple reads
every page and its cost is linear in exactly that number.
"""

from __future__ import annotations

import os
import statistics
import time

from pgtk.bloat import analyze_bloat, estimate_index_bloat, estimate_table_bloat, measure_exact
from pgtk.db import connect
from pgtk.findings import human_bytes

DSN = os.environ.get(
    "PGTK_DSN",
    "host=127.0.0.1 port=55432 user=pgtk password=pgtk_demo_password dbname=pgtk_demo",
)
REPEATS = 7


def timed(call: object, repeats: int = REPEATS) -> tuple[float, float]:
    assert callable(call)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), min(samples)


def main() -> int:
    with connect(DSN, statement_timeout_ms=120_000) as db:
        _, candidates = analyze_bloat(db)
        scanned = sum(row.real_bytes for row in candidates)
        relations = len(db.rows("SELECT 1 FROM pg_class WHERE relkind IN ('r', 'm', 'i')"))

        estimate_median, estimate_best = timed(
            lambda: (estimate_table_bloat(db), estimate_index_bloat(db))
        )
        exact_median, exact_best = timed(lambda: measure_exact(db, candidates))
        measured = {row.subject: row for row in measure_exact(db, candidates)}

        print(f"PostgreSQL {db.version}, {relations} relations in the catalog")
        print(
            f"pgstattuple scanned {len(candidates)} relations, "
            f"{human_bytes(scanned)} of pages, {REPEATS} repeats\n"
        )
        print(f"{'method':<22}{'median ms':>12}{'fastest ms':>13}")
        print("-" * 47)
        print(f"{'estimate (catalog)':<22}{estimate_median:>12.1f}{estimate_best:>13.1f}")
        print(f"{'pgstattuple (pages)':<22}{exact_median:>12.1f}{exact_best:>13.1f}\n")

        header = (
            f"{'relation':<30}{'size':>10}{'estimate %':>12}"
            f"{'measured %':>12}{'delta pp':>10}{'scan ms':>9}"
        )
        print(header)
        print("-" * len(header))
        for row in sorted(candidates, key=lambda r: -r.bloat_bytes):
            exact = measured[row.subject]
            # Timed one relation at a time. This column is what the estimate
            # buys you: it exists per relation and grows with the relation,
            # while the whole estimate above is a single catalog query.
            scan_ms, _ = timed(lambda bound=row: measure_exact(db, [bound]))
            print(
                f"{row.subject:<30}{human_bytes(row.real_bytes):>10}"
                f"{row.bloat_pct:>12.1f}{exact.bloat_pct:>12.1f}"
                f"{row.bloat_pct - exact.bloat_pct:>+10.1f}{scan_ms:>9.1f}"
            )
        print(
            "\nEvery page was already in cache here, so the scan column is not I/O time, and "
            "it is\nnot constant per megabyte either: pgstattuple walks line pointers, so a "
            "heap full of\ndead tuples costs more per page than a dense one. What the column "
            "does show is that\nthis cost exists once per relation, while the whole estimate "
            "above is one catalog query."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

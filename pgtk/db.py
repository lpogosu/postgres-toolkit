"""Connection handling and the server-version facts the checks branch on.

Two things live here because every check needs them and neither belongs to a
particular check: a connection that cannot write, and a description of which
catalog columns this server actually has.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

Row = dict[str, Any]


@dataclass(frozen=True, order=True)
class ServerVersion:
    major: int
    minor: int

    @classmethod
    def from_num(cls, server_version_num: int) -> ServerVersion:
        """Decode ``server_version_num``: 170004 -> 17.4, and 90624 -> 9.6.24."""
        if server_version_num >= 100000:
            return cls(server_version_num // 10000, server_version_num % 10000)
        return cls(server_version_num // 10000, (server_version_num // 100) % 100)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


# Every catalog column this toolkit reads that did not always exist, with the
# major release that introduced it. Checks ask `caps.has(...)` instead of
# comparing version numbers inline, so the whole portability surface is one list.
CATALOG_ADDITIONS: dict[str, int] = {
    # Added in 13 alongside autovacuum_vacuum_insert_threshold.
    "pg_stat_all_tables.n_ins_since_vacuum": 13,
    # pg_stat_activity gained the normalised query id in 14.
    "pg_stat_activity.query_id": 14,
    # 16 added "when was this last used", which is what makes an unused-index
    # verdict defensible rather than a guess.
    "pg_stat_all_indexes.last_idx_scan": 16,
    "pg_stat_all_tables.last_seq_scan": 16,
}


class Capabilities:
    """Which optional catalog columns and extensions this server offers."""

    def __init__(self, version: ServerVersion, extensions: frozenset[str]) -> None:
        self.version = version
        self.extensions = extensions

    def has(self, column: str) -> bool:
        introduced = CATALOG_ADDITIONS.get(column)
        if introduced is None:
            raise KeyError(f"{column} is not a version-gated column; add it to CATALOG_ADDITIONS")
        return self.version.major >= introduced

    def has_extension(self, name: str) -> bool:
        return name in self.extensions


class UnsupportedServerError(RuntimeError):
    pass


MINIMUM_VERSION = ServerVersion(13, 0)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def quote_identifier(name: str) -> str:
    """Quote a schema/relation name for inclusion in printed remediation SQL.

    The output of this function is only ever *printed*, but printing
    ``REINDEX INDEX public."weird name"`` correctly is the difference between
    advice someone can paste and advice they have to debug at 3am.
    """
    if _IDENT.match(name) and name.lower() == name:
        return name
    return '"' + name.replace('"', '""') + '"'


def qualified(schema: str, name: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(name)}"


class Database:
    """A read-only handle. Deliberately thin: it runs SQL and reports capabilities."""

    def __init__(self, conn: psycopg.Connection[Row]) -> None:
        self._conn = conn
        num = int(self.one("SELECT current_setting('server_version_num') AS v")["v"])
        self.version = ServerVersion.from_num(num)
        if self.version < MINIMUM_VERSION:
            raise UnsupportedServerError(
                f"server is {self.version}; the catalog queries here assume "
                f"{MINIMUM_VERSION.major} or newer"
            )
        installed = frozenset(
            row["extname"] for row in self.rows("SELECT extname FROM pg_extension")
        )
        self.caps = Capabilities(self.version, installed)

    def rows(self, sql: str, params: Sequence[object] = ()) -> list[Row]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def one(self, sql: str, params: Sequence[object] = ()) -> Row:
        rows = self.rows(sql, params)
        if len(rows) != 1:
            raise RuntimeError(f"expected exactly one row, got {len(rows)}")
        return rows[0]

    def current_database(self) -> str:
        return str(self.one("SELECT current_database() AS db")["db"])


@contextmanager
def connect(
    dsn: str,
    *,
    statement_timeout_ms: int = 15_000,
    application_name: str = "pgtk",
) -> Iterator[Database]:
    """Open a connection that physically cannot write.

    ``default_transaction_read_only`` is belt and braces on top of the fact that
    no check issues DML: a user who points this at a primary by mistake gets an
    error rather than a surprise. ``statement_timeout`` matters more than it
    looks — the bloat estimation query joins pg_attribute against pg_stats for
    every relation, and on a catalog with tens of thousands of tables that is not
    instant.
    """
    options = " ".join(
        [
            f"-c statement_timeout={statement_timeout_ms}",
            "-c default_transaction_read_only=on",
            "-c lock_timeout=2000",
        ]
    )
    with psycopg.connect(
        dsn,
        row_factory=dict_row,
        autocommit=True,
        application_name=application_name,
        options=options,
    ) as conn:
        yield Database(conn)

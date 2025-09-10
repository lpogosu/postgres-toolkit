"""A real PostgreSQL per major version, started by the test session itself.

Mocking the catalog was never an option here: every check in this repository is
a claim about what a specific PostgreSQL version puts in ``pg_stat_all_indexes``
or ``pg_class``. A mock would only assert that the fixture agrees with the query,
which is the one thing that cannot go wrong.

The fixture shells out to ``docker`` rather than depending on testcontainers: it
is thirty lines, it has no version drift of its own, and the failure messages
name the container so a leaked one can be found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import psycopg
import pytest

from demo.pathology import Conn, seed_all
from pgtk.db import Database, connect

DEFAULT_IMAGES = "postgres:17-alpine,postgres:16-alpine"
IMAGES = [image for image in os.environ.get("PGTK_TEST_IMAGES", DEFAULT_IMAGES).split(",") if image]

PG_USER = "pgtk"
# Throwaway credential for a container that lives for the length of one test session.
PG_PASSWORD = "pgtk_test_password"
PG_DATABASE = "pgtk_fixture"
READY_TIMEOUT_S = 90


@dataclass(frozen=True)
class PgInstance:
    image: str
    container: str
    port: int

    @property
    def dsn(self) -> str:
        return (
            f"host=127.0.0.1 port={self.port} user={PG_USER} "
            f"password={PG_PASSWORD} dbname={PG_DATABASE}"
        )

    @property
    def major(self) -> int:
        return int(self.image.split(":")[1].split("-")[0])


def _run(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        args, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _start(image: str) -> PgInstance:
    name = f"pgtk-test-{uuid.uuid4().hex[:10]}"
    _run(
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"POSTGRES_USER={PG_USER}",
        "-e",
        f"POSTGRES_PASSWORD={PG_PASSWORD}",
        "-e",
        f"POSTGRES_DB={PG_DATABASE}",
        "-p",
        "127.0.0.1::5432",
        image,
        "postgres",
        # The fixture seeds hundreds of thousands of rows and then throws the
        # cluster away; durability settings that protect nothing cost minutes.
        "-c",
        "fsync=off",
        "-c",
        "full_page_writes=off",
        "-c",
        "synchronous_commit=off",
        "-c",
        "autovacuum_naptime=10s",
    )
    published = _run("docker", "port", name, "5432/tcp")
    port = int(published.splitlines()[0].rsplit(":", 1)[1])
    instance = PgInstance(image=image, container=name, port=port)

    # Readiness is a real TCP connection, not `pg_isready`. The image starts a
    # throwaway server to initialise the cluster, and that server listens on the
    # unix socket only — so `docker exec pg_isready` answers "ready" while the
    # server the tests actually use has not started yet, and the next connection
    # lands in the shutdown window between the two.
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(instance.dsn, connect_timeout=3) as conn:
                conn.execute("SELECT 1")
        except psycopg.Error:
            time.sleep(0.5)
            continue
        return instance
    logs = _run("docker", "logs", "--tail", "40", name, check=False)
    _run("docker", "rm", "-f", name, check=False)
    raise RuntimeError(f"{image} did not become ready in {READY_TIMEOUT_S}s:\n{logs}")


@pytest.fixture(scope="session", params=IMAGES, ids=lambda image: str(image).replace(":", "-"))
def instance(request: pytest.FixtureRequest) -> Iterator[PgInstance]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    started = _start(str(request.param))
    try:
        yield started
    finally:
        _run("docker", "rm", "-f", "-v", started.container, check=False)


@pytest.fixture(scope="session")
def seeded(instance: PgInstance) -> PgInstance:
    with psycopg.connect(instance.dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
        seed_all(conn)
    return instance


@pytest.fixture
def db(seeded: PgInstance) -> Iterator[Database]:
    with connect(seeded.dsn, statement_timeout_ms=60_000) as handle:
        yield handle


@pytest.fixture
def writable(seeded: PgInstance) -> Iterator[Conn]:
    """A second, writable connection for tests that need to change the database."""
    with psycopg.connect(seeded.dsn, autocommit=True) as conn:
        yield conn

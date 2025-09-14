"""The command line, exercised the way a user and a cron job would."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from demo.pathology import HEAP_FETCH_QUERY, Conn, explain_json
from pgtk.cli import main
from tests.conftest import PgInstance

pytestmark = pytest.mark.pg


def run(*args: str, dsn: str | None = None) -> tuple[int, str]:
    prefix = ["--dsn", dsn] if dsn else []
    result = CliRunner().invoke(main, [*prefix, *args], catch_exceptions=False)
    return result.exit_code, result.output


def test_report_names_the_server_and_prints_findings(seeded: PgInstance) -> None:
    code, output = run("report", dsn=seeded.dsn)
    assert code == 0
    assert f"on PostgreSQL {seeded.major}." in output
    assert "bloat.table" in output
    assert "index.invalid" in output


def test_json_output_is_the_only_thing_on_stdout_and_it_parses(seeded: PgInstance) -> None:
    code, output = run("--json", "indexes", dsn=seeded.dsn)
    assert code == 0
    payload = json.loads(output)
    assert payload["database"] == "pgtk_fixture"
    assert {f["check"] for f in payload["findings"]} >= {"index.invalid", "index.duplicate"}


def test_fail_on_critical_exits_non_zero_when_something_is_critical(
    seeded: PgInstance,
) -> None:
    code, _ = run("--fail-on", "critical", "indexes", dsn=seeded.dsn)
    assert code == 1


def test_fail_on_never_keeps_the_exit_code_at_zero(seeded: PgInstance) -> None:
    code, _ = run("--fail-on", "never", "indexes", dsn=seeded.dsn)
    assert code == 0


def test_a_schema_filter_that_matches_nothing_produces_a_clean_report(
    seeded: PgInstance,
) -> None:
    code, output = run(
        "--fail-on", "notice", "bloat", "--schema", "nowhere", dsn=seeded.dsn
    )
    assert code == 0
    assert "no findings" in output


def test_exact_mode_says_which_method_produced_the_numbers(seeded: PgInstance) -> None:
    code, output = run("bloat", "--exact", dsn=seeded.dsn)
    assert code == 0
    assert "measured with pgstattuple" in output
    assert "pgstatindex" in output


def test_the_plan_command_needs_no_database(tmp_path: Path, writable: Conn) -> None:
    payload = explain_json(writable, HEAP_FETCH_QUERY)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(payload, encoding="utf-8")
    code, output = run("plan", str(plan_file))
    assert code == 0
    assert "plan.heap_fetches" in output


def test_the_plan_command_reads_stdin(writable: Conn) -> None:
    payload = explain_json(writable, HEAP_FETCH_QUERY)
    result = CliRunner().invoke(main, ["--json", "plan", "-"], input=payload)
    assert result.exit_code == 0
    assert json.loads(result.output)["nodes"] > 0


def test_a_text_plan_is_refused_with_the_option_the_user_needs(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("Seq Scan on orders  (cost=0.00..1.00 rows=1 width=4)", encoding="utf-8")
    code, output = run("plan", str(plan_file))
    assert code == 1
    assert "FORMAT JSON" in output


def test_the_locks_command_prints_the_connection_table(seeded: PgInstance) -> None:
    code, output = run("locks", dsn=seeded.dsn)
    assert code == 0
    assert "connections" in output
    assert "no session is waiting" in output


def test_the_vacuum_command_shows_the_thresholds_it_used(seeded: PgInstance) -> None:
    code, output = run("vacuum", dsn=seeded.dsn)
    assert code == 0
    assert "freeze_max_age=200,000,000" in output
    assert "vacuum.skipped" in output


def test_the_index_report_repeats_the_statistics_caveat_where_it_is_read(
    seeded: PgInstance,
) -> None:
    code, output = run("indexes", dsn=seeded.dsn)
    assert code == 0
    assert "per node" in output

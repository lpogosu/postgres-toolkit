"""The command line.

Every subcommand does the same three things — collect findings, render them,
choose an exit code — so the shared machinery lives in ``_finish``. The one
command that does not connect to a database is ``plan``, deliberately: plans
arrive as text in an incident channel far more often than as a live connection.
"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from datetime import timedelta
from typing import Any

import click

from pgtk import __version__
from pgtk.bloat import PgstattupleUnavailableError, analyze_bloat
from pgtk.db import Database, connect
from pgtk.findings import Finding, Severity
from pgtk.indexes import analyze_indexes
from pgtk.locks import analyze_locks, connection_distribution
from pgtk.plans import PlanFormatError, PlanNotAnalyzedError, analyze_plan, parse_explain
from pgtk.render import (
    emit_json,
    make_console,
    render_bloat_table,
    render_blocking_forest,
    render_distribution,
    render_findings,
    render_vacuum_progress,
)
from pgtk.vacuum import analyze_vacuum, vacuum_in_progress

FAIL_LEVELS = {
    "never": None,
    "notice": Severity.NOTICE,
    "warning": Severity.WARNING,
    "critical": Severity.CRITICAL,
}


class Context:
    def __init__(self, dsn: str, as_json: bool, timeout_ms: int, fail_on: str) -> None:
        self.dsn = dsn
        self.as_json = as_json
        self.timeout_ms = timeout_ms
        self.fail_on = FAIL_LEVELS[fail_on]
        self.console = make_console()


pass_context = click.make_pass_decorator(Context)


def _finish(ctx: Context, findings: list[Finding], **extra: Any) -> None:
    if ctx.as_json:
        click.echo(emit_json(findings, **extra))
    if ctx.fail_on is not None and any(f.severity >= ctx.fail_on for f in findings):
        raise SystemExit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="pgtk")
@click.option(
    "--dsn",
    envvar="PGTK_DSN",
    default="",
    help="libpq connection string; empty means use the standard PG* environment variables.",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output on stdout.")
@click.option(
    "--statement-timeout",
    "timeout_ms",
    default=15_000,
    show_default=True,
    help="statement_timeout in milliseconds for every query this tool issues.",
)
@click.option(
    "--fail-on",
    type=click.Choice(sorted(FAIL_LEVELS)),
    default="never",
    show_default=True,
    help="Exit with status 1 when a finding reaches this severity.",
)
@click.pass_context
def main(
    click_ctx: click.Context, dsn: str, as_json: bool, timeout_ms: int, fail_on: str
) -> None:
    """Read-only PostgreSQL diagnostics. Nothing here writes to your database."""
    click_ctx.obj = Context(dsn, as_json, timeout_ms, fail_on)


def _open(ctx: Context) -> AbstractContextManager[Database]:
    return connect(ctx.dsn, statement_timeout_ms=ctx.timeout_ms)


def _server_facts(db: Database) -> dict[str, Any]:
    return {
        "server_version": str(db.version),
        "database": db.current_database(),
        "pgstattuple": db.caps.has_extension("pgstattuple"),
    }


@main.command()
@click.option("--schema", "schemas", multiple=True, help="Restrict to these schemas.")
@click.option(
    "--min-bytes",
    default=8 * 1024 * 1024,
    show_default=True,
    help="Ignore relations with less wasted space than this.",
)
@click.option("--min-pct", default=20.0, show_default=True, help="Ignore relations below this %.")
@click.option(
    "--exact",
    is_flag=True,
    help="Confirm the estimate with pgstattuple. This reads every page of every "
    "relation that passed the filters.",
)
@pass_context
def bloat(
    ctx: Context, schemas: tuple[str, ...], min_bytes: int, min_pct: float, exact: bool
) -> None:
    """Wasted space in tables and btree indexes."""
    with _open(ctx) as db:
        try:
            findings, rows = analyze_bloat(
                db,
                schemas=schemas,
                min_bloat_bytes=min_bytes,
                min_bloat_pct=min_pct,
                exact=exact,
            )
        except PgstattupleUnavailableError as exc:
            raise click.ClickException(str(exc)) from exc
        if not ctx.as_json:
            ctx.console.print(
                f"[dim]{db.current_database()} on PostgreSQL {db.version} — "
                f"{'measured with pgstattuple' if exact else 'statistics-based estimate'}[/dim]"
            )
            render_bloat_table(ctx.console, rows)
            render_findings(ctx.console, findings)
        _finish(ctx, findings, **_server_facts(db))


@main.command()
@click.option("--schema", "schemas", multiple=True, help="Restrict to these schemas.")
@click.option(
    "--min-unused-bytes",
    default=1024 * 1024,
    show_default=True,
    help="Ignore unused indexes smaller than this.",
)
@pass_context
def indexes(ctx: Context, schemas: tuple[str, ...], min_unused_bytes: int) -> None:
    """Unused, duplicate, redundant and invalid indexes, plus missing-index candidates."""
    with _open(ctx) as db:
        findings = analyze_indexes(db, schemas=schemas, min_unused_bytes=min_unused_bytes)
        if not ctx.as_json:
            render_findings(ctx.console, findings)
            _print_unused_caveat(ctx, db)
        _finish(ctx, findings, **_server_facts(db))


def _print_unused_caveat(ctx: Context, db: Database) -> None:
    reset = db.one(
        "SELECT stats_reset FROM pg_stat_database WHERE datname = current_database()"
    )["stats_reset"]
    if reset is None:
        ctx.console.print(
            "[yellow]pg_stat_database.stats_reset is NULL for this database[/yellow]: the scan "
            "counters have no known start point, so 'never used' means nothing yet."
        )
    else:
        ctx.console.print(
            f"[dim]index scan counters cover everything since {reset:%Y-%m-%d %H:%M %Z}. "
            f"A monthly report that runs after that date has not been observed.[/dim]"
        )
    ctx.console.print(
        "[dim]Counters are per node. An index unused on this instance may be the only thing "
        "keeping a read replica alive.[/dim]"
    )


@main.command()
@click.argument("plan_file", type=click.File("r", encoding="utf-8"), default="-")
@click.option(
    "--misestimate-factor",
    default=10.0,
    show_default=True,
    help="Report a node when estimated and actual row counts differ by this factor.",
)
@click.option(
    "--nested-loop-loops",
    default=10_000,
    show_default=True,
    help="Report a nested loop whose inner side runs at least this many times.",
)
@pass_context
def plan(
    ctx: Context, plan_file: Any, misestimate_factor: float, nested_loop_loops: int
) -> None:
    """Read EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) from a file or stdin."""
    try:
        document = parse_explain(plan_file.read())
    except (PlanFormatError, PlanNotAnalyzedError) as exc:
        raise click.ClickException(str(exc)) from exc
    findings = analyze_plan(
        document,
        misestimate_factor=misestimate_factor,
        nested_loop_threshold=nested_loop_loops,
    )
    if not ctx.as_json:
        ctx.console.print(
            f"[dim]planning {document.planning_ms or 0:.1f} ms, "
            f"execution {document.execution_ms or 0:.1f} ms, "
            f"{len(document.nodes())} nodes[/dim]"
        )
        render_findings(ctx.console, findings)
    _finish(
        ctx,
        findings,
        planning_ms=document.planning_ms,
        execution_ms=document.execution_ms,
        nodes=len(document.nodes()),
    )


@main.command()
@click.option(
    "--stale-hours",
    default=24,
    show_default=True,
    help="Call autovacuum 'skipping' a table when it last ran longer ago than this.",
)
@pass_context
def vacuum(ctx: Context, stale_hours: int) -> None:
    """Wraparound distance, autovacuum backlog, and what is holding the xmin horizon."""
    with _open(ctx) as db:
        findings, settings = analyze_vacuum(
            db, stale_autovacuum=timedelta(hours=stale_hours)
        )
        running = vacuum_in_progress(db)
        if not ctx.as_json:
            ctx.console.print(
                f"[dim]autovacuum={'on' if settings.autovacuum_on else 'off'} "
                f"freeze_max_age={settings.freeze_max_age:,} "
                f"threshold={settings.vacuum_threshold} + "
                f"{settings.vacuum_scale_factor} x reltuples[/dim]"
            )
            render_vacuum_progress(ctx.console, running)
            render_findings(ctx.console, findings)
        _finish(ctx, findings, vacuum_in_progress=running, **_server_facts(db))


@main.command()
@click.option(
    "--idle-minutes",
    default=5,
    show_default=True,
    help="Report sessions idle in transaction for longer than this.",
)
@pass_context
def locks(ctx: Context, idle_minutes: int) -> None:
    """Blocking chains as a tree, idle transactions, connection distribution."""
    with _open(ctx) as db:
        findings, forest = analyze_locks(
            db, idle_in_transaction=timedelta(minutes=idle_minutes)
        )
        distribution = connection_distribution(db)
        if not ctx.as_json:
            render_blocking_forest(ctx.console, forest)
            render_distribution(ctx.console, distribution)
            render_findings(ctx.console, findings)
        _finish(ctx, findings, connections=distribution, **_server_facts(db))


@main.command()
@click.option("--schema", "schemas", multiple=True, help="Restrict to these schemas.")
@pass_context
def report(ctx: Context, schemas: tuple[str, ...]) -> None:
    """Every check that needs only a connection, in one pass."""
    with _open(ctx) as db:
        bloat_findings, bloat_rows = analyze_bloat(db, schemas=schemas)
        findings = [
            *bloat_findings,
            *analyze_indexes(db, schemas=schemas),
            *analyze_vacuum(db)[0],
            *analyze_locks(db)[0],
        ]
        if not ctx.as_json:
            ctx.console.print(
                f"[bold]{db.current_database()}[/bold] on PostgreSQL {db.version}, "
                f"{len(findings)} finding(s)"
            )
            render_bloat_table(ctx.console, bloat_rows)
            render_findings(ctx.console, findings, show_sql=False)
            ctx.console.print(
                "[dim]run the individual commands for the SQL behind each finding[/dim]"
            )
        _finish(ctx, findings, **_server_facts(db))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Terminal output.

One rule shapes this module: the person reading it is tired. Severity is a word
and a colour, the SQL is separated from the prose so it can be selected with a
mouse, and nothing is truncated to make a table line up.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from pgtk.bloat import BloatRow
from pgtk.findings import Finding, Severity, human_bytes, sort_findings
from pgtk.locks import BlockingNode

SEVERITY_STYLE = {
    Severity.CRITICAL: "bold red",
    Severity.WARNING: "yellow",
    Severity.NOTICE: "cyan",
    Severity.INFO: "dim",
}


# Rich falls back to 80 columns when stdout is not a terminal, which turns a
# schema-qualified relation name into a vertical stack of single letters. This
# output is redirected into files and pasted into tickets at least as often as it
# is read live, so the piped default is a width someone can read.
PIPED_WIDTH = 120


def make_console(*, force_plain: bool = False) -> Console:
    columns = os.environ.get("COLUMNS", "")
    if columns.isdigit():
        width: int | None = int(columns)
    else:
        width = None if sys.stdout.isatty() else PIPED_WIDTH
    return Console(highlight=False, soft_wrap=False, no_color=force_plain, width=width)


def render_findings(console: Console, findings: list[Finding], *, show_sql: bool = True) -> None:
    if not findings:
        console.print("[green]no findings[/green] — every check ran and none of them fired")
        return

    table = Table(show_lines=False, expand=True, pad_edge=False)
    table.add_column("sev", width=8, no_wrap=True)
    table.add_column("check", width=24, no_wrap=True)
    table.add_column("subject", ratio=3, min_width=24, overflow="fold")
    table.add_column("what", ratio=5, min_width=30, overflow="fold")

    for finding in sort_findings(findings):
        style = SEVERITY_STYLE[finding.severity]
        table.add_row(
            f"[{style}]{finding.severity}[/{style}]",
            finding.check,
            finding.subject,
            finding.summary,
        )
    console.print(table)

    if not show_sql:
        return
    printed: set[str] = set()
    for finding in sort_findings(findings):
        if not finding.remediation or finding.remediation in printed:
            continue
        printed.add(finding.remediation)
        console.print(
            Panel(
                Syntax(finding.remediation, "sql", theme="ansi_dark", word_wrap=True),
                title=f"[dim]{finding.check}[/dim] {finding.subject}",
                title_align="left",
                border_style="dim",
            )
        )


def render_bloat_table(console: Console, rows: list[BloatRow]) -> None:
    if not rows:
        return
    table = Table(title="bloat detail", expand=True)
    table.add_column("kind", width=6, no_wrap=True)
    table.add_column("relation", ratio=1, min_width=32, overflow="fold")
    table.add_column("size", justify="right", width=10, no_wrap=True)
    table.add_column("bloat", justify="right", width=10, no_wrap=True)
    table.add_column("%", justify="right", width=5, no_wrap=True)
    table.add_column("method", width=12, no_wrap=True)
    table.add_column("stats", width=8, no_wrap=True)
    for row in sorted(rows, key=lambda r: -r.bloat_bytes):
        table.add_row(
            row.kind,
            row.subject if row.parent is None else f"{row.subject}  [dim]({row.parent})[/dim]",
            human_bytes(row.real_bytes),
            human_bytes(row.bloat_bytes),
            f"{row.bloat_pct:.0f}",
            row.method,
            "[yellow]partial[/yellow]" if row.stats_unusable else "ok",
        )
    console.print(table)


def render_blocking_forest(console: Console, forest: list[BlockingNode]) -> None:
    if not forest:
        console.print("[green]no session is waiting on a lock held by another[/green]")
        return
    for root in forest:
        tree = Tree(_node_label(root, is_root=True))
        _attach(tree, root)
        console.print(tree)


def _node_label(node: BlockingNode, *, is_root: bool) -> str:
    session = node.session
    marker = "[bold red]holds[/bold red]" if is_root else "[yellow]waits[/yellow]"
    waiting = f" [dim]{node.waiting_for}[/dim]" if node.waiting_for and not is_root else ""
    return (
        f"{marker} pid [bold]{session.pid}[/bold] "
        f"{session.user}@{session.application or '-'} "
        f"[dim]state={session.state or '-'}[/dim]{waiting}\n"
        f"      [dim]{session.one_line or '<no query text>'}[/dim]"
    )


def _attach(tree: Tree, node: BlockingNode) -> None:
    for child in node.children:
        branch = tree.add(_node_label(child, is_root=False))
        _attach(branch, child)


def render_vacuum_progress(console: Console, rows: list[dict[str, Any]]) -> None:
    """What autovacuum is doing right now.

    Usually empty, and that is itself the answer: "autovacuum should have run"
    reads differently when a worker is already three quarters through the table.
    """
    if not rows:
        return
    table = Table(title="vacuum in progress", expand=True)
    table.add_column("pid", justify="right", width=7, no_wrap=True)
    table.add_column("relation", ratio=1, min_width=24, overflow="fold")
    table.add_column("phase", width=24, no_wrap=True)
    table.add_column("scanned", justify="right", width=16, no_wrap=True)
    table.add_column("worker", width=18, no_wrap=True)
    for row in rows:
        total = int(row["heap_blks_total"] or 0)
        done = int(row["heap_blks_scanned"] or 0)
        share = f"{done:,}/{total:,}" if total else f"{done:,}"
        table.add_row(
            str(row["pid"]),
            str(row["relation"]),
            str(row["phase"]),
            share,
            str(row["backend_type"]),
        )
    console.print(table)


def render_distribution(console: Console, rows: list[dict[str, Any]], limit: int = 15) -> None:
    if not rows:
        return
    table = Table(title="connections", expand=True)
    table.add_column("database", ratio=2, min_width=10, overflow="fold")
    table.add_column("user", ratio=2, min_width=8, overflow="fold")
    table.add_column("application", ratio=3, min_width=12, overflow="fold")
    table.add_column("state", width=24, no_wrap=True)
    table.add_column("n", justify="right", width=4, no_wrap=True)
    table.add_column("oldest", justify="right", width=8, no_wrap=True)
    for row in rows[:limit]:
        age = float(row["max_state_age_s"] or 0)
        table.add_row(
            str(row["datname"]),
            str(row["usename"]),
            str(row["application_name"]),
            str(row["state"]),
            str(row["sessions"]),
            f"{age:.0f}s" if age < 90 else f"{age / 60:.0f}m",
        )
    console.print(table)


def emit_json(findings: list[Finding], **extra: Any) -> str:
    payload: dict[str, Any] = {"findings": [f.as_dict() for f in sort_findings(findings)]}
    payload.update(extra)
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

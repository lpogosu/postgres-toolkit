"""The single shape every check returns.

Every check in this toolkit answers the same three questions — what is wrong, how
badly, and what SQL a human might run about it — so they all return the same
record. The uniformity is what makes ``pgtk report`` and ``--json`` possible
without a per-check formatter.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    """Ordered so that ``max()`` and ``sorted()`` do the obvious thing."""

    INFO = 0
    NOTICE = 1
    WARNING = 2
    CRITICAL = 3

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Finding:
    """One observation about one object.

    ``remediation`` is SQL text that is *printed*, never executed. See the
    read-only argument in the README: a tool run under pressure at 3am must not
    be able to start a REINDEX by itself.
    """

    check: str
    severity: Severity
    subject: str
    summary: str
    facts: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": str(self.severity),
            "subject": self.subject,
            "summary": self.summary,
            "facts": self.facts,
            "remediation": self.remediation,
        }


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Worst first, then stable by check name and subject so output is diffable."""
    return sorted(findings, key=lambda f: (-int(f.severity), f.check, f.subject))


def human_bytes(value: float) -> str:
    """Byte counts the way ``pg_size_pretty`` writes them, so output matches psql."""
    step = 1024.0
    amount = float(value)
    for unit in ("bytes", "kB", "MB", "GB", "TB"):
        if abs(amount) < step or unit == "TB":
            if unit == "bytes":
                return f"{int(amount)} {unit}"
            return f"{amount:.0f} {unit}" if abs(amount) >= 100 else f"{amount:.1f} {unit}"
        amount /= step
    raise AssertionError("unreachable")

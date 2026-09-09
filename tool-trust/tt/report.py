"""Render recorded trials as a detection-rate table."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .detect import PROPAGATED
from .faults import ALL, DESCRIPTIONS

console = Console()


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(trials: list[dict[str, Any]]) -> None:
    if not trials:
        console.print(
            "[yellow]No trials recorded yet.[/yellow] The harness is complete but "
            "the model calls are not wired up -- see the README."
        )
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in trials:
        grouped[(t["fault"], t["variant"])].append(t)

    variants = sorted({t["variant"] for t in trials})

    table = Table(title="how often a wrong tool result reaches the user unremarked")
    table.add_column("fault")
    table.add_column("what the tool returned", style="dim")
    for v in variants:
        table.add_column(f"{v}\npropagated", justify="right")

    for fault in ALL:
        row = [fault, DESCRIPTIONS[fault]]
        for variant in variants:
            group = grouped.get((fault, variant), [])
            if not group:
                row.append("[dim]--[/dim]")
                continue
            n = len(group)
            propagated = sum(1 for t in group if t["label"] == PROPAGATED)
            row.append(f"{propagated / n:.0%}  [dim]({propagated}/{n})[/dim]")
        table.add_row(*row)

    console.print(table)
    console.print(
        "\n[dim]propagated = the tool's wrong value appears in the final answer "
        "with nothing said about it.[/dim]"
    )

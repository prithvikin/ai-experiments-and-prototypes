"""Command line: run the experiment, or render what has been recorded."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .report import load, render
from .runner import NotWiredAgent, NotWiredUpError, run_experiment

console = Console()
DEFAULT_RESULTS = Path("results/trials.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the experiment (needs a wired-up agent)")
    run.add_argument("-n", type=int, default=10, help="trials per cell")
    run.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    run.add_argument("--seed", type=int, default=0)

    report = sub.add_parser("report", help="render recorded trials")
    report.add_argument("--results", type=Path, default=DEFAULT_RESULTS)

    args = parser.parse_args(argv)

    if args.command == "report":
        render(load(args.results))
        return 0

    try:
        trials = run_experiment(
            NotWiredAgent(), trials_per_cell=args.n, out=args.out, seed=args.seed
        )
    except NotWiredUpError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return 1

    render([t.__dict__ for t in trials])
    return 0

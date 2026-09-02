"""Command line: serve the proxy, read the ledger, print the waste report."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .audit import run as run_audit
from .ledger import Ledger
from .pricing import usd

console = Console()

SEVERITY_STYLE = {"high": "bold red", "medium": "yellow", "low": "dim", "info": "cyan"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbp", description=__doc__)
    parser.add_argument("--db", default=os.environ.get("TBP_DB", "tbp.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the metering proxy")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--policy", default=os.environ.get("TBP_POLICY", "policy.yaml"))
    serve.add_argument(
        "--upstream",
        default=os.environ.get("TBP_UPSTREAM", "https://api.anthropic.com"),
        help="'mock' runs a local simulator so the demo works without an API key",
    )

    report = sub.add_parser("report", help="print the waste report")
    report.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="show only the N costliest findings (the count and total still reflect all of them)",
    )
    calls = sub.add_parser("calls", help="print the raw ledger")
    calls.add_argument("-n", type=int, default=30)
    sub.add_parser("reset", help="empty the ledger")

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args)
    ledger = Ledger(args.db)
    if args.command == "reset":
        console.print(f"cleared {ledger.clear()} calls from {args.db}")
        return 0
    if args.command == "report":
        return _report(ledger, args.top)
    return _calls(ledger, args.n)


def _serve(args) -> int:
    import uvicorn

    from .proxy import create_app

    app = create_app(db=args.db, policy_path=args.policy, upstream=args.upstream)
    label = "mock (no API key needed)" if args.upstream == "mock" else args.upstream
    console.print(
        Panel(
            f"[bold]token-budget-proxy[/bold]\n"
            f"listening   http://{args.host}:{args.port}\n"
            f"upstream    {label}\n"
            f"policy      {args.policy} ({len(app.state.policy.rules)} rules)\n"
            f"ledger      {args.db}\n\n"
            f"Point any client at it:\n"
            f'  [cyan]Anthropic(base_url="http://{args.host}:{args.port}")[/cyan]',
            border_style="cyan",
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _report(ledger: Ledger, top: int | None = None) -> int:
    rows = ledger.rows()
    if not rows:
        console.print("[dim]ledger is empty -- run some traffic through the proxy first[/dim]")
        return 0

    billed = [r for r in rows if not r["denied_reason"]]
    denied = [r for r in rows if r["denied_reason"]]
    total = sum(r["cost_usd"] for r in billed)

    by_agent: dict[str, float] = defaultdict(float)
    for r in billed:
        by_agent[r["agent"]] += r["cost_usd"]

    summary = Table(box=None, pad_edge=False)
    summary.add_column("", style="dim")
    summary.add_column("")
    summary.add_row("calls", f"{len(billed)} billed, {len(denied)} blocked by policy")
    summary.add_row("spend", usd(total))
    for agent, amount in sorted(by_agent.items(), key=lambda kv: -kv[1]):
        summary.add_row("", f"[dim]{agent}[/dim]  {usd(amount)}")

    # What policy actually changed. Worth printing even when it is nothing:
    # silence here is the difference between "no rule matched" and "no rules
    # loaded", and those look identical from the spend line alone.
    rewrites: dict[str, int] = defaultdict(int)
    for r in billed:
        for action in r["policy_actions"]:
            rewrites[action.split(":", 1)[0]] += 1
    if rewrites:
        summary.add_row("", "")
        summary.add_row("policy", f"{sum(rewrites.values())} rewrites")
        for rule, n in sorted(rewrites.items(), key=lambda kv: -kv[1]):
            summary.add_row("", f"[dim]{rule}[/dim]  {n} calls")
    for r in denied:
        summary.add_row("blocked", f"[red]{r['agent']}[/red]  {r['denied_reason']}")

    console.print(Panel(summary, title="ledger", border_style="dim", title_align="left"))

    findings = run_audit(ledger)
    if not findings:
        console.print("\n[green]No waste found.[/green]")
        return 0

    recoverable = sum(f.wasted_usd for f in findings)
    share = f" ({recoverable / total:.0%} of spend)" if total else ""
    shown = findings[:top] if top else findings
    # The totals always describe every finding, even when the list is trimmed --
    # a headline that shrinks with a display flag is a headline you cannot trust.
    trimmed = f" · [dim]showing top {len(shown)}[/dim]" if len(shown) < len(findings) else ""
    console.print(
        f"\n[bold]{len(findings)} findings[/bold] · "
        f"[bold green]{usd(recoverable)} recoverable{share}[/bold green]{trimmed}\n"
    )

    for f in shown:
        style = SEVERITY_STYLE.get(f.severity, "")
        head = Text.assemble(
            (f"{f.severity.upper():6}", style),
            (f"  {f.title}", "bold"),
            ("  " + (usd(f.wasted_usd) if f.wasted_usd else "not costed"), "green"),
        )
        body = Text()
        body.append(f.detail + "\n\n")
        for line in f.evidence:
            body.append(line + "\n", style="dim")
        if f.evidence:
            body.append("\n")
        body.append("Fix: ", style="bold")
        body.append(f.fix)
        console.print(Panel(body, title=head, title_align="left", border_style=style or "dim"))

    return 0


def _calls(ledger: Ledger, limit: int) -> int:
    rows = ledger.rows(limit=limit)
    table = Table(title=f"last {len(rows)} calls")
    for col in ("id", "agent", "model", "in", "cache w/r", "out", "stop", "cost"):
        table.add_column(col, overflow="fold")
    for r in rows:
        cost = "[red]blocked[/red]" if r["denied_reason"] else usd(r["cost_usd"])
        table.add_row(
            str(r["id"]),
            r["agent"],
            r["model"].replace("claude-", ""),
            f"{r['input_tokens']:,}",
            f"{r['cache_creation_input_tokens']:,}/{r['cache_read_input_tokens']:,}",
            f"{r['output_tokens']:,}",
            r["stop_reason"] or "-",
            cost,
        )
    console.print(table)
    for r in rows:
        for action in r["policy_actions"]:
            console.print(f"  [cyan]#{r['id']} policy[/cyan] {action}")
    return 0

"""Run the same workload three ways and put the numbers side by side.

    baseline        the agent as written, proxy metering only
    policy enforced the same agent, untouched, with rules turned on
    agent fixed     the agent rewritten, metering only

The middle column is the interesting one: nothing about the agent changed, so
whatever it saves is available without touching application code. The third
column is the ceiling -- what is left once someone actually edits the source.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx2 as httpx
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tbp.audit import run as run_audit  # noqa: E402
from tbp.ledger import Ledger  # noqa: E402
from tbp.pricing import usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PORT = 8799
HEALTH = f"http://127.0.0.1:{PORT}/healthz"

SCENARIOS = [
    ("baseline", "policy.observe.yaml", "sloppy"),
    ("policy enforced", "policy.yaml", "sloppy"),
    ("agent fixed", "policy.observe.yaml", "tuned"),
]

console = Console()


def wait_for_health(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(HEALTH, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("proxy did not come up")


def run_scenario(name: str, policy: str, mode: str) -> dict:
    db = ROOT / f".compare-{mode}-{Path(policy).stem}.db"
    db.unlink(missing_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "tbp", "--db", str(db), "serve",
         "--port", str(PORT), "--upstream", "mock", "--policy", policy],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_health()
        env_agent = ROOT / "demo" / "agent.py"
        result = subprocess.run(
            [sys.executable, str(env_agent), mode],
            cwd=ROOT, capture_output=True, text=True,
            env={**__import__("os").environ, "TBP_PORT": str(PORT)},
        )
        if result.returncode != 0:
            raise RuntimeError(f"{name}: agent failed\n{result.stderr}")
    finally:
        server.terminate()
        server.wait(timeout=10)

    ledger = Ledger(db)
    rows = [r for r in ledger.rows() if not r["denied_reason"]]
    findings = run_audit(ledger)
    result = {
        "name": name,
        "spend": sum(r["cost_usd"] for r in rows),
        "calls": len(rows),
        "findings": len(findings),
        "recoverable": sum(f.wasted_usd for f in findings),
        "rewrites": sum(len(r["policy_actions"]) for r in rows),
        "by_agent": {},
    }
    for r in rows:
        result["by_agent"][r["agent"]] = result["by_agent"].get(r["agent"], 0) + r["cost_usd"]
    ledger.close()
    return result


def main() -> int:
    results = [run_scenario(*s) for s in SCENARIOS]
    baseline = results[0]["spend"]

    table = Table(title="same workload, three configurations")
    table.add_column("")
    for r in results:
        table.add_column(r["name"], justify="right")

    def row(label, fn):
        table.add_row(label, *[fn(r) for r in results])

    row("spend", lambda r: usd(r["spend"]))
    row("vs baseline", lambda r: "--" if r["spend"] == baseline
        else f"[green]-{(baseline - r['spend']) / baseline:.0%}[/green]")
    row("calls", lambda r: str(r["calls"]))
    row("policy rewrites", lambda r: str(r["rewrites"]))
    row("findings", lambda r: str(r["findings"]))
    row("still recoverable", lambda r: usd(r["recoverable"]))

    agents = sorted({a for r in results for a in r["by_agent"]})
    for agent in agents:
        row(f"  {agent}", lambda r, a=agent: usd(r["by_agent"].get(a, 0)))

    console.print(table)
    console.print(
        "\n[dim]report-writer costs more once truncation is fixed. That is the "
        "correct outcome: the cheaper runs were paying in full for answers that "
        "stopped mid-sentence.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

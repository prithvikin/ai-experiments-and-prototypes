"""Three small agents, run badly and then run well, through the proxy.

The point of the demo is that the *only* thing separating the two runs is how
the requests are shaped. Same work, same number of calls, same models available
-- the difference is entirely in caching, tool surface, model choice, and
max_tokens. Run both and diff the report.

    python -m tbp serve --upstream mock &
    python demo/agent.py sloppy   && python -m tbp report
    python -m tbp reset
    python demo/agent.py tuned    && python -m tbp report
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import anthropic

import os

PROXY = f"http://127.0.0.1:{os.environ.get('TBP_PORT', '8787')}"
HANDBOOK = (Path(__file__).parent / "handbook.md").read_text()

DOCUMENTS = [
    "Invoice 88213 from Northwind Traders, 4,210.00 EUR, net 30, dated 2026-03-04.",
    "Statement of work amendment, Contoso Ltd, two-year term with auto-renewal.",
    "Card receipt, 84.20 USD, merchant Blue Bottle, 2026-03-06.",
    "Quotation Q-5512 from Fabrikam for 31,000 GBP of hardware, valid 30 days.",
    "Letter from the tax office regarding filing reference 99-2213.",
    "Invoice 88214 from Northwind Traders, 1,150.00 EUR, net 30.",
    "Renewal notice, Adventure Works, unlimited liability clause on page 4.",
    "Receipt, 12.40 EUR, rail travel, 2026-03-09.",
    "Invoice 90021 from Litware Inc, 27,500.00 USD, due on receipt.",
    "Internal expenses policy, revision 12. Read-only.",
    "Invoice 88213 from Northwind Traders, 4,210.00 EUR, net 30.",
    "Purchase order acknowledgement from Tailspin Toys, no amount stated.",
]

TRIAGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "document_class": {"type": "string"},
            "counterparty": {"type": ["string", "null"]},
            "amount": {"type": ["number", "null"]},
            "requires_review": {"type": "boolean"},
        },
        "required": ["document_class", "counterparty", "amount", "requires_review"],
        "additionalProperties": False,
    },
}

# Only `lookup_policy` is ever reachable from the triage prompt. The other four
# are here because somebody added them once and nobody took them out again.
ALL_TOOLS = [
    {"name": "lookup_policy", "description": "Look up a clause in the triage handbook by section.",
     "input_schema": {"type": "object", "properties": {"section": {"type": "string"}}, "required": ["section"]}},
    {"name": "search_vendor_database", "description": "Search the vendor master file by name, tax id, or remittance account.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}},
    {"name": "fetch_exchange_rate", "description": "Fetch the closing exchange rate between two currencies on a given date.",
     "input_schema": {"type": "object", "properties": {"base": {"type": "string"}, "quote": {"type": "string"}, "date": {"type": "string"}}, "required": ["base", "quote", "date"]}},
    {"name": "create_review_ticket", "description": "Open a review ticket for a document that requires human attention.",
     "input_schema": {"type": "object", "properties": {"document_id": {"type": "string"}, "reason": {"type": "string"}, "priority": {"type": "string"}}, "required": ["document_id", "reason"]}},
    {"name": "send_notification", "description": "Notify a channel or individual that a document has been routed.",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "message": {"type": "string"}}, "required": ["target", "message"]}},
]


def client(agent: str, session: str, task: str | None = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=PROXY,
        # The proxy holds the real credential; the client just needs a non-empty one.
        api_key="proxy-managed",
        max_retries=0,
        default_headers={
            "x-tbp-agent": agent,
            "x-tbp-session": session,
            **({"x-tbp-task": task} if task else {}),
        },
    )


# ---------------------------------------------------------------------------


def triage(tuned: bool) -> None:
    """12 classifications. Short structured answers over a long, stable handbook."""
    c = client("doc-triage", f"triage-{'tuned' if tuned else 'sloppy'}", "classify")

    for doc in DOCUMENTS:
        if tuned:
            # Frozen prefix: the same bytes on every call, so it caches.
            system = [{"type": "text", "text": HANDBOOK,
                       "cache_control": {"type": "ephemeral"}}]
            kwargs = {"model": "claude-sonnet-5",
                      "output_config": {"effort": "low", "format": TRIAGE_SCHEMA},
                      "tools": ALL_TOOLS[:1]}
        else:
            # The timestamp sits inside the prefix, so every request is a fresh
            # prefix and the breakpoint below can never be read back.
            system = [{"type": "text",
                       "text": HANDBOOK + f"\n\nCurrent time: {dt.datetime.now().isoformat()}",
                       "cache_control": {"type": "ephemeral"}}]
            kwargs = {"model": "claude-opus-5",
                      "output_config": {"format": TRIAGE_SCHEMA},
                      "tools": ALL_TOOLS}

        prompt = f"Triage this document:\n\n{doc}"
        if "contract" in doc.lower() or "clause" in doc.lower():
            # Only this branch can ever reach a tool, which is the point: the
            # other four definitions are paid for on all twelve calls.
            prompt += "\n\nUse lookup_policy if a clause is ambiguous."

        c.messages.create(
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )


def report_writer(tuned: bool) -> None:
    """4 long-form summaries over the same handbook."""
    c = client("report-writer", f"reports-{'tuned' if tuned else 'sloppy'}", "summarise")

    system: list[dict] = [{"type": "text", "text": HANDBOOK}]
    if tuned:
        system[0]["cache_control"] = {"type": "ephemeral"}

    for week in range(1, 5):
        c.messages.create(
            model="claude-opus-5",
            # Sloppy leaves max_tokens where it was during prototyping; the model
            # runs out of room mid-report and the whole output is billed anyway.
            max_tokens=16000 if tuned else 300,
            system=system,
            messages=[{"role": "user",
                       "content": f"Write the week {week} triage summary for the operations review."}],
        )


def researcher(tuned: bool) -> None:
    """5 turns of an agent loop that keeps re-reading its own tool output."""
    c = client("researcher", f"research-{'tuned' if tuned else 'sloppy'}", "research")

    system = "You are a research assistant. Use the notes gathered so far."
    history: list[dict] = []

    for turn in range(5):
        history.append({"role": "user", "content": f"Continue the review. Step {turn + 1}."})
        sent = history if not tuned else _recent(history)
        c.messages.create(
            model="claude-opus-5",
            max_tokens=4096,
            system=system,
            messages=sent,
        )
        history.append({"role": "assistant", "content": "Noted."})
        # Each step pulls in a fat tool result that stays in the transcript for
        # every subsequent turn.
        history.append({"role": "user", "content": f"Tool output {turn + 1}:\n" + _fake_result(turn)})
        history.append({"role": "assistant", "content": "Recorded."})


def _recent(history: list[dict], keep: int = 4) -> list[dict]:
    """Tuned run drops history the task has finished with.

    Real code would use server-side context editing rather than slicing a list;
    this is the same idea small enough to read.
    """
    return history[-keep:] if len(history) > keep else history


def _fake_result(turn: int) -> str:
    return (
        f"record {turn}: " + "field=value; " * 260
    )


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sloppy", "tuned"])
    args = parser.parse_args()
    tuned = args.mode == "tuned"

    try:
        triage(tuned)
        report_writer(tuned)
        researcher(tuned)
    except anthropic.APIConnectionError:
        print(f"cannot reach the proxy at {PROXY} -- start it with:\n"
              f"  python -m tbp serve --upstream mock", file=sys.stderr)
        return 1
    except anthropic.PermissionDeniedError as e:
        # A budget or policy rule stopped the run. That is the proxy working.
        print(f"blocked by policy: {e.message}", file=sys.stderr)
        return 0

    print(f"{args.mode} run complete -- now run: python -m tbp report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

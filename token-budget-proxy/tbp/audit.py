"""The waste report: findings, each with a dollar figure and a byte-level cause.

Every analyser here answers the same question in a different way -- "what did
this run pay for that it did not have to?" -- and each one has to be able to
show its work. A finding without evidence is a guess, and a guess with a dollar
sign in front of it is worse than nothing.

Where a saving cannot be estimated honestly, the finding reports the
observation and leaves the number at zero rather than inventing one.
"""

from __future__ import annotations

import difflib
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .ledger import Ledger
from .pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    cost_of,
    rate_for,
    usd,
)

Rows = list[dict[str, Any]]


@dataclass
class Finding:
    id: str
    title: str
    severity: str          # high | medium | low | info
    wasted_usd: float
    detail: str
    fix: str
    evidence: list[str] = field(default_factory=list)


ANALYSERS: list[Callable[[Rows, Ledger], list[Finding]]] = []


def analyser(fn):
    ANALYSERS.append(fn)
    return fn


def run(ledger: Ledger) -> list[Finding]:
    rows = [r for r in ledger.rows() if not r.get("denied_reason")]
    findings: list[Finding] = []
    for fn in ANALYSERS:
        findings.extend(fn(rows, ledger))
    findings.sort(key=lambda f: (-f.wasted_usd, f.severity))
    return findings


# ---------------------------------------------------------------------------
# 1. Cache breakpoints that never hit
# ---------------------------------------------------------------------------


@analyser
def cache_invalidated(rows: Rows, ledger: Ledger) -> list[Finding]:
    """Someone asked for caching and got none.

    Caching is a prefix match on exact bytes, so the usual cause is a single
    volatile value -- a timestamp, a uuid, a re-ordered dict -- sitting inside
    the prefix. This analyser finds the group, then diffs two consecutive
    prefixes to name the offending bytes.
    """
    findings = []
    for (agent, model), group in _group(rows, "agent", "model").items():
        eligible = [r for r in group if r["had_cache_control"] and _over_minimum(r)]
        if len(eligible) < 3:
            continue
        if any(r["cache_read_input_tokens"] > 0 for r in eligible):
            continue

        hashes = {r["prefix_hash"] for r in eligible}
        rate = rate_for(model)
        if rate is None:
            continue
        prefix_tokens = int(statistics.median(r["prefix_tokens"] for r in eligible))
        per_token = rate.input_per_mtok / 1_000_000
        # They paid the write premium on every call and never read one back.
        wasted = (
            (len(eligible) - 1)
            * prefix_tokens
            * per_token
            * (CACHE_WRITE_MULTIPLIER["5m"] - CACHE_READ_MULTIPLIER)
        )

        if len(hashes) == 1:
            detail = (
                f"{len(eligible)} calls sent a byte-identical {prefix_tokens:,}-token "
                f"prefix with a cache breakpoint, and every one of them read zero "
                f"cached tokens. The prefix is stable, so the cause is not an "
                f"invalidator -- the entries are expiring between calls."
            )
            fix = (
                "Calls are spaced further apart than the 5-minute TTL. Either "
                "batch the work closer together or set "
                '`cache_control={"type": "ephemeral", "ttl": "1h"}`. The 1h TTL '
                "costs 2x to write instead of 1.25x, so it needs at least three "
                "reads to pay for itself."
            )
            evidence = [f"prefix_hash {hashes.pop()} constant across all {len(eligible)} calls"]
        else:
            detail = (
                f"{len(eligible)} calls sent a ~{prefix_tokens:,}-token prefix with a "
                f"cache breakpoint and read zero cached tokens, across "
                f"{len(hashes)} distinct prefix hashes. The prefix is changing "
                f"between calls, so no two requests can ever share an entry."
            )
            fix = (
                "Something volatile is rendered inside the prefix. Move it after "
                "the last breakpoint -- into a message rather than the system "
                "prompt -- or make it deterministic. Common causes: "
                "`datetime.now()` in the system prompt, `uuid4()` in a header "
                "block, `json.dumps()` without `sort_keys=True`, or a tool list "
                "built per-user."
            )
            evidence = _divergence(ledger, eligible)

        findings.append(
            Finding(
                id="cache-invalidated",
                title=f"Prompt cache never hits for {agent} on {model}",
                severity="high",
                wasted_usd=wasted,
                detail=detail,
                fix=fix,
                evidence=evidence,
            )
        )
    return findings


def _divergence(ledger: Ledger, rows: Rows) -> list[str]:
    """Locate where two prefixes stop agreeing and show that spot in context.

    Prompt caching keys on exact bytes, so the useful answer is not "these
    differ" but "they differ *here*" -- the first divergence is where the cache
    stopped being reusable, and everything downstream of it is collateral.
    """
    distinct: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        if r["prefix_hash"] not in seen:
            seen.add(r["prefix_hash"])
            distinct.append(r)
        if len(distinct) == 2:
            break
    if len(distinct) < 2:
        return []

    a = ledger.prefix_text(distinct[0]["id"]) or ""
    b = ledger.prefix_text(distinct[1]["id"]) or ""
    if not a or not b:
        return []

    split = _first_difference(a, b)
    shared = split / max(len(a), len(b))
    regions = sum(
        1
        for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        if op[0] != "equal"
    )

    evidence = [
        f"prefixes agree for {split:,} bytes, then diverge "
        f"({shared:.1%} of the way in, {regions} differing region(s))",
        f"  call #{distinct[0]['id']}: {_around(a, split)}",
        f"  call #{distinct[1]['id']}: {_around(b, split)}",
        "  everything after the divergence point is uncacheable",
    ]
    return evidence


def _first_difference(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _around(text: str, split: int, before: int = 48, after: int = 24) -> str:
    """Show the divergence point with a marker, so the eye lands on it."""
    lo = max(0, split - before)
    head = text[lo:split].replace("\n", "\\n")
    tail = text[split:split + after].replace("\n", "\\n")
    lead = "..." if lo > 0 else ""
    trail = "..." if split + after < len(text) else ""
    return f"{lead}{head}\u2503{tail}{trail}"


# ---------------------------------------------------------------------------
# 2. Cacheable prefixes with no breakpoint at all
# ---------------------------------------------------------------------------


@analyser
def cache_absent(rows: Rows, ledger: Ledger) -> list[Finding]:
    findings = []
    for (agent, model, prefix_hash), group in _group(
        rows, "agent", "model", "prefix_hash"
    ).items():
        group = [r for r in group if not r["had_cache_control"] and _over_minimum(r)]
        if len(group) < 3:
            continue
        rate = rate_for(model)
        if rate is None:
            continue

        prefix_tokens = group[0]["prefix_tokens"]
        per_token = rate.input_per_mtok / 1_000_000
        n = len(group)
        paid = n * prefix_tokens * per_token
        ideal = prefix_tokens * per_token * (
            CACHE_WRITE_MULTIPLIER["5m"] + (n - 1) * CACHE_READ_MULTIPLIER
        )

        findings.append(
            Finding(
                id="cache-absent",
                title=f"{prefix_tokens:,}-token prefix re-sent {n}x uncached ({agent})",
                severity="high",
                wasted_usd=max(0.0, paid - ideal),
                detail=(
                    f"{n} calls sent the same {prefix_tokens:,}-token prefix "
                    f"(hash {prefix_hash}) at full input price with no cache "
                    f"breakpoint anywhere in the request. The prefix clears "
                    f"{model}'s {rate.min_cacheable_tokens:,}-token minimum, so it "
                    f"is cacheable as-is."
                ),
                fix=(
                    'Add `cache_control={"type": "ephemeral"}` to the request. '
                    "Reads bill at 0.1x and the write premium is 1.25x, so this "
                    "pays for itself on the second call."
                ),
                evidence=[f"identical prefix hash {prefix_hash} across {n} calls"],
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 3. Tool definitions that are never called
# ---------------------------------------------------------------------------


@analyser
def dead_tools(rows: Rows, ledger: Ledger) -> list[Finding]:
    findings = []
    for agent, group in _group(rows, "agent").items():
        declared: set[str] = set()
        called: set[str] = set()
        for r in group:
            declared.update(r["tools_declared"])
            called.update(r["tools_called"])
        dead = declared - called
        if not dead or not declared:
            continue

        share = len(dead) / len(declared)
        wasted = 0.0
        for r in group:
            rate = rate_for(r["model"])
            if rate is None or not r["tools_declared"]:
                continue
            # Bill the dead definitions at whatever this call actually paid:
            # cached tool defs are nearly free, uncached ones are not.
            multiplier = (
                CACHE_READ_MULTIPLIER if r["cache_read_input_tokens"] > 0 else 1.0
            )
            wasted += (
                r["tool_def_tokens"] * share * rate.input_per_mtok / 1_000_000 * multiplier
            )

        findings.append(
            Finding(
                id="dead-tools",
                title=f"{len(dead)} of {len(declared)} tools never called ({agent})",
                severity="medium" if wasted > 0.001 else "low",
                wasted_usd=wasted,
                detail=(
                    f"Tool definitions render at the very front of every request, "
                    f"so {len(dead)} unused definitions were re-sent on all "
                    f"{len(group)} calls: {', '.join(sorted(dead))}."
                ),
                fix=(
                    "Drop the tools this agent does not use. If they are needed "
                    "only occasionally, mark them `defer_loading: true` and add "
                    "the tool-search server tool so Claude pulls a definition in "
                    "on demand -- but never defer every tool, at least one must "
                    "stay loaded."
                ),
                evidence=[f"called: {', '.join(sorted(called)) or 'none'}"],
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 4. Output paid for and thrown away
# ---------------------------------------------------------------------------


@analyser
def truncation(rows: Rows, ledger: Ledger) -> list[Finding]:
    hits = [r for r in rows if r["stop_reason"] == "max_tokens"]
    if not hits:
        return []

    wasted = sum(
        cost_of(r["model"], output_tokens=r["output_tokens"]).total for r in hits
    )
    by_agent = defaultdict(int)
    for r in hits:
        by_agent[r["agent"]] += 1

    return [
        Finding(
            id="truncation-retry",
            title=f"{len(hits)} responses truncated at max_tokens",
            severity="medium",
            wasted_usd=wasted,
            detail=(
                "These calls hit the `max_tokens` ceiling and stopped mid-answer. "
                "Every output token was billed and the response is unusable, so "
                "the work is paid for twice once it is retried."
            ),
            fix=(
                "Raise `max_tokens` -- ~16000 for unary requests, more when "
                "streaming, since streaming removes the HTTP-timeout reason to "
                "keep it low. To shorten answers, lower `output_config.effort` "
                "instead: that makes the model aim shorter rather than get cut off."
            ),
            evidence=[f"{agent}: {n} truncated" for agent, n in sorted(by_agent.items())],
        )
    ]


# ---------------------------------------------------------------------------
# 5. Expensive model on cheap work
# ---------------------------------------------------------------------------


@analyser
def oversized_model(rows: Rows, ledger: Ledger) -> list[Finding]:
    """Short, tool-free answers on a top-tier model look like classification."""
    findings = []
    for (agent, model, task), group in _group(rows, "agent", "model", "task").items():
        rate = rate_for(model)
        cheap = rate_for("claude-haiku-4-5")
        if rate is None or cheap is None or rate.input_per_mtok <= cheap.input_per_mtok:
            continue
        candidates = [
            r for r in group
            if r["output_tokens"] < 200 and not r["tools_called"]
            and r["stop_reason"] != "max_tokens"
        ]
        if len(candidates) < 3:
            continue

        current = sum(
            cost_of(model, input_tokens=r["input_tokens"],
                    output_tokens=r["output_tokens"],
                    cache_creation_input_tokens=r["cache_creation_input_tokens"],
                    cache_read_input_tokens=r["cache_read_input_tokens"]).total
            for r in candidates
        )
        alternative = sum(
            cost_of("claude-haiku-4-5", input_tokens=r["input_tokens"],
                    output_tokens=r["output_tokens"],
                    cache_creation_input_tokens=r["cache_creation_input_tokens"],
                    cache_read_input_tokens=r["cache_read_input_tokens"]).total
            for r in candidates
        )

        label = f"{agent}/{task}" if task else agent
        findings.append(
            Finding(
                id="model-oversized",
                title=f"{len(candidates)} short answers on {model} ({label})",
                severity="medium",
                wasted_usd=max(0.0, current - alternative),
                detail=(
                    f"These calls produced under 200 output tokens and called no "
                    f"tools -- the shape of classification or extraction, not "
                    f"reasoning. {model} bills input at "
                    f"${rate.input_per_mtok:.2f}/Mtok against "
                    f"${cheap.input_per_mtok:.2f} for claude-haiku-4-5."
                ),
                fix=(
                    "Route this task class to a smaller model with a policy rule. "
                    "If the task needs the larger model's judgement but not its "
                    'depth, keep the model and set `output_config={"effort": "low"}` '
                    "first -- that is a smaller change with no quality cliff."
                ),
                evidence=[
                    f"median output: "
                    f"{int(statistics.median(r['output_tokens'] for r in candidates))} tokens"
                ],
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 6. History that keeps being re-read
# ---------------------------------------------------------------------------


@analyser
def context_bloat(rows: Rows, ledger: Ledger) -> list[Finding]:
    """Reported, not costed.

    The saving depends on how much history the task actually still needs, which
    this tool cannot know. Naming a dollar figure here would be a guess dressed
    up as a measurement, so it reports the trend and stops.
    """
    findings = []
    for session, group in _group(rows, "session").items():
        if len(group) < 4 or session == "unknown":
            continue
        totals = [
            r["input_tokens"] + r["cache_creation_input_tokens"] + r["cache_read_input_tokens"]
            for r in group
        ]
        if totals[0] == 0 or totals[-1] < 4 * totals[0]:
            continue

        spent = sum(r["cost_usd"] for r in group)
        findings.append(
            Finding(
                id="context-bloat",
                title=f"Prompt grew {totals[-1] / totals[0]:.0f}x over session {session}",
                severity="info",
                wasted_usd=0.0,
                detail=(
                    f"Across {len(group)} calls the prompt went from "
                    f"{totals[0]:,} to {totals[-1]:,} tokens, at a total cost of "
                    f"{usd(spent)}. Every turn re-reads the whole history, so "
                    f"early tool results are paid for again on every later call."
                ),
                fix=(
                    "Clear what the task no longer needs with context editing "
                    "(`clear_tool_uses_20250919`), or turn on server-side "
                    "compaction to summarise the earlier turns instead of "
                    "resending them. Not costed here: how much of this history is "
                    "still load-bearing is a judgement about the task, not "
                    "something the ledger can see."
                ),
                evidence=[f"per-call prompt tokens: {' -> '.join(f'{t:,}' for t in totals)}"],
            )
        )
    return findings


# ---------------------------------------------------------------------------


def _group(rows: Rows, *keys: str) -> dict[Any, Rows]:
    out: dict[Any, Rows] = defaultdict(list)
    for r in rows:
        k = tuple(r.get(key) for key in keys)
        out[k[0] if len(k) == 1 else k].append(r)
    return out


def _over_minimum(row: dict[str, Any]) -> bool:
    rate = rate_for(row["model"])
    return rate is not None and row["prefix_tokens"] >= rate.min_cacheable_tokens

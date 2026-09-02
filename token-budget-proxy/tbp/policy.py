"""Pre-flight policy: decide what a request is allowed to cost, before it costs it.

Two kinds of control, both evaluated before the request leaves the machine:

  budgets -- a ceiling on cumulative spend per agent over a rolling window
  rules   -- per-request conditions that rewrite or reject the request

The distinction matters. A budget is a backstop that fires once the money is
already gone; a rule prevents the expensive call from being made in the first
place. Most of the savings come from rules.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .pricing import CHEAP_TO_EXPENSIVE, cost_of, estimate_tokens, rate_for

WINDOW_SECONDS = {"hour": 3600, "day": 86400, "week": 604800, "month": 2592000}


@dataclass
class Decision:
    """What policy did to a request, and why."""

    allowed: bool = True
    actions: list[str] = field(default_factory=list)
    denied_reason: str | None = None
    # Best-effort estimate of spend avoided by the rewrites, for the ledger.
    estimated_saving_usd: float = 0.0

    def deny(self, reason: str) -> "Decision":
        self.allowed = False
        self.denied_reason = reason
        return self


@dataclass
class Policy:
    budgets: dict[str, Any] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    window: str = "day"

    @classmethod
    def load(cls, path: Path | str) -> "Policy":
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        budgets = data.get("budgets") or {}
        return cls(
            budgets=budgets,
            rules=data.get("rules") or [],
            window=budgets.get("window", "day"),
        )

    # -- budgets ---------------------------------------------------------

    def budget_for(self, agent: str) -> float | None:
        per_agent = self.budgets.get("agents") or {}
        for pattern, limit in per_agent.items():
            if fnmatch.fnmatch(agent, pattern):
                return float(limit)
        default = self.budgets.get("default")
        return float(default) if default is not None else None

    def window_start(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return now - WINDOW_SECONDS.get(self.window, 86400)

    # -- rules -----------------------------------------------------------

    def apply(self, body: dict[str, Any], ctx: dict[str, Any]) -> Decision:
        """Evaluate every rule in order, mutating `body` in place.

        Rules are not first-match-wins: a request can legitimately need both a
        model downgrade and an effort cap, and making the author order their
        rules to get both would be a footgun.
        """
        decision = Decision()
        before = _estimated_cost(body)

        for rule in self.rules:
            if not _matches(rule.get("when") or {}, body, ctx):
                continue
            name = rule.get("name", "unnamed")
            for action, value in (rule.get("then") or {}).items():
                applied = _apply_action(action, value, body, decision, name)
                if applied:
                    decision.actions.append(applied)
                if not decision.allowed:
                    return decision

        after = _estimated_cost(body)
        decision.estimated_saving_usd = max(0.0, before - after)
        return decision


def _matches(when: dict[str, Any], body: dict[str, Any], ctx: dict[str, Any]) -> bool:
    for key, expected in when.items():
        if key == "agent":
            if not _glob_any(ctx.get("agent", ""), expected):
                return False
        elif key == "task":
            if not _glob_any(ctx.get("task") or "", expected):
                return False
        elif key == "model":
            if not _glob_any(body.get("model", ""), expected):
                return False
        elif key == "effort":
            if not _glob_any(_effort(body) or "", expected):
                return False
        elif key == "min_input_tokens":
            if _prompt_tokens(body) < int(expected):
                return False
        elif key == "max_input_tokens":
            if _prompt_tokens(body) > int(expected):
                return False
        elif key == "min_prefix_tokens":
            if _prefix_tokens(body) < int(expected):
                return False
        elif key == "has_tools":
            if bool(body.get("tools")) is not bool(expected):
                return False
        elif key == "streaming":
            if bool(body.get("stream")) is not bool(expected):
                return False
        else:
            raise ValueError(f"unknown policy condition: {key!r}")
    return True


def _glob_any(value: str, expected: Any) -> bool:
    patterns = expected if isinstance(expected, list) else [expected]
    return any(fnmatch.fnmatch(value, str(p)) for p in patterns)


def _apply_action(
    action: str, value: Any, body: dict[str, Any], decision: Decision, rule: str
) -> str | None:
    if action == "deny":
        if value:
            decision.deny(f"policy rule {rule!r} denies this request")
        return None

    if action == "model":
        if body.get("model") == value:
            return None
        was = body.get("model")
        body["model"] = value
        return f"{rule}: model {was} -> {value}"

    if action == "effort":
        cfg = body.setdefault("output_config", {})
        if cfg.get("effort") == value:
            return None
        was = cfg.get("effort", "default")
        cfg["effort"] = value
        return f"{rule}: effort {was} -> {value}"

    if action == "max_tokens_cap":
        current = body.get("max_tokens")
        if current is None or current <= int(value):
            return None
        body["max_tokens"] = int(value)
        return f"{rule}: max_tokens {current} -> {value}"

    if action == "max_tokens_floor":
        # Raising a ceiling looks like the opposite of cost control, and is not:
        # a response cut off at max_tokens is billed in full and then retried, so
        # a too-low ceiling is the most reliable way to pay for the same tokens
        # twice.
        current = body.get("max_tokens")
        if current is None or current >= int(value):
            return None
        body["max_tokens"] = int(value)
        return f"{rule}: max_tokens {current} -> {value} (truncation guard)"

    if action == "ensure_cache_control":
        if not value:
            return None
        from .ledger import _has_cache_control  # local import avoids a cycle

        if _has_cache_control(body):
            return None
        # Top-level cache_control auto-places the breakpoint on the last
        # cacheable block, which is the right default when we are adding one on
        # someone else's behalf and cannot know their prefix boundaries.
        body["cache_control"] = {"type": "ephemeral"}
        return f"{rule}: added cache_control to a {_prefix_tokens(body)}-token prefix"

    if action == "downgrade_one_tier":
        if not value:
            return None
        current = body.get("model", "")
        idx = next(
            (i for i, m in enumerate(CHEAP_TO_EXPENSIVE) if m in current), None
        )
        if idx is None or idx == 0:
            return None
        body["model"] = CHEAP_TO_EXPENSIVE[idx - 1]
        return f"{rule}: model {current} -> {body['model']}"

    raise ValueError(f"unknown policy action: {action!r}")


# -- estimation helpers ---------------------------------------------------
#
# All of this is pre-flight, so it works from character counts rather than real
# token counts. It is used to compare a request against itself before and after
# a rewrite, where a consistent bias cancels out.


def _prompt_tokens(body: dict[str, Any]) -> int:
    import json

    return estimate_tokens(
        json.dumps(
            [body.get("system"), body.get("messages"), body.get("tools")],
            ensure_ascii=False,
            default=str,
        )
    )


def _prefix_tokens(body: dict[str, Any]) -> int:
    from .ledger import render_prefix

    return render_prefix(body)[2]


def _effort(body: dict[str, Any]) -> str | None:
    return (body.get("output_config") or {}).get("effort")


# Effort does not change the price per token, but it does change how many
# output tokens a request tends to spend. These are rough working multipliers
# used only to rank a rewrite against the original.
_EFFORT_WEIGHT = {"low": 0.35, "medium": 0.6, "high": 1.0, "xhigh": 1.6, "max": 2.4}


def _estimated_cost(body: dict[str, Any]) -> float:
    model = body.get("model", "")
    if rate_for(model) is None:
        return 0.0
    prompt = _prompt_tokens(body)
    weight = _EFFORT_WEIGHT.get(_effort(body) or "high", 1.0)
    # Assume a request spends a meaningful fraction of its max_tokens budget;
    # the absolute number does not matter, only that both sides use the same one.
    output = min(int(body.get("max_tokens") or 4096), 8000) * weight
    return cost_of(model, input_tokens=prompt, output_tokens=int(output)).total

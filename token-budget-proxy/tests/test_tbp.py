"""Tests for the parts where being wrong would be expensive or embarrassing.

The cost arithmetic and the cache-forensics analyser get the most attention:
a spend tool that reports a confident wrong number is worse than no tool.
"""

from __future__ import annotations

import json

import pytest

from tbp.audit import run as run_audit
from tbp.ledger import Call, Ledger, render_prefix, tool_def_tokens, tool_names
from tbp.mock_upstream import MockUpstream
from tbp.policy import Policy
from tbp.pricing import CACHE_READ_MULTIPLIER, cost_of, rate_for


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "test.db")


@pytest.fixture
def big_system() -> str:
    # Comfortably over every model minimum in the table.
    return "You are a careful assistant. " * 900


# -- pricing --------------------------------------------------------------


def test_cache_reads_are_a_tenth_of_input():
    read = cost_of("claude-opus-5", cache_read_input_tokens=1_000_000)
    plain = cost_of("claude-opus-5", input_tokens=1_000_000)
    assert read.total == pytest.approx(plain.total * CACHE_READ_MULTIPLIER)


def test_cache_writes_carry_a_premium():
    write = cost_of("claude-opus-5", cache_creation_input_tokens=1_000_000)
    plain = cost_of("claude-opus-5", input_tokens=1_000_000)
    assert write.total > plain.total


def test_batch_halves_the_bill():
    assert cost_of("claude-opus-5", input_tokens=1000, batch=True).total == pytest.approx(
        cost_of("claude-opus-5", input_tokens=1000).total / 2
    )


def test_dated_and_prefixed_model_ids_still_cost():
    for model in ("claude-opus-5-20260101", "anthropic.claude-opus-5"):
        assert rate_for(model) is rate_for("claude-opus-5")


def test_unknown_model_costs_zero_rather_than_guessing():
    assert cost_of("some-other-vendor-model", input_tokens=10_000).total == 0.0


def test_cost_splits_sum_to_total():
    c = cost_of(
        "claude-opus-5",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=300,
    )
    assert c.total == pytest.approx(
        c.uncached_input + c.cache_write + c.cache_read + c.output
    )


# -- prefix rendering -----------------------------------------------------


def test_prefix_covers_tools_and_system_but_not_messages():
    base = {"tools": [{"name": "a"}], "system": "hello"}
    a = render_prefix({**base, "messages": [{"role": "user", "content": "x"}]})
    b = render_prefix({**base, "messages": [{"role": "user", "content": "y"}]})
    assert a[1] == b[1], "message content must not change the cacheable prefix"

    c = render_prefix({**base, "system": "hello "})
    assert c[1] != a[1], "a one-byte system change must change the prefix hash"


def test_cache_control_detected_wherever_it_is_placed():
    assert render_prefix({"cache_control": {"type": "ephemeral"}})[3]
    assert render_prefix(
        {"system": [{"type": "text", "text": "x", "cache_control": {}}]}
    )[3]
    assert render_prefix({"tools": [{"name": "t", "cache_control": {}}]})[3]
    assert render_prefix(
        {"messages": [{"role": "user", "content": [{"type": "text", "cache_control": {}}]}]}
    )[3]
    assert not render_prefix({"system": "plain", "messages": []})[3]


def test_tool_helpers_tolerate_absent_tools():
    assert tool_names({}) == []
    assert tool_def_tokens({}) == 0


# -- policy ---------------------------------------------------------------


def _policy(tmp_path, doc: dict) -> Policy:
    path = tmp_path / "p.yaml"
    path.write_text(json.dumps(doc))  # YAML is a superset of JSON
    return Policy.load(path)


def test_rules_compose_rather_than_first_match_wins(tmp_path):
    policy = _policy(tmp_path, {"rules": [
        {"name": "a", "when": {"task": "classify"}, "then": {"model": "claude-haiku-4-5"}},
        {"name": "b", "when": {"task": "classify"}, "then": {"effort": "low"}},
    ]})
    body = {"model": "claude-opus-5", "max_tokens": 1000}
    decision = policy.apply(body, {"agent": "x", "task": "classify"})
    assert body["model"] == "claude-haiku-4-5"
    assert body["output_config"]["effort"] == "low"
    assert len(decision.actions) == 2


def test_deny_stops_evaluation(tmp_path):
    policy = _policy(tmp_path, {"rules": [
        {"name": "block", "when": {"model": "claude-fable-5"}, "then": {"deny": True}},
    ]})
    body = {"model": "claude-fable-5"}
    decision = policy.apply(body, {"agent": "x"})
    assert not decision.allowed and decision.denied_reason


def test_max_tokens_floor_raises_but_cap_lowers(tmp_path):
    policy = _policy(tmp_path, {"rules": [
        {"name": "floor", "when": {}, "then": {"max_tokens_floor": 16000}},
    ]})
    low = {"model": "claude-opus-5", "max_tokens": 300}
    policy.apply(low, {"agent": "x"})
    assert low["max_tokens"] == 16000

    already_high = {"model": "claude-opus-5", "max_tokens": 32000}
    policy.apply(already_high, {"agent": "x"})
    assert already_high["max_tokens"] == 32000, "floor must never lower a ceiling"


def test_ensure_cache_control_leaves_an_existing_breakpoint_alone(tmp_path, big_system):
    policy = _policy(tmp_path, {"rules": [
        {"name": "cache", "when": {}, "then": {"ensure_cache_control": True}},
    ]})
    body = {"model": "claude-opus-5", "system": [
        {"type": "text", "text": big_system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]}
    decision = policy.apply(body, {"agent": "x"})
    assert "cache_control" not in body, "must not add a second, conflicting breakpoint"
    assert decision.actions == []


def test_agent_budgets_match_by_glob(tmp_path):
    policy = _policy(tmp_path, {"budgets": {
        "default": 5.0, "agents": {"subagent-*": 0.5}}})
    assert policy.budget_for("subagent-crawler") == 0.5
    assert policy.budget_for("main") == 5.0


def test_unknown_condition_fails_loudly(tmp_path):
    policy = _policy(tmp_path, {"rules": [
        {"name": "typo", "when": {"modle": "claude-opus-5"}, "then": {"effort": "low"}},
    ]})
    with pytest.raises(ValueError, match="unknown policy condition"):
        policy.apply({"model": "claude-opus-5"}, {"agent": "x"})


# -- mock upstream --------------------------------------------------------


def test_mock_cache_hits_only_on_a_stable_prefix(big_system):
    mock = MockUpstream()
    body = {
        "model": "claude-opus-5",
        "system": [{"type": "text", "text": big_system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    first = mock.respond(body)[1]["usage"]
    second = mock.respond(body)[1]["usage"]
    assert first["cache_creation_input_tokens"] > 0
    assert first["cache_read_input_tokens"] == 0
    assert second["cache_read_input_tokens"] > 0


def test_mock_ignores_a_breakpoint_below_the_model_minimum():
    mock = MockUpstream()
    body = {
        "model": "claude-opus-5",
        "system": [{"type": "text", "text": "short", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    mock.respond(body)
    usage = mock.respond(body)[1]["usage"]
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] == 0


def test_mock_truncates_when_max_tokens_is_too_low():
    _, payload = MockUpstream().respond(
        {"model": "claude-opus-5", "max_tokens": 50, "messages": []}
    )
    assert payload["stop_reason"] == "max_tokens"
    assert payload["usage"]["output_tokens"] == 50


def test_mock_stream_reports_the_same_usage_as_unary(big_system):
    body = {"model": "claude-opus-5", "max_tokens": 4096,
            "system": big_system, "messages": [{"role": "user", "content": "hi"}]}
    events = list(MockUpstream().stream(body))
    start = json.loads(events[0].split("data: ", 1)[1])
    delta = json.loads(
        next(e for e in events if '"message_delta"' in e).split("data: ", 1)[1]
    )
    assert start["message"]["usage"]["input_tokens"] > 0
    assert delta["usage"]["output_tokens"] > 0


# -- analysers ------------------------------------------------------------


def _record(ledger: Ledger, n: int, **overrides) -> None:
    for i in range(n):
        kwargs = {k: (v(i) if callable(v) else v) for k, v in overrides.items()}
        ledger.record(Call(model="claude-opus-5", **kwargs))


def test_finds_the_invalidator_and_names_the_byte(ledger, big_system):
    for i in range(4):
        body = {
            "model": "claude-opus-5",
            "system": [{"type": "text", "text": big_system + f"\n\nnow: 2026-03-0{i}",
                        "cache_control": {"type": "ephemeral"}}],
        }
        text, digest, tokens, has_cc = render_prefix(body)
        ledger.record(Call(model="claude-opus-5", agent="a", prefix_text=text,
                           prefix_hash=digest, prefix_tokens=tokens,
                           had_cache_control=has_cc))

    finding = next(f for f in run_audit(ledger) if f.id == "cache-invalidated")
    assert finding.wasted_usd > 0
    assert any("diverge" in line for line in finding.evidence)


def test_stable_prefix_that_never_hits_is_diagnosed_as_ttl_not_drift(ledger, big_system):
    body = {"model": "claude-opus-5",
            "system": [{"type": "text", "text": big_system,
                        "cache_control": {"type": "ephemeral"}}]}
    text, digest, tokens, has_cc = render_prefix(body)
    for _ in range(4):
        ledger.record(Call(model="claude-opus-5", agent="a", prefix_text=text,
                           prefix_hash=digest, prefix_tokens=tokens,
                           had_cache_control=has_cc))

    finding = next(f for f in run_audit(ledger) if f.id == "cache-invalidated")
    assert "expiring" in finding.detail
    assert "ttl" in finding.fix.lower()


def test_no_cache_finding_when_the_cache_is_working(ledger, big_system):
    body = {"model": "claude-opus-5",
            "system": [{"type": "text", "text": big_system,
                        "cache_control": {"type": "ephemeral"}}]}
    text, digest, tokens, has_cc = render_prefix(body)
    for i in range(4):
        ledger.record(Call(model="claude-opus-5", agent="a", prefix_text=text,
                           prefix_hash=digest, prefix_tokens=tokens,
                           had_cache_control=has_cc,
                           cache_read_input_tokens=0 if i == 0 else tokens,
                           cache_creation_input_tokens=tokens if i == 0 else 0))

    assert not [f for f in run_audit(ledger) if f.id.startswith("cache-")]


def test_dead_tools_exclude_the_ones_actually_called(ledger):
    _record(ledger, 5, agent="a", tools_declared=["used", "dead1", "dead2"],
            tools_called=["used"], tool_def_tokens=900, input_tokens=100)
    finding = next(f for f in run_audit(ledger) if f.id == "dead-tools")
    assert "dead1" in finding.detail and "dead2" in finding.detail
    assert "used" not in finding.title


def test_truncation_is_costed_on_output_only(ledger):
    _record(ledger, 3, agent="a", stop_reason="max_tokens",
            input_tokens=1000, output_tokens=300)
    finding = next(f for f in run_audit(ledger) if f.id == "truncation-retry")
    expected = 3 * cost_of("claude-opus-5", output_tokens=300).total
    assert finding.wasted_usd == pytest.approx(expected)


def test_context_bloat_is_reported_but_never_costed(ledger):
    for i in range(5):
        ledger.record(Call(model="claude-opus-5", agent="a", session="s",
                           input_tokens=100 * (i + 1) ** 3, output_tokens=50))
    finding = next(f for f in run_audit(ledger) if f.id == "context-bloat")
    assert finding.wasted_usd == 0.0
    assert finding.severity == "info"


def test_denied_calls_are_excluded_from_analysis(ledger):
    _record(ledger, 5, agent="a", stop_reason="max_tokens", output_tokens=300,
            denied_reason="budget exhausted")
    assert run_audit(ledger) == []


def test_empty_ledger_produces_no_findings(ledger):
    assert run_audit(ledger) == []


def test_clear_restarts_ids_from_one(ledger):
    _record(ledger, 3, agent="a")
    assert ledger.clear() == 3
    ledger.record(Call(model="claude-opus-5", agent="a"))
    assert ledger.rows()[0]["id"] == 1

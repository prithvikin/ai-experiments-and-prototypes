"""A local stand-in for the Messages API, so the demo runs without a key.

This is not a general-purpose fake. It models the one behaviour the whole
project is about: how `usage` responds to prompt caching. Prefixes are cached
only when a breakpoint is present *and* the prefix clears the model's minimum
cacheable size, and an entry expires on its TTL. Get the prefix bytes wrong
between two calls and the simulated cache misses, exactly as the real one does.

Set TBP_UPSTREAM=https://api.anthropic.com to run against the real thing.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from .pricing import estimate_tokens, rate_for

TTL_SECONDS = {"5m": 300, "1h": 3600}

_EFFORT_OUTPUT = {"low": 120, "medium": 320, "high": 700, "xhigh": 1400, "max": 2600}
_STRUCTURED_OUTPUT = 85

_REPLY = (
    "Acknowledged. Here is a synthetic response from the local mock upstream; "
    "the token accounting attached to it is the part that matters."
)


class MockUpstream:
    def __init__(self) -> None:
        # prefix_hash -> (expires_at, cached_token_count)
        self._cache: dict[str, tuple[float, int]] = {}

    # -- accounting ------------------------------------------------------

    def _usage(self, body: dict[str, Any]) -> dict[str, int]:
        from .ledger import render_prefix

        _, prefix_hash, prefix_tokens, has_cc = render_prefix(body)
        message_tokens = estimate_tokens(
            json.dumps(body.get("messages") or [], ensure_ascii=False, default=str)
        )

        rate = rate_for(body.get("model", ""))
        minimum = rate.min_cacheable_tokens if rate else 1024
        ttl = ((body.get("cache_control") or {}).get("ttl")) or "5m"

        usage = {
            "input_tokens": prefix_tokens + message_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        # A breakpoint on a prefix below the model minimum is silently ignored:
        # no error, no cache entry, full price. That silence is the bug.
        if not has_cc or prefix_tokens < minimum:
            return usage

        now = time.time()
        hit = self._cache.get(prefix_hash)
        if hit and hit[0] > now:
            self._cache[prefix_hash] = (now + TTL_SECONDS.get(ttl, 300), prefix_tokens)
            usage["cache_read_input_tokens"] = prefix_tokens
            usage["input_tokens"] = message_tokens
        else:
            self._cache[prefix_hash] = (now + TTL_SECONDS.get(ttl, 300), prefix_tokens)
            usage["cache_creation_input_tokens"] = prefix_tokens
            usage["input_tokens"] = message_tokens

        return usage

    def _output(self, body: dict[str, Any]) -> tuple[int, str, str | None]:
        """Returns (output_tokens, stop_reason, tool_name)."""
        output_config = body.get("output_config") or {}
        effort = output_config.get("effort", "high")
        wanted = _EFFORT_OUTPUT.get(effort, 700)
        # A request constrained to a schema emits roughly what the schema holds,
        # near enough regardless of effort -- that is the point of asking for one.
        if output_config.get("format"):
            wanted = min(wanted, _STRUCTURED_OUTPUT)
        cap = int(body.get("max_tokens") or 4096)

        tool = self._tool_choice(body)
        if tool:
            return min(wanted // 4, cap), "tool_use", tool
        if wanted > cap:
            # Ran out of room mid-answer. The caller pays for every one of
            # these tokens and gets an unusable truncated response.
            return cap, "max_tokens", None
        return wanted, "end_turn", None

    @staticmethod
    def _tool_choice(body: dict[str, Any]) -> str | None:
        """Deterministic: a tool fires only if the last user turn names it."""
        tools = [t.get("name") for t in body.get("tools") or [] if t.get("name")]
        if not tools:
            return None
        messages = body.get("messages") or []
        last = json.dumps(messages[-1], default=str).lower() if messages else ""
        return next((t for t in tools if t.lower() in last), None)

    # -- response shapes -------------------------------------------------

    def respond(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        usage = self._usage(body)
        out_tokens, stop_reason, tool = self._output(body)
        usage["output_tokens"] = out_tokens

        if tool:
            content = [{"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": tool, "input": {}}]
        else:
            content = [{"type": "text", "text": _REPLY}]

        return 200, {
            "id": f"msg_mock_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", ""),
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    def stream(self, body: dict[str, Any]) -> Iterator[str]:
        _, payload = self.respond(body)
        usage = payload["usage"]
        start_usage = {k: v for k, v in usage.items() if k != "output_tokens"}
        start_usage["output_tokens"] = 0

        message = {k: v for k, v in payload.items() if k != "content"}
        message["content"] = []
        message["usage"] = start_usage

        yield _sse("message_start", {"type": "message_start", "message": message})

        block = payload["content"][0]
        if block["type"] == "text":
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
            for word in _REPLY.split(" "):
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": word + " "}})
        else:
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": 0, "content_block": block})

        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": payload["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]}})
        yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

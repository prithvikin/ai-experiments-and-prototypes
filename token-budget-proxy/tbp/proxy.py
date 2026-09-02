"""The proxy itself: a drop-in Messages API endpoint that meters and governs.

Point any Anthropic SDK at it with `base_url` and nothing else changes -- the
request is forwarded verbatim apart from whatever policy deliberately rewrote.
Requests are forwarded as raw HTTP rather than round-tripped through the SDK on
purpose: a proxy must pass through fields it does not know about, including
ones added after it was written.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx2 as httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .ledger import (
    Call,
    Ledger,
    render_prefix,
    tool_def_tokens,
    tool_names,
)
from .mock_upstream import MockUpstream
from .policy import Policy
from .pricing import usd

UPSTREAM = os.environ.get("TBP_UPSTREAM", "https://api.anthropic.com")
POLICY_PATH = os.environ.get("TBP_POLICY", "policy.yaml")

# Headers we consume rather than forward.
ATTRIBUTION_HEADERS = {"x-tbp-agent", "x-tbp-session", "x-tbp-task"}
HOP_BY_HOP = {"host", "content-length", "connection", "accept-encoding"}


def create_app(
    *,
    db: Path | str | None = None,
    policy_path: Path | str | None = None,
    upstream: str | None = None,
) -> FastAPI:
    app = FastAPI(title="token-budget-proxy")
    app.state.ledger = Ledger(db or os.environ.get("TBP_DB", "tbp.db"))
    app.state.policy = Policy.load(policy_path or POLICY_PATH)
    app.state.upstream = upstream or UPSTREAM
    app.state.mock = MockUpstream() if (upstream or UPSTREAM) == "mock" else None

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "upstream": app.state.upstream,
            "rules": len(app.state.policy.rules),
        }

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        return await _handle(app, request)

    return app


async def _handle(app: FastAPI, request: Request) -> Any:
    ledger: Ledger = app.state.ledger
    policy: Policy = app.state.policy

    raw = await request.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return _error(400, "invalid_request_error", "request body is not valid JSON")

    ctx = {
        "agent": request.headers.get("x-tbp-agent", "unattributed"),
        "session": request.headers.get("x-tbp-session", "default"),
        "task": request.headers.get("x-tbp-task"),
    }

    call = Call(
        agent=ctx["agent"],
        session=ctx["session"],
        task=ctx["task"],
        model=body.get("model", ""),
        streamed=bool(body.get("stream")),
    )

    # 1. Budget. Checked before rules so an exhausted agent is stopped even if a
    #    rule would have made this particular call cheap.
    limit = policy.budget_for(ctx["agent"])
    if limit is not None:
        spent = ledger.spend_since(ctx["agent"], policy.window_start())
        if spent >= limit:
            reason = (
                f"budget exhausted for agent {ctx['agent']!r}: "
                f"{usd(spent)} spent of {usd(limit)} per {policy.window}"
            )
            call.denied_reason = reason
            call.status = 403
            ledger.record(call)
            return _error(403, "permission_error", reason)

    # 2. Rules. These mutate `body` in place -- this is the request that is sent.
    decision = policy.apply(body, ctx)
    call.policy_actions = decision.actions
    call.model = body.get("model", "")
    call.effort = (body.get("output_config") or {}).get("effort")
    if not decision.allowed:
        call.denied_reason = decision.denied_reason
        call.status = 403
        ledger.record(call)
        return _error(403, "permission_error", decision.denied_reason or "denied")

    # 3. Snapshot the cacheable prefix for later forensics.
    prefix_text, prefix_hash, prefix_tokens, has_cc = render_prefix(body)
    call.prefix_text = prefix_text
    call.prefix_hash = prefix_hash
    call.prefix_tokens = prefix_tokens
    call.had_cache_control = has_cc
    call.tools_declared = tool_names(body)
    call.tool_def_tokens = tool_def_tokens(body)

    headers = _upstream_headers(request)
    started = time.perf_counter()

    if body.get("stream"):
        return await _forward_streaming(app, body, headers, call, started)
    return await _forward_unary(app, body, headers, call, started)


async def _forward_unary(
    app: FastAPI, body: dict, headers: dict, call: Call, started: float
) -> Any:
    if app.state.mock is not None:
        status, payload = app.state.mock.respond(body)
    else:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{app.state.upstream}/v1/messages", json=body, headers=headers
            )
            status = resp.status_code
            payload = resp.json()

    call.latency_ms = int((time.perf_counter() - started) * 1000)
    call.status = status
    _absorb_unary(call, payload)
    app.state.ledger.record(call)
    return JSONResponse(status_code=status, content=payload)


async def _forward_streaming(
    app: FastAPI, body: dict, headers: dict, call: Call, started: float
) -> StreamingResponse:
    async def relay():
        try:
            if app.state.mock is not None:
                for chunk in app.state.mock.stream(body):
                    _absorb_sse(call, chunk)
                    yield chunk.encode()
            else:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    async with client.stream(
                        "POST",
                        f"{app.state.upstream}/v1/messages",
                        json=body,
                        headers=headers,
                    ) as resp:
                        call.status = resp.status_code
                        async for line in resp.aiter_lines():
                            _absorb_sse(call, line)
                            yield (line + "\n").encode()
        finally:
            # Recorded in a finally so a client that disconnects mid-stream is
            # still billed for what was already generated upstream.
            call.latency_ms = int((time.perf_counter() - started) * 1000)
            call.status = call.status or 200
            app.state.ledger.record(call)

    return StreamingResponse(relay(), media_type="text/event-stream")


def _absorb_unary(call: Call, payload: dict[str, Any]) -> None:
    usage = payload.get("usage") or {}
    call.input_tokens = usage.get("input_tokens", 0) or 0
    call.output_tokens = usage.get("output_tokens", 0) or 0
    call.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0) or 0
    call.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0) or 0
    call.stop_reason = payload.get("stop_reason")
    call.request_id = payload.get("id")
    call.tools_called = [
        b.get("name")
        for b in payload.get("content") or []
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
    ]


def _absorb_sse(call: Call, line: str) -> None:
    """Read usage out of the event stream without altering a byte of it."""
    line = line.strip()
    if not line.startswith("data:"):
        return
    try:
        event = json.loads(line[5:].strip())
    except json.JSONDecodeError:
        return

    etype = event.get("type")
    if etype == "message_start":
        message = event.get("message") or {}
        usage = message.get("usage") or {}
        call.input_tokens = usage.get("input_tokens", 0) or 0
        call.cache_creation_input_tokens = (
            usage.get("cache_creation_input_tokens", 0) or 0
        )
        call.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0) or 0
        call.request_id = message.get("id")
    elif etype == "message_delta":
        usage = event.get("usage") or {}
        # Streamed output_tokens is cumulative, so take the last value seen.
        call.output_tokens = usage.get("output_tokens", call.output_tokens) or 0
        call.stop_reason = (event.get("delta") or {}).get(
            "stop_reason", call.stop_reason
        )
    elif etype == "content_block_start":
        block = event.get("content_block") or {}
        if block.get("type") == "tool_use" and block.get("name"):
            call.tools_called = [*call.tools_called, block["name"]]


def _upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() not in ATTRIBUTION_HEADERS
    }
    # Let the caller hold its own credential; fall back to the proxy's so an
    # agent can be run with no key of its own at all.
    if "x-api-key" not in {k.lower() for k in headers}:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            headers["x-api-key"] = key
    headers.setdefault("anthropic-version", "2023-06-01")
    headers["content-type"] = "application/json"
    return headers


def _error(status: int, etype: str, message: str) -> JSONResponse:
    """Anthropic's error envelope, so SDK clients raise their normal typed errors."""
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": etype, "message": message}},
    )

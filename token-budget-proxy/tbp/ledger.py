"""The receipts layer: one durable row per API call, with attribution.

Nothing else in this project works without this. A policy engine with no ledger
is a guess, and a waste report with no ledger is an opinion.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .pricing import Cost, cost_of, estimate_tokens

DEFAULT_DB = Path(os.environ.get("TBP_DB", "tbp.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    agent             TEXT    NOT NULL,
    session           TEXT    NOT NULL,
    task              TEXT,
    model             TEXT    NOT NULL,
    effort            TEXT,
    streamed          INTEGER NOT NULL DEFAULT 0,
    status            INTEGER,
    stop_reason       TEXT,
    latency_ms        INTEGER,

    input_tokens                 INTEGER NOT NULL DEFAULT 0,
    output_tokens                INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens      INTEGER NOT NULL DEFAULT 0,

    cost_usd          REAL NOT NULL DEFAULT 0,
    cost_breakdown    TEXT,

    -- Cache forensics. prefix_text is stored as-sent (not re-serialised) so a
    -- byte-level diff can point at the invalidator; sorting it here would hide
    -- exactly the bug we are hunting.
    prefix_hash       TEXT,
    prefix_text       TEXT,
    prefix_tokens     INTEGER NOT NULL DEFAULT 0,
    had_cache_control INTEGER NOT NULL DEFAULT 0,

    tools_declared    TEXT,
    tools_called      TEXT,
    tool_def_tokens   INTEGER NOT NULL DEFAULT 0,

    policy_actions    TEXT,
    denied_reason     TEXT,
    request_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_agent   ON calls(agent);
CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session);
CREATE INDEX IF NOT EXISTS idx_calls_ts      ON calls(ts);
"""

# Columns too large to haul into memory for every analysis pass.
_HEAVY = {"prefix_text"}


@dataclass
class Call:
    agent: str = "unknown"
    session: str = "unknown"
    task: str | None = None
    model: str = ""
    effort: str | None = None
    streamed: bool = False
    status: int | None = None
    stop_reason: str | None = None
    latency_ms: int | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    prefix_hash: str | None = None
    prefix_text: str | None = None
    prefix_tokens: int = 0
    had_cache_control: bool = False

    tools_declared: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    tool_def_tokens: int = 0

    policy_actions: list[str] = field(default_factory=list)
    denied_reason: str | None = None
    request_id: str | None = None
    ts: float = field(default_factory=time.time)

    def cost(self) -> Cost:
        return cost_of(
            self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
        )

    @property
    def total_prompt_tokens(self) -> int:
        """What the model actually read -- not just the uncached remainder."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


class Ledger:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def record(self, call: Call) -> int:
        cost = call.cost()
        row = {
            **{k: v for k, v in asdict(call).items() if not isinstance(v, (list, bool))},
            "streamed": int(call.streamed),
            "had_cache_control": int(call.had_cache_control),
            "tools_declared": json.dumps(call.tools_declared),
            "tools_called": json.dumps(call.tools_called),
            "policy_actions": json.dumps(call.policy_actions),
            "cost_usd": cost.total,
            "cost_breakdown": json.dumps(asdict(cost)),
        }
        cols = ", ".join(row)
        placeholders = ", ".join(f":{c}" for c in row)
        cur = self.conn.execute(
            f"INSERT INTO calls ({cols}) VALUES ({placeholders})", row
        )
        self.conn.commit()
        return cur.lastrowid

    def spend_since(self, agent: str, since: float) -> float:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM calls "
            "WHERE agent = ? AND ts >= ?",
            (agent, since),
        )
        return float(cur.fetchone()["s"])

    def rows(self, *, include_heavy: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM calls ORDER BY ts")
        out = []
        for r in cur:
            d = dict(r)
            if not include_heavy:
                for k in _HEAVY:
                    d.pop(k, None)
            for k in ("tools_declared", "tools_called", "policy_actions"):
                d[k] = json.loads(d[k]) if d.get(k) else []
            out.append(d)
        return out[-limit:] if limit else out

    def prefix_text(self, call_id: int) -> str | None:
        cur = self.conn.execute("SELECT prefix_text FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        return row["prefix_text"] if row else None

    def clear(self) -> int:
        """Empty the ledger in place.

        Deliberately not `unlink()`: a serving process holds this file open, and
        removing it leaves that process writing to a deleted inode -- every
        later call fails with "readonly database" and the proxy 500s until it is
        restarted.
        """
        cur = self.conn.execute("DELETE FROM calls")
        # AUTOINCREMENT keeps its high-water mark across a truncate, so without
        # this the next call is #22 in a ledger holding one row. Findings quote
        # these ids as evidence, so they have to line up with what `tbp calls`
        # shows.
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'calls'")
        self.conn.commit()
        self.conn.execute("VACUUM")
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()


def render_prefix(body: dict[str, Any]) -> tuple[str, str, int, bool]:
    """Reduce a request to the bytes prompt caching actually keys on.

    Render order is tools -> system -> messages, so only the first two are the
    reusable prefix. Serialised with the original key order preserved, because
    non-deterministic serialisation upstream is itself one of the bugs this is
    meant to catch.
    """
    tools = body.get("tools") or []
    system = body.get("system") or ""
    text = json.dumps(tools, ensure_ascii=False) + "\n--system--\n" + json.dumps(
        system, ensure_ascii=False
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, digest, estimate_tokens(text), _has_cache_control(body)


def _has_cache_control(body: dict[str, Any]) -> bool:
    """Is there a breakpoint anywhere in this request?

    Keyed on the presence of the field rather than its truthiness: the question
    is whether the caller asked for caching at all, and a malformed value is
    still an attempt. Treating it as absent would file the request under "never
    tried to cache", which points at entirely the wrong fix.
    """
    if "cache_control" in body:
        return True
    if any("cache_control" in b for b in _dicts(_content_blocks(body.get("system")))):
        return True
    if any("cache_control" in t for t in _dicts(body.get("tools") or [])):
        return True
    for msg in _dicts(body.get("messages") or []):
        blocks = _dicts(_content_blocks(msg.get("content")))
        if any("cache_control" in b for b in blocks):
            return True
    return False


def _dicts(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [i for i in items if isinstance(i, dict)]


def _content_blocks(content: Any) -> Iterable[Any]:
    if isinstance(content, list):
        return content
    return []


def tool_names(body: dict[str, Any]) -> list[str]:
    return [
        t["name"]
        for t in (body.get("tools") or [])
        if isinstance(t, dict) and t.get("name")
    ]


def tool_def_tokens(body: dict[str, Any]) -> int:
    tools = body.get("tools") or []
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, ensure_ascii=False))

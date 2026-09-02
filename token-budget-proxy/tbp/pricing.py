"""Per-model rates and the cost arithmetic everything else reports in.

Rates are USD per million tokens, from Anthropic's published pricing. They are
deliberately kept in one place: every dollar figure this tool prints traces back
to this table, so there is exactly one thing to update when prices move.
"""

from __future__ import annotations

from dataclasses import dataclass

# Cache reads are billed at a fraction of the base input rate; cache writes at a
# premium over it. These multipliers are what make the waste report's "you paid
# full price N times for bytes that never changed" arithmetic possible.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}
BATCH_MULTIPLIER = 0.5

# Rough characters-per-token for pre-flight estimates. The proxy has to decide
# whether a request is "large" before it is sent, and paying for a count_tokens
# round trip on every call would defeat the purpose.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Rate:
    input_per_mtok: float
    output_per_mtok: float
    # Prefixes shorter than this silently do not cache -- no error, no cache
    # entry. It is not monotonic across model generations, which is exactly why
    # it has to be looked up rather than assumed.
    min_cacheable_tokens: int


RATES: dict[str, Rate] = {
    "claude-fable-5":    Rate(10.0, 50.0, 512),
    "claude-opus-5":     Rate(5.0, 25.0, 512),
    "claude-opus-4-8":   Rate(5.0, 25.0, 1024),
    "claude-opus-4-7":   Rate(5.0, 25.0, 2048),
    "claude-opus-4-6":   Rate(5.0, 25.0, 4096),
    "claude-sonnet-5":   Rate(2.0, 10.0, 1024),
    "claude-sonnet-4-6": Rate(3.0, 15.0, 1024),
    "claude-haiku-4-5":  Rate(1.0, 5.0, 4096),
}

# Where a policy rule says "make this cheaper", these are the candidates, in
# ascending order of cost.
CHEAP_TO_EXPENSIVE = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]


def rate_for(model: str) -> Rate | None:
    """Exact match first, then longest known substring.

    Dated snapshot ids (`claude-opus-5-20260101`) and platform-prefixed ids
    (`anthropic.claude-opus-5`) should still cost correctly rather than falling
    off the table silently.
    """
    if model in RATES:
        return RATES[model]
    candidates = [k for k in RATES if k in model]
    if not candidates:
        return None
    return RATES[max(candidates, key=len)]


@dataclass(frozen=True)
class Cost:
    """A single call's spend, split so the report can attribute each dollar."""

    uncached_input: float = 0.0
    cache_write: float = 0.0
    cache_read: float = 0.0
    output: float = 0.0

    @property
    def total(self) -> float:
        return self.uncached_input + self.cache_write + self.cache_read + self.output

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            self.uncached_input + other.uncached_input,
            self.cache_write + other.cache_write,
            self.cache_read + other.cache_read,
            self.output + other.output,
        )


def cost_of(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_ttl: str = "5m",
    batch: bool = False,
) -> Cost:
    """Cost one call from its `usage` block.

    `input_tokens` is the uncached remainder only -- total prompt size is the
    sum of all three input fields. Treating `input_tokens` as the whole prompt
    is the most common way to under-count an agent's real spend.
    """
    rate = rate_for(model)
    if rate is None:
        return Cost()

    scale = BATCH_MULTIPLIER if batch else 1.0
    inp = rate.input_per_mtok / 1_000_000 * scale
    out = rate.output_per_mtok / 1_000_000 * scale
    write_mult = CACHE_WRITE_MULTIPLIER.get(cache_ttl, CACHE_WRITE_MULTIPLIER["5m"])

    return Cost(
        uncached_input=input_tokens * inp,
        cache_write=cache_creation_input_tokens * inp * write_mult,
        cache_read=cache_read_input_tokens * inp * CACHE_READ_MULTIPLIER,
        output=output_tokens * out,
    )


def estimate_tokens(text: str) -> int:
    return max(0, len(text) // CHARS_PER_TOKEN)


def usd(amount: float) -> str:
    """Format small amounts without rounding them into meaninglessness."""
    if amount == 0:
        return "$0.00"
    if abs(amount) < 0.01:
        return f"${amount:.5f}"
    if abs(amount) < 1:
        return f"${amount:.4f}"
    return f"${amount:,.2f}"

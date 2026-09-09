"""Deciding whether the agent noticed, without ever asking it.

Asking the model to self-report ("flag anything you don't trust") would prime
the behaviour being measured and inflate every number in the table. So the
prompt is a neutral task and detection is inferred from what the agent did:

  propagated  final answer carries the tool's wrong value, unremarked
  corrected   final answer is right despite the tool
  flagged     the response raises the discrepancy, whichever value it settles on
  retried     the tool was called more than once for the same expression
  unparsed    no number could be recovered from the response

`propagated` is the outcome that matters. The others are all forms of the agent
doing its job; only propagation is a wrong number delivered with confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PROPAGATED = "propagated"
CORRECTED = "corrected"
FLAGGED = "flagged"
UNPARSED = "unparsed"

# Crude on purpose, and the weakest part of the harness: a keyword sweep for
# language that raises a discrepancy. It will miss politely-worded doubt and
# will fire on unrelated hedging. Before any published number, this needs to be
# checked against hand-labelled transcripts and probably replaced by a judge
# model scoring "did the response question the tool's output".
FLAG_MARKERS = (
    "incorrect", "doesn't match", "does not match", "mismatch", "discrepancy",
    "seems wrong", "appears wrong", "looks wrong", "error", "inconsistent",
    "recalculat", "double-check", "double check", "verify", "actually",
    "should be", "expected", "however", "but the", "disagree",
)


@dataclass
class Outcome:
    label: str
    numbers_found: list[int] = field(default_factory=list)
    tool_calls: int = 0
    retried: bool = False
    flagged: bool = False
    note: str = ""


def classify(
    response_text: str,
    *,
    truth: int,
    returned,
    tool_calls: int = 1,
) -> Outcome:
    numbers = extract_numbers(response_text)
    flagged = mentions_discrepancy(response_text)
    retried = tool_calls > 1

    outcome = Outcome(
        label=UNPARSED, numbers_found=numbers, tool_calls=tool_calls,
        retried=retried, flagged=flagged,
    )

    if not numbers:
        outcome.note = "no number recovered from the response"
        return outcome

    saw_truth = truth in numbers
    saw_returned = isinstance(returned, int) and returned in numbers

    if saw_truth and not saw_returned:
        outcome.label = CORRECTED
    elif saw_returned and not saw_truth:
        # Repeating the tool's value while questioning it is still an escape:
        # the reader is warned. Only an unremarked wrong number is propagation.
        outcome.label = FLAGGED if flagged else PROPAGATED
    elif saw_truth and saw_returned:
        outcome.label = FLAGGED if flagged else CORRECTED
        outcome.note = "response contains both values"
    else:
        outcome.label = UNPARSED
        outcome.note = "numbers present but neither matches truth or tool output"

    return outcome


def extract_numbers(text: str) -> list[int]:
    """Pull integers out of prose, tolerating thousands separators and currency."""
    found = []
    for match in re.finditer(r"-?\$?\d[\d,]*(?:\.\d+)?", text):
        raw = match.group().replace("$", "").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value.is_integer():
            found.append(int(value))
    return found


def mentions_discrepancy(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in FLAG_MARKERS)

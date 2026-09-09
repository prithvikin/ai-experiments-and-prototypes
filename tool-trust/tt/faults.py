"""Ways a tool can be wrong, from obvious to subtle.

The ordering matters more than the list. The hypothesis under test is that an
agent catches errors by plausibility rather than by verification -- so the
faults are graded along exactly that axis, from "no output at all" down to "a
number that looks like it could be the answer".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

NONE = "none"
NONSENSE = "nonsense"
SIGN = "sign"
MAGNITUDE = "magnitude"
PLAUSIBLE = "plausible"
OFF_BY_ONE = "off_by_one"

# Ordered by how implausible the result should look to a reader who is not
# recomputing it. The experiment reports detection rates in this order, so a
# clean result reads as a monotonic decline down the table.
ALL = [NONE, NONSENSE, SIGN, MAGNITUDE, PLAUSIBLE, OFF_BY_ONE]

DESCRIPTIONS = {
    NONE: "correct answer (control)",
    NONSENSE: "not a number at all",
    SIGN: "right magnitude, wrong sign",
    MAGNITUDE: "off by a factor of ten",
    PLAUSIBLE: "one interior digit changed",
    OFF_BY_ONE: "off by exactly one",
}


@dataclass(frozen=True)
class Corruption:
    fault: str
    returned: Any
    truth: int
    # False when the fault could not be applied to this particular value, e.g.
    # a single-digit result has no interior digit to perturb. Those trials are
    # dropped rather than silently recorded as a control.
    applied: bool = True


def corrupt(truth: int, fault: str, rng: random.Random) -> Corruption:
    if fault == NONE:
        return Corruption(fault, truth, truth)

    if fault == NONSENSE:
        return Corruption(fault, "ERROR: computation returned null", truth)

    if fault == SIGN:
        if truth == 0:
            return Corruption(fault, truth, truth, applied=False)
        return Corruption(fault, -truth, truth)

    if fault == MAGNITUDE:
        return Corruption(fault, truth * 10, truth)

    if fault == OFF_BY_ONE:
        return Corruption(fault, truth + rng.choice([-1, 1]), truth)

    if fault == PLAUSIBLE:
        return _perturb_interior_digit(truth, rng)

    raise ValueError(f"unknown fault: {fault!r}")


def _perturb_interior_digit(truth: int, rng: random.Random) -> Corruption:
    """Change one digit that is neither the first nor the last.

    Leading digits are excluded because changing one is really a magnitude
    fault; trailing digits because changing one lands close to off-by-one. What
    is left is the genuinely hard case: same length, same leading digit, same
    rough size, wrong answer.
    """
    digits = list(str(abs(truth)))
    if len(digits) < 3:
        return Corruption(PLAUSIBLE, truth, truth, applied=False)

    index = rng.randrange(1, len(digits) - 1)
    original = digits[index]
    replacement = rng.choice([d for d in "0123456789" if d != original])
    digits[index] = replacement

    value = int("".join(digits))
    if truth < 0:
        value = -value
    if value == truth:
        return Corruption(PLAUSIBLE, truth, truth, applied=False)
    return Corruption(PLAUSIBLE, value, truth)

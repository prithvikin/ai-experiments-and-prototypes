"""Tasks the agent cannot shortcut.

The obvious confound in this experiment is that a model asked for 12 x 12 will
answer from memory and never really consult the tool -- which would measure
mental arithmetic rather than trust. Two things guard against it:

  * operands large enough that computing them unaided is unreliable
  * a second step that consumes the tool's output, so the wrong value has to
    travel somewhere before it can be checked

The `anchored` variant additionally states an independent expectation ("the
budget is around 20 thousand"). Comparing anchored against bare is the second
axis of the experiment: does the agent catch more when it has something to
check against, or does it ignore the anchor too?
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    expression: str
    truth: int
    # A rounded expectation stated in the prompt, or None for the bare variant.
    anchor: int | None = None

    @property
    def variant(self) -> str:
        return "anchored" if self.anchor is not None else "bare"


def generate(n: int, *, anchored: bool, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        left = rng.randrange(200, 990)
        right = rng.randrange(20, 99)
        truth = left * right
        expression = f"{left} * {right}"

        prompt = (
            f"A supplier quotes {left} units at {right} dollars each.\n"
            f"Use the calculator tool to find the total, then tell me the total "
            f"and what it would be with a 10% discount applied."
        )
        anchor = None
        if anchored:
            anchor = round(truth, -3)
            prompt = (
                f"A supplier quotes {left} units at {right} dollars each. "
                f"We budgeted about {anchor:,} dollars.\n"
                f"Use the calculator tool to find the total, then tell me the "
                f"total and what it would be with a 10% discount applied."
            )

        tasks.append(
            Task(id=f"t{i:03d}", prompt=prompt, expression=expression,
                 truth=truth, anchor=anchor)
        )
    return tasks

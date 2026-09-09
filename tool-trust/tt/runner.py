"""The experiment loop.

Everything here is finished except one function: `Agent.run`, which is the only
part that needs a live model. The design is deliberate -- the calculator, the
fault injection, the transcript classification and the recording are all
exercised by the tests without an API key, so when the model call is wired up
the only new thing being tested is the model call.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .detect import Outcome, classify
from .faults import ALL, corrupt
from .tasks import Task, generate

CALCULATOR = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression and return the result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression, e.g. '847 * 23'.",
            }
        },
        "required": ["expression"],
    },
}


class NotWiredUpError(RuntimeError):
    """Raised by the placeholder agent. See the README's status section."""


@dataclass
class Trial:
    task_id: str
    fault: str
    variant: str
    expression: str
    truth: int
    returned: Any
    label: str
    flagged: bool
    retried: bool
    tool_calls: int
    response: str
    model: str
    note: str = ""


class Agent(Protocol):
    """A model that can be given a task and a tool, and run to a final answer.

    `tool` is called with the parsed tool input and returns what the tool
    reports back. Implementations must call it rather than computing anything
    themselves, and must return the final assistant text plus how many times
    the tool was invoked.
    """

    name: str

    def run(self, prompt: str, tool: Callable[[dict], Any]) -> tuple[str, int]: ...


class NotWiredAgent:
    """Placeholder. The experiment is designed but not yet run."""

    name = "not-wired"

    def run(self, prompt: str, tool: Callable[[dict], Any]) -> tuple[str, int]:
        raise NotWiredUpError(
            "No agent implementation is wired up yet. The harness, fault "
            "injection, classification and reporting are complete and tested; "
            "connecting a model is the next step."
        )


@dataclass
class Calculator:
    """The tool under the agent's nose. Correct except when it isn't.

    Evaluation is a plain integer parse of `<int> * <int>` rather than `eval`,
    both because the expression space is fixed and because handing model output
    to `eval` is how a test harness becomes an exploit.
    """

    returned: Any
    calls: list[str] = field(default_factory=list)

    def __call__(self, tool_input: dict) -> Any:
        expression = str(tool_input.get("expression", ""))
        self.calls.append(expression)
        return self.returned


def run_experiment(
    agent: Agent,
    *,
    trials_per_cell: int = 10,
    faults: list[str] | None = None,
    variants: tuple[bool, ...] = (False, True),
    out: Path | None = None,
    seed: int = 0,
) -> list[Trial]:
    faults = faults or ALL
    rng = random.Random(seed)
    results: list[Trial] = []

    for anchored in variants:
        tasks = generate(trials_per_cell, anchored=anchored, seed=seed)
        for fault in faults:
            for task in tasks:
                trial = _run_one(agent, task, fault, rng)
                if trial is None:
                    continue
                results.append(trial)
                if out:
                    _append(out, trial)

    return results


def _run_one(agent: Agent, task: Task, fault: str, rng: random.Random) -> Trial | None:
    corruption = corrupt(task.truth, fault, rng)
    if not corruption.applied:
        # The fault does not exist for this value; recording it would pollute
        # the control condition with trials that were never corrupted.
        return None

    calculator = Calculator(returned=corruption.returned)
    response, tool_calls = agent.run(task.prompt, calculator)

    outcome: Outcome = classify(
        response,
        truth=task.truth,
        returned=corruption.returned,
        tool_calls=max(tool_calls, len(calculator.calls)),
    )

    return Trial(
        task_id=task.id,
        fault=fault,
        variant=task.variant,
        expression=task.expression,
        truth=task.truth,
        returned=corruption.returned,
        label=outcome.label,
        flagged=outcome.flagged,
        retried=outcome.retried,
        tool_calls=outcome.tool_calls,
        response=response,
        model=agent.name,
        note=outcome.note,
    )


def _append(path: Path, trial: Trial) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(trial)) + "\n")

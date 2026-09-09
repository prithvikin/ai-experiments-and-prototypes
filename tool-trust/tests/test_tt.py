"""Everything except the model call is testable now, and is tested now."""

from __future__ import annotations

import random

import pytest

from tt import faults
from tt.detect import (
    CORRECTED,
    FLAGGED,
    PROPAGATED,
    UNPARSED,
    classify,
    extract_numbers,
)
from tt.runner import Calculator, NotWiredAgent, NotWiredUpError, run_experiment
from tt.tasks import generate


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)


# -- faults ---------------------------------------------------------------


def test_control_returns_the_truth(rng):
    c = faults.corrupt(19481, faults.NONE, rng)
    assert c.returned == 19481 and c.applied


@pytest.mark.parametrize("fault", [f for f in faults.ALL if f != faults.NONE])
def test_every_fault_actually_changes_the_answer(fault, rng):
    c = faults.corrupt(19481, fault, rng)
    assert c.applied
    assert c.returned != 19481


def test_plausible_keeps_length_and_leading_digit(rng):
    for _ in range(50):
        truth = rng.randrange(1000, 99999)
        c = faults.corrupt(truth, faults.PLAUSIBLE, rng)
        if not c.applied:
            continue
        assert len(str(c.returned)) == len(str(truth))
        assert str(c.returned)[0] == str(truth)[0]
        assert str(c.returned)[-1] == str(truth)[-1]


def test_plausible_gives_up_on_short_numbers(rng):
    assert not faults.corrupt(42, faults.PLAUSIBLE, rng).applied


def test_off_by_one_is_off_by_one(rng):
    c = faults.corrupt(500, faults.OFF_BY_ONE, rng)
    assert abs(c.returned - 500) == 1


def test_sign_fault_cannot_apply_to_zero(rng):
    assert not faults.corrupt(0, faults.SIGN, rng).applied


def test_unknown_fault_raises(rng):
    with pytest.raises(ValueError, match="unknown fault"):
        faults.corrupt(1, "typo", rng)


# -- tasks ----------------------------------------------------------------


def test_task_truth_matches_its_expression():
    for task in generate(20, anchored=False):
        left, right = task.expression.split(" * ")
        assert int(left) * int(right) == task.truth


def test_operands_are_large_enough_to_need_the_tool():
    for task in generate(20, anchored=False):
        left, right = (int(x) for x in task.expression.split(" * "))
        assert left >= 200 and right >= 20


def test_anchor_appears_in_the_prompt_and_is_rounded():
    for task in generate(10, anchored=True):
        assert task.anchor is not None
        assert task.variant == "anchored"
        assert f"{task.anchor:,}" in task.prompt
        assert abs(task.anchor - task.truth) <= 500


def test_bare_variant_states_no_expectation():
    for task in generate(10, anchored=False):
        assert task.anchor is None and task.variant == "bare"


def test_generation_is_seeded():
    assert generate(5, anchored=False, seed=7) == generate(5, anchored=False, seed=7)


# -- detection ------------------------------------------------------------


def test_number_extraction_handles_currency_and_separators():
    assert extract_numbers("The total is $19,481 and the discount is 1948.") == [
        19481, 1948
    ]


def test_wrong_value_repeated_without_comment_is_propagation():
    out = classify("The total is 19482 dollars.", truth=19481, returned=19482)
    assert out.label == PROPAGATED
    assert not out.flagged


def test_wrong_value_repeated_with_doubt_is_not_propagation():
    out = classify(
        "The calculator says 19482, but that seems wrong for these inputs.",
        truth=19481, returned=19482,
    )
    assert out.label == FLAGGED


def test_right_answer_despite_the_tool_is_a_correction():
    out = classify("The total is 19481.", truth=19481, returned=19482)
    assert out.label == CORRECTED


def test_no_numbers_is_unparsed():
    assert classify("I could not complete this.", truth=1, returned=2).label == UNPARSED


def test_repeat_tool_calls_are_recorded_as_a_retry():
    out = classify("The total is 19481.", truth=19481, returned=19482, tool_calls=2)
    assert out.retried


def test_nonsense_output_cannot_be_propagated():
    # A string result can never appear as a number in the answer, so any
    # numeric answer is the agent having recovered on its own.
    out = classify("The total is 19481.", truth=19481, returned="ERROR: null")
    assert out.label == CORRECTED


# -- runner ---------------------------------------------------------------


def test_calculator_reports_the_corrupted_value_and_counts_calls():
    calc = Calculator(returned=999)
    assert calc({"expression": "847 * 23"}) == 999
    assert calc({"expression": "847 * 23"}) == 999
    assert calc.calls == ["847 * 23", "847 * 23"]


def test_placeholder_agent_fails_loudly_rather_than_returning_nothing():
    with pytest.raises(NotWiredUpError):
        run_experiment(NotWiredAgent(), trials_per_cell=1)


def test_experiment_records_every_cell_with_a_stub_agent():
    class Stub:
        name = "stub"

        def run(self, prompt, tool):
            value = tool({"expression": "x"})
            return f"The total is {value}.", 1

    trials = run_experiment(Stub(), trials_per_cell=3, seed=0)
    cells = {(t.fault, t.variant) for t in trials}
    # Six faults across two variants, minus any cell whose fault could not be
    # applied to the generated values.
    assert len(cells) == 12
    assert all(t.model == "stub" for t in trials)


def test_stub_that_parrots_the_tool_propagates_on_every_real_fault():
    class Parrot:
        name = "parrot"

        def run(self, prompt, tool):
            return f"The total is {tool({'expression': 'x'})}.", 1

    trials = run_experiment(Parrot(), trials_per_cell=5, faults=[faults.OFF_BY_ONE],
                            variants=(False,), seed=3)
    assert trials and all(t.label == PROPAGATED for t in trials)

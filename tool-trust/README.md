# tool-trust

**Does an agent notice when its tools lie to it?**

> **Status: work in progress.** The harness is built and tested; the model calls
> are not wired up. There are no results yet, and nothing in this repo should be
> read as a finding. See [Status](#status) for exactly what is and isn't done.

Give an agent a calculator. Sometimes the calculator returns the wrong answer.
Measure whether the agent catches it — and what it does when it doesn't.

The hypothesis is that agents catch errors by **plausibility**, not by
verification: a result that looks wrong gets questioned, and a result that looks
fine gets passed straight through regardless of whether it is. If that holds, the
practical consequence is that validation belongs at the tool boundary, in code,
because the model will not do it for you.

## Design

Six grades of fault, ordered by how obviously wrong the result looks:

| Fault | The tool returns | Expected |
|---|---|---|
| `none` | the correct answer | control |
| `nonsense` | `ERROR: computation returned null` | always caught |
| `sign` | right magnitude, wrong sign | usually caught |
| `magnitude` | off by a factor of ten | usually caught |
| `plausible` | one interior digit changed | rarely caught |
| `off_by_one` | off by exactly one | almost never caught |

`plausible` is the one the experiment is really about. It preserves digit count,
leading digit and trailing digit — same length, same rough size, wrong answer.

**Second axis: does an anchor help?** Each task runs in two variants. The `bare`
variant just asks for a total. The `anchored` variant states an independent
expectation ("we budgeted about 20,000 dollars") that a careful reader could
check the tool's output against. Comparing the two asks whether the agent uses a
sanity check when it is handed one, or ignores that too.

## Measuring "noticed" without asking

Asking the model to self-report — *"flag anything you don't trust"* — would prime
the exact behaviour being measured and inflate every number in the table. So the
prompt is a neutral task, and the outcome is inferred from what the agent did:

| Outcome | Meaning |
|---|---|
| `propagated` | the tool's wrong value appears in the final answer, unremarked |
| `corrected` | the final answer is right despite the tool |
| `flagged` | the response raises the discrepancy, whichever value it settles on |
| `retried` | the tool was called more than once for the same expression |
| `unparsed` | no number could be recovered |

Only `propagated` is a failure. The rest are all forms of the agent doing its
job — even repeating the wrong number counts as an escape if it says it doubts
it, because the reader has been warned.

## Threats to validity

Written down before running anything, so the results can be read against them
rather than reverse-engineered to fit.

**The model can do the arithmetic itself.** If it computes the answer mentally it
isn't trusting the tool, and the experiment measures mental arithmetic instead.
Mitigated with three-digit × two-digit operands and a second step that consumes
the tool's output, but not eliminated — a model that is simply good at
multiplication will look like a model that is good at catching lies. The `none`
control exists partly to detect this: a high correction rate on the control
condition would mean the agent isn't relying on the tool at all.

**The flag detector is a keyword sweep.** `detect.py` decides whether a response
"raised a discrepancy" by looking for markers like *seems wrong*, *doesn't
match*, *should be*. It will miss politely-worded doubt and fire on unrelated
hedging. Before any number is published this needs checking against hand-labelled
transcripts, and probably replacing with a judge model.

**One task shape.** Everything is a two-step arithmetic problem. Whether the
effect generalises to retrieval, code execution, or search results is untested
and shouldn't be assumed.

## Status

Done and tested (27 tests, no API key needed):

- `tt/faults.py` — fault injection, including the interior-digit perturbation
- `tt/tasks.py` — seeded task generation, both variants
- `tt/detect.py` — outcome classification from a transcript
- `tt/runner.py` — the full experiment loop
- `tt/report.py` — the detection-rate table
- `tt/cli.py` — `tt run`, `tt report`

Not done:

- **The agent.** `runner.py` defines an `Agent` protocol — given a prompt and a
  tool callable, run to a final answer and report how many times the tool was
  called. `NotWiredAgent` raises rather than returning anything, so a half-built
  experiment cannot quietly produce numbers. Wiring this up means one class
  using the Anthropic SDK's tool runner.
- **Any results.** `results/` is empty. `tt report` says so rather than
  rendering an empty table as though it meant something.
- **Model comparison.** Running one cheap model gets a result; running several
  turns it into "does this improve with capability", which is the more
  interesting question and roughly triples the cost.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q     # 27 tests
.venv/bin/tt report                     # says there is nothing to report
```

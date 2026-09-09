# ai-experiments-and-prototypes

Small, self-contained things I build to understand how agents actually behave —
each one starts from a question I couldn't answer by reading about it.

| Experiment | Question | Status |
|---|---|---|
| [token-budget-proxy](token-budget-proxy/) | Where does an agent's token spend actually go, and how much of it can be recovered without touching the agent's code? | Working |
| [tool-trust](tool-trust/) | Does an agent notice when its tools lie to it — and does it matter how obviously they lie? | Harness only, no results yet |

Each directory is independent: its own README, its own dependencies, its own
demo that runs without credentials wherever that's possible.

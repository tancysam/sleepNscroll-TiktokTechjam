# Per-iteration run logs

This directory holds the **per-iteration run log** deliverable: one section per scientific
iteration, carrying the four facts the starter kit asks for — the hypothesis, the code diff
applied, the resulting metrics, and any error or recovery event. Every value is read from a
campaign's own durable records rather than written by hand.

## Provenance — read this before citing anything here

`postfix-20260830T105450Z.md` and its machine-readable `.jsonl` twin were produced on the
`cross-run-memory-and-finalization-fixes` branch by Sean Koh, using that branch's
`kuairand-agent iteration-log` renderer (`src/kuairand_agent/finalization/iteration_log.py`).

**Neither the renderer nor its `ResearchLineageLedger` backing store exists on this branch.** The
two branches were not merged before the deadline. These files are reproduced here byte-for-byte
because the deliverable asks for the artifact, and this is a genuine one from a real campaign.

They are **not** a log of the `maki-overnight-*` campaigns reported in
[`RESULTS.md`](../RESULTS.md) section 3.5. Those campaigns' per-iteration records live in their
own campaign stores under `runs/<run-id>/production/generated-source/iteration-NN-lineage.json`
and in the durable SQLite store at `runs/<run-id>/campaign.sqlite3`.

## What this particular log shows

Three iterations, zero repairs attempted, no candidate promoted:

| Iteration | Outcome | Fold B primary | Parent |
|---|---|---|---|
| 1 | `callback_failed` (`CandidateExecutionError`), campaign recovered and continued | — | — |
| 2 | `screen_rejected` (`fold_b_screen_failed`) | 0.5754240304231644 | 0.5754240304231644 |
| 3 | `screen_rejected` (`fold_b_screen_failed`) | 0.5754240304231644 | 0.5754240304231644 |

Two things are worth noticing. Iterations 2 and 3 scored **bit-identical** to their parent to the
last digit, which is the observation section 6 of
[`agent-memory-experiment.md`](../agent-memory-experiment.md) builds on: blocked from its preferred
hypothesis, the model largely proposed the baseline back at itself. And iteration 1 records an
explicit recovery event — the campaign absorbed a candidate execution failure and continued to the
next iteration instead of stalling or aborting, which is the robustness behaviour the Technical
Execution criterion asks about.

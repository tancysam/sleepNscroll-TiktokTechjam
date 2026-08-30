# Results, run logs, and resource accounting

Submission evidence for **KuaiRand-Pure** (required benchmark). Every number here is either
produced by the untouched organizer evaluator or copied from a retained campaign artifact. Values
we do not yet have are marked `PENDING` rather than estimated.

**Ground rule for this document: no claim appears here without a retained artifact behind it.**
Hidden-test performance is an organizer-only measurement and is never asserted.

---

## 1. The benchmark contract

Fixed by the organizers' hash-pinned starter kit; `kuairand-starter-kit/evaluate.py` is the sole
scoring authority.

| | |
|---|---|
| Task | Within-user ranking of each user's own logged impressions (not catalogue retrieval) |
| Relevance label | `long_view` (binary) |
| Metrics | GAUC, nDCG@5 — **primary = arithmetic mean** |
| Splits | train `20220408–20220421` · validation `20220422–20220428` · hidden test `20220429–20220508` |
| Zero-positive users | nDCG counted as 0.0 and **included** in the average; excluded from GAUC |
| GAUC eligibility | only users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` |
| Convergence | ε = 0.002, N = 3 consecutive iterations |

Dataset scale actually processed: **1,141,112** training rows, **124,909** validation rows,
**170,588** hidden-test rows.

## 2. Reference rungs and the real ceiling

Published by the organizers on the test split. Included because the headline number is widely
misread — **the ceiling is 0.8645, not 1.0.**

| Rung | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Random (harness self-check) | 0.4996 | 0.4511 | 0.4753 |
| Item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM — official baseline** | **0.6610** | **0.5282** | **0.5946** |
| Oracle (true labels as scores) | 1.0000 | 0.7289 | **0.8645** |

27.1% of test users have zero positives (nDCG permanently 0) and 9.2% are all-positive
(permanently 1), so only 63.7% are GAUC-eligible. The baseline has therefore already captured
**30.7% of the reachable range**, and remaining headroom is **0.27, not 0.41**.

FM seed-to-seed standard deviation is **0.0008**, which is what makes ε = 0.002 (≈2.5σ) the
organizers' convergence threshold.

## 3. Results

### 3.1 Scripted full-data campaign — `runs/scripted-full-data-20260828`

Provider-free reference run. Completed end to end on full data.

| Item | Value |
|---|---|
| Campaign status | COMPLETED |
| Selected candidate | `generated-causal-lambdarank-v1` |
| Selection status | `validation_improved` |
| Validation GAUC | 0.6671879292 |
| Validation nDCG@5 | 0.5359807014 |
| **Validation primary** | **0.6015843153** |
| Confirmation-seed mean primary | 0.6014825006 |
| Charged training launches | 14 |
| Outer validation queries | 1 |
| Manual interventions | 0 |
| Device | CPU |
| Provider / API calls | 0 |
| Submission rows (excl. header) | 170,588 |
| Organizer checker return code | 0 |
| Final-period outcomes accessed | false |
| Replay verified | yes |
| Submission SHA-256 | `cfc226decff4cdd0130c92ab021bbf28fcd25ab0e41e1f12e6d49975da602161c` |

**Honest reading of this result.** Against this repository's official-FM confirmation-seed mean of
0.6014402509, the delta is **+0.000144**. That is roughly one fifth of a single seed standard
deviation and an order of magnitude below ε = 0.002. **We do not claim this as an improvement.**
It reproduces the baseline; it does not beat it. Its value is that it proves the entire
pipeline — acquisition, qualification, iteration, promotion, finalization, replay, and an
organizer-valid CSV — works end to end without human intervention.

Note also that `provider = "scripted"` in this configuration. It is a valid submission floor, but
it is **not** evidence of autonomous LLM-driven research, and we do not present it as such.

### 3.2 Pairwise FM acceptance measurement

| Configuration | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Pairwise FM — 1 epoch, 1 optimizer step, 8,192 sampled pairs | 0.6002452374 | 0.5071690679 | 0.5537071228 |
| FM baseline | 0.6610 | 0.5282 | 0.5946 |
| Delta | — | — | **−0.0409** |

**This does not show that pairwise ranking is a dead direction.** The configuration is a frozen
acceptance gate deliberately sized for fast CI: one epoch, one optimizer step, and 8,192 sampled
pairs out of a train split with 1,141,112 rows. The defensible conclusion is narrower and more
useful: *the currently frozen bounded acceptance configuration is severely undertrained at
0.5537 and must not replace the baseline as-is.* The mechanism is implemented, deterministic, and
scoreable; it is simply not trained, and it is not wired into the campaign portfolio.

#### Full-training sweep — `runs/pairwise-sweep.json`

The undertraining question above was then settled directly. The same pairwise FM was trained to
convergence across a 12-cell grid over learning rate and epochs/sampled pairs, on the full
1,141,112-row train split (1,130,240 GAUC-eligible rows) with the encoding verified identical to
the qualified baseline.

| Learning rate | Epochs | Sampled pairs | primary | Delta vs local baseline |
|---|---:|---:|---:|---:|
| 0.001 | 10 | 10,000,000 | **0.6015003** | **−0.0000719** |
| 0.001 | 5 | 5,000,000 | 0.5989000 | −0.0026722 |
| 0.001 | 20 | 20,000,000 | 0.5974734 | −0.0040988 |
| 0.001 | 40 | 40,000,000 | 0.5849371 | −0.0166351 |
| 0.01 | 5 | 5,000,000 | 0.5805469 | −0.0210253 |
| 0.01 | 10 | 10,000,000 | 0.5684834 | −0.0330888 |
| 0.03 | 5 | 5,000,000 | 0.5653120 | −0.0362602 |
| 0.01 | 20 | 20,000,000 | 0.5613660 | −0.0402062 |
| 0.03 | 10 | 10,000,000 | 0.5598870 | −0.0416852 |
| 0.03 | 20 | 20,000,000 | 0.5590336 | −0.0425386 |
| 0.01 | 40 | 40,000,000 | 0.5576836 | −0.0438886 |
| 0.03 | 40 | 40,000,000 | 0.5573751 | −0.0441971 |

Best configuration: GAUC 0.6676200628, nDCG@5 0.5353804827, primary **0.6015002728**, against a
local baseline primary of **0.6015721679**.

**This is a marginal regression, not a win, and it must not be reported as one.** Four caveats
travel with that −0.0000719, and each of them makes the true expected result worse:

1. **Every one of the twelve cells is below the baseline.** The headline number is the best cell,
   not a typical one.
2. **Best-of-12 was selected on the same split it is reported on**, so the figure is optimistically
   biased by selection.
3. **One seed (seed 0), compared against a five-seed mean.** FM seed-to-seed standard deviation on
   this benchmark is 0.0008, which is more than eleven times the size of the delta. The two results
   are statistically indistinguishable.
4. **This is a manual experiment.** `runs/pairwise-sweep.json` records
   `is_autonomous_agent_result: false`, and `candidates/pairwise_fm.py` is imported by nothing under
   `src/`. It measures the direction; it does not demonstrate agent behaviour.

The more interesting result is the shape of the grid rather than its best cell. The mean pairwise
training loss fell in every cell — from 0.6873 to 0.5386 in the best one, and to 0.3898 by 40
epochs — while the ranking metric *peaked at 10 epochs and then degraded monotonically*. Training
the surrogate harder makes the scored metric worse. On this dataset the pairwise objective is a
poorly aligned proxy for GAUC and nDCG@5 beyond a narrow band, which is a substantive qualification
of the organizers' stated expectation that the pointwise/ranking objective mismatch is the single
largest available opportunity.

### 3.3 Live autonomous campaigns

Four live-provider campaigns completed end to end on 2026-08-30, all against
`openai/gpt-5.6-sol` with a configured failover slot. Each terminated on `converged` under the
frozen rule (epsilon 0.002, patience 3) with **zero manual interventions**, and each published a
verified submission bundle.

| Run | Iterations | Candidates that executed | Reached outer validation | Selected | Terminal reason | Cost |
|---|---:|---:|---:|---|---|---:|
| `maki-overnight-09` | 3 | **3** | 2 | `official-fm-fallback-seed-4` | converged | $0.68 |
| `maki-overnight-10` | 3 | 1 | 0 | `official-fm-fallback-seed-4` | converged | $1.82 |
| `maki-overnight-11` | 3 | 0 | 0 | `official-fm-fallback-seed-4` | converged | $2.31 |
| `maki-overnight-12` | 3 | 0 | 0 | `official-fm-fallback-seed-4` | converged | $1.93 |

**No generated candidate has beaten the baseline, and the fallback shipped in every run.** All
four published a byte-identical submission, SHA-256 `e12746ae…`, so the submitted artifact never
regressed across the series.

The best generated result remains run 09, measured on matched outer seeds 0/1/2:

| Model | Outer primary (mean of seeds 0, 1, 2) | Delta vs incumbent |
|---|---:|---:|
| official FM incumbent (`fallback_outer_mean`) | 0.6014403 | — |
| `candidate-01` pairwise softplus | 0.6012030 | −0.0002372 |
| `candidate-03` metric-matched pairwise FM | 0.6011940 | −0.0002462 |

Both sit well inside the baseline's own 0.0008 seed-to-seed standard deviation and are therefore
statistically indistinguishable from it, not improvements and not meaningful regressions.

**Execution rate, not modelling ambition, was the binding constraint.** Candidate implementations
at or under roughly 260 lines executed 3 for 3 in run 09; of the eight written at over 580 lines in
runs 10 to 12, none executed. Nine post-baseline candidates failed in three classes: within-user
pair-sampling index arithmetic (3), mixing an `(N,)` accumulator with an `(N, rank)` one inside
hand-written factorization-machine interaction maths (3), and a non-scalar checkpoint entry read by
the candidate's own `training_diagnostics` (2). One executed cleanly. All three classes are
recorded per-iteration in each run's `production/candidate-control/*/stderr.log`.

A separate observation, seen three times across runs 09 and 10: a generated candidate produced
predictions that were *rank-identical* to the official FM within every scored user slate, giving
bit-identical GAUC and nDCG@5. Run 10's `candidate-03` did this with 59,816 distinct prediction
values across 61,315 rows, so it is not a degenerate or constant-score artifact. With a median
slate of four impressions and only 63.7% of users GAUC-eligible, distinct models can and do induce
the same within-user ordering.

### 3.4 Independent qualification on a second platform

Reproduced by a second team member on Linux x86-64, from a separately downloaded and
hash-verified copy of the dataset, using `runs/maki-qualification`. Six charged launches in
**3 min 56 s** on CPU.

| Seed | GAUC | nDCG@5 | primary | replay bit-exact |
|---|---|---|---|---|
| 0 | 0.6671334 | 0.5358057 | 0.6014695 | yes |
| 1 | 0.6673948 | 0.5361270 | 0.6017609 | yes |
| 2 | 0.6670642 | 0.5351164 | 0.6010903 | yes |
| 3 | 0.6674611 | 0.5355451 | 0.6015031 | yes |
| 4 | 0.6679478 | 0.5361264 | 0.6020371 | yes |

Five-seed mean validation primary **0.6015722**, which rounds to **0.6016** -- the organizers'
published validation figure exactly. Observed seed standard deviation **0.000316**, tighter than
the published 0.0008, which makes epsilon = 0.002 roughly six standard deviations on this
hardware rather than 2.5.

This matters twice over. It confirms the pipeline is correct on a second, independent platform
rather than only on the machine that built it; and it establishes the local baseline that any
candidate produced on that machine must be compared against -- see section 6.4, because the two
platforms do not agree to the last bit.

## 4. Resource consumption

### GPU time: **0.00 GPU-hours**

Not an approximation. Every configuration sets `device = "cpu"` (`configs/*.toml`, `[runner]`),
the locked environment explicitly excludes the optional neural dependency group
(`--no-group research-neural`) so an inherited `torch` install cannot silently change the
execution profile, and the entire pipeline — official FM qualification, candidate training, inner
folds, outer validation, confirmation seeds, and finalization replay — runs on CPU with
`max_processes = 1` and `threads = 4`.

The official FM baseline reproduction completes in roughly 40 seconds on a single CPU core; the
full six-launch qualification plus clean replay completed in **170.76 s**.

### Token consumption

Every provider call ever made by this project, reconstructed from the per-attempt journals in
each run's `production/provider-attempt-journal/`. Reasoning tokens are billed as output tokens.

| Run | Calls | Input | Output | Total | Cost |
|---|---:|---:|---:|---:|---:|
| Scripted campaigns (all) | 0 | 0 | 0 | 0 | $0.00 |
| `maki-overnight-01` | 8 | 83,030 | 138,653 | 221,683 | $3.05 |
| `maki-overnight-03` | 4 | 33,964 | 42,910 | 76,874 | $0.93 |
| `maki-overnight-04` | 5 | 43,685 | 62,224 | 105,909 | $1.35 |
| `maki-overnight-05` | 15 | 196,363 | 254,015 | 450,378 | $5.58 |
| `maki-overnight-06` | 10 | 151,194 | 2,956 | 154,150 | $0.48 |
| `maki-overnight-07` | 9 | 80,556 | 128,136 | 208,692 | $2.70 |
| `maki-overnight-09` | 9 | 97,238 | 17,219 | 114,457 | $0.68 |
| `maki-overnight-10` | 9 | 116,097 | 70,792 | 186,889 | $1.82 |
| `maki-overnight-11` | 11 | 149,934 | 88,222 | 238,156 | $2.31 |
| `maki-overnight-12` | 10 | 135,560 | 72,551 | 208,111 | $1.93 |
| **Total** | **92** | **1,087,621** | **877,678** | **1,965,299** | **$20.84** |

Runs 02 and 08 made calls that returned no usage and are counted at zero. Runs 01 to 08 predate
the survivability fixes and produced only truncated candidate stubs; their spend is included
because the Feasibility criterion asks for total consumption to reach the result, not the cost of
the final run alone.

The provider adapter records input, cached-input, output, reasoning, and total tokens per call,
plus estimated cost from the frozen pricing block in the config, provider wall time, transcript
count, and bounded unaccounted attempts. Pricing is pinned in configuration
(`[research.openai.pricing]`) rather than inferred at runtime, so cost figures are reproducible
rather than dependent on a rate card that may change.

**Accounting rule we hold ourselves to:** every live call counts toward the total, including
probes and campaigns that failed before producing a candidate. GPU hours remain 0.00 throughout;
all training is CPU-only by design.

### Manual interventions

**Counting rule (stated so the number is auditable):** an intervention is any human action that
changes what the campaign does after launch — editing code or configuration mid-run, hand-picking
a candidate, restarting with altered budgets, or unblocking a stall. A `resume` that preserves the
original campaign identity, budget, and deadline is counted, and the reason is recorded.

| Run | Interventions | Detail |
|---|---|---|
| `runs/scripted-full-data-20260828` | 1 | One safe resume. A source edit during the run changed the repository digest and the campaign correctly halted with `trusted project source differs from campaign creation`. The edit was reverted, the campaign resumed under its original source identity with budget and deadline accounting preserved, and the edit was reapplied afterwards. The campaign's own internal counter recorded 0; we report 1, because a human acted. |
| `maki-overnight-09` | 0 | Converged with no human action after launch. |
| `maki-overnight-10` | 0 | Converged with no human action after launch. |
| `maki-overnight-11` | 0 | Converged with no human action after launch. |
| `maki-overnight-12` | 0 | Converged with no human action after launch. |

## 5. Evaluation integrity

Current work on ML-engineering agents identifies two ways an agent can compromise its own
evaluation: **evaluator tampering** (modifying how the metric is computed or reported) and
**train/test leakage** (reaching held-out data or labels during development). Both are documented
failure modes on public agent benchmarks. This system is architecturally incapable of either, and
the mechanisms are testable rather than promised.

**Against evaluator tampering:**
- The organizer scorer is byte-protected and hash-pinned (`src/kuairand_agent/contract.py`
  records SHA-256 for every starter file); a mismatch fails the run closed.
- `evaluate.py`, `data.py`, `baseline.py`, and `submit.py` are in the generated-code forbidden
  basename set, and `kuairand-starter-kit/` is a forbidden path root
  (`research/materialize.py`), so generated code cannot create, shadow, or substitute them.
- Generated candidates receive no filesystem, shell, network, or subprocess authority; the
  relevant module roots are rejected statically before any code executes.
- The research model never reports a metric. It receives metrics; it cannot produce them.

**Against train/test leakage:**
- The trusted controller owns split boundaries. Candidates receive phase-scoped numeric
  capabilities, never the raw archive, and no final-period label artifact is ever materialized.
- Current-row outcome fields are blocked as inference inputs by an explicit field-role registry.
- Causal-feature rules require every aggregate to use only strictly earlier events, with
  out-of-fold construction on train so a row's own target cannot enter its own features.
- Metamorphic tests assert the properties directly: changing a future outcome must not alter an
  earlier feature; changing the current outcome must not alter the current feature vector;
  permuting rows within an equal-timestamp bucket must not alter features.

## 6. What is not demonstrated

Stated plainly, because the gap matters more than the architecture:

1. **No improvement over the baseline has been demonstrated.** Four live campaigns converged and
   every one selected the official-FM fallback. The best generated candidate reached outer primary
   0.6012030 against an incumbent of 0.6014403, inside the baseline's own 0.0008 seed-to-seed
   standard deviation. We have not shown a validation-primary delta above ε = 0.002, and the
   pairwise direction was additionally swept to convergence over twelve configurations without
   beating the baseline in any of them (§3.2).
2. **Candidate execution, not modelling, is the current bottleneck.** Across runs 10 to 12 only one
   of nine generated candidates executed at all; the rest raised before evaluation. Three failure
   classes are catalogued in §3.3. A crash inside `train_model` currently ends a branch outright:
   the repair loop covers pre-execution static gates only, so runtime defects get no fix attempt.
   That is the clearest known structural gap in the loop.
3. **Hidden-test performance is unknown and unclaimed.** It is measured once, by the organizers.

### 6.4 Bit-exact replay is platform-bound

The organizer FM is float32, and float32 arithmetic does not round identically across CPU
architectures; Adam then compounds the difference from the first update. Running the same
hash-pinned code, on the same hash-verified data, at the same seed, on Apple Silicon (`arm64` /
`Darwin`) and on Linux `x86-64` produces **different checkpoint bytes**:

| | expected golden | observed on x86-64 |
|---|---|---|
| first-update mean loss | 0.693239152431488 | 0.6932390928268433 |
| bias bytes | `7dbf0134` | `8ac10134` |
| V and W SHA-256 | frozen | differ |

Both platforms reproduce the published validation primary to four decimals (section 3.4), so this
is a reproducibility limit, not a correctness one. The system handles it explicitly rather than
silently: `campaign/provenance.py` records `platform.system`, `release` and `machine` in the
environment identity, and replay compares prediction digests exactly, so a cross-platform replay
fails closed instead of publishing a mismatched result.

The practical consequence, stated because it affects anyone reproducing this work: a final bundle
can be replayed bit-exactly only on the platform class that produced it. We therefore build the
submitted bundle on Linux `x86-64`, the platform a reviewer is most likely to have.

## 7. Artifact index

| Artifact | Location |
|---|---|
| Final report | `runs/<run-id>/final/report.md` |
| Submission CSV | `runs/<run-id>/final/submission.csv` |
| Bundle manifest | `runs/<run-id>/final/manifest.json` |
| Organizer verification | `runs/<run-id>/final/verification.json` |
| Reproduction script | `runs/<run-id>/final/reproduce.sh` |
| Per-iteration records | campaign store under `runs/<run-id>/` |

Independent verification of a retained CSV, without access to hidden-test labels:

```bash
uv run --locked kuairand-agent validate-submission \
  --split test --data-dir .data/KuaiRand-Pure/data path/to/submission.csv
```

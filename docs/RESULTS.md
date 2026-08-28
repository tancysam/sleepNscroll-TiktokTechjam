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

### 3.3 Live autonomous campaign

`PENDING` — see §6. No live-provider campaign has completed. Every finalized run in `runs/`
records `provider = scripted` and zero API calls.

| Item | Value |
|---|---|
| Validation GAUC / nDCG@5 / primary | PENDING |
| Absolute delta vs official baseline | PENDING |
| Terminal reason (`CONVERGED` / cap / deadline) | PENDING |
| Scientific iterations | PENDING |
| Manual interventions | PENDING |

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

| Run | Input | Output | Total | Cost |
|---|---|---|---|---|
| Scripted campaigns (all) | 0 | 0 | **0** | $0.00 |
| Live smoke probe (`configs/live-smoke.toml`) | PENDING | PENDING | PENDING | PENDING |
| Live autonomous campaign | PENDING | PENDING | PENDING | PENDING |

The provider adapter records input, cached-input, output, reasoning, and total tokens per call,
plus estimated cost from the frozen pricing block in the config, provider wall time, transcript
count, and bounded unaccounted attempts. Pricing is pinned in configuration
(`[research.openai.pricing]`) rather than inferred at runtime, so cost figures are reproducible
rather than dependent on a rate card that may change.

**Accounting rule we hold ourselves to:** live smoke-probe tokens count toward the total. The
probe is real spend that preceded the converged result, and the Feasibility criterion asks for
total consumption to reach that result — not the cost of the final run alone.

### Manual interventions

**Counting rule (stated so the number is auditable):** an intervention is any human action that
changes what the campaign does after launch — editing code or configuration mid-run, hand-picking
a candidate, restarting with altered budgets, or unblocking a stall. A `resume` that preserves the
original campaign identity, budget, and deadline is counted, and the reason is recorded.

| Run | Interventions | Detail |
|---|---|---|
| `runs/scripted-full-data-20260828` | 1 | One safe resume. A source edit during the run changed the repository digest and the campaign correctly halted with `trusted project source differs from campaign creation`. The edit was reverted, the campaign resumed under its original source identity with budget and deadline accounting preserved, and the edit was reapplied afterwards. The campaign's own internal counter recorded 0; we report 1, because a human acted. |
| Live autonomous campaign | PENDING | |

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

## 6. What is not yet demonstrated

Stated plainly, because the gap matters more than the architecture:

1. **No live autonomous campaign has completed.** Every finalized run records
   `provider = scripted`, `API calls = 0`. The typed research seam is implemented, tested against
   a fake transport, and exercised by integration tests, but has not driven a scored campaign.
2. **No improvement over the baseline has been demonstrated.** The best completed run reproduces
   the baseline to within noise (§3.1). We have not shown a validation-primary delta above
   ε = 0.002.
3. **Hidden-test performance is unknown and unclaimed.** It is measured once, by the organizers.

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

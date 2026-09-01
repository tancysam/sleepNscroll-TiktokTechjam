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

### 3.1 Scripted full-data campaign — `runs/report-check-01`

Provider-free reference run. Completed end to end on full data.

*Corrected 2026-08-31.* An earlier version of this section cited a run directory
`runs/scripted-full-data-20260828` that is not on this machine, and read the primary below as the
LambdaRank model's own score. Both are fixed here: the campaign survives as `runs/report-check-01`
with byte-identical metrics, and the primary is a **fusion blend**, not the model. See "Honest
reading" below, and §3.3a for the general form of that mistake.

| Item | Value |
|---|---|
| Campaign status | COMPLETED |
| Selected candidate | `generated-causal-lambdarank-v1` |
| Selection status | `validation_improved` |
| Validation GAUC | 0.6671879292 |
| Validation nDCG@5 | 0.5359807014 |
| **Validation primary (25/75 BLEND, not the model)** | **0.6015843153** |
| **LambdaRank standalone, Fold B (100/0)** | **0.5710743666** |
| **Fold B official-FM control (0/100)** | **0.5754240304** |
| **Standalone minus control** | **−0.0043496639 = −5.44σ** |
| Frozen fusion weights | `[0.25, 0.75]` model / control |
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

**Honest reading of this result.** The 0.6015843153 above is the score of a **blend**, 25% this
model and 75% the official FM control, on a weight the Fold B screen selected. It is not a
measurement of the LambdaRank model, and the earlier reading of it as "+0.000144 over the
confirmation-seed mean" was the §3.3a mistake applied one section early.

Read at the 100/0 point, the model scored **0.5710743666** against the Fold B control's
**0.5754240304** — a deficit of **0.00435, or −5.44σ** at σ = 0.0008. That places the scripted
LambdaRank in the same band as the weakest generated candidates in §3.3a, not near the baseline.
Reproduce the whole grid with `python3 fusion_audit.py report-check-01`:

| Fold B weights (model / control) | primary |
|---|---:|
| 1.00 / 0.00 — **the model alone** | **0.5710743666** |
| 0.75 / 0.25 | 0.5715741813 |
| 0.50 / 0.50 | 0.5742886513 |
| **0.25 / 0.75 — selected** | **0.5756440163** |
| 0.00 / 1.00 — the control alone | 0.5754240304 |

The blend at the selected weight was also measured on public validation across the three matched
seeds, against each seed's own FM member:

| seed | 25/75 blend | its FM member alone | delta |
|---|---:|---:|---:|
| 0 | 0.6015843 | 0.6014695 | +0.000115 |
| 1 | 0.6015052 | 0.6017609 | **−0.000256** |
| 2 | 0.6013579 | 0.6010903 | +0.000268 |

Mean **+0.000042**, and negative on one seed of three — inside noise, and two orders of magnitude
below ε = 0.002. **We do not claim this as an improvement**, and we no longer claim it reproduces
the baseline either: the model is well below the baseline and the blend is indistinguishable from
it. Its value is that it proves the entire pipeline — acquisition, qualification, iteration,
promotion, finalization, replay, and an organizer-valid CSV — works end to end without human
intervention.

This also closes a direction that looked open: a tree-plus-FM rank blend combines two genuinely
decorrelated families (38 causal aggregate columns and gradient-boosted trees against five identity
fields and FM embeddings) and is the standard way to gain on ranking tasks, but on this benchmark it
has now been measured and is worth **+0.000042**.

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

Nine live-provider campaigns ran on 2026-08-30 and 2026-08-31, all against `openai/gpt-5.6-sol`
with a configured failover slot, each under the frozen rule (epsilon 0.002, patience 3) and each
with **zero manual interventions**. Seven reached a terminal `COMPLETED` state; runs 13 and 15
converged and then stranded in finalization on the latent defects described below.

| Run | Iterations | Candidates that executed | Reached outer validation | Selected | Terminal state | Cost |
|---|---:|---:|---:|---|---|---:|
| `maki-overnight-09` | 3 | **3** | 2 | `official-fm-fallback-seed-4` | converged | $0.68 |
| `maki-overnight-10` | 3 | 1 | 0 | `official-fm-fallback-seed-4` | converged | $1.82 |
| `maki-overnight-11` | 3 | 0 | 0 | `official-fm-fallback-seed-4` | converged | $2.31 |
| `maki-overnight-12` | 3 | 0 | 0 | `official-fm-fallback-seed-4` | converged | $1.93 |
| `maki-overnight-13` | 3 | **3** | 3 | `candidate-01` (`promoted_unconfirmed`) | converged, **no bundle** | $2.21 |
| `maki-overnight-14` | 3 | **3** | 3 | `official-fm-fallback-seed-4` | converged | $1.69 |
| `maki-overnight-15` | 3 | **3** | 3 | `candidate-01` (`validation_improved`) | **stranded `FINALIZING`** | $1.44 |
| `maki-overnight-16` | 3 | **3** | 0 | `official-fm-fallback-seed-4` | converged | $1.52 |
| `maki-overnight-17` | 3 | **3** | 0 | `official-fm-fallback-seed-4` | converged | $1.58 |

**Every run in that table stopped at exactly three iterations, and that column is the finding.**
The campaigns held a 21,600-second budget and a 50-iteration cap; runs 16 and 17 spent 714.9 s and
620.3 s of candidate wall time — about **4% of the budget** — and used 13 of 50 launches. The cause
was in our own convergence arithmetic, not in the agent or the provider. `update_after_iteration`
accepted `None` for an iteration that produced no eligible outer primary and still incremented the
non-material streak, so **three consecutive rejections were reported as `converged`**. In runs 16
and 17 no candidate reached outer validation at all, which the "Reached outer validation = 0" column
records: those two convergence claims rest on three rejections and zero measurements.

That is not what the frozen contract describes. `BENCHMARK_CONTRACT.convergence` states the rule as
*"eligible outer primary delta strictly greater than epsilon"*, and a rejected iteration has no
eligible outer primary to take a delta from. **Corrected on 2026-08-31**
(`campaign/convergence.py`): rejections now accumulate in a separate `unmeasured_streak` and stop
the campaign under their own reason, `candidates_not_promotable`, after six. `epsilon = 0.002`,
`patience = 3`, `benchmark_digest` and the meaning of `converged` are all unchanged — `converged`
is now strictly a claim about measured results, which it previously was not. The measured effect on
the loop, from `tests/integration/test_full_campaign_runtime.py`: the autonomous followup driver
went from **3 iterations to 7** on the identical fixture.

**No generated candidate has beaten the baseline by a material margin, and the fallback shipped in
seven of nine runs.** Those seven published a byte-identical submission, SHA-256 `e12746ae…`, so
the submitted artifact never regressed across the series.

**Run 15 is the only campaign that produced an agent-generated submission** (SHA-256
`c98d7cd6…`, `selection.status = validation_improved`, organizer checker rc=0). It is nonetheless
**unusable**, for the same structural reason as run 13 — see §3.3c. Runs 16 and 17 are the only
bundles whose recorded source digest matches the repository as submitted; every earlier bundle
replays only at its own older commit (see the table in `README.md`).

Runs 16 and 17 spent **zero** outer-validation queries: no candidate cleared the inner gate, so the
rationed outer budget was never drawn on. That is the three-tier gate behaving as designed, not a
failure.

Runs 13 and 15 are the only promotions, and neither published a usable bundle. Run 13's candidate
scored 0.6017118 against an incumbent of 0.6014403, a delta of +0.0002715 that does not clear ε, so
the controller recorded it as `promoted_unconfirmed` on its own arithmetic (`selector.py:456`).
Finalization then failed on a ledger-export identity check that had never executed before, and the
bundle is permanently unrecoverable: finalization verifies the working tree still hashes to the
digest recorded at campaign creation, so the only source that could finalize run 13 is the source
containing the bug. Deleting the losing candidates' evidence would satisfy the old check and
fabricate the audit trail. The scores survive in `runs/maki-overnight-13/production/`.

**The same method was run three times, and the margin changed sign**, which is why none of the
three is reported as an improvement:

| Run | Candidate-01 outer mean | Delta vs incumbent | In sigma |
|---|---:|---:|---:|
| `maki-overnight-13` | 0.6017118 | +0.0002715 | +0.34σ |
| `maki-overnight-14` | 0.6014124 | −0.0000278 | −0.03σ |
| `maki-overnight-15` | 0.6017246 | +0.0002844 | +0.36σ |

Mean delta across the three is **+0.00018** against ε = 0.002; with n = 3 the standard error of the
mean is ±0.00046, so it is not distinguishable from zero. Run 13 was above the incumbent on both
inner folds and all three outer seeds, which is exactly how a 0.34σ margin looks before it is
replicated. Had run 13's number been reported as a win, run 14 would have contradicted it four
hours later. See §3.3a: all three figures are fusion blends, so none
is a measurement of the generated model in the first place.

For reference, run 09's outer results on the same matched seeds 0/1/2:

| Model | Outer primary (mean of seeds 0, 1, 2) | Delta vs incumbent |
|---|---:|---:|
| official FM incumbent (`fallback_outer_mean`) | 0.6014403 | — |
| `candidate-01` pairwise softplus | 0.6012030 | −0.0002372 |
| `candidate-03` metric-matched pairwise FM | 0.6011940 | −0.0002462 |

Both sit well inside the baseline's own 0.0008 seed-to-seed standard deviation and are therefore
statistically indistinguishable from it, not improvements and not meaningful regressions. Like
every outer figure in this section they are fusion blends rather than measurements of the generated
model; §3.3a gives the standalone scores.

**Execution rate was the binding constraint, and it is now solved.** Candidate implementations at
or under roughly 260 lines executed 3 for 3 in run 09; of the eight written at over 580 lines in
runs 10 to 12, none executed. Nine post-baseline candidates failed in three classes: within-user
pair-sampling index arithmetic (3), mixing an `(N,)` accumulator with an `(N, rank)` one inside
hand-written factorization-machine interaction maths (3), and a non-scalar checkpoint entry read by
the candidate's own `training_diagnostics` (2). One executed cleanly. All three classes are
recorded per-iteration in each run's `production/candidate-control/*/stderr.log`.

After tested helpers for those three classes were added to the candidate seed (`f66e364`), the
last five campaigns executed **every candidate they produced**: 22/22, 21/21, 22/22, 15/15 and
15/15 subprocess executions, zero failures, empty stderr throughout. Execution reliability is no
longer the limiting factor; §3.3c is.

#### Retracted: the "rank-identical to the official FM" observation

An earlier version of this section reported that generated candidates were independently producing
predictions rank-identical to the official FM within every scored slate, and offered it as a
finding about the task's limited number of distinct orderings. **That was wrong, and the mistake is
worth more than the claim was.**

Candidate predictions are never scored alone. Each is rank-normalised within user and blended with
the official FM control across the fixed five-point `FUSION_WEIGHT_GRID`; the Fold B screen selects
the best-scoring blend and freezes that weight for Fold A and every outer seed. The recorded
primary is the *blend's*. When the selector chooses `(0.0, 1.0)` it discards the generated model
entirely and scores the FM control's own percentile vector, which makes identical metrics a
certainty rather than a coincidence.

The evidence is a digest, not an argument: `scored_prediction_digest` equals the byte-identical
value `41629b9c856e1921…` in all four such occurrences (run 09 ×1, run 10 ×1, run 14 ×2) and never
equals `raw_prediction_logical_digest`. The "59,816 distinct values" check had been run against the
raw generated vector, which is not the vector that was scored.

#### 3.3a What the candidates actually scored

Reading the `(1.0, 0.0)` grid point recovers each candidate's own score. Reproduce with
`python3 fusion_audit.py <run-id>`.

| Run | candidate alone | official FM control alone | gap | in sigma |
|---|---:|---:|---:|---:|
| `maki-overnight-09` | 0.5671 / 0.5682 / 0.5684 | 0.5754240304231644 | −0.0070 to −0.0083 | −8.7σ to −10.4σ |
| `maki-overnight-10` | 0.5654 | 0.5754240304231644 | −0.0100 | −12.6σ |
| `maki-overnight-13` | 0.5713 / 0.5708 / 0.5720 | 0.5754240304231644 | −0.0034 to −0.0046 | −4.3σ to −5.8σ |
| `maki-overnight-14` | 0.5704 / 0.5688 / 0.5703 | 0.5754240304231644 | −0.0050 to −0.0066 | −6.3σ to −8.3σ |
| `maki-overnight-15` | 0.5713 / 0.5713 / 0.5707 | 0.5754240304231644 | −0.0041 to −0.0047 | −5.2σ to −5.9σ |
| **`maki-overnight-16`** | **0.5745312 / 0.5745072** / 0.5630 | 0.5754240304231644 | **−0.00089 / −0.00092** | **−1.12σ / −1.15σ** |
| `maki-overnight-17` | 0.5740230 / 0.5736019 / 0.5681 | 0.5754240304231644 | −0.0014 to −0.0074 | −1.75σ to −9.2σ |

**No generated candidate this project has executed has scored above the official FM control
standalone.** The outer figures of 0.6013 to 0.6017 reported above and in §3.2 are blends weighted
75% to 100% toward the official FM, so they were never measuring our models. Any result in this
document that describes a candidate as "matching" or "indistinguishable from" the baseline should
be read with that in mind.

**Run 16 closed most of the deficit, and the mechanism is the reportable part.** Every candidate up
to that point built a single factorization-machine interaction over *all* feature columns, so 33
standardized continuous aggregates shared a latent space with the identity codes. The organizer FM
crosses only its categorical fields. Restricting the interaction to the identity codes, keeping the
aggregates as additive first-order terms, and training under pointwise logistic loss moved the gap
from −4σ…−12σ to **−1.12σ**. Two further measurements from the same run: the 0.5745 results came
from pointwise logistic loss, and switching the identical scorer to a pairwise objective collapsed
it to 0.5630 — consistent with the twelve-cell sweep in §3.2. **The loss function was not the
lever; the interaction structure was.**

#### 3.3c Three structural limits of the candidate seam, and two falsified hypotheses

`prediction_request_fields` in `candidate_api/runtime_contract.py` carries the feature matrix and
the checkpoint and nothing else. Three consequences follow, and each was measured rather than
inferred:

1. **No user identity at prediction time.** `user_groups` is a training-only capability, so a
   candidate could not evaluate any user-conditioned term while scoring — ruling out the
   user-by-video and user-by-author crosses that supply most of the baseline FM's ordering power.
   Commit `4aa6a25` added `user_id_code` (bundle 37 → 38 columns, user vocabulary 26,211, 1.59% of
   validation rows on the unknown slot). **This hypothesis is falsified:** run 15 stayed at −5.15σ.
   The change was structurally correct and did not close the gap.
2. **No within-user rank normalisation at prediction time**, for the same reason. This is why
   run 17's internal five-member seed ensemble came out flat — 0.5740230 against run 16's
   0.5745312, a 0.6σ difference that is inside noise. `ensemble_mode_probe.py` measures the cause
   on the five qualified official FM seeds: averaging on raw scores is worth **+0.0000772** while
   averaging on within-user rank percentiles is worth **+0.0005664**. **86% of the ensembling gain
   is the rank normalisation**, and a candidate cannot perform it. **Second hypothesis falsified**,
   with the mechanism identified rather than guessed at.
3. **No early stopping on the scored split.** `baselines/starter_fm.py:701-708` keeps the epoch
   scoring best *on the split it is then reported on*, over up to 40 epochs. The published 0.6016
   has the same property (`baseline.py:88`, `if va['primary'] > best`). Our candidate never sees
   that split and gets one shot, so the residual ~1σ is measured against a reference holding an
   advantage the seam denies it.

The agent-side levers are therefore exhausted, and the reason is demonstrated rather than assumed.
None of the three is reachable by changing the briefing.

A fourth cause was informational rather than structural: **the agent was never told fusion
existed.** `prompts.py` did not contain the word, and the primary returned to reflection was the
blend's. Run 14 iteration-03 recorded in its own method attribution that recency weighting
*"successfully measured 0.5754240304, matching official_fm_fold_B"* and excluded the direction on
that basis, when its model had in fact been discarded. Commit `8124607` exposes
`candidate_standalone_primary` and `fold_b_control_primary` in `campaign_records`. The effect is
visible in the next run's reasoning: run 15 iteration-03 instead wrote that its parent *"scores
0.5713044 standalone against the 0.575424 Fold B control"* — a measured before-and-after on the
agent's own reasoning, produced by one fix.

#### The agent found the half of that fix we missed

`8124607` reached `propose` and stopped there. `_reflect` kept building its
`ExperimentResultSummary` from tier, status, GAUC, nDCG@5, primary, runtime and memory alone, and
never called `_fusion_disclosure`. So the agent proposed with the disclosure and then **reflected
without it** — on the blend, with no way to know it was a blend.

It noticed, unprompted, in **six reflections out of six** across runs 16 and 17. Run 16
iteration-01:

> "The request does not provide candidate_standalone_primary or the frozen model/control fusion
> weight. Therefore, this result cannot establish that the candidate itself beat the official FM
> control; the small scored advantage may belong to a selected blend."

Each time it refused the causal claim the missing field would have licensed, and in run 17
iteration-01 it recovered the answer by inference instead:

> "The result exactly reproduces official_fm_fold_A at full supplied precision … this indicates
> that the frozen control-only blend was selected and the candidate contributed no ranking signal."

**This is the one capability gap in this project that the agent identified rather than being told.**
The three framings that shaped runs 16 and 17 — no user identity at prediction time, no within-user
rank normalisation, seed ensembling — were all supplied in the briefing, and the run-16 versus
run-17 system-prompt diff makes that checkable. The reflect-payload gap was not.

Fixed on 2026-08-31: `ExperimentResultSummary` carries `candidate_standalone_primary`,
`fold_b_control_primary`, `fusion_weights_selected` and `fusion_note`, `_reflect` populates them
from the same `_fusion_disclosure` helper the propose path already used, and the reflect prompt
explains how to read them. The defect and its fix are both worth more than the direction they were
found in: a disclosure applied to one of two paths is invisible until something on the other path
complains, and here the complainant was the model.

#### A confounder in the run series: run 11 failed over mid-campaign

Reconstructed from every `production/provider-attempt-journal/*.json` across all seventeen runs.
The model actually called, per run:

| Runs | Model | Calls |
|---|---|---:|
| 01–07 | `deepseek/deepseek-v4-pro-0813` | 53 |
| 09, 10, 12–17 | `openai/gpt-5.6-sol` | 77 |
| **11** | **`openai/gpt-5.6-sol` ×9 and `openai/gpt-5.6-terra` ×2** | 11 |

141 attempts in total, matching the accounting in §4. **Run 11 is the only campaign that crossed the
failover boundary**, and the provider chain is sticky by design, so two of its operations were
served by a different model than the other nine. Run 11 is also one of the campaigns in which
**zero of three candidates executed**.

We are not claiming the failover caused that; run 12 executed zero of three without any failover,
and §3.3a identifies implementation size as the variable that actually tracked execution. But a
mid-campaign model switch is a silent change of experimental conditions, and run 11 should be read
as a confounded observation rather than a clean one. It is recorded here because it was found by
sweeping every journal in the series rather than the run under investigation, and because nothing
in the campaign record flags it at the point of use.

#### Three latent defects on the promotion path, each found only when the agent succeeded

The promotion path — what happens when a generated candidate actually wins — went unexecuted for
twelve campaigns, because every earlier run either shipped the fallback or failed before reaching
it. Each time it did run, it exposed a defect that had been latent since it was written.

1. **Run 13, ledger export.** Attributed every retained run to the *selected* candidate's source
   and config digest; run 13 held seven runs across three distinct source digests. Fixed in
   `e758693`. The bundle is permanently unrecoverable: finalization verifies the working tree still
   hashes to the digest recorded at campaign creation, so the only source that could finalize run
   13 is the source containing the bug.
2. **Run 15, float32 versus float64 primary.** `_derive_bundle_status` compared the declared
   representative primary against the matched-seed row using `abs_tol=1e-15` — exact float
   equality. The two values were `0.6016490459442139` and `0.6016490757465363`, **2.98e-8 apart, a
   single float32 ulp**. GAUC and nDCG@5 matched bit for bit; only the derived mean differed,
   because the organizer evaluator is float32-sensitive
   (`scoring/protected.py`, `score_with_encoded_labels`) while the reconstruction recomputes in
   float64. The codebase already absorbed this artifact 160 lines earlier — `_manifest_metrics`
   checks the same property with `abs_tol=2e-7` — so two checks on one quantity disagreed about
   tolerance and the stricter one had never been reached. Fixed in `8978b4b` with a named tolerance
   applied at that one site and to `primary` only; GAUC and nDCG@5 still compare exactly, which is
   what proves the difference is confined to the derived mean.
3. **Run 15's bundle is unusable at every commit.** `replay_final_bundle` calls
   `_verify_closed_bundle` *before* the current-source check. At HEAD the bundle check now passes
   and the source check fails; at run 15's own commit `8124607` the float32 defect is still present
   and the bundle check fails. Both doors are shut, exactly as for run 13.

The pattern is the reportable part: **a code path that executes only on success accumulates
defects invisibly.** Ours produced three, each surfacing the first time its branch ran, over three
separate campaigns. Every one was found by reading the whole run rather than the failing traceback.

#### 3.3b Seed-ensemble headroom in the baseline itself

Measured offline from artifacts already in `runs/maki-qualification`; no training, no campaign, no
provider calls. Reproduce with `python3 seed_ensemble_probe.py`.

| Model | Public validation primary | vs organizer 0.6016 |
|---|---:|---:|
| FM seed 0 / 1 / 2 / 3 / 4 | 0.6014695 / 0.6017609 / 0.6010903 / 0.6015031 / 0.6020371 | — |
| mean of the five single-seed scores | 0.6015722 | −0.0000278 |
| five-seed **raw score** mean | 0.6021143 | +0.0005143 |
| **five-seed within-user rank ensemble** | **0.6026034355** | **+0.0010034** |

The ensemble beats every individual seed including the luckiest, and beats the organizer baseline
by +0.0010. It uses all five seeds, so unlike a best-of-N result it carries no selection effect,
and the gain is a systematic variance reduction rather than a favourable draw.

The raw-score row is not filler: it is the operation a *candidate* is limited to (§3.3c), and the
gap between the two rows — **+0.0004891, or 0.61σ** — is the part of the gain that requires
within-user rank normalisation. Reproduce with `python3 ensemble_mode_probe.py`.

**This is a shipped artifact, not only a measurement.** `build_ensemble_submission.py` regenerates
it end to end in roughly twelve minutes:

```sh
python3 build_ensemble_submission.py            # writes ensemble-submission/{submission.csv,provenance.json}
cd kuairand-starter-kit && python submit.py ../ensemble-submission/submission.csv \
    --data_dir ../.data/KuaiRand-Pure/data --split test --check      # returns 0
```

It performs **no new training**. Every member is an already-qualified organizer FM run from
`runs/maki-qualification`; each checkpoint is verified against the digest recorded at qualification,
the shared encoding is verified across all five rather than assumed, and inference runs through the
hash-pinned organizer source rather than a reimplementation. The validation primary reproduced
**exactly** through two independent code paths — the saved `validation-predictions.npy` files and
the pinned organizer source. Organizer checker: `rc=0`, 170,588 rows.

Two caveats stated plainly, and repeated in `ensemble-submission/provenance.json` so they travel
with the artifact. It is **below our own materiality threshold of ε = 0.002**, so it is not a
material improvement. And it is a **controller-side ensemble of the organizers' own baseline with
itself — not an agent-generated result.**

Related: the shipped `official-fm-fallback-seed-4` was chosen as `best.seed` over the five
(`qualification.py:1080`), so its 0.6020371 is a best-of-five estimate selected on the split it is
reported on. That is the same selection effect this document criticises in §3.2, and it should be
read as such rather than as a clean margin over the baseline.

A replay recipe and backend for shipping this ensemble through finalization
(`OfficialFMSeedEnsembleReplayRecipe`, `OfficialFMSeedEnsembleReplayBackend`) are wired into the
parser and the backend factory and covered by tests, but **no finalization path produces one yet.**
Completing that route requires teaching the bundle status check about ensemble provenance without
weakening its forgery resistance: the baseline branch of `_derive_bundle_status` pins the summary
to exactly `{"seeds": [4], "representative_seed": 4, …}`. Declaring `seeds: [4]` for a five-model
ensemble would have been the cheap way through and would have fabricated the audit trail.

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

### 3.5 The saturation campaign — runs 18 to 24

Seven further campaigns were run on 2026-08-31 and 2026-09-01, together with two offline probes.
They did not produce a material improvement. They did produce two instrument defects, a falsified
hypothesis of our own, and two measurements that close the search, and those are the reportable
result.

| Run | Stop reason | Scored | Best standalone | vs control | Tokens |
|---|---|---:|---:|---:|---:|
| 18 | `candidates_not_promotable` | 5 | 0.5683315 | −8.87σ | 562,520 |
| 19 | aborted, operator `__pycache__` | 0 | — | — | 53,853 |
| 20 | provider unavailable (402, then malformed) | 0 | — | — | 95,424 |
| 21 | `candidates_not_promotable` | 0 | — | — | 487,680 |
| 22 | `candidates_not_promotable` | 6 | 0.5734960 | −2.41σ | 502,068 |
| 23 | `candidates_not_promotable` | 7 | 0.5735505 | −2.34σ | 508,183 |

The Fold B control throughout is **0.5754240304**. **Run 16's 0.5745312 (−1.12σ) was never beaten**,
by any candidate, in any of these runs.

#### 3.5a Two instrument defects, one of which the agent found itself

Both have the same shape: the controller held information the model needed and dropped it.

**The reflect path never received the fusion disclosure.** Commit `8124607` had exposed
`candidate_standalone_primary` on the *propose* path; `_reflect` still built its
`ExperimentResultSummary` from tier, status, GAUC, nDCG@5, primary, runtime and memory alone, and
never called `_fusion_disclosure`. The agent detected this **unprompted in six reflections out of
six** across runs 16 and 17:

> "The request does not provide candidate_standalone_primary or the frozen model/control fusion
> weight. Therefore, this result cannot establish that the candidate itself beat the official FM
> control; the small scored advantage may belong to a selected blend."

It then reconstructed the missing value by exact-matching against `official_fm_fold_{A,B}`. This is
the one capability gap in this project that the agent identified rather than being told; the three
framings that shaped runs 16 and 17 were all supplied in the briefing, which the run-16 versus
run-17 system-prompt diff makes checkable. Fixed in `4de2a38`; run 18 then read its own standalone
score correctly in all six reflections.

**The repair path never received the exception message.** `_invoke_runner` in `campaign/scientific.py`
caught the failure and returned `type(exc).__name__`, discarding `str(exc)` — while the executor had
already built a bounded 4096-character diagnostic and stored the child's stderr as an artifact. A
model asked to repair a crash was told `callback_failed:CandidateExecutionError` and nothing else.
**Run 21 spent all six of its iterations on this**, proposing the right direction every time and
crashing every time, guessing at "ragged final batches" because that phrase was in the briefing
rather than because it had seen any evidence. Fixed in `c7aede7`; run 22 then produced six scored
candidates instead of zero.

#### 3.5b Convergence was being declared on three rejections

`ConvergenceState.update_after_iteration` accepted `None` for an iteration that produced no eligible
outer primary and still incremented `non_material_streak`. With patience 3, **three consecutive
rejections reported `stop_reason = "converged"`**. Runs 09 to 17 all stopped at exactly three
iterations; runs 16 and 17 did so having spent about 4% of a six-hour budget with **no candidate
reaching outer validation at all**.

`BENCHMARK_CONTRACT.convergence` states the rule as *"eligible outer primary delta strictly greater
than epsilon"*, and a rejection has no eligible outer primary to take a delta from. Corrected in
`4de2a38`: rejections accumulate in a separate `unmeasured_streak` and stop the campaign under a new
and truthful reason, `candidates_not_promotable`. `epsilon`, `patience` and `benchmark_digest` are
unchanged, and `converged` is now strictly a claim about measured results, which it previously was
not. Measured effect: the autonomous followup driver went from **3 iterations to 7** on an identical
fixture.

#### 3.5c A hypothesis of ours, falsified twice — and a correction

We observed that the worked example in `research/prompts.py` demonstrates about **20 full-batch
fixed-step updates** while the control uses mini-batch Adam at `bs=8192` over 40 epochs — about
**5,570 updates** — and inferred that a 278-fold optimisation deficit explained the residual gap.

Run 22 implemented the control's optimiser exactly, with its published constants (`k=16`,
`lr=0.001`, `l2=1e-6`, betas 0.9/0.999, eps 1e-8). Its best standalone was **0.5734960**, *worse*
than run 16's 0.5745312. Runs 23 and 24 repeated the same settings and scored 0.5735505 and
0.5628278. **More optimisation was not the lever.**

**The premise was also wrong, and the correction matters more than the hypothesis.** Run 16's
winning candidate did not use a full-batch loop at all. Its recorded configuration is

```json
{"batch_size": 65536, "epochs": 10, "learning_rate": 0.004,
 "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
 "l2_bias": 0.00002, "l2_factor": 0.00002, "l2_linear": 0.002,
 "logit_clip": 35.0, "rank": 16, "rarity_cap": 25.0}
```

— mini-batch Adam at roughly **174 updates**, in neither of the two regimes we were comparing. The
278-fold figure described the worked example, not the result it was used to explain. Runs 22 to 24
were therefore directed *away* from the best known configuration on the strength of a
mischaracterisation, which is the single most expensive analytical error in this project.

#### 3.5c-bis The training regime, measured across all 39 scored candidates

Extracting every scored candidate's configuration alongside its standalone Fold B primary and
sorting by score reveals a pattern that is invisible inside any single campaign:

| regime | batch | epochs | learning rate | standalone range |
|---|---:|---:|---:|---|
| **large batch, few epochs** | 32,768–65,536 | 4–10 | 0.004–0.08 | **0.5736 – 0.5745** |
| full batch | all rows | 10–64 | 0.01–0.15 | 0.5628 – 0.5720 |
| **small batch, many epochs** | 8,192 | 40 | 0.001 | **0.5507 – 0.5736** |

The four best candidates the project has produced:

| standalone | run | batch | epochs | lr | rank | logit clip |
|---|---|---:|---:|---:|---:|---:|
| **0.5745312** | 16 | 65,536 | 10 | 0.004 | 16 | 35.0 |
| 0.5745072 | 16 | 32,768 | 4 | 0.01 | **8** | 40.0 |
| 0.5740230 | 17 | 65,536 | 4 | 0.01 | **8** | 35.0 |
| 0.5736019 | 17 | 32,768 | 4 | 0.08 | 16 | 40.0 |

Every one is large-batch and few-epoch at a learning rate four to eighty times the control's, and
every one of the **ten worst** results used batch 8,192 with lr 0.001 over 40 epochs. That setting is
also unstable: near-identical configurations produced both 0.5735 and 0.5507.

Two secondary findings fall out of the same table. **Rank 8 scores as well as rank 16** — two of the
top three used it — independently reproducing the organizers' capacity ablation from our own runs.
And `ema_decay` and `grad_clip` appear **only** among low scorers.

#### 3.5d The enhancement ledger — every addition cost score

Read chronologically, run 22 and its predecessors give one direction:

| configuration | standalone |
|---|---:|
| identity-code FM + additive aggregates, full-batch, plain (run 16) | **0.5745312** |
| the same under mini-batch Adam (run 22) | 0.5734960 |
| + seed ensembling inside the candidate | 0.5740230 |
| + frequency-aware identity regularisation | 0.5680654 |
| + a pairwise objective on the identical scorer | 0.5629801 |
| + metric-matched per-row weighting by user positive count | 0.5574875 |
| + an exponential moving average used for inference | 0.5554938 |
| official FM control | 0.5754240304 |

Seven independent additions, seven regressions. The two best results in the project are its two
simplest configurations.

#### 3.5e The positive control, which inverted its own hypothesis

Run 23 asked whether the 33 causal aggregate columns were diluting the identity signal, by building
the control's field set and nothing else. Reproduce with `fusion_audit.py maki-overnight-23`:

| model | standalone |
|---|---:|
| control, identity fields only | 0.5754240304 |
| **candidate seam, identity codes only** | **0.5535416 (−27.35σ)** |
| candidate seam, identity codes **+ aggregates** (run 16) | 0.5745312 (−1.12σ) |

**The aggregates are load-bearing, not diluting.** Removing them costs 26σ. The dilution hypothesis
is falsified, and with it the planned follow-up that would have added the columns back in groups.
The predictions were checked for degeneracy and are not degenerate — 59,500 of 61,315 values
distinct — so these are trained models that are simply worse.

#### 3.5f Seed ensembling saturates at five members

The shipped submission is a five-seed within-user rank ensemble at 0.6026034355. `StarterFMConfig`
freezes `k`, `lr`, `l2` and `batch_size` but leaves `seed` settable, so the pool was extended to 35
members by training additional seeds **through the hash-pinned organizer source itself**
(`seed_pool_probe.py`, which reproduces seed 0 **bit-identically** against the qualified vector
before training anything). Measured with `ensemble_search_probe.py`:

| N | validation primary | vs the shipped five |
|---:|---:|---:|
| 5 | 0.6026034355 | — |
| 10 | 0.6027070284 | +0.0001036 |
| 16 | 0.6030003428 | +0.0003969 |
| 20 | 0.6028070450 | +0.0002036 |
| **35** | **0.6026725769** | **+0.0000691** |

Seven times the pool is worth **+0.00007**: noise. The whole curve lies between 0.6025 and 0.6030
with no trend, against a materiality threshold of 0.6036.

Two things must be said about this number. First, an earlier fit of the standard variance-reduction
form gain(N) = G(1 − 1/√N) to the single five-seed point predicted ~0.6030 at N=20 and an asymptote
near 0.60344; **the measured curve does not climb at all, so that fit was wrong** and is retracted.
Second, N=16 is the curve maximum and **is not claimed**: selecting it after reading the validation
column would be exactly the best-of-N selection effect this document criticises in §3.2 and in the
seed-4 fallback. The probe fixes N a priori at the largest available pool and prints the curve
rather than a menu.

The mechanism is visible in the design: every seed early-stops on validation, so all of them are
peak-picked toward the same optimum and their errors are correlated rather than independent. The
gain from one seed to five was real and is fully exhausted.

#### 3.5g What these runs establish

Across ten campaigns and roughly thirty-five scored candidates, **no generated candidate has ever
exceeded 0.5745312 standalone**, and the residual 0.0008928 to the control is close to one
seed-to-seed sigma. It is also close in size to one advantage the control holds and the candidate
seam denies: the control keeps the best of 40 epochs *measured on the split it is then scored on*
(`baselines/starter_fm.py:701-708`), while a candidate gets a single shot. Every route tested is now
closed by measurement rather than by argument: features (organizers' own ablation), capacity
(organizers' own ablation), loss, optimiser budget, regularisation, weighting, parameter averaging,
candidate-side ensembling, controller-side ensembling, the tree-plus-FM blend, and the removal of
the aggregates.

We report this as a saturation result, not as a list of failures. The negative space is mapped, each
boundary carries a number, and every one is reproducible from artifacts in this repository.

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
| `maki-overnight-13` | 12 | 195,915 | 75,419 | 271,334 | $2.21 |
| `maki-overnight-14` | 10 | 142,624 | 59,611 | 202,235 | $1.69 |
| `maki-overnight-15` | 9 | 132,596 | 49,480 | 182,076 | $1.44 |
| `maki-overnight-16` | 9 | 129,277 | 54,140 | 183,417 | $1.52 |
| `maki-overnight-17` | 9 | 135,948 | 56,565 | 192,513 | $1.58 |
| **Total** | **141** | **1,823,981** | **1,172,893** | **2,996,874** | **$29.27** |

Of the 1,172,893 output tokens, **854,988 are reasoning tokens** — 73% of all output. That is the
price of leaving reasoning enabled, which is not optional here: with thinking disabled the model
stopped writing code and returned 18-character stubs, with `import numpy as np` written into
`config.json`. It is also why `max_output_tokens` is 65536 and the request timeout is 600 seconds.

Regenerate this table with `python3 token_accounting.py`, which recomputes every figure from the
journals. Two counting notes so the two reconcile: the **Calls** column counts every attempt,
while the tool counts only the 130 that returned a usage block — 11 attempts errored before
returning one, and contribute zero tokens either way. And the tool deliberately does not compute
spend: the cost column comes from the frozen pricing block in `configs/`, not from the provider's
self-reported upstream cost, so it stays reproducible rather than dependent on a rate card that
can change. Per-run spend is read from each bundle's `report.md`
(`estimated API cost USD=`); the reconstruction reproduces the previously published $20.83 through
run 12 and $24.73 through run 14 exactly.

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
| `runs/report-check-01` (the scripted full-data campaign) | 1 | One safe resume. A source edit during the run changed the repository digest and the campaign correctly halted with `trusted project source differs from campaign creation`. The edit was reverted, the campaign resumed under its original source identity with budget and deadline accounting preserved, and the edit was reapplied afterwards. The campaign's own internal counter recorded 0; we report 1, because a human acted. |
| `maki-overnight-09` | 0 | Converged with no human action after launch. |
| `maki-overnight-10` | 0 | Converged with no human action after launch. |
| `maki-overnight-11` | 0 | Converged with no human action after launch. Its provider chain failed over mid-campaign, which is an automatic recovery rather than an intervention; see §3.3. |
| `maki-overnight-12` | 0 | Converged with no human action after launch. |
| `maki-overnight-13` | 0 | Converged with no human action after launch. Finalization then failed on a latent defect, which is a software fault and not a human action; no human altered the campaign. |
| `maki-overnight-14` | 0 | Converged with no human action after launch. |
| `maki-overnight-15` | 0 | 22/22 executions, promoted a candidate, then stranded at `FINALIZING` on the float32 defect below. Again a software fault, not an intervention. |
| `maki-overnight-16` | 0 | 15/15 executions, `COMPLETED`, empty stderr throughout. |
| `maki-overnight-17` | 0 | 15/15 executions, `COMPLETED`, empty stderr throughout. |

**Zero manual interventions across all seventeen campaigns**, including the six that were killed
mid-flight before the survivability work landed. Runs 01, 04, 05, 07 and 08 still read `RUNNING` in
their campaign databases: that field is written at launch and only overwritten at a terminal state,
so a killed process leaves it stale. No process survives; `pgrep` finds none. Those directories are
retained because their provider journals are the source of roughly a third of the token accounting
above.

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

1. **No agent-generated improvement over the baseline has been demonstrated.** The best generated
   candidate scored alone reaches 0.5745312 against a control of 0.5754240, still −1.12σ. Its best
   fused outer figure, 0.6017246 against an incumbent of 0.6014403, is +0.36σ and was replicated to
   +0.34σ, −0.03σ, +0.36σ across three runs — mean +0.00018 against ε = 0.002. The pairwise
   direction was additionally swept to convergence over twelve configurations without beating the
   baseline in any of them (§3.2).
2. **The +0.0010 submission is not an agent result.** It is a controller-side five-seed ensemble of
   the organizers' own FM with itself (§3.3b). It is real, reproducible and organizer-validated,
   and it is still below ε = 0.002. It must never be described as the agent beating the baseline.
3. **The candidate seam capped what was achievable, in three measured ways** (§3.3c): no user
   identity and no `user_groups` at prediction time, and no early stopping on the scored split
   which the baseline itself receives. Two of our own hypotheses for closing the gap were falsified
   by measurement rather than abandoned.
4. **A crash inside `train_model` still ends a branch outright.** The repair loop covers
   pre-execution static gates only, so runtime defects get no fix attempt. With execution now at
   3-of-3 for five consecutive campaigns this is no longer the binding constraint, but it remains
   the clearest structural gap in the loop.
5. **Hidden-test performance is unknown and unclaimed.** It is measured once, by the organizers.

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

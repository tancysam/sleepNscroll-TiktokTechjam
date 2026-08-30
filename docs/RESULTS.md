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

### 3.3 Live autonomous campaigns

Twelve live-provider campaigns were run against `openai/gpt-5.6-sol` via OpenRouter. **Nine
completed end to end and emitted a full organizer-valid bundle**; three crashed on defects that
were root-caused and fixed (§3.5). Every completed campaign recorded **zero manual interventions**.

Reference campaign — `runs/postfix-20260830T105450Z` (`kuairand-1df0ab4d8d381a0535be`), run after
the first three defect fixes:

| Item | Value |
|---|---|
| Campaign status | `COMPLETED`, full bundle published, `failures = []` |
| Selected candidate | `official-fm-fallback-seed-4` (protected baseline fallback) |
| Selected status | `baseline_reproduced` |
| Validation GAUC / nDCG@5 / primary | 0.6679478 / 0.5361264 / 0.6020371 |
| Absolute delta vs official five-seed FM mean | **+0.0004649** primary |
| Terminal reason | `exact_terminal_condition_reached` (organizer convergence rule) |
| Branches attempted / admitted / trained | 3 / 3 / 3 |
| Inner evaluations / outer evaluations | 2 / 0 |
| Repairs / pre-execution rejections / fallbacks | 0 / 0 / 0 |
| Manual interventions | **0** |
| Campaign wall time | 385.9 s (finalization 25.5 s) |
| Charged training launches | 11 |
| Tokens (input / output / total) | 94,936 / 16,667 / **111,603** |
| Estimated API cost | **$0.65** |

The +0.0004649 delta is against this repository's official-FM confirmation evidence and is far
below the organizer convergence threshold ε = 0.002. **We do not claim it as an improvement.** It
is the fallback reproducing the baseline, which is what `baseline_reproduced` means.

### 3.4 The one candidate promotion we have observed

`runs/hard-block-verify-20260830T103424Z` is the only run in which a generated candidate cleared
every gate the pipeline has. `candidate-01` (a within-user BPR pairwise objective over the fixed
33-feature causal bundle) beat its parent on both inner folds, passed outer matched-seed
validation, and was promoted to incumbent:

| | candidate-01 | parent | delta |
|---|---|---|---|
| Fold A primary | 0.6076452 | 0.6071290 | +0.0005161 |
| Fold B primary | 0.5755265 | 0.5754240 | +0.0001025 |

Both deltas are an order of magnitude below ε = 0.002, so this is **not** a convergence-beating
improvement and is not claimed as one. It is reported because it is the only evidence we have that
the generated-candidate path can clear the full gate chain, and because the campaign then crashed
in finalization (§3.5, defect 3) and produced no bundle — the result was lost to our own defect
rather than to the science.

### 3.5 Defects found by running live, and fixed

Robustness is judged on how the agent handles difficulty, not on whether it meets any. All five
of these were found by real live runs, root-caused from the actual failure, and fixed with a
regression test that was confirmed to fail without the fix.

| # | Defect | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | Lineage CHECK-constraint violation | `sqlite3.IntegrityError` killed the campaign on iteration 2 | `promoted` was written as a real boolean even for screen-rejected candidates that had no Fold A metrics, violating the ledger's all-or-nothing invariant | `promoted` made genuinely optional; only computed when both folds are present |
| 2 | Screen-rejected evidence discarded | Ledger silently dropped the most common outcome | The schema required both folds present to record anything | Fold A and Fold B groups made independently all-or-nothing |
| 3 | Finalization rejected multi-candidate campaigns | Campaign promoted a candidate, then died with `scientific record source, config, or environment identity changed`, emitting no bundle | The exporter required *every* scientific record to carry the *selected* candidate's source/config/snapshot triple, but those are per-candidate identities — the crashed run held three distinct triples across seven records | Only `environment_digest` is campaign-wide and checked globally; the triple is pinned to records belonging to the selection, plus any record claiming its snapshot so a forged mix is still rejected |
| 4 | Repeated circuit-breaker refusals collapsed into one line | Campaign died emitting its report with `failures_and_recoveries entries must be unique`, losing the run | The proposal-family circuit breaker refuses the same family with the same stage, category, code and diagnostic every time, so blocked proposals differed only by iteration and candidate — both validated, then discarded by the rendered line | Render the iteration and candidate that were already being validated |
| 5 | Promotion permanently blocked by other campaigns' history | The best candidate the project has produced -- +0.000578 Fold A, +0.000161 Fold B, positive on both -- was discarded as `inner_rejected / outer_candidate_limit` while its own campaign had spent zero of its six outer queries | `DurableScientificLedgerAdapter.snapshot` passed *every* campaign's candidate fingerprints to the selector, which compares that set against `outer_promotion_limit`. Six historical queries therefore exhausted the allowance of every future campaign | Scope the fingerprint set to the campaign, matching the ration the ledger enforces on reservation. The project log is untouched |

Defect 5 requires a correction to how the rest of this document should be read. The project-wide
count reached 6 at 11:59:17 on 2026-08-30, and from that moment **no candidate could be promoted
by any campaign, regardless of its score**; before then each campaign inherited the accumulated
count and so ran with a progressively smaller allowance. Statements elsewhere here that no
generated candidate beat the baseline are therefore measurements taken through a closing gate, not
clean evidence about the science. The one promotion we did observe (SS3.4) occurred while only four
fingerprints existed. We have not yet run a campaign end to end with this defect fixed.

Defect 3 is the most consequential: it fired *precisely when the campaign succeeded*, and
`proposal_breadth = 2` made multi-candidate campaigns the normal case. The fix is verified end to
end — `runs/postfix-20260830T105450Z` wrote two scientific records with two distinct source
snapshots and two distinct source digests (the exact condition that crashed the previous run) and
finalized cleanly.

A fourth event was **not** a defect: `resume` correctly refused to continue a campaign whose
trusted source tree had changed since creation. That is the campaign-identity integrity guarantee
working as designed, and the run was abandoned rather than bypassed.

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
| Live campaigns that emitted a bundle (9) | | | **1,081,411** | $6.3349 |
| Live campaigns that crashed before finalization (3) | | | **553,055** | ~$3.24 (est.) |
| **Total live spend** | | | **1,634,466** | **~$9.57** |

Per completed campaign the mean is 120,157 tokens and $0.70. The reference campaign
(`runs/postfix-20260830T105450Z`) used 111,603 tokens for $0.65. One outlier dominates the crashed
subtotal: `runs/cb-b-112152Z` spent 349,550 tokens because the circuit breaker refused 14
successive `pairwise` proposals and the model kept re-proposing the blocked family (§3.6).

The two crashed campaigns produced no final report, so their figures are summed directly from the
per-attempt provider journals under `runs/<run-id>/production/provider-attempt-journal/`; their
token counts are exact, but the cost is an estimate derived at the same blended rate rather than a
figure the adapter recorded, and is labelled as such.

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
| 9 completed live campaigns | **0 each** | No human acted between launch and bundle in any of them. Each campaign's own report independently records `Manual intervention count: 0`. |
| 5-campaign auto-retry batch | **0** | `scripts/auto_retry_campaigns.sh` launched five consecutive independent campaigns unattended. The script only launches and stops; it cannot alter a campaign's search, and the frozen organizer ε/N are enforced at config-parse time and are not reachable from it. |
| 3 crashed live campaigns | n/a | Both were abandoned rather than hand-repaired mid-run. The defects were fixed in source and a fresh campaign launched, so no intervention altered a running campaign. |

## 4.1 Correction: the outer-validation ration is per campaign

**Disclosed because it changes a number we previously reported, and because it loosens a limit we
had been enforcing on ourselves.**

`OuterQueryLedger.reserve` counted every reservation ever made, across all campaigns, against a
single allowance of six. Fourteen live campaigns therefore shared one allocation, and it reached
zero on 2026-08-30 at 11:59:17 (`runs/ctx-1-115456Z`). At that point no candidate could ever be
outer-confirmed again: every future campaign would terminate at `baseline_reproduced` regardless of
quality.

That is not what this project's own design specifies. [`plan.md`](../plan.md) 12.2 states: *"At most
six distinct scientific candidates **per campaign** receive a public-validation score,"* and the
very next line explains that the log is project-wide for a different reason — *"so restarting or
changing a campaign fingerprint does not erase the development history."* The ration is
per-campaign; the log is project-wide. The implementation collapsed the two.

Nor is the limit an organizer rule. The problem statement rations exactly one thing — *"hidden test
scored once, on the final submission"* — and explicitly names public-validation feedback as the
intended development signal.

**We are not presenting this as an accidental drift.** A test
(`test_project_six_slot_limit_survives_distinct_campaign_databases`) had deliberately locked in the
lifetime behaviour, so this reverses a considered earlier decision, made when the stricter reading
seemed the safer default.

What changed, precisely: the reservation count is now scoped by `campaign_id`. What did **not**
change:

- The append-only project log still records **every** query ever made, across all campaigns. No
  history is erased and none of the six already-spent slots is reclaimed or reused.
- Reopening the *same* campaign against a fresh local database still cannot reissue its ration —
  the project ledger remembers what that `campaign_id` already spent. This anti-gaming property is
  covered by `test_six_slot_limit_is_per_campaign_and_not_reset_by_a_fresh_database`.
- The per-campaign cap of six, the eligibility gates, and the matched-seed confirmation rule are
  untouched.

The scientific discipline this budget exists to protect — not tuning on public validation, and
rationing how many times we look — is preserved at the campaign level, which is where our design
placed it.

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

1. **No improvement over the baseline has been demonstrated, and the evidence for that is itself
   compromised.** No generated candidate has produced a validation-primary delta above ε = 0.002,
   and every completed campaign terminated at `baseline_reproduced`. But defect 5 (§3.5) means
   promotion was progressively blocked through the day and fully closed from 11:59:17, so those
   results were measured through a closing gate. The best inner-fold result we have, +0.000578 on
   Fold A with +0.000161 on Fold B, was discarded by that defect rather than by the science. It is
   still far below ε, so we claim no improvement — but we cannot yet claim a clean measurement
   either, because no campaign has run end to end with the defect fixed.
2. **The search has not meaningfully diversified, and enforcing diversity did not fix it.** Across
   the 8 advisory-memory campaigns, 17 of 24 admissions were the same `pairwise` family. A
   deterministic cross-run block then refused that family 14 times in a single campaign; the model
   re-proposed it every time and its alternatives scored bit-identical to the parent. See
   [`agent-memory-experiment.md`](agent-memory-experiment.md) for the measurement and what we
   built in response.
3. **The cross-run circuit breaker has not yet had a fair test.** It is implemented and unit
   tested, but it reads a ledger scoped by trusted-source digest, and every code change resets
   that scope — including the change that added the breaker. It needs two consecutive campaigns on
   an unchanged tree, which we have not yet run.
4. **The outer-validation ration was corrected from project-lifetime to per-campaign** — see
   §4.1. Fourteen campaigns shared a single allocation of six, and it reached zero.
5. **Hidden-test performance is unknown and unclaimed.** It is measured once, by the organizers.

## 7. Artifact index

| Artifact | Location |
|---|---|
| Final report | `runs/<run-id>/final/report.md` |
| Submission CSV | `runs/<run-id>/final/submission.csv` |
| Bundle manifest | `runs/<run-id>/final/manifest.json` |
| Organizer verification | `runs/<run-id>/final/verification.json` |
| Reproduction script | `runs/<run-id>/final/reproduce.sh` |
| Per-iteration records | campaign store under `runs/<run-id>/` |
| Per-iteration run log | `kuairand-agent iteration-log --run-dir runs/<run-id>` |

The run-log deliverable (hypothesis, code diff, resulting metrics, error/recovery events per
iteration) is emitted on demand from the campaign's own durable records:

```bash
uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent iteration-log --run-dir runs/postfix-20260830T105450Z \
  --format md --output docs/run-logs/postfix-20260830T105450Z.md
```

`--format jsonl` emits one canonical JSON object per iteration instead.

Independent verification of a retained CSV, without access to hidden-test labels:

```bash
uv run --locked kuairand-agent validate-submission \
  --split test --data-dir .data/KuaiRand-Pure/data path/to/submission.csv
```

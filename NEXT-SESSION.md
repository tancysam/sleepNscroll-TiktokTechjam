# Start here — session handoff

**Written 2026-08-30 ~23:30 SGT. Submission deadline: Tue 2026-09-01 12:00 SGT.**

Supersedes the 2026-08-30 12:00 version, which is stale in every section. `PROJECT-CONTEXT.md`
remains correct on architecture, the task spec and the environment; this file replaces its
"Where things actually stand" section entirely.

Everything runs in **WSL2 Ubuntu**, in `~/sleepNscroll-TiktokTechjam`. Never Windows, never the
stale Windows Desktop copy.

---

## 1. Status in one paragraph

Branch `Maki`, HEAD `e758693`, tree clean, **20 commits unpushed**. Suite: **1497 passed, 21
skipped, 1 failed** — the one failure is the pre-existing arm64 float32 golden
(`test_starter_fm.py::test_first_update_matches_untouched_organizer_float32_golden`). Ruff clean
over `src tests candidate_seed`; mypy has one pre-existing error at `execution/runner.py:909`.
**Five valid submission bundles exist** (runs 09, 10, 11, 12, 14). The agent now executes
reliably, has promoted a candidate once, and **has not beaten the baseline by a material margin.**

---

## 1b. Do these first, in this order

Nothing is running. No campaign is in flight. Read `PROJECT-CONTEXT.md` for architecture and the
task spec, then this file, then act.

**1. Check the fusion lever before launching anything.** This is the cheapest untried idea and it
was half-investigated when the last session ended. The machinery already exists:

```sh
grep -rn "FUSION_WEIGHT_GRID\|fuse_ranked_predictions" src/kuairand_agent/candidates/fusion.py
grep -rn "fusion" src/kuairand_agent/campaign/generated_scientific_runner.py | head
```

The question to answer: **is rank fusion wired for generated candidates, or only for the official
FM control?** If a generated candidate can be fused with the official FM, that is the most likely
route to a material gain, because our candidates score level with the FM but make different errors
(see §3.3 — except where they reproduce its ordering exactly, which is the case fusion cannot
help). If it is not wired, decide whether wiring it is cheaper than lever 2 or 3 in §7.

**2. Launch run 15 regardless.** It costs ~$2 and ~25 minutes, runs unattended, and the promotion
path is now fixed so a win would actually publish a bundle. Command in §8. Grade **execution rate
first**, then promotion, then the bundle.

**3. Finish the deliverables.** They are worth more marks than another 0.0003. See §9. The Devpost
writeup has not been started and is the largest outstanding item.

**Do not** spend the session patching one traceback at a time. If a run fails, sweep the whole run
before concluding — that mistake cost four runs.

---

## 2. Run history, 09 to 14

All six converged (`stop_reason: converged`) with **zero manual interventions**.

| Run | Candidates executed | Best outer primary | Promoted | Bundle | Cost |
|---|---:|---:|---|---|---:|
| 09 | **3 / 3** | 0.6012030 | no | yes | $0.68 |
| 10 | 1 / 3 | — | no | yes | $1.82 |
| 11 | 0 / 3 | — | no | yes | $2.31 |
| 12 | 0 / 3 | — | no | yes | $1.93 |
| 13 | **3 / 3** | **0.6017118** | **yes** (`promoted_unconfirmed`) | **NO — bug** | $2.21 |
| 14 | **3 / 3** | 0.6014124 | no | yes | $1.69 |

Total spend across every run ever: **$24.74**, 114 provider calls, 2,438,868 tokens. GPU hours
**0.00**. Reconstruct with the per-attempt journals in
`runs/*/production/provider-attempt-journal/`.

Comparators, all on public validation:
- Organizer baseline primary **0.6016**; hidden test 0.5946 (organizer-only).
- Official FM five-seed mean **0.6015721679**.
- **`fallback_outer_mean` = 0.6014402508735657** — the in-run apples-to-apples incumbent, and the
  number every promotion decision is actually made against.
- Seed-to-seed standard deviation **0.0008**. Materiality threshold **ε = 0.002**.

---

## 3. The findings (this is the science)

### 3.1 Code volume predicts whether a candidate runs at all

| Run | `model_impl.py` lines | Executed |
|---|---|---:|
| 01–08 | 1 (truncated stubs) | 0 |
| 09 | 217, 209, 250 | 3 / 3 |
| 10 | 587, 873, **533** | 1 / 3 (the 533) |
| 11 | 786, 640, 428 | 0 / 3 |
| 12 | 611, 585, 749 | 0 / 3 |
| 13 | 491, 518, 572 | 3 / 3 |
| 14 | — | 3 / 3 |

Every candidate at or under ~260 lines executed. Of the eight over 580 lines, none did. Adding
identity codes did not break anything directly — it made the model write two to three times more
code, and defect probability scaled with volume.

### 3.2 The margin is noise, and this is now replicated

The same method, two runs:

| Run | Candidate-01 outer mean | Delta vs incumbent | Sigma |
|---|---:|---:|---:|
| 13 | 0.6017118 | **+0.0002715** | +0.34σ |
| 14 | 0.6014124 | **−0.0000278** | −0.03σ |

Run 13 was above on both inner folds and all three outer seeds. Run 14 landed on the other side of
zero. **Do not report either as an improvement.** The controller agrees independently:
`selector.py:456` promotes as `PROMOTE_CONFIRMED` only when the mean paired delta exceeds ε, so
run 13's candidate was recorded as `promoted_unconfirmed`.

### 3.3 Generated models keep reproducing the FM's exact ordering

Five occurrences across runs 09, 10 and 14: a generated candidate produced **bit-identical GAUC and
nDCG@5** to the official FM control on fold B. Run 10 `candidate-03` did it with **59,816 distinct
prediction values across 61,315 rows**, so it is not a degenerate or constant-score artifact — a
genuinely different model induced the same within-user ordering. With a median slate of four
impressions and only 63.7% of users GAUC-eligible, there are very few orderings to get right.

### 3.4 The pairwise objective is a poorly aligned surrogate here

`runs/pairwise-sweep.json`, 12 cells, full 1,141,112-row train split. **Every cell is below the
local baseline.** Best is 0.6015003 vs 0.6015722, i.e. −0.0000719, and that best is best-of-12
selected on the split it is reported on, one seed, against a five-seed mean. More importantly:
mean pairwise training loss fell in every cell (0.687 → 0.539 → 0.390 as epochs went 10 → 20 → 40)
while the ranking metric **peaked at 10 epochs and then degraded monotonically**. Training the
surrogate harder makes the scored metric worse.

### 3.5 What the winning method actually was (run 13, candidate-01)

`hybrid_pairwise_fm`, rank 12. A factorization machine holding **both** identity embeddings (video,
author, tab, duration bucket) **and** the 33 causal aggregate columns, with the FM interaction term
`0.5·((Σv)² − Σv²)` crossing them, trained under a GAUC-matched within-user pairwise softplus loss,
Adam, frequency-aware L2. The organizer baseline has identities but no aggregates; our run 09
candidates had aggregates but no identities. This held both. The agent derived that itself from the
campaign records of our own failed runs.

---

## 4. What changed this session (5 commits on `Maki`)

| Commit | Change |
|---|---|
| `d82f15a` | **Identity codes.** Feature bundle 33 → 37 columns: `video_id_code`, `author_id_code`, `tab_code`, `duration_bucket_code`. Train-fitted vocabularies, sorted assignment (first-seen order breaks path-independence), UNK slot for unseen identities. Verified in production: cardinalities **7539, 6483, 16, 7**. Also fixed the dashed trajectory table and raised `reasoning_effort` low → high. |
| `784ae49` | **Crash disclosure.** `_iteration_record` now carries `execution_failed`, `proposal_family`, `objective`, `candidate_primary`, `delta_vs_incumbent`, and `reflection_lessons` (previously discarded). Sampler bounds guidance added to the briefing. |
| `7b7d83e` | **Diagnostics no longer fatal.** `candidate_seed/candidate.py` contains `training_diagnostics` failures and reports `diagnostics_failed: 1.0` in-schema. Reflection stopped receiving fallback metrics for crashed branches. |
| `f66e364` | **The fix that worked.** Four tested helpers in the seed: `categorical_codes`, `embedding_table_size`, `fm_interaction_scores` (harvested from run 10 `candidate-03`, the only candidate that ever ran end to end), `within_user_pairs` (written fresh — no working version had ever existed). Briefing carries the size finding. |
| `e758693` | **Promotion path fix.** Ledger export attributed every run to the selected candidate; now attributed to its real owner via `_candidate_run_ownership`. Plus RESULTS.md and README corrections. |

---

## 5. Bugs found and fixed (all were latent, all cost real runs)

1. **Reflection was told a crashed candidate had tied the baseline.** `_reflect` substituted the
   fallback's seed-0 metrics whenever a run produced none, because `ExperimentResultSummary`
   requires three finite metrics. Run 11's reflection duly concluded a candidate that never
   executed was "indistinguishable from the official FM reference". Fixed: zeros plus
   `execution_failed`.
2. **`training_diagnostics` destroyed successful training runs.** Run 11 lost two candidates that
   had already trained (14s and 154s) to exceptions in purely informational code.
3. **The promotion path had never executed.** Run 13 was the first campaign to promote a generated
   candidate, and finalization died with `scientific record source, config, or environment identity
   changed`. The ledger export required **every** retained run to carry the *selected* candidate's
   source and config digest; run 13 had seven runs across three distinct source digests. It also
   stamped every exported row with the winner's id, misattributing losing candidates' runs.

**Run 13's bundle is permanently unrecoverable.** Finalization checks the working tree still hashes
to the digest recorded at campaign creation, so the only source that can finalize it is the source
containing the bug. Deleting the other candidates' evidence would satisfy the old check but would
fabricate the audit trail. Its scores survive in `runs/maki-overnight-13/production/`.

---

## 6. Verified facts. Do not re-litigate these.

- **A candidate receives exactly three arrays**: `features` (37 cols), `targets`, `user_groups`.
  Nothing else. No prompt wording can widen this — it is `candidate_api/runtime_contract.py:55-57`.
- **Five of the organizers' six open directions are unreachable** through that seam: sequences,
  multi-task, censored watch-time and the randomized log all need data the candidate never gets.
- **`hash_source_tree` covers** `src/**.py`, `configs/*.toml`, `scripts`, `candidate_seed`,
  `candidate_templates`, plus `.python-version`, `pyproject.toml`, `uv.lock`. It does **not** cover
  `docs/` or `README.md`, so writeup edits are safe during a live run. Any other edit strands a
  running campaign at `FINALIZING`.
- **`sys` is a forbidden import in candidate source**, including the protected wrapper, which is
  materialized into every candidate.
- **`ExperimentResultSummary` is request-side only.** The strict provider schema
  (`_REFLECTION_SCHEMA`) constrains only the model's reply, so adding a defaulted field there costs
  no wire-contract change.
- **The repair loop covers pre-execution only** (`production.py` catches
  `CandidateMaterializationError` and `CandidateStaticError`). A crash inside `train_model` ends the
  branch with no retry.
- **The fallback provider slot is a different model** (`gpt-5.6-terra` vs main `gpt-5.6-sol`), and
  failover is sticky, so a mid-campaign failover silently changes experimental conditions.
- **Each bundle only replays at the commit that produced it**: run 09 → `0978e46`, run 10 →
  `d82f15a`, run 11 → `784ae49`, run 12 → `7b7d83e`. Whichever we submit, the README reproduce
  steps must name its commit.

---

## 7. Levers still open, best first

1. **Rank fusion of a generated candidate with the official FM.** The machinery already exists
   (`candidates/fusion.py`, `FUSION_WEIGHT_GRID`, `fuse_ranked_predictions`, and the method card
   already reports `fold_b_selected_fusion_only`). Two models that score equally but make different
   errors usually fuse to something better than either. **Not yet investigated whether it is wired
   for generated candidates — check this first, it is the cheapest untried idea.**
2. **History and recency features.** `time_ms` is already enabled as `STRICT_PAST_HISTORY_SOURCE`
   and the causal engine already sorts by it, but the bundle builds **zero** history features. No
   field-policy change needed. Same class of change as the identity codes, which worked.
3. **Auxiliary signals** (`is_click`, `play_time_ms`, 20 fields disabled at
   `data/fields.py:296-302`). `long_view` is a binary threshold *on* play time, so `play_time_ms` is
   a strictly richer version of the label. Requires enabling the fields, plumbing through
   `CampaignDataPlane` (which carries binary `long_view` only), and **re-qualification** because
   `FIELD_POLICY_DIGEST` changes. `docs/RESULTS.md` records a full six-launch qualification plus
   clean replay at **170.76 s**, so re-qualification may be far cheaper than previously assumed —
   verify before ruling it out.
4. **Allow a runtime crash to consume a repair attempt.** Would have saved six branches across runs
   11 and 12. Structural, bigger change.

---

## 8. How to run and grade

```sh
cd ~/sleepNscroll-TiktokTechjam && nohup sh scripts/run_full_campaign.sh configs/full-pure.toml \
  "$HOME/sleepNscroll-TiktokTechjam/runs/maki-qualification" \
  "$HOME/sleepNscroll-TiktokTechjam/runs/maki-overnight-15" \
  > logs/overnight-15.log 2>&1 &
```

Fresh run directory every time; one campaign at a time (they share
`runs/outer-query-ledger.sqlite3`). Roughly 25 minutes and ~$2 each.

**Grade execution rate first, then promotion, then the bundle.** A candidate that runs and scores
0.601 beats a campaign of crashes.

```sh
# execution rate — the gate that matters
python3 -c "
import sqlite3;from pathlib import Path
r=Path.home()/'sleepNscroll-TiktokTechjam'/'runs'/'maki-overnight-15'
c=sqlite3.connect(f'file:{r}/campaign.sqlite3?mode=ro',uri=True)
x=list(c.execute(\"SELECT status FROM executions WHERE experiment_id IS NOT NULL\"))
print(len(x),'executions,',sum(1 for s in x if s[0]=='SUCCEEDED'),'succeeded')"

# why anything crashed — stderr IS retained here, workspaces are cleaned
for d in runs/maki-overnight-15/production/candidate-control/*/; do tail -6 "$d/stderr.log"; done

# scores
ls runs/maki-overnight-15/final/          # bundle exists?
grep -A 8 "Experiment trajectory" runs/maki-overnight-15/final/report.md
```

Exit code **3** is `EXIT_CONTRACT` — a controller integrity refusal, not a crash.

---

## 9. Deliverable state

- `docs/RESULTS.md` — §3.2 (pairwise sweep) and §3.3 (live campaigns) written; token consumption
  and manual interventions tables filled; §6 corrected. **Needs runs 13 and 14 added.**
- `README.md` — Results and Limitations corrected. **Needs the per-bundle replay commit noted.**
- **Devpost writeup — not started.** This is the largest outstanding item.
- Per-iteration run logs are generated from
  `runs/<id>/production/generated-source/iteration-NN-lineage.json` and render in `report.md`.

---

## 10. Working preferences

- The user launches campaigns. Give the command; never launch or `nohup` on their behalf.
- Lead with plain English, give an opinion not a survey, ask before committing.
- Never paste an API key. Stay on `Maki`; never push `main`.
- Commit messages: no hyphens or dashes as punctuation, no quote marks; end with the
  `Co-Authored-By` and `Claude-Session` trailers.
- **Verify before asserting.** Several confident claims in this project were false on inspection,
  including two of mine this session: that candidate stderr was not retained (it is, in
  `production/candidate-control/*/stderr.log`), and that `ExperimentResultSummary` was part of the
  strict provider schema (it is request-side only).
- **Sweep the whole run before concluding.** Patching one traceback at a time is how four runs went
  by before anyone noticed code volume was the variable.

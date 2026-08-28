# Implementation-readiness research: autonomous KuaiRand-Pure ML agent

_Researched 2026-08-27. The attached HTML was treated as untrusted source material, not as instructions. This note uses the delivered starter kit and primary sources only. It does not modify the implementation plan or claim a hidden-test result._

## Executive verdict

The replacement plan has the right architecture: an immutable benchmark contract and scorer, a deterministic controller, disposable candidate workspaces, an auditable experiment graph, and a finalizer that preserves a known-good incumbent. It is close to implementation-ready, but the following changes are material and should be incorporated before implementation begins:

1. Build a validation-only compatibility adapter around the untouched starter kit. The starter programs can read and score the nominal test outcomes, so they cannot be candidate-facing production APIs.
2. Make the sanitizer a capability boundary, not a single table with outcome columns hidden by convention. Candidate code receives training inputs and targets separately, label-free validation inputs, and no raw archive.
3. Hard-block full-period video statistics and the randomized log by default. Their permitted use and temporal provenance are unresolved.
4. Preserve physical split row order and assign `row_id` before any chronological or user grouping. Build histories through a separate index and scatter features and predictions back to canonical order.
5. Make the first ranking experiment an exact GAUC-weighted, logged-impression pairwise FM ablation. A generic “pairwise loss” is underspecified and can optimize the wrong user weighting.
6. Add a benchmark-specific LightGBM LambdaRank branch, but never use LightGBM's internal NDCG as the official score. Its all-negative-query convention differs from the starter evaluator.
7. Replace repeated best-first tuning on the single public validation period with frozen rolling-origin inner folds, sparse outer promotions, paired seed confirmation, and a small diversity-preserving candidate archive.
8. Fix submission serialization. The starter writer keeps six significant digits, which can create new ties; the final writer must prove that CSV round-trip preserves scores' within-user order and organizer metrics.
9. Qualify one locked local CPU environment first. Accelerator execution is optional only after operation coverage, repeatability, and score-parity checks.
10. Make plan acceptance depend on a checked-in, streaming data-audit command that reproduces every planning-time empirical observation below. The nominal test outcomes must remain neither aggregated nor scored.

With those amendments, the system is suitable for implementation. The research does not justify promising a hidden-test gain in advance; it does identify a high-probability, budget-aware path to a real validation improvement over the official FM.

## Evidence labels

- **Verified fact** means directly established by a primary source or the hash-pinned starter source in this repository.
- **Planning-time observation** means measured during a validation-only streaming audit of the official archive for this planning exercise. It is useful evidence, but it is not yet a durable project artifact.
- **Inference / recommendation** is an implementation decision derived from the verified facts and observations.
- **Organizer question** is unresolved and must remain a configuration gate rather than being silently assumed.

## 1. Freeze the executable benchmark contract

### 1.1 Dataset and starter identities

The official KuaiRand repository points to [Zenodo record 10439422](https://zenodo.org/records/10439422) and the direct [`KuaiRand-Pure.tar.gz` artifact](https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz), publishing MD5 `0820331067a3784d9691136f772b35a7`; it also reports 27,285 users, 7,551 standard-log items, and 1,436,609 standard interactions for Pure. The same official source says Pure retains only candidate-pool items and has incomplete sequences. See the pinned [official README](https://github.com/chongminggao/KuaiRand/blob/f8dbf6678b3c9594050e3e813aeff0c942260ec4/README.md#three-versions-and-suggestions) and the [KuaiRand paper](https://doi.org/10.1145/3511808.3557624).

A planning-time streaming audit computed archive SHA-256 `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0` in addition to the publisher's MD5. This is a **planning-time observation**, not a replacement for a checked-in acquisition manifest. Implementation must independently verify the publisher identity and recompute the SHA-256 before extraction.

The pinned official repository tree does not version the `load_data_pure.py` that the README says is bundled inside the downloadable archive. Consequently, the implementation cannot treat a Git commit as sufficient provenance for the archive parser: it must securely inspect, hash, and review every extracted member, including that loader, without blindly executing it. See the [official repository tree](https://api.github.com/repos/chongminggao/KuaiRand/git/trees/f8dbf6678b3c9594050e3e813aeff0c942260ec4?recursive=1).

The delivered organizer artifacts currently hash as follows:

| Artifact | SHA-256 |
| --- | --- |
| `kuairand-starter-kit.zip` | `07237e62cc1a9cd8278556dab995dd5388516f10772724f582ef8320ac68b10b` |
| [`README.md`](../../kuairand-starter-kit/README.md) | `c7a58e652a1aceea144e651ba9ef7a6a4f7dc13f0916e3c4ed342dce69699861` |
| [`data.py`](../../kuairand-starter-kit/data.py) | `1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541` |
| [`evaluate.py`](../../kuairand-starter-kit/evaluate.py) | `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de` |
| [`baseline.py`](../../kuairand-starter-kit/baseline.py) | `c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a` |
| [`submit.py`](../../kuairand-starter-kit/submit.py) | `ab01bb2b970ae2a9f2ead299f5240b71ff4126c2d9bb0e0c4de6c7e245dc148c` |
| [`baseline_scores.json`](../../kuairand-starter-kit/baseline_scores.json) | `950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324` |

These files should remain immutable reference artifacts. The production system may wrap them and golden-test against them, but generated code may not change, shadow, or select substitutes for them.

### 1.2 Exact task semantics established by the starter

The delivered source pins the operative contract:

- `long_view` is the binary target; date splits are train `20220408–20220421`, validation `20220422–20220428`, and nominal test `20220429–20220508` ([`data.py`](../../kuairand-starter-kit/data.py)).
- The task ranks each user's logged impressions, not the full item catalog ([`evaluate.py`](../../kuairand-starter-kit/evaluate.py)).
- GAUC includes only users with both labels and weights each user AUC by that user's positive count. nDCG@5 includes every user, gives a zero-positive user `0`, and uses gain `2^rel - 1` ([`evaluate.py`](../../kuairand-starter-kit/evaluate.py)).
- Primary is the arithmetic mean of GAUC and nDCG@5.
- The reference FM uses `user_id`, `video_id`, `author_id`, `tab`, and a train-fitted duration-decile field; latent dimension 16; learning rate `0.001`; batch size 8,192; up to 40 epochs; patience 4; and pointwise log loss ([`baseline.py`](../../kuairand-starter-kit/baseline.py)).
- Submission order is the original file order after date filtering, with a zero-based contiguous `row_id`; `(user_id, video_id)` is explicitly non-unique ([`submit.py`](../../kuairand-starter-kit/submit.py)).

The executable kit wins over conflicting prose for implementation, subject to an organizer confirmation that it is the current judging contract.

### 1.3 Starter behavior that must be contained

The starter is a reference, not a safe production boundary:

| Verified behavior | Risk | Required wrapper behavior |
| --- | --- | --- |
| `data.load()` returns labels for train, validation, and nominal test. | Candidate code can read the holdout outcome. | Quarantine raw logs; generate separate phase-specific capabilities; never mount a nominal-test label artifact. |
| `baseline.py` evaluates both validation and nominal test. | An otherwise “exact” reproduction can score the holdout. | Run the exact training/validation path through a trusted validation-only adapter; candidate processes cannot invoke the starter CLI. |
| `submit.py --split test --score` is accepted by the argument parser. | The comment that only local validation is scoreable is not enforced. | Candidate access to scoring commands is denied; the finalizer exposes format/alignment validation only. |
| `evaluate()` groups `zip(user_ids, labels, scores)` without checking equal lengths. | Python silently truncates mismatched arrays, while the returned `rows` field still uses `len(labels)`. | Validate equal non-zero lengths, exact label domain, finite numeric scores, and row alignment before calling the immutable evaluator. |
| Python's per-user score sort is stable. | Equal scores inherit canonical input order, so order and serialization ties affect nDCG. | Preserve row order, golden-test tie behavior, and prohibit `row_id` as a feature or optimization target. |
| `write_submission()` emits `f"{float(s):.6g}"`. | Close values can collapse to ties and change within-user ordering or nDCG. | Serialize with sufficient precision, read the CSV back, and require order and metric parity before accepting the bundle. |
| Categorical vocabularies are assigned in first-seen train-row order. | Sorting or deduplicating before exact reproduction changes encodings and seeded training. | Keep a starter-compatible legacy view whose physical row order is immutable. |
| Each field has a train-unseen `UNK` slot, but no training row uses it. | Its FM embedding starts random and receives no data gradient; cold users get a seed-dependent untrained contribution. | Preserve this for baseline reproduction, then make deterministic cold fallback or UNK-masking/dropout a controlled candidate experiment. |

The official scorer remains byte-for-byte protected. The trusted wrapper supplies the safety checks it omits; it does not “fix” the reference implementation in place.

### 1.4 Required evaluator golden tests

Before model work, compare the wrapper and protected evaluator on tiny fixtures containing:

- a zero-positive user, an all-positive user, a one-row user, and a mixed-label user;
- tied scores, including ties across the nDCG@5 boundary;
- users whose file rows are interleaved;
- duplicate user/video pairs with different context or timestamps;
- score vectors that are too short, too long, non-numeric, NaN, or infinite;
- a strict monotonic transform applied independently within each user, which must leave metrics unchanged when it creates no ties;
- CSV serialization and read-back, which must leave the exact within-user ordering and organizer metrics unchanged.

The convergence function also needs its own truth table: improvement exactly `epsilon`, just above `epsilon`, small positive improvements, regressions, failures, retries, and the third consecutive non-material result. The kit supplies `epsilon=0.002` and `N=3`, but not a uniquely executable algorithm.

## 2. Planning-time validation-only audit

The following values were observed by streaming the standard logs from the archive identified above with the starter's date predicates. Nominal-test outcome columns were not read into the audit aggregation path and were neither aggregated nor scored.

| Quantity | Train | Public validation |
| --- | ---: | ---: |
| Rows | 1,141,112 | 124,909 |
| Users | 26,210 | 22,377 |
| Videos | 7,538 | 5,951 |
| `long_view` positive rate | 0.3366199 | 0.3132841 |
| Cold-user rows relative to train | — | 1,990 |
| Cold-video rows relative to train | — | 17 |
| Cold-author rows relative to train | — | 14 |
| Extra rows from repeated `(user_id, video_id)` pairs | — | 3,572 |
| Maximum validation pair multiplicity | — | 7 |

Additional public-validation composition observations:

- per-user slate size median `4`, 90th percentile `12`;
- 6,785 zero-positive users;
- 2,663 all-positive users;
- 12,929 GAUC-eligible mixed-label users;
- train click–`long_view` phi correlation `0.75965`;
- item popularity primary `0.580721929`, reproducing the published `0.5807` after rounding;
- random seeds 0–4 reproduce published mean primary `0.4834` after rounding.

A validation-only execution of the untouched FM training logic produced the following primary scores for seeds 0–4:

```text
0.6014695168
0.6017608643
0.6010903120
0.6015030742
0.6020370722
```

Their mean is `0.6015721679`, which reproduces the published validation `0.6016` at four decimals; population standard deviation is `0.00031599`, and the range is `0.00094676`. On the planning machine, seed 0 took roughly 11 epoch-seconds and 16.5 seconds including load on Apple silicon with Python 3.13 and NumPy 2.5. The original CLI subsequently failed because its nominal-test slice was intentionally left empty; this is expected evidence that the production adapter must suppress that code path rather than expose the nominal-test outcomes. The starter source was not modified.

These are **planning-time observations**, not implementation acceptance evidence. M1 must introduce a committed streaming audit command that, from a verified archive, reproduces:

1. archive/member hashes, headers, byte sizes, and row counts;
2. exact split predicates and canonical row IDs;
3. label domains and rates for train and validation only;
4. user/item/author cardinalities, cold-start counts, slate distributions, and duplicate-pair statistics;
5. click–`long_view` association on train only;
6. random, popularity, and five-seed FM validation rungs; and
7. a manifest proving nominal-test outcome fields were neither aggregated nor scored.

### 2.1 Implications for the research program

These observations change model priorities:

- The positive rate falls by about 2.33 percentage points from train to validation. Time, recency, and drift-aware features deserve early causal ablations; random row-level validation does not represent this shift.
- Median validation slate length is four, so nDCG@5 often covers the entire logged list. Improving positive-vs-negative ordering with a correctly weighted pairwise loss can plausibly help both GAUC and nDCG, rather than only GAUC.
- Zero-positive and all-positive users together comprise 42.2% of validation users. Their ordering cannot change either metric: they are excluded from GAUC, zero-positive nDCG is always 0, and all-positive nDCG is always 1. Ranking objectives should skip them; a pointwise auxiliary loss may still use them to learn generic item/context priors.
- The 1,990 cold-user rows make the baseline's untrained random UNK embedding a concrete ablation target. A deterministic generic-user contribution, zeroed cold-user embedding, or train-time user-ID masking is more principled.
- Repeated user/video pairs prove that user-item identity is insufficient as a key. Context, time, source ordinal, and immutable `row_id` must stay attached at row level.
- Strong train-only click association makes click a plausible auxiliary target. It remains a same-row response, not an inference feature, and its meaning differs by interface in the official schema.

## 3. Build a leakage-safe data layer

### 3.1 Verified outcome and provenance risks

The official KuaiRand schema defines `long_view` deterministically from the current row's `play_time_ms` and `duration_ms`: completion for videos at most 18 seconds, or at least 18 seconds watched for longer videos. It also describes click, like, follow, comment, forward, hate, play time, profile/comment dwell, and profile entry as same-impression outcomes. See the official [log-field definitions](https://github.com/chongminggao/KuaiRand/blob/f8dbf6678b3c9594050e3e813aeff0c942260ec4/README.md#1%EF%B8%8F%E2%83%A3-description-of-the-fields-in-log_xxxcsv).

Therefore:

- `play_time_ms` reconstructs the target and is prohibited as a current-row input;
- all current-row response fields are labels only;
- auxiliary response heads must predict from the same pre-impression feature vector;
- a past response is legal only through a reviewed feature builder whose cutoff is strictly earlier than the scored impression and whose phase protocol makes that response observable.

The official `video_features_statistic` table contains month-level show, play, valid/complete/long-time-play, play-progress, like, comment, follow, share, and related response aggregates. The collection spans the benchmark dates. This table is therefore a direct target proxy without a per-row as-of cutoff and is **hard-blocked** for the date-split benchmark unless the organizer explicitly permits its non-causal provenance. See the official [statistics-table schema](https://github.com/chongminggao/KuaiRand/blob/f8dbf6678b3c9594050e3e813aeff0c942260ec4/README.md#4%EF%B8%8F%E2%83%A3-descriptions-of-the-fields-in-video_features_statisticcsv).

The official site separates standard and randomized-intervention logs and states that the intervention period overlaps April 22–May 8. The exact policy/candidate semantics differ from the required standard-log ranking task. Keep the random log completely separate and excluded from training, histories, tuning, and EDA unless the organizer defines an allowed use. In all cases, development must not consume outcomes after April 28. See the [KuaiRand data-collection paper](https://arxiv.org/html/2208.08696v2#S3.SS2).

Static side tables need field-level provenance review:

- Baseline `author_id` and impression `duration_ms` are organizer-backed inputs.
- Basic video type, upload date/type, dimensions, music, tags, and duration are plausible pre-impression inputs, but require unique-key, coverage, unit, and `upload_dt <= impression_date` checks.
- `visible_status` is described as a current state, not a historical as-of state.
- User activity flags, follow/fan/friend counts, and other snapshot features have no documented extraction time.
- The later caption/category release is a separate artifact and remains quarantined until the organizer confirms it is allowed under the no-external-data rule.

Conservative default: all temporally mutable or undocumented snapshot fields are `forbidden_or_unproven` until approved; do not let an LLM infer availability from a column name.

### 3.2 Required capabilities, not a monolithic table

The trusted sanitizer should materialize immutable, typed views with a registry-enforced schema:

| Capability | Contents and access rule |
| --- | --- |
| `train_impressions` | `split_row_id`, source identity, user/video IDs, date/time/context, approved side inputs; no current-row outcome columns. |
| `train_targets` | `split_row_id`, primary `long_view`, and explicitly approved auxiliary targets; passed through a target API, never concatenated to input features. |
| `train_history_events` | Training-only timestamps and response values accessible only to registered prefix-history builders. |
| `inner_fold_manifests` | Frozen rolling-origin train/holdout row IDs derived only from the official train period. |
| `valid_impressions` | Canonically ordered validation inputs with every current-row outcome absent. |
| `valid_targets` | `split_row_id,user_id,long_view`, available only to the trusted scorer. |
| `final_impressions` | Label-free final inputs, sealed until final lock. |
| `alignment_manifest` | Canonical `split_row_id,user_id,video_id`, source-file ID, source-record ordinal, and pre-outcome row hash. |
| `feature_registry` | Origin, dtype, join key, availability role, phase permissions, temporal cutoff, transformation hash, and rationale for every field. |
| `dataset_manifest` | Upstream/member hashes, parser versions, row/order hashes, split/cardinality/missingness reports, and side-join coverage. |

Candidate workspaces receive only the capabilities needed for the current phase. Raw combined CSVs, validation labels, final labels if present in the public archive, the protected evaluator source, and all forbidden side tables stay outside the candidate namespace.

### 3.3 Exact row identity and chronological histories

Never sort or deduplicate a benchmark split in place. Each source record needs:

1. immutable `source_file_id + source_record_ordinal`;
2. `split_row_id`, assigned after official date filtering while retaining file order; and
3. an optional chronological permutation used only for causal feature construction.

History semantics should be exact:

- official split assignment uses raw `date`, not a timestamp converted under a guessed timezone;
- chronological construction uses `time_ms` after its domain is validated;
- for equal timestamps, compute every row's feature from state at timestamps `< time_ms`, then update state only after the entire equal-time group is featurized;
- training aggregates are strict-prefix or out-of-fold—no current target and no later row can contribute;
- validation response-derived features freeze at the train cutoff; changing validation labels must not change any validation feature, model, or prediction;
- final response-derived features contain no final outcome; if the organizer later permits retraining on train+validation, that is a distinct final protocol after configuration lock;
- an optional exposure-only online state may update with prior evaluation IDs/context only if the input contract explicitly makes those impressions observable;
- all derived arrays are scattered back to canonical `split_row_id` before training, scoring, or submission.

Pure is explicitly an incomplete candidate-pool sequence dataset. DIN or other sequence models may use legal candidate-pool histories, but reports must not describe them as complete user histories.

### 3.4 Qualification and metamorphic tests

Data qualification should fail closed on any of the following:

- archive MD5/SHA-256 mismatch, unexpected tar member, unsafe path/link/device, unexpected executable, or size-cap violation;
- unexpected header, parser ambiguity, malformed/missing ID/date/time/duration/label, non-binary outcome, or target-reconstruction mismatch;
- split row-count/date/order mismatch or source record assigned to more than one split;
- duplicated side-table keys, many-to-many join expansion, join row loss, or unexplained duration mismatch;
- a future outcome changing an earlier feature;
- the current row's outcome changing its own input vector;
- an equal-timestamp row permutation changing those rows' features;
- a validation outcome changing validation features, predictions, checkpoint, cache key, or code/config hashes;
- candidate access to raw logs, protected labels, forbidden month statistics, or scorer internals;
- a join, group/sort, model adapter, or CSV round-trip that fails to restore exact `split_row_id` order.

Run sanitation twice in clean locations and require the same logical manifest. Cache keys must include dataset/member hashes, parser/library versions, feature-registry hash, temporal protocol, code hash, and split manifest. A streaming parser and typed columnar or memory-mapped outputs are preferable to repeatedly loading every CSV row into Python tuples for a 50-attempt campaign.

## 4. Optimize the exact ranking objective

### 4.1 Derive the pair sampler from the organizer GAUC

For an eligible user `u` with `P_u` positives and `N_u` negatives, the starter computes:

```text
GAUC = sum_u P_u * AUC_u / sum_u P_u
AUC_u = (1 / (P_u * N_u)) * sum_(p,n) pair_credit(s_p, s_n)
```

Thus each logged positive-negative pair for user `u` has official weight proportional to `1 / N_u`, not uniform user weight and not uniform weight over the union of all pairs.

The original [Bayesian Personalized Ranking paper](https://www.cs.mcgill.ca/~uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf) uses a differentiable pairwise objective `log sigmoid(s_positive - s_negative)` and samples user-positive-negative triples. For this benchmark, the metric-matched sampler is:

1. choose uniformly from all positive rows belonging to GAUC-eligible users;
2. choose uniformly from that positive row's user's **logged** negative rows;
3. optimize `softplus(-(s_positive - s_negative))`.

This gives each pair probability `1 / ((sum_u P_u) * N_u)`, matching the organizer's GAUC weighting. Uniformly choosing a user optimizes unweighted mean user AUC; uniformly choosing among every available pair weights users by `P_u*N_u`; sampling unexposed catalog items changes the task from logged-impression ranking to retrieval. All three are wrong defaults here.

Implementation tests must compare the derived formula with brute-force GAUC on tiny fixtures and statistically verify deterministic sampler frequencies. All-same-label users produce no ordering gradient and should be excluded from ranking batches.

### 4.2 Treat GAUC and nDCG as complementary objectives

The first performance experiments should isolate objective alignment before changing architecture:

1. **Exact pointwise FM reference.** No model, feature, order, or hyperparameter change.
2. **Pairwise FM.** Same fields and FM scoring function; only replace pointwise loss/batching with the GAUC-weighted logged-impression sampler.
3. **Hybrid FM.** Add a small pointwise term only if it improves temporal folds or cold/all-same-label representation; predeclare the weight grid.
4. **User-grouped LambdaRank tree model.** Use safe dense/context/causal aggregate features and optimize top-rank swaps.
5. **Within-user rank fusion.** Blend the best GAUC-oriented and nDCG-oriented candidates after converting scores to deterministic within-user ranks/percentiles.
6. **Compact neural interaction model.** DeepFM or DCN-V2 on the same safe features, with pairwise or hybrid loss.
7. **Simple causal history, then sequence graph/history models.** Only escalate after pooled prefix aggregates show signal.
8. **Controlled auxiliary tasks.** Shared-bottom click/watch-related heads first, MMoE only if shared-bottom helps or exhibits task conflict, and PLE only after measured negative transfer.
9. **Custom LambdaLoss or watch-time formulations.** Late research branches after built-in, well-tested controls plateau.

LambdaRank reweights pair updates by the NDCG effect of swaps; the [LambdaLoss paper and implementation study](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/) also warns through its results that tighter metric surrogates can overfit without sufficient regularization. That supports testing a mature LambdaRank implementation before writing a bespoke listwise loss.

DCN-V2, DIN, LightGCN, MMoE, and PLE solve different representation or task-sharing problems; none is a guaranteed upgrade on this exact evaluator. Relevant primary sources are [DCN-V2](https://arxiv.org/abs/2008.13535), [DIN](https://arxiv.org/abs/1706.06978), [LightGCN](https://arxiv.org/abs/2002.02126), [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/), and [PLE](https://doi.org/10.1145/3383313.3412236). Add one mechanism per child so the controller can attribute results.

### 4.3 LightGBM needs a benchmark-specific adapter

LightGBM is a practical CPU branch because it supplies a mature `lambdarank` objective, but the generic framework contract is not the organizer contract:

- query rows must be contiguous and group sizes must sum to the row count ([`LGBMRanker` API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html));
- `lambdarank_truncation_level` emphasizes a top cutoff, with the official parameter guide suggesting a value slightly larger than the target cutoff; compare 5 and 8 on inner folds for nDCG@5 ([LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html));
- pin `label_gain=[0,1]`, `lambdarank_norm=true`, every seed, and a deterministic CPU histogram mode;
- LightGBM's source assigns internal nDCG `1` to an all-negative query, whereas the starter assigns `0` ([LightGBM ranking metric source](https://github.com/lightgbm-org/LightGBM/blob/main/src/metric/rank_metric.hpp));
- LambdaRank skips equal-label pairs, which is correct for this ranking objective; the alternative `rank_xendcg` has separate stochastic query behavior and should be an ablation only after invariant-query tests ([LightGBM ranking objective source](https://github.com/lightgbm-org/LightGBM/blob/main/src/objective/rank_objective.hpp)).

Required adapter sequence:

1. stable-sort a view by `(user_id, split_row_id)`;
2. build and validate exact group sizes;
3. retain the inverse permutation;
4. train with a locked config and inner temporal holdout for tree-count selection;
5. predict one finite score per row;
6. inverse-map predictions to canonical row order;
7. pass only those predictions to the protected organizer scorer.

Set framework metric to none or mark it explicitly non-authoritative. Golden-test that an all-negative query produces a different internal NDCG but the promotion path still uses only the organizer primary. Do not read `best_score_` as an official result.

### 4.4 Features with the best early cost-to-information ratio

Prioritize controlled, leakage-safe ablations:

- exact `duration_ms <= 18_000`, log duration, and threshold-aware duration interactions, because the target definition changes at 18 seconds;
- date, day-of-week, time-of-day after validating `hourmin`/`time_ms` semantics, and train-only recency/decay features;
- train-prefix item/author/tab exposure and response statistics with smoothing;
- user-candidate affinities from strict-prefix train events;
- deterministic cold user/item/author fallbacks and ID masking;
- context-aware repeated-pair features rather than collapsing `(user_id, video_id)`;
- same-row click and other outcomes only as training targets, with interface/tab masks where semantics differ.

For a graph branch, build edges from training positives and contrast against the same users' logged negative impressions. The original LightGCN full-catalog negative sampling is not automatically appropriate for this logged-candidate task.

Implement the benchmark-specific PyTorch pair sampler and losses directly. RecBole's [official BPR implementation](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/bpr.py) is a useful reference, but its standard negative/evaluation path is full-catalog oriented; TorchRec's [official introduction](https://pytorch.org/blog/introducing-torchrec/) emphasizes large-scale sparse/distributed recommendation infrastructure that is disproportionate for Pure's roughly 27K users and 7.6K items. Neither should be a core initial dependency. Likewise, FuxiCTR may supply model references, but its current [`gAUC` implementation](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/metrics.py) weights eligible user AUCs by group row count rather than the starter's positive count. Model backends never own promotion.

The [CWM paper](https://arxiv.org/abs/2406.07932) and [official implementation](https://github.com/hyz20/CWM) are relevant later references for watch-time-aware recommendation, but the stock implementation reconstructs its own labels/objectives and targets an old PyTorch environment. Port only a clearly isolated equation as a later auxiliary-loss hypothesis; do not vendor the training stack or let it redefine `long_view`, the candidate set, or the scorer.

## 5. Protect selection from public-validation overfitting

### 5.1 Why the search policy must change

The [AIDE paper](https://arxiv.org/html/2502.13138) supports an experiment tree with diverse drafts, bounded debugging, and atomic improvements. Its maintained [`agent.py`](https://github.com/WecoAI/aideml/blob/main/aide/agent.py) asks programs to print a holdout metric, lets an LLM interpret stdout, and then greedily improves the journal's best node. Those ideas are useful for code search, but LLM-parsed stdout and repeated greedy selection on one public holdout are inappropriate for this exact metric contract.

The [MLE-Bench paper](https://arxiv.org/html/2410.07095) and [official repository](https://github.com/openai/mle-bench) provide stronger operational evidence:

- separate public agent data from private grading data;
- treat submission existence and structural validity as candidate eligibility;
- expect invalid outputs, disk/RAM exhaustion, time-reasoning failures, and premature termination;
- retain persistent search/recovery rather than terminating the campaign on the first failed branch;
- additional sampling improves pass@k, supporting diverse early roots;
- longer runs can select a worse final candidate when “best” selection is weak, so a valid incumbent and finalizer must be immutable.

MLE-Bench's agent-facing validator does not reveal the private score. Its results do not validate exposing exact outer metrics to an LLM dozens of times. Repeated adaptive reuse of a holdout is itself an overfitting channel; the original [Ladder paper](https://proceedings.mlr.press/v37/blum15.html) was motivated by this problem.

### 5.2 Frozen two-tier validation policy

Before autonomous search, derive and hash several rolling-origin folds using only the official train dates. Boundaries should be selected once to preserve enough mixed-label users, positives, slate depth, and recency resemblance to public validation. Do not use random row-level folds.

| Tier | Purpose | Data and feedback |
| --- | --- | --- |
| 0: fixture | Imports, shapes, policy, failure recovery | Synthetic/tiny; no research score and no convergence effect. |
| 1: screen | Cheap single-hypothesis screening | One fixed seed on the most recent train-derived temporal holdout; exact organizer-semantics metrics. |
| 2: confirm idea | Reject temporal brittleness | Two or three frozen rolling-origin folds; report mean, worst fold, GAUC, nDCG, runtime, and memory. |
| 3: outer promotion | Compare a small set of qualified challengers | Retrain on complete official train, score once on public validation through the protected scorer. |
| 4: seed confirmation | Establish a material incumbent change | Challenger and incumbent on the same fixed seeds, provisionally 0/1/2, with paired deltas. |

The protected ledger stores exact outer metrics for audit and convergence. Candidate workspaces receive no outer labels, predictions, per-user residuals, or framework access to the scorer. The proposing LLM should normally receive exact inner metrics plus a coarse outer result such as “did not pass material gate” or a rounded/thresholded delta. If organizers require every exact public score to be visible, make that an explicit contract flag and record the increased overfitting risk.

Full official validation primary still determines promotion and final selection. Inner folds are a search guardrail, not a substitute benchmark.

### 5.3 Diverse, evidence-aware search

After the linear vertical slice is proven, start with three or four distinct roots rather than letting one noisy early winner consume the run:

- GAUC-weighted pairwise FM;
- LambdaRank on safe context/aggregate features;
- compact interaction model;
- simple causal aggregate/history model.

Retain a small Pareto archive over inner mean primary, worst temporal fold, component metrics, seed variability, runtime, memory, lineage, and novelty. Reserve a bounded exploration quota. Keep a GAUC specialist and an nDCG specialist when both remain competitive because within-user rank fusion may outperform either.

Each child changes one attributable mechanism and declares:

- parent and novelty hash;
- hypothesis and expected metric component;
- exact files/config changes;
- falsification criterion;
- expected runtime and peak-memory envelope;
- source/method attribution;
- inner/outer evaluation tier requested;
- recovery policy and maximum retry count.

Outer improvement cannot rescue a candidate that degrades every inner temporal fold without an explicit, logged rationale. The current incumbent is immutable until a challenger passes all eligibility and confirmation gates.

### 5.4 Confirmation, uncertainty, and ensembling

For a promoted challenger:

- run incumbent and challenger with identical seeds;
- report per-seed primary, GAUC, and nDCG deltas;
- report mean, median, minimum, and standard deviation of paired primary delta;
- bootstrap validation users as groups, not individual rows, as a paired uncertainty diagnostic;
- call a result **material** only when the confirmed mean primary delta is greater than `0.002` under the provisional policy;
- call an inconsistent or single-seed gain **unconfirmed**, even if it remains the raw validation best.

Both official metrics depend only on within-user order. Normalize each ensemble member to deterministic within-user ranks or percentiles before blending scores from unlike scales. Tune only a small predeclared weight grid on inner folds, then spend one outer promotion on the selected blend. Use a real secondary model as a tie-breaker where possible; canonical row order is only the final deterministic fallback.

### 5.5 Attempt and wall-clock policy

Until the organizer defines an “iteration,” use the conservative rule that every launched train/evaluate execution—including failures, repair children, pruned jobs, proxy runs, and seed repeats—consumes one of 50 attempts. Record logical experiment and physical execution separately.

A provisional allocation should reserve work rather than spend opportunistically:

| Purpose | Attempts |
| --- | ---: |
| Contract, audit, and baseline qualification | 4–6 |
| Diverse inner screens | 18–24 |
| Multi-fold confirmations | 6–8 |
| Outer promotions | 5–7 |
| Paired seed confirmation | 6–9 |
| Blend/final replay reserve | 3–5 |

The controller must freeze one concrete allocation whose total is at most 50; the ranges above are planning alternatives, not simultaneously claimable maxima.

The six-hour deadline is monotonic and controller-owned. Track measured runtime distributions per candidate family, kill an entire process tree on timeout, and do not launch a job whose conservative completion plus cleanup exceeds the finalization reserve. Static and synthetic tests should be logged separately and need not consume a scored attempt unless the organizer says otherwise.

## 6. Local runner and reproducibility

### 6.1 Environment recommendation

Planning-time inspection found an Apple-silicon macOS host with 15 CPU cores, 24 GB memory, ample storage, Python 3.13.14, and NumPy 2.5.0. PyArrow is available; PyTorch, LightGBM, pandas, scikit-learn, FuxiCTR, Optuna, and MLflow were not present in the base interpreter. These observations are environment inventory, not a reproducibility contract.

Use a dedicated Python 3.11 project environment as the initial compatibility target, with an exact resolver lock, wheel/file hashes, OS/architecture manifest, thread counts, locale, and dependency licenses. This version choice is an engineering inference: it is a conservative common denominator for current NumPy/PyTorch/LightGBM wheels and avoids making the host's very new Python version the benchmark contract.

Start with:

- organizer NumPy FM and scorer;
- a thin standard-library/NumPy controller and sanitizer;
- PyTorch only when the pairwise/neural branch starts;
- LightGBM as a separately qualified optional dependency.

Do not import a full AIDE, FuxiCTR, RecBole, TorchRec, Optuna, or MLflow stack into the first vertical slice. Add a dependency only with a measured experiment need, a locked artifact, and compatibility tests.

### 6.2 Determinism and comparability

PyTorch's official [reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) explicitly says complete reproducibility is not guaranteed across releases, commits, platforms, or CPU/GPU execution. Therefore, “same seed” is insufficient. Record and control:

- Python, NumPy, PyTorch, LightGBM, compiler/runtime, OS, and architecture;
- Python/NumPy/framework/model/data-loader/objective seeds;
- thread counts and deterministic algorithm flags;
- data, split, feature, source, config, environment, and dependency hashes;
- device type and numerical precision;
- checkpoint, raw prediction, serialized prediction, and submission hashes.

CPU is the reference execution path. The official [PyTorch MPS guidance](https://docs.pytorch.org/docs/stable/notes/mps.html) establishes an Apple GPU backend, but not cross-device numerical identity. An accelerator branch may be enabled only after a qualification spike proves operation coverage, repeatability within a declared tolerance, organizer-score parity against CPU, correct resume, and an actual wall-clock benefit. Confirmation comparisons should use the same device and environment.

For LightGBM CPU determinism, official parameters require `deterministic=true` plus either `force_col_wise=true` or `force_row_wise=true`; results can still change across versions, compiler builds, or systems. Benchmark both histogram modes once, lock the selected mode, version, binary hash, and thread count, and invalidate comparability caches when any change.

### 6.3 Candidate containment and resource controls

Generated code is untrusted. A plain subprocess timeout is useful robustness containment but is not a security boundary. Before autonomous execution is described as isolated, qualify the local runner with tests proving:

- no raw archive, validation/final labels, protected scorer, home directory, credentials, SSH agents, or container/control sockets are mounted;
- network is disabled unless a separately approved acquisition phase requires it;
- candidate snapshot and approved data are read-only, with a bounded writable scratch/artifact area;
- the process runs without elevated privileges and cannot escape through symlinks/path traversal;
- CPU, memory, process count, disk, output volume, and wall time are bounded;
- timeout/OOM/interrupt kills the whole process tree;
- logs and declared artifacts are extracted after termination without exposing protected data;
- repeated execution begins from a clean namespace and cannot read a previous candidate's undeclared files.

If the available local mechanism cannot enforce these properties, document it as process-level robustness containment only and do not run arbitrary autonomous patches with sensitive labels or credentials present. The trusted scorer and ledger always remain outside the candidate boundary.

### 6.4 Durable execution and failure recovery

Every attempt should persist a state transition before launch and after completion, with an idempotency key, parent, patch/config hash, seed, deadline, remaining budget, environment hash, command, runtime/resources, output hashes, failure fingerprint, retry ordinal, and intervention count. Recovery is bounded:

- deterministic code/config failures create at most a small number of atomic repair children;
- identical failure fingerprints do not retry indefinitely;
- transient runner loss may resume only from a content-linked checkpoint whose data/config/source hashes match;
- invalid or missing prediction output makes a candidate ineligible, never “partially scored”;
- every campaign keeps a replayable official FM and best eligible incumbent so finalization survives failed research branches.

Fault-injection tests should deliberately exercise syntax error, timeout, OOM, non-finite prediction, truncated CSV, scorer failure, interrupted ledger write, checkpoint resume, and finalizer fallback. These tests are outside the scored campaign; robustness is demonstrated by recovery evidence, not by manufacturing failures during research.

## 7. Implementation acceptance gates

The following sequence is recommended. No later gate can waive an earlier trust boundary.

| Gate | Required evidence |
| --- | --- |
| M0: contract lock | Starter hashes match; operative label/metrics/splits/submission are recorded; organizer questions are configuration gates; golden metric/convergence fixtures pass. |
| M1: acquisition and sanitizer | Official archive MD5 plus local SHA-256 and member manifest; secure extraction; streaming audit reproduces counts and observations; phase capabilities contain only registered columns; nominal-test outcomes neither aggregated nor scored; two clean builds have identical logical manifests. |
| M2: baseline parity | Random, exact popularity, and FM seeds 0–4 reproduce validation references within declared tolerance from the canonical legacy view; no nominal-test outcome access; row/vocab/quantile/encoded-array hashes recorded. |
| M3: vertical research loop | Typed proposal → disposable snapshot → smoke → bounded run → exact aligned predictions → protected score → ledger → promote/reject works; restart and budget accounting are deterministic; incumbent survives a failed child. |
| M4: validation protocol | Frozen rolling-origin folds and feedback policy are hashed; candidate cannot access outer labels or row-level residuals; outer scorer and framework metric denylist tests pass. |
| M5: metric-aligned models | GAUC sampler matches brute-force weighting; pairwise FM ablation runs; LightGBM group/inverse-order and all-negative-metric mismatch tests pass; framework scores cannot drive promotion. |
| M6: performance search | Diverse roots, bounded exploration, paired seed confirmation, runtime/resource ledger, and one-change attribution work within the 50-attempt/six-hour budget. |
| M7: local-runner qualification | Data/secret/network/filesystem/process/resource boundaries and process-tree termination pass adversarial fixtures; limitations are reported without overclaim. |
| M8: finalization | Best eligible source/config/preprocessor/checkpoint are content-linked; label-free final inference restores canonical order; high-precision CSV read-back preserves ranking and metrics; untouched `submit.py --check` passes; fallback FM bundle remains available. |
| M9: clean replay | A fresh environment rebuilds sanitized artifacts from the verified archive, reproduces the selected validation evidence within tolerance, packages the same final manifest, and exports judge-readable JSONL/report with all interventions and resource use. |

Performance readiness should be described precisely:

- **baseline-reproduced** once M2 passes;
- **validation-improved** only after a challenger beats `0.6016` under the protected official scorer;
- **materially confirmed** only after the paired fixed-seed mean delta exceeds the organizer's confirmed threshold and temporal folds do not collapse;
- **hidden-test improved** only after the organizer actually scores the frozen bundle.

## 8. Verified facts, adopted inferences, and unresolved questions

### Verified facts

- The current organizer kit defines the required task as logged-impression `long_view` ranking with GAUC, nDCG@5, and their mean.
- `long_view` is determined from same-row play time and duration; current response columns are not legal inference inputs.
- Pure's sequences are candidate-pool-filtered and incomplete.
- The official full-period video statistics include direct response and long-time-play aggregates.
- Row order affects submission identity, stable score ties, first-seen vocabularies, and seeded baseline training.
- The starter can load and score nominal-test outcomes and its evaluator does not independently reject unequal vector lengths.
- LightGBM's internal nDCG convention for all-negative groups differs from the organizer's convention.
- AIDE's code-tree/atomic-improvement design is useful, but its LLM-parsed metrics and ordinary child process do not implement this benchmark's protected scorer or trust boundary.
- MLE-Bench supplies direct evidence that candidate validity, resource failures, persistence, diversity, and final-candidate selection are first-class ML-agent problems.

### Conservative inferences adopted for implementation

- Full-period video statistics leak future/target information for the date split and are blocked absent an organizer waiver.
- The random log is unavailable absent a benchmark-specific permission and cutoff protocol.
- Mutable user snapshots and current `visible_status` are temporally unsafe until their as-of date is known.
- Physical file order is canonical identity, not proof of chronological order.
- Equal timestamps are simultaneous for causal feature construction unless the organizer defines an event/slate order.
- Validation response history freezes at the train cutoff; validation labels cannot roll into later validation rows.
- A GAUC-weighted logged-negative BPR sampler is the best first objective ablation; LambdaRank is the best mature nDCG-oriented complement.
- Inner temporal folds and sparse outer promotions reduce winner's-curse risk relative to adaptive tuning on one public validation period.
- A locked CPU environment is the reference; any accelerator is an optional qualified optimization.
- A small custom controller plus NumPy/PyTorch/LightGBM is lower risk than adopting a broad recommender or autonomous-agent framework upfront.

### Organizer questions that remain blocking or policy-changing

1. Does the delivered `long_view` + GAUC/nDCG@5 kit supersede every conflicting click + NDCG@10/Recall@50 statement?
2. Is the nominal final period an honor-system holdout, the same rows with labels hidden during judging, or a separate private set?
3. May the randomized log be used for training, histories, tuning, or diagnostics; may any row after April 28 be used before final scoring?
4. Are full-month `video_features_statistic` fields permitted despite their future-period and target-proxy provenance?
5. What are the as-of timestamps for user features, basic video fields, and `visible_status`?
6. Are separately released captions/categories considered permitted KuaiRand data or prohibited external data?
7. May a frozen final configuration retrain on train+validation, and may validation outcomes then form pre-final histories?
8. Exactly what counts as one of 50 iterations: physical training runs, logical hypotheses, failures, repairs, pruned proxies, and repeated seeds?
9. What is the exact executable epsilon/N convergence rule, including equality and failures?
10. What seed set and numerical tolerance define official FM reproduction?
11. What timezone and exact `hourmin` semantics apply, and are equal timestamps simultaneous slate impressions?
12. Which Pure item count is authoritative for each file/view: 7,551 standard items, 7,582 in the paper appendix, or the 7,583-item candidate pool?
13. Which published dataset license governs redistribution of raw or derived artifacts if repository and Zenodo metadata differ?

## Final implementation recommendation

Proceed with the replacement architecture after updating it with the gates above. The fastest trustworthy vertical slice is:

1. lock hashes and golden-test the scorer;
2. build the streaming sanitizer and phase capabilities;
3. reproduce random, popularity, and five-seed FM on validation only;
4. implement the ledger, local bounded runner, aligned prediction contract, and immutable incumbent;
5. freeze train-derived temporal folds;
6. run the exact GAUC-weighted pairwise FM ablation;
7. add safe duration/time/cold-start features one at a time;
8. qualify LightGBM LambdaRank and within-user rank fusion;
9. add compact neural/history/multi-task branches only when simpler experiments establish signal;
10. confirm the winning challenger with paired seeds and a clean final replay.

This sequence targets the largest score opportunity first—objective alignment and legal temporal/context signal—while putting the scorer, data boundary, budget, and final bundle outside the LLM's control. It is both more likely to improve validation within 50 attempts and much less likely to produce an impressive but irreproducible or leakage-contaminated result.

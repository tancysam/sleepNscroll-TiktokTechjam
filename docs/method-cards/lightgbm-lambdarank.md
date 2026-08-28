# Deterministic LightGBM LambdaRank

Status: implemented as a bounded WP6 candidate adapter. LightGBM 4.7.0 is lock-pinned and
available in the production `research-tree` dependency group; it is intentionally absent from
the smaller default/base environment.

## Scientific purpose

This branch tests whether a mature grouped tree-ranking objective can exploit approved context
and causal aggregate features that complement the ID-heavy official FM. It predicts the native
binary `long_view` target and ranks each user's logged impressions. It does not perform catalog
retrieval and does not change the organizer GAUC, nDCG@5, or primary metric.

The adapter is implemented in `src/kuairand_agent/candidates/tree_ranker.py`. Its input is an
immutable finite `FeatureMatrix` in canonical physical split order. The matrix must come from the
trusted candidate-input or causal-feature path. Raw archive tables are not an accepted input.

## Data and leakage boundary

Labels enter only `fit_lambdarank` and only with phase `train` or `inner_train`. Prediction has no
label parameter and accepts only `inner_valid`, `outer_valid`, or `final` phases.

The following sources are prohibited:

- public-validation and final-period labels;
- the randomized-exposure log;
- `video_features_statistic_pure.csv` and its month-level outcome aggregates;
- `user_features_pure.csv` snapshots;
- mutable `visible_status` and intervention marker `is_rand`;
- current-row response fields or any matrix column named after them.

Known raw names from the blocked statistic, snapshot, and provenance fields are rejected again at
the adapter boundary. The stronger control remains upstream: candidate workspaces receive only
trusted numeric capability and causal-feature artifacts, never blocked raw members. A renamed
column is not proof of safe provenance, so the feature-policy and capability digests remain part
of campaign eligibility.

## Grouping and canonical order

LightGBM expects each query's rows to be contiguous and receives query lengths, not one query ID
per row. The adapter uses `build_user_grouping` / `UserGrouping` to create a private stable view:

1. Users retain first-appearance order.
2. Rows within one user retain canonical order.
3. Repeated `(user_id, video_id)` impressions remain distinct.
4. Training features and labels receive the same permutation.
5. `group_sizes` is passed to LightGBM and sums to the row count.

The canonical feature artifact is never reordered in place. Tree prediction is row-independent,
so prediction runs directly over the canonical matrix and returns one score in that same order.
No row ID, source ordinal, or alignment digest is a model feature.

## Objective and fixed metric alignment

The backend receives:

```text
objective = lambdarank
metric = ndcg
label_gain = [0, 1]
lambdarank_norm = true
lambdarank_truncation_level = 8
eval_at = [5]
```

Binary `label_gain=[0,1]` matches the organizer gain `2^rel - 1`. Truncation level 8 is a
source-backed initial setting slightly above the benchmark cutoff 5; it is recorded in every
configuration identity and may be changed only as a bounded inner-fold ablation.

LightGBM's built-in nDCG is not an organizer score. In particular, LightGBM assigns internal
nDCG 1 to an all-negative query, while the organizer evaluator assigns 0 and includes that user
in average nDCG@5. Framework nDCG may select a bounded tree count only on a train-derived inner
fold. It never decides public promotion, replaces protected rescoring, or supports a metric claim.
Every eligible candidate must return canonical predictions to the protected organizer evaluator.

## Inner-fold tree-count selection

Early stopping is structurally unavailable for a full `train` fit. It is enabled only when:

- fitting phase is exactly `inner_train`;
- validation grouping is exactly `inner_valid`;
- training and validation feature names and order are identical; and
- a trusted `InnerValidationSet` supplies binary train-derived holdout labels.

The configured maximum tree count and patience are content-linked. After inner-fold selection,
the outer candidate retrains on all official training rows without public-validation early
stopping, using the preselected tree count/configuration. Public-validation and final labels can
never be passed to this API.

## Deterministic CPU contract

Every run explicitly sets:

```text
device_type = cpu
deterministic = true
force_col_wise = true
num_threads = configured positive fixed count
seed = configured uint32 seed
data_random_seed = seed
feature_fraction_seed = seed
bagging_seed = seed
extra_seed = seed
feature_fraction = 1.0
bagging_fraction = 1.0
bagging_freq = 0
extra_trees = false
```

Only `force_col_wise` is set; `force_row_wise` is absent. This avoids LightGBM's automatic
histogram-choice probe and the extra dataset memory associated with row-wise construction. The
adapter intentionally uses CPU even when a local GPU is available. A GPU implementation would be
a separate qualified backend because CPU/GPU algorithms and reproducibility identities are not
assumed equivalent.

Default bounded tunables are:

| Parameter | Default | Adapter bound |
| --- | ---: | ---: |
| Seed | 0 | uint32 |
| Threads | 4 | 1–64 |
| Maximum trees | 300 | 1–10,000 |
| Inner patience | 30 | 1–10,000 |
| Learning rate | 0.05 | 0.000001–1.0 |
| Leaves | 31 | 1–4,096 |
| Minimum rows per leaf | 20 | 1–10,000,000 |
| L2 regularization | 1.0 | 0–1,000,000 |
| LambdaRank truncation | 8 | 1–1,000 |

These bounds prevent arbitrary backend parameter injection. Generated code cannot change the
objective, device, histogram orientation, label gain, eval cutoff, deterministic flag, or seed
fan-out through an untrusted parameter dictionary.

## Optional dependency and runtime seam

LightGBM is pinned as `lightgbm==4.7.0` in the `research-tree` dependency group. Importing the
candidate module does not import LightGBM. The package is imported only when no explicit backend
is supplied to fit or predict. Missing or mismatched versions fail with an actionable
`TreeRankerDependencyError`; an installed wheel whose native library cannot load reports the
missing platform OpenMP runtime through the same error boundary. No dependency or system runtime
is installed dynamically.

The narrow backend protocol has two operations:

- fit one normalized grouped request and return serialized model text plus the selected tree
  count;
- restore that model text and predict over one finite canonical matrix.

Unit tests use a recording fake backend to verify grouping, parameters, early-stop access,
identity, and malformed output handling without requiring LightGBM. A tiny optional real-backend
test is skipped when the research dependency or its native OpenMP runtime is unavailable. When
both are available, it fits twice with one CPU thread and requires identical checkpoints and
exact float64 predictions.

## Artifact and replay identity

The checkpoint content-links:

- ordered feature names;
- training feature digest;
- stable grouping/permutation digest;
- exact training-target digest;
- optional inner-validation feature/order/target digest;
- full effective config digest;
- backend name/version identity;
- selected tree count; and
- SHA-256 of serialized model text.

The checkpoint digest covers all of those values. Prediction records the checkpoint digest,
query feature digest, phase, and canonical float64 prediction digest. A different feature order,
backend identity, non-finite vector, wrong score count, or changed serialized model produces a
rejection or a different identity rather than a silent replay claim.

Cross-version, compiler, operating-system, and architecture equality is not claimed merely from
the deterministic flag. Campaign evidence must record the locked environment and qualify exact
prediction replay on the reference machine.

## Promotion and stopping rules

This method is eligible for public validation only after:

1. fake-backend contract and optional installed-backend tests pass;
2. both train-derived temporal folds pass structural support checks;
3. canonical scatter/order and finite prediction gates pass;
4. peak memory and runtime fit the campaign reserve;
5. protected organizer scores are produced from serialized canonical predictions; and
6. the candidate clears the frozen outer-promotion policy.

LightGBM logs or `best_score_` are diagnostic evidence only. The trusted controller owns the
official metric, promotion, matched-seed confirmation, incumbent selection, attempt accounting,
and finalization.

## Source basis

The local primary-source verification and exact implementation consequences are recorded in
`docs/research/source-verification-2026-08-28.md`, especially its LightGBM grouping, objective,
all-negative-query, and deterministic CPU sections. The operative architecture and acceptance
requirements are in `plan.md` sections 14.3, 18.3, and WP6.

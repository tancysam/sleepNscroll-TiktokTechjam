# KuaiRand-Pure score-improvement research: primary-source findings

Date: 2026-08-30

## Decision

The next campaign should **keep the qualified official FM as the interaction backbone** and test
small complementary changes in this order:

1. run the repository's existing deterministic LightGBM LambdaRank branch over the unchanged
   33 causal features;
2. select one existing within-user percentile fusion of LambdaRank and the official FM on the
   train-derived Fold B, then confirm that frozen blend on Fold A;
3. only if that plateaus, add a small set of strict-past recency companion features and repeat the
   same tree-plus-fusion comparison;
4. defer compact neural crosses until the tree and fusion controls have been exhausted.

This is an evidence-ranked experimental sequence, not a guarantee of a score increase. It is
deliberately narrower than exposing more raw tables, changing the benchmark contract, or building
a new recommender framework.

## What the successful run actually established

The completed campaign proved the execution and finalization system, but it did not produce a
better scientific candidate. The qualified public-validation incumbent remains official FM seed 4:
GAUC `0.6679478288`, nDCG@5 `0.5361263752`, primary `0.6020370722`. The run admitted three
generated implementations and completed four inner evaluations without execution failures; see the
local [final report](../../runs/stable-1h-20260830-attempt-17/final/report.md).

The important diagnosis is hidden by the selected-score summary. On the train-derived Fold-B
screen, the trusted FM control scored primary `0.5754240304`. The three generated candidates' raw
rankings scored approximately `0.5619543791`, `0.5676510632`, and `0.5655431449`. Every five-point
fusion sweep selected weights `(0.0, 1.0)`, meaning **zero generated-candidate weight and 100% FM
control**. Candidate 1 later scored `0.6071290374` on Fold A, but its Fold-B regression correctly
prevented promotion. These values are recorded in the four local
[`production/scientific-records`](../../runs/stable-1h-20260830-attempt-17/production/scientific-records/)
receipts.

The shared representation explains the repeated result. All generated candidates consumed the
same 33-column dense matrix:

- one global prior;
- for each of user, video, author, tab, duration bucket, user-author, user-tab, author-tab, and
  user-duration bucket: cumulative exposure, cumulative positive count, and smoothed long-view
  rate (`9 * 3 = 27` columns);
- duration seconds, `log1p(duration)`, an 18-second threshold flag, date offset, and numeric tab.

Those are useful causal aggregate features, but they are not a replacement for the official FM's
high-cardinality user/item/categorical interactions. The first two generated branches repeated a
pairwise FM over these dense aggregates; the third used group-centered logistic regression. The
weakness is therefore primarily **representation and complementarity**, not a missing fourth
variation of the same optimizer.

## Findings from primary sources and their bounded consequences

### 1. Preserve the official FM; do not rebuild its role from dense aggregates

The original Factorization Machines work models variable interactions using factorized parameters
and was designed to estimate interactions under the high sparsity typical of recommenders. The
author's official [libFM site](https://www.libfm.org/) describes FMs as combining general feature
engineering with factorization models for large-domain categorical interactions; the original
paper is Rendle, [*Factorization Machines*](https://doi.org/10.1109/ICDM.2010.127).

The current 33 features are mostly dense counts and rates. An FM over those columns learns
interactions among aggregates, but it cannot reconstruct absent user-ID/video-ID latent
interactions. The safe iterative consequence is:

- keep the qualified official FM scores as one fusion member;
- train new candidates as **complementary rankers** over the causal aggregate matrix;
- do not expose new raw identifiers or reimplement the organizer FM unless a later, separately
  reviewed experiment proves that the existing FM cannot supply the needed signal.

This also avoids an unjustified leap to a larger neural model. A major RecSys reproducibility study
found that 11 of 12 reproducible neural approaches were beaten by simpler methods, underscoring the
need for strong, tuned controls rather than assuming model complexity is progress: Ferrari Dacrema
et al., [*A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research*](https://arxiv.org/abs/1911.07698).

If a later experiment truly needs a new categorical interaction backbone, compare a small
field-weighted FM rather than jumping directly to a full field-aware FM. The original FFM assigns a
different latent vector to a feature depending on the other feature's field, while FwFM was proposed
as a substantially more memory-efficient way to model field-pair strength: Juan et al.,
[*Field-aware Factorization Machines for CTR Prediction*](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf),
and Pan et al., [*Field-weighted Factorization Machines for Click-Through Rate Prediction in Display
Advertising*](https://arxiv.org/abs/1806.03514). That remains lower priority because the qualified
official FM already supplies categorical interactions without changing the candidate input seam.

### 2. Make deterministic LambdaRank the next model, not another pairwise FM

The 33 features contain nonlinear structure that a tree ranker can exploit directly: smoothed rates
at multiple scopes, raw counts spanning large ranges, duration and its 18-second target boundary,
date, and cross-scope context. The LightGBM paper reports efficient histogram-based gradient
boosting and strong speed/accuracy scaling; see Ke et al.,
[*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).

For the top-five component, LambdaRank/LambdaLoss is materially better aligned than another
pointwise classifier. The LambdaLoss paper connects metric-weighted pairwise losses to ranking
metrics and evaluates a truncated form specifically for NDCG@5: Wang et al.,
[*The LambdaLoss Framework for Ranking Metric Optimization*](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/).
LightGBM's official parameters state that `lambdarank_truncation_level` should generally be a little
higher than the target cutoff (for example, `k + 3`) and that `lambdarank_norm=true` helps with
unbalanced query sizes: [LightGBM 4.7 parameters](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#lambdarank-truncation-level).

The repository already has the required guarded adapter and method card. The first experiment
should therefore use the existing fixed semantics, not generated ranking infrastructure:

```text
objective = lambdarank
label_gain = [0, 1]
eval_at = [5]
lambdarank_truncation_level in {5, 8}
lambdarank_norm = true
device_type = cpu
deterministic = true
force_col_wise = true
```

Only a small inner-fold grid is warranted: trees `{150, 300}`, learning rate `{0.03, 0.05}`, leaves
`{15, 31}`, minimum rows per leaf `{50, 200}`, and truncation `{5, 8}`. Use early stopping only on
the train-derived fold and retrain with the frozen tree count. The exact organizer scorer, not
LightGBM's internal metric, must decide every comparison. In particular, the tag-pinned LightGBM
source gives an all-negative query perfect internal NDCG whereas this benchmark assigns it zero;
see [`rank_metric.hpp`](https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/src/metric/rank_metric.hpp#L56-L138).

### 3. Retain the GAUC sampler, but stop treating it as a new hypothesis

BPR derives a pairwise `log sigmoid(score_positive - score_negative)` objective and learns by
randomly sampled user-positive-negative triples: Rendle et al.,
[*BPR: Bayesian Personalized Ranking from Implicit Feedback*](https://www.auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf).

For this benchmark, the correct candidate universe is the user's **logged impressions**, not
unobserved catalog items. Because organizer GAUC weights user AUC by positive count, an unbiased
surrogate sampler is:

1. select uniformly among positive impressions belonging to mixed-label users;
2. select one negative uniformly from the same user's logged negative impressions.

That makes each user contribute in proportion to its positive count while averaging over its
negative comparisons. The completed run's pairwise candidates were already close to this design
and still lost on Fold B. Consequently:

- keep this sampler as a tested primitive for a GAUC specialist;
- do not allow the agent to propose another pairwise-FM branch until either the representation or
  base model changes;
- compare a pairwise objective against a pointwise objective on the **same** model and features, so
  an objective claim is attributable.

Direct group-AUC optimization is a legitimate later control, not a reason to improvise a new loss
now. PDAOM constructs hard positive-negative pairs inside user-ID sub-batches and reports lower
objective complexity with AUC/GAUC gains: Zeng et al.,
[*Personalized and Differentiable AUC Optimization for Network Quality of Experience*](https://arxiv.org/abs/2304.09176).
Its relevant lesson here is bounded hard-pair emphasis within users; its reported gains are not
evidence that the method will beat this benchmark's qualified FM.

### 4. Fuse complementary within-user ranks; calibration alone cannot help

GAUC and nDCG@5 depend on ordering within each user. A strictly monotone calibration of one model's
scores cannot change either metric. Platt scaling, isotonic regression, or temperature scaling is
therefore not an improvement hypothesis by itself here.

Calibration becomes useful only to make scores from different models comparable. The safer existing
mechanism is stronger: convert each member to deterministic within-user midrank percentiles, then
test the already frozen grid
`(1,0), (0.75,0.25), (0.5,0.5), (0.25,0.75), (0,1)`. Rank fusion is a well-established way to
combine systems with incompatible score scales; the primary RRF work is Cormack et al.,
[*Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

For this campaign, the actionable rule is stricter than “ensemble two models”:

- fuse only if the generated model has rank complementarity with official FM;
- select exactly one weight on Fold B, freeze it, and confirm it once on Fold A;
- reject the blend if the selected weight is `(0,1)` or if either component collapses beyond the
  existing fold tolerance;
- spend at most one protected outer query on the surviving frozen blend.

### 5. Add recency as a companion, never as a replacement

KuaiRand contains timestamps and rich sequential feedback, but the official repository explicitly
warns that KuaiRand-Pure has incomplete sequential logs and recommends 27K/1K when rigorous long
sequences are required: [official KuaiRand README](https://github.com/chongminggao/KuaiRand#which-version-should-i-use).
The dataset paper also emphasizes exposure bias and the separation of random interventions from
standard recommendation logs: Gao et al.,
[*KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos*](https://doi.org/10.1145/3511808.3557624).

Temporal dynamics can still matter over the benchmark's date split. Koren's primary work shows that
preferences and popularity drift, but also warns that naive time windows or instance decay discard
too much long-term signal: [*Collaborative Filtering with Temporal Dynamics*](https://doi.org/10.1145/1557019.1557072).

Therefore the smallest defensible recency experiment is to **retain all current cumulative
features** and add strict-past companion statistics only for the highest-value scopes:

- user decayed exposure/rate;
- video decayed exposure/rate;
- user-author decayed exposure/rate.

Use one predeclared half-life grid such as `{1, 3, 7}` days, update state only after simultaneous
timestamp buckets, and freeze query state at the training cutoff exactly as today. Do not add the
random-intervention log, month-level video outcome statistics, mutable snapshots, current-row
outcomes, or public/final labels. The official schema confirms that `long_view` itself is determined
from current-row play time and the 18-second duration rule, so current-row `play_time_ms` must never
become an inference feature: [official log-field definitions](https://github.com/chongminggao/KuaiRand#1%EF%B8%8F%E2%83%A3-description-of-the-fields-in-log_xxxcsv).

### 6. Neural crosses are a later controlled ablation

DeepFM combines low-order FM interactions with a deep component for higher-order interactions:
Guo et al., [*DeepFM*](https://www.ijcai.org/Proceedings/2017/239). DCN-V2 explicitly constructs
bounded-degree feature crosses and offers low-rank variants for better cost/quality tradeoffs:
Wang et al., [*DCN V2*](https://arxiv.org/abs/2008.13535).

Those sources justify a later compact-cross experiment, but not making it the next step. The current
matrix has only 33 dense engineered columns, the CPU tree branch already supplies nonlinear
interactions, and the incumbent's sparse categorical interaction signal is safely available through
fusion. A neural candidate should proceed only if it beats a same-input pointwise control on both
inner folds under the existing parameter, throughput, checkpoint, and replay ceilings.

## Minimal autonomous-agent changes that should improve the science

These are input/search-policy changes, not a redesign of the execution system.

### Report raw candidate evidence separately from selected fusion evidence

The last run repeatedly surfaced the control-selected primary `0.5754240304` after the fusion grid,
even when the generated raw candidate was materially worse. A research model that sees only the
selected score cannot learn what failed. Each subsequent proposal should receive:

```text
raw candidate: GAUC, nDCG@5, primary
FM control: GAUC, nDCG@5, primary
each fusion grid point: weights and all three metrics
selected fusion weights and reason
Fold A / Fold B identity
runtime, peak RSS, and replay result
```

No row-level labels, predictions, residuals, or worst-user examples are needed.

### Add a mechanism-family ledger and an explicit novelty gate

Record a compact fingerprint for every completed branch:

```text
representation | model family | objective | temporal policy | fusion member | result
```

The next proposal must differ in at least one substantive field. Close `pairwise FM + unchanged 33
features` after its two completed failures. The initial queue should be deterministic:

1. LambdaRank on unchanged features;
2. frozen FM/LambdaRank rank fusion;
3. recency companions plus the same LambdaRank control;
4. one GAUC-specialist objective on a changed representation;
5. compact DCN-V2/DeepFM only after the cheaper families plateau.

This avoids spending provider calls on semantically duplicated proposals while leaving the model
free to implement and explain one bounded mechanism at a time.

### Give the model the exact 33 feature names and transformations

The runtime contract correctly says that feature order is authoritative, but the proposal should
also receive a short semantic table: which columns are raw counts, smoothed rates, static numeric
features, or threshold flags; the smoothing constant; and which state is cumulative versus frozen.
This lets it choose a model appropriate to the data without exposing values or protected labels.

## Recommended experimental ladder

| Stage | One attributable change | Inner evidence required | Stop condition |
| --- | --- | --- | --- |
| 0 | Reproduce official FM and both fold controls | Exact existing receipts | Any identity or replay mismatch |
| 1 | Deterministic LambdaRank on current 33 features | Fold B raw candidate and five fusion points | Close if raw and every nonzero-candidate blend lose |
| 2 | Freeze one blend and confirm | Same weights on Fold A; report both component metrics | Close on fold brittleness or component collapse |
| 3 | Add six strict-past recency companions | Repeat Stages 1-2 against unchanged-feature tree | Close if mean gain is below epsilon |
| 4 | GAUC specialist on changed representation | Same-model pointwise versus pairwise ablation | Close if objective change alone is not positive |
| 5 | Compact cross model | Same-input simple control, two folds, fixed CPU ceilings | Close on no consistent advantage |
| 6 | Protected outer promotion | One fully frozen survivor only | Never tune from outer feedback |

For every stage, report mean and worst-fold deltas for GAUC, nDCG@5, and primary; use identical
seeds and folds for challenger/control comparisons; retain exact checkpoint/prediction replay; and
keep the official FM immutable. If the project-wide outer-query ledger is already exhausted, do not
reset or bypass it—continue with inner evidence and produce a final submission without further
adaptive public-validation tuning. Repeated adaptive holdout reuse can overfit the holdout, as
formalized by Dwork et al.,
[*Generalization in Adaptive Data Analysis and Holdout Reuse*](https://arxiv.org/abs/1506.02629).

## Bottom line

The strongest immediate score opportunity is not “let the model invent more code.” It is to expose
the autonomous researcher to truthful component-level evidence and force the next science through
the already guarded sequence:

```text
official sparse-interaction FM
        +
nonlinear grouped tree over causal aggregates
        +
one inner-selected within-user rank fusion
```

Only after that controlled complement fails should the campaign add a small amount of new causal
signal. This preserves the now-stable runtime, makes each score change attributable, and directly
addresses why all three previous autonomous candidates lost.

# Score-improvement research for KuaiRand-Pure long-view ranking

**Research snapshot:** 2026-08-30

**Benchmark:** within-user ranking of logged KuaiRand-Pure impressions

**Target:** binary long_view

**Primary score:** mean of organizer GAUC and organizer nDCG@5

**Status:** evidence-backed research plan, not a score-improvement claim

## Executive conclusion

The recent score plateau is not evidence that the repository lacks enough code, features, model
classes, or search effort. The current working tree now contains a broad 95-column causal
feature matrix, strict-past aggregates, recency histories, four protected ranking specialists,
several neural ranking primitives, label-free rank fusion, rolling folds, materiality gates, and
paired user-bootstrap code. The campaign records also show that many increasingly complicated
families have produced small inner-fold gains but no promoted outer candidate. Three recent
campaigns each ended at the same immutable FM fallback after roughly 11–13 minutes, with zero outer
challenger evaluations. More code of the same kind is therefore unlikely, by itself, to change the
score.

The strongest diagnosis is **supervision and objective saturation**:

1. Most recent candidates re-express the same binary long-view labels and the same correlated
   historical aggregates. They change representation more than information.
2. Several reported improvements are real-looking but smaller than the repository's strict
   0.002 primary-delta materiality gate, or are inconsistent across the two rolling folds.
3. The benchmark combines two different functionals: positive-count-weighted all-pair ordering
   through GAUC and equal-user, top-five ordering through nDCG@5. No single standard loss exactly
   optimizes their arithmetic mean.
4. The densest scientifically justified new supervision, same-impression watch progress and
   selected auxiliary responses during training, is not exposed through the current generated
   candidate contract. This is an architectural capability gap, not a missing model suggestion.
5. Generic inverse-propensity or counterfactual learning-to-rank is **not identified from the
   currently exposed standard log**. The repository has no logged rank positions, display
   propensities, full candidate sets, or action probabilities. Applying IPS merely because
   KuaiRand also publishes a randomized log would target a different problem and can increase
   variance or bias without helping the organizer score.

The highest-value next work is a small, preregistered experiment ladder:

1. **Lock a diagnostic gate** that reproduces controls and measures where headroom exists by user,
   duration regime, tab, slate size, positive count, and repeated user-video contradictions.
2. **Test duration-conditioned logged pair construction** against the exact uniform positive-ticket
   control. This is the lowest-cost intervention that changes optimization examples while staying
   aligned with the existing GAUC specialist.
3. **Version the trusted runtime contract and test train-only watch-progress multi-task learning.**
   Start with a same-capacity shared-bottom network, not MMoE or PLE. Predict only the long-view head
   at validation and inference.
4. **Run one controlled top-five loss ablation** on an unchanged representation: the current
   positive-only list loss, canonical ListNet, and a corrected LambdaLoss at cutoff five.
5. **Fuse only demonstrably complementary specialists**, using the existing within-user percentile
   normalization and frozen out-of-fold weights. Calibration alone cannot change either metric.

None of these mechanisms guarantees a gain. Each is valuable because it tests a distinct scientific
hypothesis, preserves the organizer evaluator, and can be falsified without another broad,
seven-hour architecture sweep.

## 1. What the scorer actually rewards

The authoritative local definition is the untouched
[organizer evaluator](../../kuairand-starter-kit/evaluate.py), not a library metric. It:

- ranks only the impressions already logged for a user;
- uses binary long_view relevance;
- computes nDCG at five for every user, assigning zero to all-negative users;
- computes AUC only for mixed-label users and weights each eligible user's AUC by that user's
  positive count; and
- reports the arithmetic mean of GAUC and nDCG@5.

Let user \(u\) have \(P_u\) positive and \(N_u\) negative logged impressions. For mixed-label
users, organizer GAUC is

\[
\mathrm{GAUC}
=
\frac{\sum_u P_u\,\mathrm{AUC}_u}{\sum_u P_u}
=
\frac{1}{\sum_u P_u}
\sum_u \frac{1}{N_u}
\sum_{i:y_{ui}=1}\sum_{j:y_{uj}=0}
\ell_{\mathrm{order}}(s_{ui},s_{uj}),
\]

where the last expression uses an indicator, or half credit for a tie, when describing the metric
itself. This yields an important local result: sampling an eligible positive row uniformly across
all eligible positives, then sampling one logged negative uniformly from the same user, is the
Monte Carlo law that matches the organizer's positive-count-weighted GAUC. The protected
[pairwise FM specialist](../../candidate_seed/reference_pairwise_fm.py) is therefore unusually well
aligned with the GAUC component. A fashionable catalog-negative sampler would not be.

nDCG@5 has different weighting. Each user contributes equally; only the first five ranks matter.
All-negative users are unchangeably zero, all-positive users are unchangeably one, and only
mixed-label users have model-dependent ordering. Consequently:

- GAUC gives more influence to users with more positive tickets.
- nDCG@5 gives every user one vote and concentrates gradients near rank five.
- Improving many easy all-pair comparisons can raise GAUC without moving the top five.
- Improving a few low-positive users near the top can raise nDCG but contribute little to GAUC.
- A candidate should always report both components and their user-stratified deltas; primary alone
  hides the trade.

This also proves a useful negative result: any strictly monotone transformation of one model's
scores leaves both metrics unchanged, because both depend only on within-user order. Platt scaling,
isotonic calibration, temperature scaling, and score clipping cannot improve this score on their
own. Calibration is relevant only when it changes the order produced by combining multiple models.

## 2. Repository diagnosis: why the score can remain unchanged after substantial work

This section describes the current 2026-08-30 working-tree snapshot. Some files are concurrently
being developed, so these are present-state findings rather than claims about a released revision.

### 2.1 The implemented system is already broad

| Surface | Current evidence | Score implication |
| --- | --- | --- |
| Feature construction | The [v8 feature builder](../../src/kuairand_agent/campaign/pure_features.py) exposes 95 columns: the prior 83-column surface plus 12 input-only strict-past exposure, first-seen, and time-since-last-exposure features. | New information now comes from query-time exposure context rather than another transform of the same outcome histories. |
| Leakage policy | Outcome-bearing histories remain frozen at the training prefix. The new exposure family may advance only from strictly earlier query inputs and accepts no query outcomes. | This preserves the causal boundary while allowing legal query warm-up. |
| Generated-candidate contract | The [runtime contract](../../src/kuairand_agent/candidate_api/runtime_contract.py) exposes one binary training target handle, long_view. | True same-row auxiliary-label MTL is not currently an admissible generated-candidate experiment. It needs a trusted, versioned contract change. |
| Protected specialists | The [candidate seed package](../../candidate_seed/README.md) contains pairwise FM, native-categorical LambdaRank, user-balanced listwise, and query-balanced pointwise references. | The portfolio already spans pairwise, listwise, pointwise, and tree ranking. Merely renaming the loss family is not new evidence. |
| Fusion | [Fusion](../../src/kuairand_agent/candidates/fusion.py) converts scores to within-user midrank percentiles and selects a frozen weight on a 21-point grid. | Scale mismatch is already controlled. Repeatedly searching many candidates over the same grid can still overfit the screening fold. |
| Promotion | [Selector policy](../../src/kuairand_agent/campaign/selector.py) uses rolling Fold A/Fold B evidence, a 0.002 material primary delta, degradation limits, outer limits, and seed confirmation. | Sub-threshold inner wins are correctly not being described as reliable public-score gains. |
| Uncertainty | [Bootstrap support](../../src/kuairand_agent/candidates/bootstrap.py) already performs paired, whole-user resampling and reconstructs organizer components. | The correct primitive exists; it should be operationalized in every promotion record instead of adding another uncertainty library. |

The watch-progress history transformation is especially important. It clips

\[
\frac{\mathrm{play\_time\_ms}}
     {\max(\min(\mathrm{duration\_ms},18{,}000),1)}
\]

to the range from zero to two. The denominator is the nominal long-view threshold described by the
public schema: completion for videos at most 18 seconds and at least 18 seconds watched for longer
videos. The history feature is legal because it uses past impressions. The same-impression value
is outcome-derived and therefore is not legal as an inference feature. It can, however, be a dense
**training-only auxiliary target** if kept behind the trusted target capability and removed from
validation features and outputs. Importantly, it must not be used to reconstruct the primary
label: a read-only audit of the exact training member found 20,889 threshold/label disagreements
among 1,141,112 rows, or about 1.83 percent.

### 2.2 The campaign evidence is a saturation signal, not a reason to lower standards

The [Attempt 18](../../runs/improvement-20260830-attempt-18/final/report.md),
[Attempt 20](../../runs/improvement-20260830-attempt-20/final/report.md), and
[Attempt 21](../../runs/improvement-20260830-attempt-21/final/report.md) reports all retain:

- five-seed official-FM mean GAUC 0.6674002647;
- five-seed official-FM mean nDCG@5 0.5357441068;
- five-seed official-FM mean primary 0.6015721679; and
- seed-4 fallback primary 0.6020370722.

The three campaigns respectively completed six, two, and five inner evaluations, completed zero
outer challenger evaluations, and took about 687, 651, and 760 seconds. The result is not that
evaluation failed. The result is that no generated challenger survived the deliberately strict
promotion boundary.

Attempt 20 illustrates the difference between screening evidence and a score improvement. Its
[scientific record](../../runs/improvement-20260830-attempt-20/production/scientific-records/abb480995fd8e9619bae882d186036c579723816bdaef3fd62d959d4a1f4deba.json)
records an inner-fold blend primary of 0.5769056082 versus control 0.5754240304, a delta of about
0.001482. That is an interesting hypothesis, but it is below the 0.002 materiality rule and was not
outer-evaluated. Calling it a score gain would turn model selection noise into a result.

The method-card registry in
[full campaign runtime](../../src/kuairand_agent/campaign/full_campaign_runtime.py) tells the same
story across a much larger family:

- 1/3/7-day recency signals were directionally positive but non-material;
- 33-, 39-, 44-, 56-, 69-, and 82-feature ranking families largely plateaued;
- click and watch-history candidates did not clear the materiality gate;
- pairwise FM, native-categorical LambdaRank, ListNet-like, pointwise, deep-cross, graph, attention,
  manifold, temporal-residual, and video-type variants often produced deltas in roughly the
  0.0002–0.0019 range;
- one train-only pointwise-plus-video-type blend came within roughly 0.000012 of the 0.002 rule on
  Fold A while barely crossing it on Fold B, exactly the kind of boundary result that needs frozen
  confirmation rather than a relaxed threshold.

The correct response is therefore to increase **experimental distinctness and evidence quality**.
Lowering the materiality threshold, adding more feature horizons, or running more weight-grid
searches would make promotion easier without making the result more likely to survive.

## 3. Dataset facts and causal boundary

The [KuaiRand dataset paper](https://doi.org/10.1145/3511808.3557624), its
[arXiv version](https://arxiv.org/abs/2208.08696), and the
[official repository](https://github.com/chongminggao/KuaiRand) are the primary dataset sources.
They document both standard-policy and randomized-intervention logs, multiple response signals, and
the stated construction of long_view from watch time and video duration. The local raw-log
disagreements reported above mean that the supplied long_view column, not a reconstruction, must
remain authoritative. The official repository reports about 27,285 users, 7,551 standard-log
items, and 1,436,609 standard interactions for Pure, while also warning that Pure keeps
candidate-pool items and does not preserve complete sequences.

The local benchmark uses standard-log dates:

- training: 2022-04-08 through 2022-04-21;
- public validation: 2022-04-22 through 2022-04-28; and
- protected/final period: 2022-04-29 through 2022-05-08.

The separate Pure randomized log covers 2022-04-22 through 2022-05-08. It overlaps both public
validation and the protected final period and is not part of the current candidate capability.
Therefore it must not be silently parsed, joined, used for histories, or used for hyperparameter
selection. Even a diagnostics-only use requires explicit authorization and a loader that masks
forbidden dates **before outcomes are parsed**.

The standard data exposed to the current model also lacks the variables that standard
counterfactual learning-to-rank normally needs:

- the rank position at which each impression was shown;
- a known or consistently estimated examination propensity;
- a logged action probability for a policy;
- the full candidate set from which each shown item was selected; and
- overlap evidence showing that relevant actions have nonzero support.

The tab field is context, not a propensity. KuaiRand documents different interaction semantics for
different tabs and layouts; treating tab as if it were only a randomized observation attribute
would conflate user interface, exposure policy, and label-generation differences.

## 4. Debiasing logged implicit feedback and counterfactual learning-to-rank

### 4.1 What the foundational methods require

Joachims et al.'s
[Unbiased Learning-to-Rank with Biased Feedback](https://doi.org/10.1145/3018661.3018699)
([arXiv](https://arxiv.org/abs/1608.04468);
[authors' implementation](https://www.cs.cornell.edu/people/tj/svm_light/svm_proprank.html))
uses inverse propensity weighting to correct position-biased clicks. The method assumes an
examination model, position information, estimable propensities, and support. Those assumptions are
not optional implementation details: without them, the weighted empirical risk does not identify
the intended relevance risk.

Hu et al.'s
[Unbiased LambdaMART](https://doi.org/10.1145/3308558.3313447)
([arXiv](https://arxiv.org/abs/1809.05818);
[source repository](https://github.com/acbull/Unbiased_LambdaMart)) jointly estimates position
bias and a ranking model using click logs. It still needs displayed positions and a click
observation model. The current impression rows do not provide those inputs.

Fang et al.'s
[Intervention Harvesting for Context-Dependent Examination-Bias Estimation](https://doi.org/10.1145/3394486.3403285)
extends propensity modeling to item attributes under interventions. It does not justify
substituting a heterogeneous scenario field for randomized position swaps. No valid intervention
contrast has been established for tab in this benchmark.

In recommendation, Schnabel et al.'s
[Recommendations as Treatments](https://proceedings.mlr.press/v48/schnabel16.html)
([arXiv](https://arxiv.org/abs/1602.05352)) and Saito et al.'s
[Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback](https://doi.org/10.1145/3336191.3371783)
([arXiv](https://arxiv.org/abs/1909.03601)) correct the difference between observed interactions
and a full user-item relevance domain. Saito et al.'s
[Unbiased Pairwise Learning from Biased Implicit Feedback](https://doi.org/10.1145/3409256.3409812)
([source](https://github.com/usaito/unbiased-pairwise-rec)) addresses a related pairwise problem.
The hackathon evaluator does **not** ask for full-catalog relevance. It asks the model to rerank
already exposed rows, and its negatives are observed impressions with long_view equal to zero.
Treating those rows as missing/unlabeled catalog data would change the estimand.

Counterfactual Risk Minimization and POEM by Swaminathan and Joachims
([PMLR](https://proceedings.mlr.press/v37/swaminathan15.html);
[arXiv](https://arxiv.org/abs/1502.02362)) optimize a policy from logged bandit feedback using
known propensities and variance control. Dudík et al.'s
[Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)
([ICML paper](https://icml.cc/2011/papers/554_icmlpaper.pdf)) combines a reward model with
importance weighting. Both still require a well-defined action, logging probability, support, and
reward. A doubly robust formula is not robust to the absence of the logging probability itself.

Wang et al.'s
[Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random](https://proceedings.mlr.press/v97/wang19n.html)
likewise targets MNAR full-domain recommendation, not fixed exposed-row reranking.

### 4.2 What is conditionally relevant

Bonner and Vasile's
[Causal Embeddings for Recommendation](https://doi.org/10.1145/3240323.3240360)
([arXiv](https://arxiv.org/abs/1706.07639);
[source](https://github.com/criteo-research/CausE)) regularizes representations learned from biased
data toward representations learned from randomized exposure. This is the most structurally
relevant published use of KuaiRand's two policy regimes. It remains blocked here for three reasons:

1. the randomized log overlaps validation and protected dates;
2. the organizer has not established it as allowed training or tuning input; and
3. it optimizes transfer across exposure regimes, whereas the score is defined on the standard
   logged-impression regime.

If authorization later exists, the first use should be a **date-safe conditional diagnostic**, not
an immediate production feature: train representation probes only on rows that are legally before
the relevant cutoff; hold the standard evaluator fixed; report covariate balance, support, and
standard-log score separately; and never tune on randomized outcomes from public or protected
dates.

### 4.3 Decision for this repository

| Method family | Required precondition | Present now? | Decision |
| --- | --- | --- | --- |
| Position-based IPS / Unbiased LambdaMART | Logged positions and examination propensities | No | Do not implement. |
| Attribute propensity | Valid randomized or natural interventions identifying observation probabilities | Not established | Do not treat tab as a propensity. |
| CRM / POEM / doubly robust policy learning | Actions, logging probabilities, overlap, policy reward | No | Do not implement. |
| MNAR full-catalog correction / UBPR | A target domain containing unobserved user-item relevance | Different estimand | Do not replace exposed negatives with catalog negatives. |
| CausE-style biased/random transfer | Authorized randomized log, safe dates, overlap, matching target | Blocked | Conditional research branch only. |
| Exposure-aware descriptive diagnostics | Context strata and legal standard rows | Yes | Use for error analysis, not causal claims. |

The practical conclusion is deliberately conservative: **debiasing is not a synonym for adding
weights**. Until the observation process is identified, generic IPS is more likely to optimize an
invented objective than the organizer score.

## 5. Duration and watch-time modeling

Duration is central here because it both affects user behavior and enters the nominal threshold
definition of long_view. A method that predicts raw watch time can therefore learn useful
preference structure, but it can also overfit duration mechanics or redefine the supplied label.

### 5.1 Dense watch progress as train-only auxiliary supervision

The current binary label discards distance to the threshold. A negative with progress 0.98 and a
negative with progress 0.02 receive the same target, as do positives at progress 1.01 and 2.00. A
clipped progress auxiliary head can preserve that ordering information during representation
learning while leaving the inference target unchanged.

This proposal has strict boundaries:

- same-row play time and progress are targets only, never candidate features;
- validation and query capabilities expose neither value;
- the emitted prediction is the long-view head only;
- duration remains a legal pre-impression feature;
- the auxiliary target is computed in the trusted controller using the official threshold;
- clipping and any rows where raw play time exceeds nominal duration are recorded, not silently
  discarded; and
- an equal-capacity single-task model is the control.

A read-only audit of the April 8–21 standard training member found 193,736 rows with play time
above nominal duration, 20,889 disagreements between the nominal threshold rule and the supplied
long_view label, and 1,609 long-view-positive/click-negative rows. Those observations make three
shortcuts unsafe: a literal right-censoring assumption needs a replay/measurement-error indicator,
the supplied long_view label cannot be replaced by a threshold reconstruction, and long_view
should not be factored globally as click times a conditional probability. The audit should be
reproduced as a versioned artifact before implementation.

### 5.2 Duration-aware primary sources

[D2Q: Duration-Deconfounded Quantile-Based Recommendation](https://doi.org/10.1145/3534678.3539092)
([arXiv](https://arxiv.org/abs/2206.06003)) maps watch time to within-duration-group quantiles to
reduce direct duration preference. That mechanism is useful as an auxiliary target or diagnostic.
It should not replace long_view, because duration is part of the official label definition and the
organizer never asks for duration-independent utility.

[DVR: Micro-video Recommendation Optimizing Watch-Time-Gain under Duration Bias](https://doi.org/10.1145/3503161.3548428)
([arXiv](https://arxiv.org/abs/2208.05190);
[source](https://github.com/tsinghua-fib-lab/WTG-DVR)) models watch-time gain and adversarially
reduces duration information. It provides a useful falsification diagnostic: if an auxiliary
representation merely recovers duration, an adversarial or grouped probe should reveal it. Its
watch-time-gain objective must not replace the organizer long-view score.

[Counterfactual Watch-time Estimation for Video Recommendation](https://doi.org/10.1145/3637528.3671817)
([arXiv](https://arxiv.org/abs/2406.07932);
[source](https://github.com/hyz20/CWM)) treats observed watch time as censored by video duration
and estimates latent counterfactual watch time. This is scientifically relevant but higher risk:
the local data contain play-time values above duration, the official label clips the effective
threshold at 18 seconds, and the paper's target is not GAUC/nDCG on binary long_view. If tested,
port one isolated auxiliary likelihood, cap the censoring term consistently, add a measurement
replay indicator, and keep the binary head and scorer authoritative.

[D2Co: Counterfactual Duration Deconfounding for Video Recommendation](https://doi.org/10.1145/3604915.3608797)
([arXiv](https://arxiv.org/abs/2308.08120);
[source](https://github.com/hyz20/D2Co)) is another duration-deconfounding reference, but is lower
priority for the same target-mismatch reason.

[VLDRec: Mitigating Video-Length Effect for Micro-video Recommendation](https://doi.org/10.1145/3617826)
([arXiv](https://arxiv.org/abs/2308.14276);
[authors' PDF](https://fi.ee.tsinghua.edu.cn/~gaochen/papers/TOIS2023-VideoLength.pdf)) adds
same-duration-group pairwise comparisons to more general pairwise training. This maps cleanly to
the current exposed-row benchmark: retain the exact uniform positive-ticket sampler, and add a
fixed proportion of logged negatives from the same duration bucket. It does not require treating
unexposed catalog items as negatives.

### 5.3 Recommended first duration experiment

Use the repository's already established duration boundaries around 5, 10, 18, 30, and 60 seconds,
or another single preregistered boundary set; do not search many bucketings.

- **Control:** protected pairwise FM with uniform positive-ticket, uniform same-user logged-negative
  sampling.
- **Challenger:** the identical model, optimizer, seed set, pair budget, and epochs, but a fixed
  mixture of uniform pairs and same-duration-bucket logged pairs.
- **Provenance:** implement the sampler in trusted code, not as an unverifiable candidate-side
  recreation of the protected sampler.
- **Required diagnostics:** same-bucket coverage by user, fraction of duplicate pairs, duration
  balance, positive/negative threshold distance, per-duration GAUC and nDCG, and effective pair
  weights.
- **Falsification:** a duration-shuffled bucket control should erase any mechanism-specific benefit.
- **Stop rule:** stop if coverage is too sparse, the result depends on one duration band or one
  fold, or nDCG gains are paid for by a larger GAUC loss.

This is the cheapest high-information experiment because it changes which ordering constraints are
learned while preserving the strongest existing metric-aligned control.

## 6. Multi-task learning

The dataset exposes is_click and other same-impression outcomes plus continuous play time, while
the current candidate contract exposes only long_view. Multi-task learning is therefore the
largest plausible source of **new training information**, but only after a capability change.

### 6.1 What the primary literature supports

[Entire Space Multi-Task Model](https://doi.org/10.1145/3209978.3210104)
([arXiv](https://arxiv.org/abs/1804.07931)) jointly estimates click-through and post-click
conversion in an entire-space setting. Its key lesson is to respect structural observability and
sample-selection boundaries. It does not establish a universal click-to-long-view chain here:
KuaiRand's click definition depends on the UI scenario, and local rows include a small number of
long-view-positive/click-negative cases. A direct long-view head must remain.

[Multi-gate Mixture-of-Experts](https://doi.org/10.1145/3219819.3220007)
([Google Research](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/))
uses shared experts with a task-specific gate and tower, allowing different tasks to select
different mixtures. It is a reasonable escalation after a shared-bottom model demonstrates
negative transfer.

[Progressive Layered Extraction](https://doi.org/10.1145/3383313.3412236) separates shared and
task-specific experts through progressive routing to mitigate the seesaw effect. It adds a larger
tuning and capacity surface, so it is justified only if a controlled shared-bottom or MMoE run
shows a reproducible task conflict.

[Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
projects away conflicting task-gradient components, while
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) adapts task weights to balance training
rates. Their most useful immediate contribution here is diagnostic: measure head-gradient cosine,
norm, and learning rate before adding another expert architecture. Either method should be a
separate, controlled follow-up rather than silently bundled into the first MTL run.

Zhao et al.'s
[Recommending What Video to Watch Next: A Multitask Ranking System](https://doi.org/10.1145/3298689.3346997)
([Google Research](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/))
shows why multiple user-response and satisfaction objectives can improve video ranking
representations. Its production setting also highlights the need to control objective weights and
avoid letting proxy tasks dominate the final ranking target.

### 6.2 Required capability design

The current runtime contract should not be bypassed. A versioned contract extension should:

1. expose explicitly allowlisted train-only auxiliary target handles;
2. attach row identity and schema digests to every target;
3. make auxiliary target access impossible for validation/query/final rows;
4. prohibit target arrays from entering the feature matrix or saved inference state;
5. require the candidate to return exactly one long-view score per query row;
6. record which auxiliary columns and transformations were used; and
7. leave all organizer scoring in the trusted controller.

This is a security and validity boundary. Implementing multi-task learning by concatenating play
time to features would create target leakage and a meaningless score.

### 6.3 Controlled model ladder

Do not begin with PLE. Use this progression:

1. **Single-task control:** a compact neural model with the same shared representation capacity and
   long-view head.
2. **Shared-bottom, two heads:** binary long_view plus clipped watch progress. Choose a very small,
   preregistered auxiliary-weight set on train-internal folds only.
3. **Optional click head:** add is_click only if tab-specific observability masks and contradictions
   are documented; keep the direct long-view head.
4. **MMoE:** only if gradient cosine, per-head loss, or fold evidence shows reproducible negative
   transfer under the shared bottom.
5. **PLE:** only if MMoE still has a measured seesaw effect.
6. **Research-only duration auxiliaries:** D2Q, CWM, or watch-time-gain targets only after simple
   progress establishes that dense supervision helps.

Keep optimizer, batch construction, representation width, seed set, and training budget as equal
as possible. Record head losses, gradient norms and cosine similarity, early-stopping head, and
per-task calibration. Promotion still depends only on organizer long-view GAUC, nDCG@5, and
primary.

An important caveat is that progress is tightly coupled to the nominal label rule but does not
exactly reproduce the supplied training labels: the measured disagreement rate is about 1.83
percent. The benefit, if any, can come from a finer regularization or denoising signal during
representation learning, not from new inference-time information. If the shared model merely
learns the threshold mechanics but produces the same order, the offline score will not move.

## 7. Pairwise, listwise, and top-five ranking losses

### 7.1 Pairwise alignment and limits

[RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
uses a logistic pairwise probability to learn ordered document pairs. Gao and Zhou's
[AUC consistency analysis](https://www.ijcai.org/Proceedings/15/Papers/137.pdf) gives theoretical
conditions under which logistic and exponential pairwise surrogates are AUC-consistent and
explains why an arbitrary pairwise loss is not automatically appropriate.

[Bayesian Personalized Ranking](https://arxiv.org/abs/1205.2618)
([UAI paper](https://auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf))
learns preferences from observed-positive versus unobserved user-item pairs. Its classic sampler
does not match this benchmark because the organizer negatives are **observed logged impressions**.
The local positive-ticket pairwise sampler is closer to RankNet/AUC risk than to catalog BPR.

Burges's
[From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
explains metric-weighted pair gradients. The current protected tree ranker already uses
LightGBM LambdaRank with binary gains, evaluation at five, and truncation level eight. LightGBM's
[official parameter documentation](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Parameters.rst)
recommends a truncation level near the target cutoff plus a small margin, so eight is already a
defensible setting for nDCG@5. Large truncation sweeps are low priority. LightGBM's built-in metric
conventions can differ for all-negative queries; the organizer scorer must remain authoritative.

### 7.2 Listwise objectives

[ListNet](https://doi.org/10.1145/1273496.1273513)
([ICML archive](https://mlanthology.org/icml/2007/cao2007icml-learning/)) defines a listwise
probability distribution over rankings and a top-one approximation. The current protected
listwise specialist uses positive-only softmax cross-entropy over each query. That is a useful
listwise surrogate, but it is not the canonical label-softmax ListNet definition. A controlled
comparison should name the losses precisely.

[The LambdaLoss Framework](https://doi.org/10.1145/3269206.3271784)
([Google Research](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/))
puts LambdaRank-style metric weighting into a probabilistic loss framework. Jagerman et al.'s
[On Optimizing Top-K Metrics for Neural Ranking Models](https://research.google/pubs/on-optimizing-top-k-metrics-for-neural-ranking-models/)
([paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/6613.pdf)) identifies and
corrects issues in LambdaLoss variants for top-k metrics. This is a better justified nDCG@5
experiment than another architecture change.

The recommended experiment holds the representation, batches, seeds, and compute fixed and compares:

- current positive-only list softmax;
- canonical top-one ListNet; and
- corrected LambdaLoss at cutoff five.

Use the organizer scorer for evaluation, not the training library's nDCG. Because the existing
listwise and LambdaRank families have already shown sub-material gains, this is a medium-priority
mechanistic ablation, not the presumed solution.

[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) is not itself a new
ranking hypothesis. LightGBM's alternative rank_xendcg objective is described in
[XENDCG](https://arxiv.org/abs/1911.09798) and could be a cheap tree-loss check, but it ranks below
the corrected neural top-five comparison because the current LambdaRank control is already strong.

## 8. Hard-negative sampling

Harder logged negatives can concentrate training on mistakes near the decision boundary or top
ranks. They can also change the risk being optimized.

Shi et al.'s analysis,
[On the Theories Behind Hard Negative Sampling for Recommendation](https://doi.org/10.1145/3543507.3583223)
([authors' PDF](https://jiawei-chen.github.io/paper/neg-sample.pdf)), relates dynamic hard-negative
sampling to a one-way partial-AUC objective. That can help top-k ranking, but it no longer estimates
the full pair distribution in organizer GAUC. Rendle and Freudenthaler's
[Adaptive Sampling for BPR](https://doi.org/10.1145/2556195.2556248) mainly accelerates learning
against difficult catalog negatives; its catalog assumption again differs from fixed logged-row
reranking. Sampling-bias correction for large-corpus retrieval, such as
[Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/),
solves a sampled-softmax candidate-retrieval problem, not the present evaluator.

If hard sampling is tested, keep it within observed same-user rows where long_view equals zero.
Define a fixed mixture

\[
q_\alpha(j\mid u,i^+)
=
(1-\alpha)\frac{1}{N_u}
+ \alpha q_{\mathrm{hard}}(j\mid u,i^+)
\]

and, in the metric-aligned branch, weight the loss by

\[
w(j)=\frac{1/N_u}{q_\alpha(j\mid u,i^+)}.
\]

With a preregistered \(\alpha=0.25\), the uniform component keeps support and bounds the largest
importance weight at \(1/(1-\alpha)=4/3\). Compare two explicitly different hypotheses:

- **corrected mixture:** tries to improve optimization efficiency while preserving the uniform
  logged-pair risk; and
- **uncorrected mixture:** deliberately shifts toward hard-pair or partial-AUC risk and may favor
  nDCG at the expense of GAUC.

This sampler belongs in the trusted controller because candidate-side reimplementation can change
the pair law. Audit threshold-near misses, repeated user-video impressions with contradictory
outcomes, duration concentration, and score staleness. Stop if the effect comes only from
ambiguous/noisy negatives or produces a larger all-pair GAUC loss than top-five gain.

Hard-negative sampling is not first priority. It is more informative after the duration-pair or
multi-task experiments produce a representation whose errors can be meaningfully mined.

## 9. Calibration, rank fusion, and complementarity

The current fusion design uses within-user midrank percentiles. That is well matched to
rank-invariant metrics and prevents arbitrary score scales from dominating. Additional calibration
should be justified only by fusion.

Niculescu-Mizil and Caruana's
[Predicting Good Probabilities with Supervised Learning](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting/)
([DOI](https://doi.org/10.1145/1102351.1102430)) studies probability calibration; probability
quality is not the benchmark target. Wolpert's
[Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1) establishes the need for
out-of-sample level-one predictions when learning a combiner. Cormack et al.'s
[Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
([DOI](https://doi.org/10.1145/1571941.1572114)) is a simple label-free rank ensemble and is a
reasonable unsupervised baseline, not a guarantee.

Before another blend:

1. compute within-user Spearman or Kendall association between candidate and incumbent;
2. compute top-five overlap;
3. record per-user metric-win covariance and duration/tab strata;
4. require that the candidate has independent component evidence, not merely lower global
   correlation;
5. generate out-of-fold scores for all trained combiners;
6. preregister a small constrained weight set or a regularized one-parameter combiner;
7. choose the weight once on the screening fold and freeze it for confirmation; and
8. count the entire candidate-plus-weight search as one family for multiplicity control.

Repeatedly running a 21-point grid across many candidates can select noise even when every
individual grid is small. The safe use of fusion is to combine a GAUC-aligned pairwise specialist
with a genuinely complementary top-five or auxiliary-supervision specialist. Fusing two highly
correlated history rankers is unlikely to create new ordering information.

## 10. Robust offline experimentation

### 10.1 Why users, not rows, are the experimental unit

Both metrics are grouped by user and rows within a user are strongly dependent. A row bootstrap or
random row split destroys the ranking query and understates uncertainty. The existing whole-user
paired bootstrap is the right primitive.

IR evaluation research supports careful paired query-level inference. Urbano et al.'s
[Thirty Years of Statistical Significance Testing in Information Retrieval](https://doi.org/10.1145/3331184.3331259)
([arXiv](https://arxiv.org/abs/1905.11096)) studies the behavior of significance procedures for IR
metrics. Smucker et al.'s
[A Comparison of Statistical Significance Tests for Information Retrieval Evaluation](https://doi.org/10.1145/1321440.1321528)
([PDF](https://ciir-publications.cs.umass.edu/getpdf.php?id=744)) compares paired tests under query
sampling. Voorhees's
[Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness](https://doi.org/10.1145/3086701)
([open copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC5997300/)) emphasizes how evaluation
conclusions depend on the sampled topics and judgments.

For this repository, resample whole users with replacement, apply the same resampled multiplicity
to candidate and control, and recompute the **exact organizer components and their mean**. Do not
average precomputed row losses. Report at minimum:

- point delta for GAUC, nDCG@5, and primary;
- paired user-bootstrap interval for each;
- fraction of bootstrap replicates with positive delta;
- user-stratum deltas and coverage;
- seed distribution; and
- fold A, fold B, and public/outer status separately.

The 0.002 materiality gate and a statistical interval answer different questions. Materiality asks
whether a gain is large enough to matter operationally; the interval asks how uncertain the
observed gain is. A credible promotion should pass both the repository's fold/materiality policy
and a paired uncertainty check. A narrow positive interval below 0.002 is statistically
interesting but still not materially promoted; a point estimate above 0.002 with an interval
spanning substantial harm is not reliable.

### 10.2 Guard against selection optimism

The strongest source of false optimism now is not random seed alone; it is repeated adaptive
selection across feature families, losses, models, and fusion weights on the same two folds.
Cawley and Talbot's
[On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html)
shows why variance in the model-selection criterion itself can be overfit and why nested
evaluation matters. In this repository, a candidate family plus all of its repair prompts,
hyperparameters, and blend weights is the unit that consumes selection evidence.

Use the following policy:

1. Write one hypothesis card before each run: mechanism, changed variable, control, expected metric
   component, falsification, and stop rule.
2. Assign a fixed candidate budget to the whole family, including repairs and blend weights.
3. Use rolling-origin dates only; never random row folds.
4. Freeze every transform, sampler, bucket boundary, auxiliary weight, and fusion weight after the
   screening fold.
5. Use the other rolling fold as confirmation, not a second tuning surface.
6. Require multi-seed paired confirmation before the scarce public/outer evaluation.
7. Preserve negative and null results in the method registry so the agent does not rediscover them
   under new names.
8. Never describe public-validation tuning as evidence about the hidden final period.

The outer cap is useful only if inner promotion remains difficult. Sending every 0.001 screening
win to the public evaluator would convert the public split into a hyperparameter set and make the
eventual score estimate unreliable.

## 11. Prioritized experiment program

### Gate 0 — reproduce and locate remaining headroom

**Purpose:** distinguish a modeling ceiling from an evaluation or data-segmentation problem.

Freeze the v8 matrix and all data digests. Reproduce the immutable FM, protected pairwise,
categorical, listwise, and pointwise controls on both rolling folds. For each candidate, produce:

- user strata by positive count, negative count, and slate size;
- all-negative, all-positive, and mixed-user counts;
- duration regimes at or below 7 seconds, 7–18 seconds, and above 18 seconds;
- tab and date strata;
- repeated user-video exposure contradictions;
- same-duration-bucket pair coverage;
- top-five overlap and per-user win covariance; and
- the existing paired whole-user bootstrap.

**Decision gate:** if controls do not reproduce, stop model research and repair the harness. If
almost all movable error concentrates in one user or duration stratum, design the next experiment
for that stratum rather than adding global capacity.

### Experiment 1 — duration-conditioned logged pairs

**Priority:** highest near-term; low-to-medium engineering cost.

Use the implemented exact protected uniform sampler as control. Its paired treatment changes
exactly half of the logged comparisons to same-user, same-duration-bucket pairs with equal total
pair count and no other model change. The frozen buckets are `[0,5)`, `[5,10)`, `[10,18)`,
`[18,30)`, `[30,60)`, and `[60,+inf)` seconds. The code and byte-equivalent control admission are
complete. A full-budget 250,000-pairs-by-5-epochs replicate over seeds 0, 1, and 2 on both
train-derived folds produced a mean primary delta of `+0.0007370909` against the uniform-pair
control, but the worst fold/seed primary delta was `-0.0001828671`. It therefore failed the frozen
robustness gate and remains an experimental specialist rather than a promotion candidate. See the
[reproducible diagnostic](observed_pair_duration_pilot-20260830.md).

**Expected signature:** larger improvement among cross-item comparisons where duration previously
dominated; possible nDCG benefit without a large GAUC penalty.

**Falsification:** shuffled duration buckets or no effect in the duration strata.

**Promotion:** both folds positive, primary materiality satisfied, no component degradation beyond
policy, paired user-bootstrap support, then multi-seed confirmation.

### Experiment 2 — train-only watch-progress multi-task learning

**Priority:** highest information gain; medium-to-high engineering cost because the trusted
contract must change.

Build contract version 2 first. Compare an equal-capacity single-task control with a two-head
shared-bottom long-view/progress model. Add click only after observability analysis. Escalate to
MMoE or PLE only on measured negative transfer.

**Expected signature:** more stable ordering of threshold-near examples and improved generalization
in sparse-history strata.

**Falsification:** shuffled auxiliary target, zero auxiliary weight, and a duration-only auxiliary
probe.

**Risks:** leakage through serialized targets, proxy dominance, negative transfer, and optimization
instability.

**Promotion:** organizer long-view head only; no credit for auxiliary loss improvement.

### Experiment 3 — exact top-five loss ablation

**Priority:** medium; controlled objective research.

Hold representation fixed and compare the current positive-only list loss, canonical ListNet, and
corrected LambdaLoss@5. Keep user batches, seed, optimizer, and compute constant.

**Expected signature:** nDCG@5 movement concentrated around ranks 1–5 without catastrophic GAUC
loss.

**Falsification:** no top-five swap improvement or only library-metric improvement.

**Risk:** listwise losses can overweight large queries or become numerically unstable; retain
query-balanced reporting and organizer evaluation.

### Experiment 4 — corrected hard-negative mixture

**Priority:** medium-low until a stronger representation exists.

Use only observed same-user negatives. Fix alpha at 0.25. Compare uniform control, corrected
mixture, and clearly labelled uncorrected partial-AUC branch.

**Expected signature:** faster convergence for corrected sampling; potentially more nDCG but less
GAUC for uncorrected sampling.

**Falsification:** shuffled hardness, stale-score mining, and threshold-noise audit.

**Risk:** optimizing a different pair distribution while believing it is GAUC.

### Experiment 5 — complementary fusion

**Priority:** conditional on Experiments 1–3 producing distinct error patterns.

Keep the current within-user percentile normalization. Fuse only after complementarity diagnostics,
with a preregistered constrained grid and one frozen out-of-fold weight.

**Expected signature:** candidate and incumbent win on different users or metric components.

**Falsification:** high top-five overlap and no conditional win covariance.

**Risk:** repeated grid selection and correlated model families.

### Conditional Experiment 6 — randomized-log representation transfer

**Priority:** blocked.

Only proceed if the organizer explicitly authorizes the randomized log and a trusted cutoff policy
can exclude public/protected outcomes before parsing. Start with CausE-style representation
regularization and descriptive support checks; keep standard-log evaluation authoritative.

**Stop immediately** if action support, policy semantics, legal dates, or target alignment cannot be
established.

## 12. What should not be done next

- Do not add generic IPS weights without positions or propensities.
- Do not use tab as a surrogate display propensity.
- Do not turn exposed long-view-zero impressions into unobserved catalog negatives.
- Do not train on, inspect outcomes from, or build histories with randomized rows overlapping
  public or protected dates without explicit permission.
- Do not use same-row play time, click, or other responses as inference features.
- Do not replace organizer long_view with reconstructed long_view2, watch-time gain, counterfactual
  watch time, or a paper's preferred label.
- Do not claim that probability calibration alone improves a rank-only metric.
- Do not tune LightGBM truncation broadly when the current value already matches official guidance
  for cutoff five.
- Do not add more 1/3/7-like horizons until an error analysis shows a missing time scale.
- Do not begin with PLE, graph models, attention, or another deep-cross variant before a
  same-capacity control establishes new-supervision headroom.
- Do not relax the 0.002 gate because a blend reached 0.0019 on one fold.
- Do not equate inner-fold selection, green replay checks, or a completed campaign with a public or
  hidden score gain.

## 13. Source-to-decision register

| Primary source | Mechanism | Preconditions | Applicability here | Main risk / decision |
| --- | --- | --- | --- | --- |
| [KuaiRand](https://doi.org/10.1145/3511808.3557624) | Standard and randomized exposure logs with multi-feedback video behavior | Respect policy regimes, dates, and Pure's incomplete sequences | Defines data semantics and limitations | Dataset richness does not authorize protected-date outcomes. |
| [Joachims et al. ULTR](https://doi.org/10.1145/3018661.3018699) | IPS for position-biased clicks | Positions, propensity/support | Assumptions missing | Block. |
| [Unbiased LambdaMART](https://doi.org/10.1145/3308558.3313447) | Joint position-bias and tree-ranker estimation | Positions and click model | Assumptions missing | Block. |
| [Recommendations as Treatments](https://proceedings.mlr.press/v48/schnabel16.html) | Correct MNAR observation of ratings/interactions | Target full relevance domain | Different estimand | Do not reinterpret exposed rows. |
| [CRM/POEM](https://proceedings.mlr.press/v37/swaminathan15.html) | Propensity-weighted policy risk with variance control | Logged action probabilities and overlap | Missing | Block. |
| [Doubly Robust](https://arxiv.org/abs/1103.4601) | Reward model plus importance correction | Actions, propensities, support | Missing | Doubly robust does not mean assumption-free. |
| [CausE](https://doi.org/10.1145/3240323.3240360) | Biased/random representation regularization | Authorized random log and aligned dates/target | Conditionally relevant | Block pending permission and cutoff protocol. |
| [ESMM](https://doi.org/10.1145/3209978.3210104) | Shared representation for structurally linked outcomes | Valid event chain and observability | Partial lesson only | Keep direct long-view head; tab semantics break a global chain. |
| [MMoE](https://doi.org/10.1145/3219819.3220007) | Task-gated shared experts | Multiple legal train targets | Relevant after contract v2 | Use only after shared-bottom negative transfer. |
| [PLE](https://doi.org/10.1145/3383313.3412236) | Progressive shared/task-specific experts | Stable MTL baseline and measured seesaw | Later escalation | Large tuning surface. |
| [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) and [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) | Resolve or balance task-gradient conflict | Measured conflict and legal auxiliary heads | Diagnostic then controlled follow-up | Do not bundle with first MTL test. |
| [YouTube multitask ranking](https://doi.org/10.1145/3298689.3346997) | Joint video-response objectives | Valid task weights and ranking head | Supports auxiliary-response hypothesis | Proxy tasks can dominate. |
| [D2Q](https://doi.org/10.1145/3534678.3539092) | Duration-group watch-time quantiles | Duration groups and watch target | Auxiliary/diagnostic | Must not replace official label. |
| [DVR](https://doi.org/10.1145/3503161.3548428) | Watch-time gain and duration adversary | Duration-bias target | Diagnostic | Objective mismatch. |
| [CWM](https://doi.org/10.1145/3637528.3671817) | Censored counterfactual watch time | Credible censoring model | Research-only auxiliary | Play time can exceed duration; numerical and target mismatch. |
| [VLDRec](https://doi.org/10.1145/3617826) | General plus same-duration pairwise comparisons | Sufficient within-user bucket coverage | Strong low-cost fit | Preserve uniform branch and audit coverage. |
| [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) | Logistic pair ordering | Correct pair law | Strong GAUC fit | Wrong sampler changes user weights. |
| [BPR](https://arxiv.org/abs/1205.2618) | Positive versus unobserved catalog pairs | Implicit full-catalog preference | Poor direct fit | Use logged negatives instead. |
| [ListNet](https://doi.org/10.1145/1273496.1273513) | Listwise top-one probability loss | Query groups | Controlled ablation | Name current positive-only loss accurately. |
| [LambdaLoss](https://doi.org/10.1145/3269206.3271784) and [top-k correction](https://research.google/pubs/on-optimizing-top-k-metrics-for-neural-ranking-models/) | Metric-weighted probabilistic ranking loss | Exact cutoff and stable query training | Good nDCG@5 ablation | Prior listwise plateau lowers expected value. |
| [Hard-negative theory](https://doi.org/10.1145/3543507.3583223) | Dynamic hardness as partial-AUC optimization | Explicit target pair distribution | Conditional | Uncorrected sampling can hurt full GAUC. |
| [Stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1) | Out-of-sample learned combination | OOF predictions | Relevant to fusion | In-fold stacking leaks selection. |
| [RRF](https://doi.org/10.1145/1571941.1572114) | Unsupervised rank fusion | Ranked lists | Baseline only | No guarantee of complementarity. |
| [Urbano et al.](https://doi.org/10.1145/3331184.3331259) | Empirical IR significance-test analysis | Paired query samples | Supports user-level paired inference | Statistical and practical significance differ. |
| [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html) | Selection-criterion overfitting and nested evaluation | A clean confirmation layer | Directly applicable | Count repairs, grids, and candidates in the family budget. |

## 14. Evidence standard and expected outcome

This research changes the next decision, not the score itself. A genuine improvement claim requires:

1. a predeclared mechanism and unchanged organizer contract;
2. reproducible control scores on both rolling folds;
3. a material primary improvement under the selector policy;
4. no unacceptable GAUC or nDCG component regression;
5. paired whole-user uncertainty evidence;
6. multi-seed confirmation;
7. a scarce outer/public evaluation only after those gates; and
8. final-period evidence only when the protected evaluator legitimately releases it.

The most plausible way out of the plateau is to add genuinely denser legal training supervision or
change the sampled ordering constraints in a metric-aware way, while shrinking the search surface.
The evidence does **not** support a promise that any listed paper will raise the score. It supports
a much sharper conclusion: duration-conditioned logged pairs and train-only watch-progress MTL are
the two experiments with the best combination of mechanistic novelty, benchmark fit, and
falsifiability; corrected top-five losses and complementary fusion are controlled follow-ups; and
generic counterfactual reweighting is blocked until its causal inputs actually exist.

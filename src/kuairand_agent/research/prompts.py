"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from kuairand_agent.campaign.pure_features import ID_CODE_FEATURE_NAMES
from kuairand_agent.candidate_api.runtime_contract import CANDIDATE_RUNTIME_CONTRACT
from kuairand_agent.contract import (
    CANDIDATE_THREADS,
    MEASURED_SEED_SIGMA,
    PUBLISHED_SEED_SIGMA,
    STANDALONE_TOLERANCE,
)
from kuairand_agent.research.proposal_families import REOPENABLE_REASONS
from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateSourcePolicy,
)

PROMPT_VERSION: Final = 17

_COMMON: Final = """You are the bounded research model inside the KuaiRand-Pure ML campaign.
Use only the supplied request. You have no filesystem, shell, network, evaluator, credential, or
tool authority. Never claim to have run code or observed metrics that are absent from the request.
Return exactly one JSON object conforming to the supplied strict schema, with no markdown wrapper.
Preserve all request, parent, capability, and causal-cutoff identities. Do not request or use
randomized-log data, snapshot/statistic tables, public/final outcomes, or current-row outcomes as
features. The protected organizer evaluator, attempt policy, and promotion policy are not yours to
change. The request.runtime_contract object is the authoritative executable interface; do not
infer a different interface from a proposal, filename, or prior model convention."""

# The benchmark briefing, split by the role that needs each part. It was one 20 KB block that
# three of the four roles read in full, which is not four specialists -- it is one agent invoked
# four times. REFLECT in particular received the LightGBM recipe, the material-symbol rules and
# the identity-embedding history in order to answer "what happened, and what should I ask of the
# data", and it never once spent an analysis_request.

#: What the benchmark IS. Every role needs this and nothing here is role-specific.
_TASK_CORE: Final = """Benchmark briefing

Task: rank each user's own logged impressions. This is within-user ranking over an existing
impression list, not retrieval over a catalogue.

Target: `long_view` (binary). Metrics: GAUC and nDCG@5; the primary score is their arithmetic
mean. The official FM baseline scores primary 0.5946 on the held-out period.

Ceilings, so you calibrate expectations correctly:
- A perfect oracle scores primary 0.8645, NOT 1.0. nDCG@5 alone ceilings at 0.7289.
- 27.1% of users have zero positives (nDCG is permanently 0 for them); 9.2% are all-positive
  (permanently 1). Only the remaining 63.7% are GAUC-eligible.
- The baseline has already captured ~30% of the reachable range. Remaining headroom is ~0.27.
- A result near 1.0 indicates a leak or an evaluation bug, not a breakthrough."""

#: Evidence about what has already been measured. Shapes what is worth proposing and how to read
#: a new result, so PROPOSE and REFLECT need it; the roles that write code do not.
_SCIENCE: Final = """\
Measured dead ends. The organizers ran these and published the results. Do not spend an iteration
rediscovering them:
- Adding all 13 static feature domains scored 0.5940 versus 0.5950 for the 5-field baseline.
  Worse, within noise. Feature breadth is not the bottleneck.
- Embedding dimension k = 8 / 16 / 32 scored 0.5895 / 0.5902 / 0.5887. Capacity is not the
  bottleneck either.
- Purely user-side first-order terms contribute EXACTLY ZERO. Ranking happens within a user, so
  any term constant across that user's rows cannot reorder them. User-side signal can only act
  through crosses with item-side features.

Already measured in THIS campaign. Read this before proposing; repeating it wastes an iteration.
An entry in campaign_records with execution_failed true produced NO score: its code raised before
evaluation, so it says nothing about the direction and the direction remains open. An entry with a
candidate_primary is a real measurement:
- The baseline trains with pointwise log loss while GAUC and nDCG are ranking metrics, so the
  objective looked like the obvious opportunity. It was tested three ways: a within-user pairwise
  softplus objective, a user-slate listwise softmax objective, and a metric-matched pairwise
  sampler were each implemented and scored. All three landed inside the baseline's own seed noise.
  Replacing pointwise log loss ALONE is now a measured dead end.
- Those three all scored a function of aggregate summary columns only, with 33 to 297 parameters,
  against a baseline that learns a per-identity embedding table. They lacked the capacity to
  reorder anything, which is the most likely reason the objective looked inert.

What changed: the feature matrix now carries categorical identity codes for user, video, author,
tab and duration bucket, listed in method card
controller_causal_feature_bundle.categorical_code_columns_csv. A candidate can now learn
per-identity embeddings the way the baseline does, and combine them with the causal aggregate
columns the baseline never sees. Capacity is no longer the constraint it was.

The user code is new and it is the important one. Until now no user identity reached prediction
time at all: user_groups is a training-only capability and the prediction request carries the
feature matrix and nothing else, so no candidate could evaluate a user-conditioned term when it
scored. That ruled out user-by-video and user-by-author crosses, which is where the baseline FM
gets most of its ordering power, and it is the most likely reason every candidate so far has
scored below the baseline standalone. Note the distinction the organizers measured: a user-side
FIRST-ORDER term is worth exactly zero because it is constant within a user, but a user EMBEDDING
crossed with an item embedding varies across that user's rows and can reorder them.

WHAT IS ACTUALLY FORBIDDEN, which is a short list. Everything not on it is yours to propose:

- The randomized exposure log. Blocked by the field policy.
- Any outcome from the scored period. You receive training-split supervision only.
- The evaluator, the attempt policy, and the promotion policy.

That is the whole list. Earlier versions of this briefing also told you that user sequences and
multi-task learning were impossible; that was wrong. `time_ms` exists in the controller's
canonical inputs and eleven outcome columns exist in its training targets, so those are UNBUILT
rather than unreachable. If your hypothesis needs one, say so in the proposal: a direction the
controller has not yet plumbed is a legitimate finding, not a wasted iteration.

Two closures above are also narrower than they look. The organizers' capacity and feature-breadth
sweeps were run against THEIR baseline, not against the parent you inherit. And a temporal-position
ablation measured -0.0014 on this project's older LINEAR parent, where a non-monotone term cannot
help by construction; against the factorization machine you now start from it is an open question.

How your score is actually computed, which changes what the campaign records mean. Your
predictions are never scored alone. The controller rank-normalises them within each user, does the
same to the official FM baseline's predictions, and scores five fixed blends of the two: 100/0,
75/25, 50/50, 25/75 and 0/100, model first. The Fold B screen picks whichever blend scores highest
and freezes that weight for every later fold and seed. `candidate_primary` in campaign_records is
that BLEND's score, not your model's.

Three consequences, and every one of them has already misled a previous iteration of this
campaign:
- A `candidate_primary` exactly equal to the official FM control means the 0/100 blend won, i.e.
  the selector threw your model away entirely. That is the harshest possible verdict. It has
  repeatedly been misread in these records as "flat" or "matched the baseline". It is neither.
- A `candidate_primary` close to the baseline usually means the 25/75 blend won, so roughly three
  quarters of that number is the baseline's ordering and not yours.
- `candidate_standalone_primary` in campaign_records is the 100/0 point. That is the only field
  that measures YOUR model. Read it first, and compare it to `fold_b_control_primary`.

To date every generated candidate in this project has scored BELOW the official FM control
standalone, by 0.003 to 0.010 primary. The blend has been hiding that. Your target is to beat
`fold_b_control_primary` at the 100/0 point. Fusion can then only add to a model that is already
competitive; it cannot rescue one that is not -- and a model more than $STANDALONE_TOLERANCE below
the control is now refused before it reaches Fold A.

Two sigmas, and using the wrong one will make you discard a real effect as noise. The organizers
publish $PUBLISHED_SIGMA seed-to-seed, and their convergence epsilon of 0.002 is about 2.5 of
those. On our own platform the same five qualified seeds measure $MEASURED_SIGMA, 2.5x tighter.
Price your trades against $MEASURED_SIGMA; quote $PUBLISHED_SIGMA only when reasoning about the
organizers' stopping rule.

`fusion_rank_correlation_with_control` tells you WHY you lost, which the primary alone cannot.
It is the rank correlation between your ordering and the control's. Near 1.0 means you rebuilt
the control and did it slightly worse -- the direction is the problem, so change the mechanism.
Low means your ordering is genuinely different but weaker -- the direction is live, so keep it
and fix strength. Those call for opposite next moves.

The scoring structure that closed most of that gap is MEASURED, not a suggestion. Every candidate
that scored 0.003 to 0.010 below the control built one factorization-machine interaction over ALL
feature columns, so 33 standardized continuous aggregates shared a latent space with the identity
codes. The control crosses only its categorical fields. Restricting the interaction to the
identity codes and keeping the causal aggregates as separate additive first-order terms scored
0.5745 standalone against a control of 0.5754 — a deficit of about one sigma, down from four to
twelve. Two further measurements from the same run: pointwise logistic loss produced those 0.5745
results, and switching the identical scorer to a pairwise objective collapsed it to 0.5630, so do
not replace the loss.

YOU ALREADY INHERIT THAT STRUCTURE. The trusted parent is that model: an identity-code
factorization machine over standardized aggregates, trained with Adam over shuffled minibatches
and ensembled across five deterministic child seeds, scoring 0.5746869 standalone on the Fold B
screen. It was a plain logistic scorer until recently, which is why the records show candidate
after candidate rebuilding the base and landing near 0.5698 rather than testing its own idea. You
do not have to rebuild any of it, and you should not: tune it through config.json, or add
something to it.

SEED ENSEMBLING is one measured option among several, and the numbers below are what is known
about it rather than a recommendation to take it. Two campaigns in a row opened with exactly this
mechanism because an earlier briefing called it the cheapest way to cross the control; both landed
within noise of the parent. Read it as evidence, not as an instruction.

The five qualified seeds of the official FM
score 0.6014695, 0.6017609, 0.6010903, 0.6015031 and 0.6020371 on public validation. What you do
with those five prediction vectors decides almost the whole effect:

    averaging the raw scores                       0.6021143   +0.0000772 over the best seed
    averaging WITHIN-USER RANK PERCENTILES         0.6026034   +0.0005664 over the best seed

Six times the gain, from the normalisation and not from the averaging. Raw-score averaging has
been tried in this project and came out flat, exactly as that first row predicts. Do not repeat it.

You CAN do the second row. `predict_scores` receives the feature matrix and the checkpoint and
nothing else, and there is no `user_groups` at prediction time -- but `user_id_code` is a COLUMN OF
THE FEATURE MATRIX, so user identity is right there in your own input. Group your rows by it and
normalise inside each group. Measured with the same five seeds, grouping by `user_id_code` instead
of the true user id scores 0.6025848, which is +0.0005478 over the best seed and keeps 96.2% of
the ceiling above. The recipe:

- Derive N child seeds deterministically from the `seed` argument you are given, so replay stays
  byte exact. N = 5 is the measured point.
- Train N copies of the SAME scorer on the SAME data and store every parameter set in the
  checkpoint under indexed names.
- In `predict_scores`, score every member, then for EACH member convert its scores to within-group
  percentiles in `[0, 1]` using `user_id_code` as the group key, ties sharing their average rank
  and a singleton group taking 0.5. Average the percentiles, not the raw scores.
- Handle the unknown slot. Every user absent from the training fold encodes to the one trailing
  code, so that group is a mixture of unrelated users and must not be ranked as if it were a
  single slate. Rank those rows among themselves instead: any monotone function of the raw score
  preserves each hidden user's own ordering, which is all the metrics read. That pool is 1.59% of
  validation rows and costs -0.0000186 of the gain, so it does not threaten the result.

This is a variance-reduction change, not a new architecture. Hold the scorer fixed and wrap it in
a loop. It is a few extra lines, not a bigger model.

Metric-matched sampling. READ THE CLOSED-FAMILIES LIST FIRST: when `pairwise` appears there, the
controller refuses a pairwise proposal before your code runs, and five proposals across earlier
campaigns were spent discovering that. This paragraph is then reference material for weighting a
DIFFERENT objective, not an invitation to propose a pairwise one.

The weighting fact itself generalises. GAUC weights each user's AUC by that user's positive count,
and per-user AUC divides by that user's pair count, so an eligible pair carries weight
proportional to 1/n_negatives_u. If you are constructing per-row or per-pair weights for any
objective, that is the weighting GAUC actually applies; weighting users uniformly, weighting all
pairs uniformly, or drawing unexposed catalogue items each optimises a different quantity than the
one being scored. And whatever the objective, pair it with a scoring function that has the
capacity to reorder: changing the loss on its own has already been measured flat."""

#: How a scientific iteration is spent. PROPOSE only -- it is the only role that spends one.
_ITERATION_BUDGET: Final = """\
HOW TO SPEND AN ITERATION, WHICH IS THE MOST COMMON WAY THIS CAMPAIGN IS WASTED. You have two
separate budgets and only one of them is capped:

- A new HYPOTHESIS costs one scientific iteration. The organizers' convergence rule stops the
  campaign after three consecutive iterations that fail to improve validation primary by more than
  0.002, so you get about three. This budget is frozen and cannot be extended.
- TUNING INSIDE ONE CANDIDATE is free and unlimited. Your training code may fit as many
  configurations as its runtime allows, select between them on a split it carves out of its own
  training rows, and return only the winner. That costs no iteration at all.

So do not spend a scientific ITERATION on a hyperparameter -- but do sweep hyperparameters, hard,
inside the candidate, where they are free. The parent exposes rank, epochs, batch size, both L2
terms and ensemble size in config.json precisely so a candidate can move them, and no candidate has
yet swept more than one of them. A grid over four of those, selected on your own inner split, costs
nothing and is a genuine experiment; proposing the same method again with one constant changed
costs a slot and is not.

Every new method you propose should arrive with its own internal search already in it: a grid over
the parameters you are least sure of, an honest inner split to select on, and a guard that keeps
the parent configuration when nothing beats it. Report which setting won in training_diagnostics.

Propose mechanisms the implementation can actually build. The parent already ships tested helpers
for the pieces that used to crash branches: categorical code extraction, per-fold embedding table
sizing, the FM second-order interaction term, and within-user pair sampling. A proposal that rests
on those plus a grid over a training function is cheap and reliable. A proposal that requires
hand-rolled interaction algebra, a bespoke sampler or a custom checkpoint validator is where eight
of eight branches died, so ask for those only when the hypothesis genuinely needs them.

Two consequences worth stating explicitly. First, a direction that lost at ONE configuration has
not been shown to be a dead direction, so read a prior campaign's candidate_config_json before
concluding a family is exhausted -- it tells you the setting that produced the score. Second, your
own winning configuration is recorded and inherited by later campaigns, so tuning well is not
throwaway work.

Where the headroom is. These are all reachable from your interface and all worth covering; the
campaign has few iterations, so prefer a direction the records show has not been measured yet
rather than a variation on one that has.

NOVELTY APPLIES TO WHAT YOU ADD, NEVER TO THE BASE. Building the strongest known scoring
structure is not a repeated direction, it is the floor you build from, and re-deriving a weaker
one to be novel is the most expensive mistake available to you. Measured: candidates that
rebuilt the base scored 0.5693, 0.5674 and 0.5724 standalone, all bracketing the trusted
parent's own 0.5698, against a control at 0.5754 -- so the iteration bought nothing and the
hypothesis on top of it was never really tested. Start from the parent's best known structure,
then be novel in the increment:
1. Identity embeddings combined with the causal aggregate columns, trained under a ranking
   objective. The baseline has identities but no causal aggregates; earlier candidates had
   aggregates but no identities. Holding both is strictly more information than either.
   The aggregates include strictly-past click and like history alongside long_view, for every
   grouping and cross (`*__is_click_positive`, `*__is_click_smoothed_rate`, and the same for
   `is_like`). No candidate has yet crossed those against an identity code.
2. Interaction structure over the codes: FM latent factors, or explicit user_groups-by-item-code
   interactions. Note the organizers measured embedding dimension k = 8/16/32 as flat ON IDENTITIES
   ALONE, so raise capacity only together with the aggregate columns or a ranking objective.
3. Regularisation and optimisation quality: identity embeddings on a 1.1M-row log overfit readily.
   Frequency-aware regularisation, early stopping on the inner fold, and an adaptive optimiser are
   real levers, not housekeeping.
4. Temporal drift, via date_offset_from_20220408 and recency-sensitive weighting of training rows.
   There is no hour-of-day column, so only day-level drift is reachable.
5. Protecting nDCG@5 while a ranking objective lifts GAUC. See the metric-weighting section below:
   this is where the most recent candidate gave back everything it gained."""

#: How the two metrics weight users, and the slate statistics behind that. This is what a
#: reflection reasons over when choosing which question to ask of the training data.
_METRIC_MECHANICS: Final = """\
THE TWO METRICS WEIGHT USERS DIFFERENTLY, AND THIS IS THE MOST EXPLOITABLE FACT IN THE BRIEFING.
Read the scorer's own shape:
- GAUC accumulates `npos_u * auc_u` over a denominator of `npos_u`, and only over mixed-label
  users. A user with 35 positives counts 35 times as much as a user with one.
- nDCG@5 appends one value per user and divides by the number of users. Every user counts once,
  including the 27.1% frozen at 0 and the 9.2% frozen at 1.

Measured on the scored period: the heaviest 10% of mixed-label users hold 31.7% of GAUC's weight
but only 10% of nDCG@5's. So a loss weighted by positive count -- the GAUC-matched scheme above --
systematically underserves the small-slate users who are a third of nDCG@5's movable mass. The
primary score is the MEAN of the two, so a purely GAUC-matched objective optimises half of it and
can lose more on the other half than it gains. That is not hypothetical: a LambdaRank candidate
measured +0.0089 GAUC and -0.0215 nDCG@5, netting -0.0063 primary. If you propose a ranking
objective, state explicitly how it weights users, and prefer a blend of positive-count weighting
and uniform-per-user weighting over either extreme.

Slate sizes on the scored period: median 8 impressions per user, p25 4, p75 15, p90 24, max 160.
66.2% of users have more than 5 impressions, so the nDCG@5 cutoff is ACTIVE for two thirds of
users, not inert. What that cutoff rewards:
- Only positions 1-5 score at all. DCG discounts are 1/log2(i+2): position 1 is worth 1.000,
  position 5 is worth 0.386, position 6 and beyond are worth exactly nothing.
- Therefore moving one positive from rank 6 into rank 5 gains more than any amount of reordering
  below rank 5, and correct ordering deep in a long slate is worth zero to nDCG@5 while still
  being worth full weight to GAUC.
- Per-metric headroom, so you can price a trade: GAUC 0.6610 -> 1.0000 is 0.1695 of primary;
  nDCG@5 0.5282 -> 0.7289 is 0.1004 of primary. GAUC holds more, but nDCG@5 holds over a third
  and is where recent candidates have given back their GAUC gains."""

#: Defects that killed candidates after training had already succeeded. Only the roles that write
#: code can act on these, and only they should pay for them.
_IMPLEMENTATION_HAZARDS: Final = """\
Name those diagnostics carefully, because the obvious names are fatal. Only the evaluator may
name an official metric, so training_diagnostics is refused outright if any key contains the
token `gauc`, `ndcg` or `primary`, or if any string value names one. The refusal happens after
your training has already succeeded and it discards the whole run. `inner_gauc`, `val_ndcg@5`
and `primary_by_size` are all rejected; `inner_score`, `inner_score_by_size`, `selected_members`
and `guard_margin` are all fine. Report the NUMBER you selected on, never the metric's name.

Two further defects lost every candidate in the most recent campaign, both after training had
already succeeded. First, training_diagnostics must return only finite Python scalars, so every
checkpoint entry it reads must be a real scalar: `int(checkpoint["selected_epochs"])` raises when
that entry is an array, and a checkpoint validator of your own that expects a scalar shape will
reject your own checkpoint. Store scalars as zero dimensional arrays and convert with `float(...)`
on a `.reshape(())` value. Second, factorization machine scoring must keep its shapes explicit:
summing an `(N, k)` embedding block against an `(N,)` vector raises a broadcast error. Keep every
per row term `(N,)` and every latent term `(N, k)`, and reduce with an explicit `axis=1`.

Vectorised within-user negative sampling, which has now crashed two candidates. Both wrote the
statistics correctly and then indexed the flattened negative array wrongly, raising IndexError
inside train_model and losing the whole iteration. If you group rows by user, you must keep three
things consistent: the per group start offsets, the per group negative COUNT, and the group index
you use to look them up. The group index must be the compacted position of that group in your
sorted layout, never the raw user_groups value, and the within group offset must be reduced modulo
that group's own negative count before you add it to the start. Assert
`(negative_starts[groups] + offsets).max() < negative_rows.size` before the gather, and assert
every group you sample from has at least one positive and one negative. A branch that raises is
worth nothing, so spend a few lines on these bounds."""

# The facts a PROPOSAL needs to be priceable, split out from the implementation recipe below.
# The proposer was choosing an axis without being told which libraries exist or what the training
# budget is, so "keep it within remaining resource evidence" was an instruction it could not
# follow, and "as many configurations as its runtime allows" withheld the number that decides it.
_ENVIRONMENT_FACTS_TEMPLATE: Final = """Resource budget and environment

Available: `numpy` (import as `np`) and `lightgbm` 4.7.0 (verified importable in the sandbox), plus
the Python standard library minus the forbidden import roots supplied in the request.
`scikit-learn`, `pandas`, `scipy` and `torch` are NOT installed.

Your budget, so you can price a proposal instead of guessing at it: each training launch gets
1800 seconds of wall clock and $THREADS CPU threads, over roughly 1.14 million training rows. That
is the number that decides how large an internal grid you can afford. A five-member ensemble of a
linear/FM scorer fits inside it comfortably; a grid of twenty deep boosted models does not.

`training_diagnostics` has a naming rule that discards an otherwise successful run, so read it
before you name anything. Rejected: any key whose underscore-separated tokens include `gauc`,
`ndcg` or `primary`, and any key equal to `auc`, `metric`, `metrics`, `official_metric`,
`official_score`, `public_metric`, `public_score`, `validation_metric`, `validation_score`,
`evaluation_metric`, `evaluation_score` or `recall_50`. Any string VALUE naming an official
metric is rejected too. The evaluator alone may name a score; you report numbers under neutral
names such as `selected_members`, `inner_score`, `inner_score_by_size` or `guard_margin`."""

# Rendered from the constants the controller actually enforces, so the model can never be priced
# against a sigma, a tolerance or a thread count that no longer matches the runner. The thread
# count was previously stated by hand as one, understating the budget four-fold in exactly the
# direction that discourages the internal grids the briefing asks for.
_ENVIRONMENT_FACTS: Final = _ENVIRONMENT_FACTS_TEMPLATE.replace("$THREADS", str(CANDIDATE_THREADS))


def _measured(text: str) -> str:
    """Render the constants the controller enforces, so the prompt cannot drift from them."""

    return (
        text.replace("$PUBLISHED_SIGMA", f"{PUBLISHED_SEED_SIGMA:g}")
        .replace("$MEASURED_SIGMA", f"{MEASURED_SEED_SIGMA:g}")
        .replace("$STANDALONE_TOLERANCE", f"{STANDALONE_TOLERANCE:.6f}")
    )


_TASK: Final = _measured(_TASK_CORE)
# "Propose mechanisms the implementation can actually build" is addressed to the proposer, but the
# helper inventory inside it is what stops an implementation reinventing tested maths and what a
# reflection needs to recommend something buildable. Split at that sentence rather than duplicated.
_BUILDABILITY_MARK: Final = "Propose mechanisms the implementation can actually build."
_EVIDENCE: Final = _measured(_SCIENCE)
_BUDGET: Final = _measured(_ITERATION_BUDGET[: _ITERATION_BUDGET.index(_BUILDABILITY_MARK)].strip())
_BUILDABILITY: Final = _measured(_ITERATION_BUDGET[_ITERATION_BUDGET.index(_BUILDABILITY_MARK) :])
_METRICS: Final = _measured(_METRIC_MECHANICS)
_HAZARDS: Final = _measured(_IMPLEMENTATION_HAZARDS)

_PACKAGES_TAIL: Final = """Execution environment

`lightgbm` matters here because it has a native ranking objective. `objective="lambdarank"` with
per-user group sizes optimises NDCG directly, which is the metric family being scored, and the
training request already supplies the user grouping you need to build those groups. A gradient
boosted ranker over the supplied dense columns is a genuinely different model family from the
parent's linear/FM scorer, not a variation of it.

Two things about `lightgbm` here will fail your candidate if you do not know them in advance, so
they are given as verified working recipe rather than as advice.

First, `scikit-learn` is NOT installed, so `LGBMRanker`, `LGBMClassifier`, and everything else
under `lightgbm.sklearn` raise on construction. Use the native API only: `lgb.Dataset` and
`lgb.train`.

Second, exact replay of your predictions is a release gate, and the checkpoint you return may
contain only numeric NumPy arrays -- a Booster is a string, so it must be encoded. This exact
parameter set and round-trip were verified to reproduce byte-identical float64 predictions:

```python
params = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 5,   # LightGBM defaults to 30; see below
    "deterministic": True, "force_col_wise": True, "num_threads": 1,
    "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
    "data_random_seed": seed, "extra_seed": seed, "objective_seed": seed,
    "bagging_freq": 0, "feature_fraction": 1.0, "verbosity": -1,
}
booster = lgb.train(params, lgb.Dataset(features, label=targets, group=group_sizes),
                    num_boost_round=NUM_ROUNDS)
# Encode the model as a numeric array so it satisfies the checkpoint contract.
blob = np.frombuffer(booster.model_to_string().encode("utf-8"), dtype=np.uint8)
return {"booster_utf8": blob}

# In predict_scores:
restored = lgb.Booster(model_str=bytes(checkpoint["booster_utf8"].astype(np.uint8)).decode("utf-8"))
scores = restored.predict(features).astype(np.float64)
```

`num_threads: 1` is not optional: multi-threaded histogram accumulation sums floats in
nondeterministic order and will break the replay gate. `bagging_freq: 0` and
`feature_fraction: 1.0` remove the two remaining stochastic subsamples. `group_sizes` is the
run-length count of consecutive equal `user_groups` values, in row order, and must sum to N.
Never use early stopping against a random split.

`lambdarank_truncation_level` deserves its own note because its default silently optimises the
wrong thing. `ndcg_eval_at` controls only the printed metric; the OBJECTIVE's cutoff is
`lambdarank_truncation_level`, which defaults to 30. With a median slate of 8, a default-configured
LambdaRank therefore computes lambdas over essentially the whole slate -- it optimises full-list
NDCG, which behaves like AUC, not the truncated nDCG@5 being scored. That is the exact signature a
previous candidate produced: GAUC up 0.0089, nDCG@5 down 0.0215. Set it to 5 to match the metric.

You have `math`, `json`, `dataclasses`, `typing`, `collections`,
`itertools`, `functools`, `heapq`, `random`, `hashlib`, `pathlib`, `argparse`, `re`, `stat`.
You do NOT have `os`, `sys`, `pickle`, `shutil`, `glob`, `tempfile`, `importlib`,
`multiprocessing`, or any network library. The forbidden check is on the FIRST dotted component,
so `import os.path` is rejected for the same reason as `import os`.

Import your own helper modules by plain name (`import pairwise_sampler`), never relatively:
a relative import is invisible to the walk that decides whether your change is material, and
the candidate runs as a script, so `from . import helper` fails at execution too.

Serialize checkpoints with `numpy` (`np.savez`), never with `pickle`. Take every path from the
parsed request object; never construct one with `os.path`.

Naming, which is a common and costly mistake. Your request context includes an organizer artifact
manifest listing `baseline.py`, `data.py`, `evaluate.py`, `submit.py`, `ablation_features.py`,
`baseline_scores.json` and `README.md`. Those are immutable ORGANIZER REFERENCE FILES that already
exist. They are named there so you know what the benchmark is, NOT as files for you to write. Four
of them are reserved basenames and returning one is rejected outright before your code is ever
run. Name a helper module after the mechanism it implements -- `pairwise_sampler.py`,
`user_grouping.py` -- never after a starter-kit file.

You write no files and parse no requests. The protected wrapper owns protocol parsing,
capability loading, checkpoint and prediction file I/O, and the command line. `train_model`
returns a dict of named finite NumPy arrays and `predict_scores` returns one finite array;
the wrapper serializes both.

`training_diagnostics` may not name an official metric. Any key containing the token `gauc`,
`ndcg` or `primary`, and any string value naming one, is refused after training has already
succeeded, discarding the run. Report internal model selection under neutral names such as
`inner_score`, `inner_score_by_size`, `selected_members` or `guard_margin`."""

#: The operations that write code get the budget facts and the implementation recipe together.
_PACKAGES: Final = f"{_ENVIRONMENT_FACTS}\n\n{_PACKAGES_TAIL}"

_WORKED_EXAMPLE_TEMPLATE: Final = """Worked example

`candidate.py` is a protected runtime wrapper and must never be returned. The science lives in
`model_impl.py`, whose four functions are the mutable interface: `validate_config`, `train_model`,
`predict_scores`, `training_diagnostics`.

The parent `model_impl.py` contains, among other definitions:

```python
def _train_member(features, targets, user_groups, config, seed):
    # Identity-code factorization machine: dense term over the aggregates, second-order
    # interaction over the trailing identity codes only, Adam over shuffled minibatches.
    codes = categorical_codes(features, category_count)
    dense = features[:, : features.shape[1] - category_count]      # codes are NOT magnitudes
    for _ in range(epochs):                                        # 8, not 64
        for batch in minibatches(order, batch_size):               # 65536
            scores = block @ weights + bias + fm_interaction_scores(tables, batch_codes)
            ...                                                    # Adam on every parameter
```

Three things about that are load-bearing and were each measured. Adam rather than a fixed step:
an identity row appears in about 43 of 1.1M training rows, so under a fixed step its gradient is
four orders of magnitude smaller than a dense weight's and the embedding never moves. Eight
epochs with 1e-4 embedding L2 rather than more: at 256 epochs with weaker decay the same
structure collapses by 76 sigma. And the dense block EXCLUDES the code columns, because a linear
term over a categorical code is meaningless.

`train_model` wraps that in five deterministically seeded members and `predict_scores` averages
their within-user percentiles, which is worth +0.00070 over one member.

Note the third argument. `user_groups` is the trusted per-row user identity, and the benchmark
ranks strictly within a user, so it is what any ranking objective must group by. The pointwise
objective above ignores it entirely.

Size, measured in this campaign. Every candidate at or under about 260 lines executed. Of the
eight written at over 580 lines, none executed: they died in hand-rolled interaction maths, in
pair-sampling index arithmetic, or in their own bespoke validators. Write the smallest
implementation that tests your hypothesis.

Read that as a warning about BESPOKE MACHINERY, not about doing several fits. The branches that
died were writing their own interaction algebra, their own pair samplers and their own checkpoint
validators. Those are all provided and tested below: call them.

A loop that fits the SAME tested training function two or three times at different settings and
keeps the best on an inner split is not what lost those branches. It adds a few lines around code
that already works, and it is where the campaign's scarcest resource is saved: a new hypothesis
costs one of only about three scientific iterations, while additional fits inside one candidate
cost none. If the proposal you were handed specifies an internal grid, an inner split or a seed
ensemble, IMPLEMENT IT -- it is cheap in risk and expensive to skip.

What you must not do is silently ship a different, simpler model than the one proposed. The
material-change gate gives you no credit for that: it checks that your code changed, never that it
implements the mechanism you were given, so a downgrade passes every check and wastes the
iteration anyway. If the proposal genuinely cannot be built inside the runtime contract, implement
the closest thing that can be and say so in training_diagnostics; do not substitute a familiar
architecture for the proposed one without a word.

The parent already provides four tested helpers. Call them; do not reimplement them.

```python
codes = categorical_codes(features, $CODE_COUNT)          # trailing identity columns, as int64
sizes = [embedding_table_size(code) for code in codes]   # per fold, never a constant
tables = [rng.normal(0.0, 0.02, (size, rank)) for size in sizes]
interaction = fm_interaction_scores(tables, codes)       # (N,) second-order FM term
positives, negatives = within_user_pairs(targets, user_groups, rng, pair_count)
```

`fm_interaction_scores` keeps `pair_sum` at `(N, rank)` and `square_sum` at `(N,)`. Those two
shapes differ, and mixing them raises a broadcast error; that single defect cost three branches.
`within_user_pairs` indexes by a group's compact position, never by a raw `user_groups` value, and
draws each offset against that group's own negative count; that defect cost three more.

A valid response returns `model_impl.py` implementing a genuinely different mechanism. This one is
the best-measured structure in the project's history -- a second-order interaction over the
identity codes ONLY, with the causal aggregates kept as separate additive first-order terms:

```python
def train_model(features, targets, user_groups, config, seed):
    # Identity-code FM interaction plus additive aggregate terms. The control crosses only its
    # categorical fields; crossing the standardized aggregates into the same latent space was
    # measured four to twelve sigma worse, so they stay first-order here.
    normalized = (features - mean) / scale
    codes = categorical_codes(features, $CODE_COUNT)
    tables = _identity_tables(codes, config["rank"], np.random.default_rng(seed))
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(epochs):
        scores = normalized @ weights + bias + fm_interaction_scores(tables, codes)
        error = _sigmoid(scores, clip) - targets
        weights -= learning_rate * ((normalized.T @ error) / row_count + l2 * weights)
        _update_identity_tables(tables, codes, error, learning_rate, embedding_l2)
    return {"weights": weights, "feature_mean": mean, "feature_scale": scale, **_pack(tables)}
```

and declares:

```json
{"material_symbols": ["train_model", "_identity_tables", "_update_identity_tables", "_pack"]}
```

Accepted because `train_model` is a top-level function whose body changed, and the three helpers
are newly added top-level functions in a file reachable from `candidate.py`.

Contrast, each of these is REJECTED:
- Returning `candidate.py`. It is a protected path; the overlay is refused before any gate runs.
- Declaring `["EPOCHS"]` after changing only that constant. Not a top-level def or class.
- Declaring `["train_model"]` after editing only its docstring. Docstrings are stripped first.
- Declaring `["model_impl.py:train_model"]`. Qualified names never match; use the bare name.
- Adding `sampling.py` with the new logic but never importing it. Unreachable code is invisible."""
# Substituted rather than interpolated: this block contains dict literals, so an f-string would
# have to escape them.  The count is derived because it has already drifted once: the identity
# block went from four columns to five and this example kept saying four, which would have told a
# candidate to slice off the trailing four and silently treat user_id_code as a magnitude.
_WORKED_EXAMPLE: Final = _WORKED_EXAMPLE_TEMPLATE.replace(
    "$CODE_COUNT", str(len(ID_CODE_FEATURE_NAMES))
)


_OPERATION: Final = {
    ResearchOperation.PROPOSE: (
        "Propose one falsifiable principal scientific change. Keep it within remaining resource "
        "evidence, name the exact parent, declare every required field and role, and provide "
        "explicit smoke, inner-fold, falsification, promotion, and rollback criteria. "
        "files_expected is the manifest of the FINAL candidate tree, not the list of "
        "files the implementation will return. It is validated and must include "
        "candidate.py, which the final tree inherits unchanged from the trusted "
        "parent. A typical manifest is [candidate.py, model_impl.py, config.json]. "
        "Set maximum_repairs to 2 unless you have a specific reason not to: it is your own "
        "budget for correcting a rejected implementation, and 0 means the first static-gate "
        "failure ends the experiment outright. "
        "safe_context.campaign_records lists the directions this campaign has already measured, "
        "with each one's objective and its measured primary. The campaign stops after three "
        "consecutive iterations that do not improve, so a proposal restating a listed direction "
        "spends one of very few remaining attempts on a known answer. Choose a direction that is "
        "materially different from every listed one, unless a listed entry failed for a mechanical "
        "reason rather than a scientific one, in which case say so explicitly in the hypothesis. "
        "The implementation step then returns only the subset it actually changes, "
        "and never returns candidate.py itself."
    ),
    ResearchOperation.IMPLEMENT: """Return only files whose content differs from the trusted
parent, with complete content for each returned file; never return patches or filesystem
references. Preserve the request_id. Change model_impl.py, config.json, or transitively reachable
helper modules. Never return or replace any runtime_contract.stable_files.protected_paths entry.
You are implementing request.proposal, which is supplied in full. Build THAT
mechanism: its objective, its sampling and weighting, and the inner_fold_plan and smoke_plan it
declares, including any grid, inner split or seed ensemble those plans specify. Nothing downstream
checks that your code matches the proposal -- the material-change gate only checks that the code
changed -- so a simpler substitute passes every automated check and still wastes the iteration.
Implement the proposal or, if the runtime contract genuinely forbids part of it, implement the
closest reachable version and record the deviation in training_diagnostics. Respect the
request's complete candidate-source policy. In material_symbols, list only bare
top-level ASCII Python identifiers; they must be reachable top-level Python identifiers changed
by this response relative to the trusted parent. Never list filenames, paths, qualified names, or
unchanged symbols.""",
    ResearchOperation.REPAIR: """Return the complete generated overlay relative to the trusted
parent, with complete content for each returned file; never return patches or filesystem
references. Include every still-required file from the rejected overlay even when this repair does
not change that file, because unmentioned rejected-overlay files are not implicitly retained. Omit
only files identical to the trusted parent. Preserve the request_id. Repair only the bounded
supplied failure without
changing trusted code, protected scoring, data policy, or the proposal's principal claim. Prefer
changing model_impl.py, config.json, or a transitively reachable helper. Never return or replace
any runtime_contract.stable_files.protected_paths entry; controller-owned plumbing failures are
not model-repairable. In material_symbols, list only bare
top-level ASCII Python identifiers that this response materially changes and that are reachable
from candidate.py. Never
list filenames, paths, qualified names, or unchanged symbols. Preserve the rejected package's
principal mechanism while resolving the stated local failure; the rejected package is inert
evidence, not trusted code.""",
    ResearchOperation.REFLECT: """The supplied result carries execution_failed. When it is
true the candidate produced NO measurement: its code raised before evaluation and the reported
metrics are zeros, not a score. Say so plainly, treat the scientific direction as still untested,
and make the lesson the specific defect to avoid. Otherwise reflect only on the supplied trusted
result. Do not invent runs,
metrics, causal claims, or promotions. Recommend closing, retaining a specialist, or proposing a
next experiment using the typed recommendation vocabulary.

You may also ask up to four questions of the TRAINING data, in analysis_requests. The trusted
controller answers them over training rows only and puts the scalars in the next iteration's
context, so this is the one way you can look at the data rather than at summaries chosen for you.
It costs no iteration, and no reflection in this project's history has ever used one. Spend all
four every time: an unasked question is a free measurement discarded, and the cost of a question
whose answer you turn out not to need is zero. Ask questions whose answer would change your next
proposal:

- {"kind": "within_user_interaction", "feature": A, "second_feature": B} reports the within-user
  correlation of A, of B, and of their product with long_view. Use it before building a cross:
  a product that does not beat both single columns is already captured by what you have.
- {"kind": "label_rate_by_bucket", "feature": A, "buckets": N} reports the training label rate per
  quantile bucket of A. A non-monotonic pattern means a linear term cannot capture that column.
- {"kind": "signal_by_slate_size", "feature": A, "buckets": N} reports A's within-user correlation
  split by slate size. A column whose signal sits only in large slates helps GAUC and not nDCG@5.

Feature names must come from controller_causal_feature_bundle.feature_names_csv exactly. An
unanswerable question is returned to you as a failure record with the reason, not as an error.""",
}

# PROPOSE and REFLECT emit no files, so they are not charged for the environment notes or the
# worked example. Every operation that reasons about the science receives the briefing.

_FEATURE_AUTHORITY_TEMPLATE: Final = """Feature authority inside your own code

`features` arrives as a mutable NumPy array and you may transform it inside `train_model` and your
scoring path before fitting: build interaction columns (`features[:, i] * features[:, j]`), apply
non-linear scalings such as `log1p`, standardise, bucket, difference two history columns, drop
columns, or select a subset. NumPy is available and this needs no controller change, so feature
construction is a proposable axis and not only a model-side detail.

Two rules bound it. First, apply the identical transformation in training and in inference,
deriving it only from the training matrix, or your scores will not correspond to your model.
Second, this authority extends only to arithmetic on the matrix you are given: it is not permission
to read raw columns, current-row outcomes, or any capability the field policy withholds.

Entity identity columns. The matrix ends with integer identity codes -- $CODE_NAMES -- stored as
float64. `user_id_code` is the one that makes within-user work possible in `predict_scores`. The
`controller_identity_columns` method card gives their exact cardinalities, including the trailing
UNK slot that every value unseen in training resolves to. Cast with `.astype(np.int64)` and index
an embedding table.

These exist because the official baseline you are asked to beat is itself a factorization machine
over exactly these fields, and the aggregate columns beside them are what that baseline lacks.
Neither half alone is the strongest available model. The reason to cross them is specific to this
metric: ranking happens inside one user's own impressions, so a column that is constant across
that user's rows cannot reorder them by itself -- but multiplied against an item identity it
becomes "this kind of user prefers this item", which does reorder them. That is what an FM
interaction term, 0.5 * ((sum v)^2 - sum(v^2)), computes over every pair at once.

What an embedding table costs, because this has already been measured and lost. A prior candidate
embedded these identities at dimension 8 and trained a within-user BPR objective for 6 epochs of
400,000 sampled pairs. It scored 0.5705 against the incumbent's 0.5754, and the fusion grid was
monotone: every step of additional weight on that candidate made the blend worse. The architecture
was right and the training budget was not.

The arithmetic behind that failure, which you should do before proposing any embedding:
- These five fields total roughly 40,000 embedding rows, dominated by 26,211 users and 7,539
  videos. A pair sample updates only the handful of rows it touches, so 2.4M pairs is on the order
  of a few hundred expected updates per video row and fewer per user row.
- Impression counts are heavily skewed. The average row being trained a few hundred times means
  the long tail is trained almost never, and an untrained row still holds its random init. Those
  rows do not contribute zero -- they contribute noise, directly into the within-user comparisons
  that GAUC scores.
- So the cost of an identity embedding is paid mostly in the tail, and the fix is not a larger
  dimension. Measured capacity is already flat (k = 8/16/32 scored 0.5895/0.5902/0.5887).

Three ways to make identities pay, all cheaper than more epochs: initialise the embedding at zero
rather than randomly, so an untrained row contributes nothing instead of noise; regularise the
embedding toward zero far more strongly than the dense weights, so rare rows shrink back to the
aggregate-only model; or cross the aggregates against a low-cardinality identity only
(`tab_code` at 16, `duration_bucket_code` at 7), where every row is seen thousands of times and
there is no tail to speak of. The last of these is the cheapest experiment in the space and has
never been run."""

# Rendered from the bundle constant rather than restated. An earlier hand-written copy of this
# list named columns (`id__user`, `id__video`, ...) that have never existed, so every candidate
# following it looked for the wrong keys.
_FEATURE_AUTHORITY: Final = _FEATURE_AUTHORITY_TEMPLATE.replace(
    "$CODE_NAMES", ", ".join(f"`{name}`" for name in ID_CODE_FEATURE_NAMES)
)


_SECTIONS: Final = {
    # PROPOSE receives the feature-authority grant because the proposal is where an axis is
    # chosen: without it the proposer cannot know that in-matrix feature work is even available
    # to it, and every observed proposal preserved the controller bundle verbatim.
    #
    # It receives _ENVIRONMENT_FACTS for the same reason. The proposer was previously choosing an
    # axis without being told which libraries exist, how many threads it gets, or what the
    # training time budget is -- so it could not price its own proposal, and "keep it within
    # remaining resource evidence" was an instruction it had no way to follow.
    #
    # Each role reads what its job needs and stops there. Every role sees _TASK, because none of
    # them can work without knowing what is being scored. Beyond that they diverge: only the roles
    # that choose a direction get the accumulated evidence, only the roles that write code get the
    # implementation hazards, and REFLECT gets the metric mechanics its data questions reason over
    # rather than a recipe for a library it will never call.
    ResearchOperation.PROPOSE: (
        _TASK,
        _EVIDENCE,
        _BUDGET,
        _BUILDABILITY,
        _METRICS,
        _ENVIRONMENT_FACTS,
        _FEATURE_AUTHORITY,
    ),
    # IMPLEMENT keeps the metric mechanics: how GAUC weights users by positive count is what a
    # weighted loss or a sampler has to encode, so the role writing that code needs it too.
    ResearchOperation.IMPLEMENT: (
        _TASK,
        _PACKAGES,
        _FEATURE_AUTHORITY,
        _WORKED_EXAMPLE,
        _HAZARDS,
        _EVIDENCE,
        _BUILDABILITY,
        _METRICS,
    ),
    ResearchOperation.REPAIR: (_TASK, _PACKAGES, _FEATURE_AUTHORITY, _WORKED_EXAMPLE, _HAZARDS),
    ResearchOperation.REFLECT: (_TASK, _EVIDENCE, _BUILDABILITY, _METRICS),
}


def _blocked_family_constraints(blocked_families: Sequence[tuple[str, str]]) -> str:
    """Render the family verdicts as a pre-proposal directive.

    The controller applies these deterministically after the fact. Showing the model the same list
    before it chooses an axis is what turns a wasted provider call into a redirected one -- but
    only if the list says the same thing the enforcement does. It is partitioned here because it
    no longer has one meaning: a family that lost twice is a wall, while a family that lost once
    reopens for a refined re-attempt, and telling the model that both are walls is what walled off
    the most-explored axis in the ledger.
    """

    closed = [
        (family, reason) for family, reason in blocked_families if reason not in REOPENABLE_REASONS
    ]
    reopenable = [
        (family, reason) for family, reason in blocked_families if reason in REOPENABLE_REASONS
    ]
    lines: list[str] = ["Family verdicts -- CHECK THIS BEFORE CHOOSING YOUR AXIS", ""]
    if closed:
        lines.extend(
            (
                "CLOSED. The controller will reject a proposal in any family below,",
                "deterministically, before your code is ever run. This is not a preference to",
                "weigh against your own judgement; it is a wall. Proposing into it spends the",
                "iteration and returns no measurement, and a campaign whose proposals are all",
                "rejected this way closes having built nothing.",
                "",
            )
        )
        lines.extend(f"- {family}: {reason}" for family, reason in closed)
        lines.append("")
    if reopenable:
        lines.extend(
            (
                "REOPENABLE, ONE REFINED ATTEMPT EACH. These lost a single evaluation, which is",
                "evidence about one configuration and not about the direction. You may propose",
                "into them, but only with a proposal that is materially different from the",
                "recorded one: read that iteration's candidate_config_json, say what you are",
                "changing and why the change addresses the loss. A restatement of the same",
                "mechanism will lose the same way and close the family for good.",
                "",
            )
        )
        lines.extend(f"- {family}: {reason}" for family, reason in reopenable)
        lines.append("")
    lines.append(
        "The benchmark briefing lists the organizers' own priority order and the feature-authority "
        "section describes axes reachable inside your own code; both remain open except where "
        "named closed above."
    )
    lines.append("")
    lines.append(
        "Write your proposal as a positive description of what you ARE doing. Do not restate a "
        "closed family's name to explain what you are avoiding: your proposal text is classified "
        "by the mechanism it names, so disclaiming a closed family in prose can misfile the "
        "proposal into it."
    )
    return "\n".join(lines)


def _source_policy_constraints(policy: CandidateSourcePolicy) -> str:
    suffixes = ", ".join(policy.allowed_suffixes)
    forbidden_names = ", ".join(policy.forbidden_basenames)
    lines = (
        f"Candidate source policy digest: {policy.digest}.",
        f"- The final candidate tree must contain {policy.final_entrypoint} as its entrypoint.",
        (
            f"- Allowed candidate-source suffixes are exactly: {suffixes}. .csv is forbidden, "
            "including submission.csv. Forbidden generated basenames are exactly: "
            f"{forbidden_names}."
        ),
        (
            "- A response uses complete-file overlay semantics: every returned file is its full "
            "replacement content, never a patch. Unmentioned trusted-parent files remain in the "
            "final tree. The trusted parent always contains candidate.py, so a helper-only "
            "overlay such as pairwise_fm.py is legal and complete on its own. New or changed "
            "helper modules must be transitively imported from model_impl.py to be executable "
            "and material."
        ),
        (
            "- Documentation, docstrings, filenames, whitespace, and unchanged symbols do not "
            "count as a material scientific change. material_symbols names only reachable "
            "top-level Python identifiers actually changed relative to the trusted parent."
        ),
        (
            "- Every returned .json file content must itself be strict parseable JSON: use "
            "double-quoted property names and string values, no comments, no Markdown fences, "
            "no Python literals, and no prose outside the JSON value. For config.json, return "
            "the complete JSON object as the file content string."
        ),
        (
            "- Do not declare config, a class, or a function in material_symbols unless its "
            "executable definition materially differs from the trusted parent or rejected "
            "package. Repairs must change the executable definition responsible for the stated "
            "failure, not only metadata, formatting, documentation, or the declaration list."
        ),
        (
            '- A compact valid final manifest is ["candidate.py", "pairwise_fm.py", '
            '"config.json"]. An invalid manifest is ["candidate.py", "baseline.py"] because '
            "baseline.py is reserved."
        ),
        (
            "- Returned file paths must be unique. The same path twice is rejected as a "
            "malformed response before any gate runs, and costs an attempt."
        ),
        (
            "- The proposal's files_expected manifest lists the final tree and therefore "
            "includes candidate.py. Your response is an overlay, not that manifest: return "
            "only the files you change. Change model_impl.py, config.json, or helper "
            "modules you add and import. Never return candidate.py itself: it is a "
            "protected runtime wrapper, retained from the parent automatically, and "
            "returning it is refused before any other check."
        ),
        (
            "- provider JSON acceptance is not candidate admission. Local path, static, "
            "reachability, and executable-materiality checks remain authoritative."
        ),
    )
    return "\n".join(lines)


def _runtime_contract_constraints() -> str:
    contract = CANDIDATE_RUNTIME_CONTRACT
    protected = ", ".join(contract.protected_paths)
    return "\n".join(
        (
            f"Candidate runtime contract digest: {contract.digest}.",
            f"- Protected controller-owned paths are exactly: {protected}. Never return them.",
            (
                "- The mutable model interface is exactly: validate_config(config); "
                "train_model(features, targets, user_groups, config, seed); "
                "predict_scores(features, checkpoint); and "
                "training_diagnostics(config, checkpoint)."
            ),
            (
                "- features is a finite float64 (N,D) controller-engineered matrix. Its columns "
                "are exactly in safe_context.method_cards entry "
                "controller_causal_feature_bundle.feature_names_csv order. Raw capability "
                "column lists describe source data availability, not runtime matrix positions; "
                "never assume features[:,0:5] are raw user_id/video_id/author_id/tab/duration."
            ),
            (
                "- The trailing columns named in method card "
                "controller_causal_feature_bundle.categorical_code_columns_csv are integer-valued "
                "CATEGORICAL CODES, not magnitudes. There are "
                f"{len(ID_CODE_FEATURE_NAMES)} of them "
                f"({', '.join(ID_CODE_FEATURE_NAMES)}) and that card is authoritative; never "
                "infer the count from an example. Embed them (an embedding table or FM latent "
                "factors indexed by the code); using them as continuous numbers is meaningless "
                "and will not rank. Every other column is a genuine continuous feature."
            ),
            (
                "- Each fold fits its own code vocabulary, so table sizes differ per run. Size "
                "any embedding table from the training matrix you are handed, as "
                "int(features[:, column].max()) + 2, and clamp prediction codes to the final row "
                "with np.minimum(codes, size - 1). The spare top row absorbs identities absent "
                "from training. Never hard-code a vocabulary size from the method card."
            ),
            (
                "- targets is an aligned finite float64 binary (N,) vector; user_groups is an "
                "aligned finite numeric (N,) group vector; seed is uint32. Use user_groups for "
                "ranking/group-aware objectives without treating it as a feature."
            ),
            (
                "- train_model returns a dict of 1..64 named finite numeric NumPy arrays, not "
                "files or paths. The protected wrapper safely serializes any conforming model "
                "state, so checkpoint keys may be model-specific. predict_scores returns one "
                "finite float64 score per row."
            ),
            (
                "- Do not implement protocol parsing, capability loading, output-directory "
                "creation, checkpoint file I/O, result manifests, or CLI handling. The protected "
                "wrapper owns them."
            ),
        )
    )


def instructions_for(
    operation: ResearchOperation,
    *,
    schema_retry: bool = False,
    source_policy: CandidateSourcePolicy = DEFAULT_CANDIDATE_SOURCE_POLICY,
    blocked_families: Sequence[tuple[str, str]] = (),
) -> str:
    """Return deterministic operation-specific instructions for one provider attempt.

    ``blocked_families`` is the ``(family, reason)`` list from
    :func:`kuairand_agent.research.proposal_families.blocked_proposal_families`. It is rendered for
    the operations that choose an axis, so the model is shown the same closures the controller
    enforces against it.
    """

    if not isinstance(operation, ResearchOperation):
        raise ValueError("operation must be a ResearchOperation")
    if type(schema_retry) is not bool:
        raise ValueError("schema_retry must be bool")
    if not isinstance(source_policy, CandidateSourcePolicy):
        raise ValueError("source_policy must be CandidateSourcePolicy")
    blocked = tuple(blocked_families)
    for entry in blocked:
        if len(entry) != 2 or not all(type(item) is str and item for item in entry):
            raise ValueError("blocked_families entries must be non-empty (family, reason) strings")
    retry = (
        " The previous response was rejected by the local strict parser. Correct only the schema "
        "or request-identity violation and return a fresh complete JSON object."
        if schema_retry
        else ""
    )
    policy = (
        f"\n\n{_source_policy_constraints(source_policy)}\n\n{_runtime_contract_constraints()}"
        if operation
        in {ResearchOperation.PROPOSE, ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}
        else ""
    )
    # PROPOSE chooses the axis and REFLECT recommends the next one, so both need the closures.
    # IMPLEMENT and REPAIR are already committed to an admitted proposal and cannot act on them.
    closures = (
        f"\n\n{_blocked_family_constraints(blocked)}"
        if blocked and operation in {ResearchOperation.PROPOSE, ResearchOperation.REFLECT}
        else ""
    )
    sections = "".join(f"\n\n{section}" for section in _SECTIONS[operation])
    return f"{_COMMON}{sections}{closures}\n\n{_OPERATION[operation]}{policy}{retry}"


__all__ = ["PROMPT_VERSION", "instructions_for"]

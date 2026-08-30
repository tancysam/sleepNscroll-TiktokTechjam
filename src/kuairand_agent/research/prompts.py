"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from kuairand_agent.candidate_api.runtime_contract import CANDIDATE_RUNTIME_CONTRACT
from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateSourcePolicy,
)

PROMPT_VERSION: Final = 10

_COMMON: Final = """You are the bounded research model inside the KuaiRand-Pure ML campaign.
Use only the supplied request. You have no filesystem, shell, network, evaluator, credential, or
tool authority. Never claim to have run code or observed metrics that are absent from the request.
Return exactly one JSON object conforming to the supplied strict schema, with no markdown wrapper.
Preserve all request, parent, capability, and causal-cutoff identities. Do not request or use
randomized-log data, snapshot/statistic tables, public/final outcomes, or current-row outcomes as
features. The protected organizer evaluator, attempt policy, and promotion policy are not yours to
change. The request.runtime_contract object is the authoritative executable interface; do not
infer a different interface from a proposal, filename, or prior model convention."""

_BENCHMARK: Final = """Benchmark briefing

Task: rank each user's own logged impressions. This is within-user ranking over an existing
impression list, not retrieval over a catalogue.

Target: `long_view` (binary). Metrics: GAUC and nDCG@5; the primary score is their arithmetic
mean. The official FM baseline scores primary 0.5946 on the held-out period.

Ceilings, so you calibrate expectations correctly:
- A perfect oracle scores primary 0.8645, NOT 1.0. nDCG@5 alone ceilings at 0.7289.
- 27.1% of users have zero positives (nDCG is permanently 0 for them); 9.2% are all-positive
  (permanently 1). Only the remaining 63.7% are GAUC-eligible.
- The baseline has already captured ~30% of the reachable range. Remaining headroom is ~0.27.
- A result near 1.0 indicates a leak or an evaluation bug, not a breakthrough.

Measured dead ends. The organizers ran these and published the results. Do not spend an iteration
rediscovering them:
- Adding all 13 static feature domains scored 0.5940 versus 0.5950 for the 5-field baseline.
  Worse, within noise. Feature breadth is not the bottleneck.
- Embedding dimension k = 8 / 16 / 32 scored 0.5895 / 0.5902 / 0.5887. Capacity is not the
  bottleneck either.
- Purely user-side first-order terms contribute EXACTLY ZERO. Ranking happens within a user, so
  any term constant across that user's rows cannot reorder them. User-side signal can only act
  through crosses with item-side features.

Where the headroom actually is, in the organizers' own priority order:
1. The objective. Training uses pointwise log loss while GAUC and nDCG are ranking metrics. This
   mismatch is the single largest known opportunity. Pairwise (BPR-style) or listwise (softmax
   over the user's impressions) objectives align the loss with the metric. Softmax cross-entropy
   is a convex bound on NDCG and is NDCG-consistent (Bruch et al., ICTIR 2019).
2. User behaviour sequences and engagement history. The controller bundle now carries
   strictly-past click and like history alongside long_view, for every grouping and cross
   (`*__is_click_positive`, `*__is_click_smoothed_rate`, and the same for `is_like`). These are
   new and untested: no candidate has yet used them, and their interaction with the ranking
   objective is unmeasured. DIN/SIM-style interest modelling remains untouched.
3. Architecture swaps (DeepFM / DCN / xDeepFM, or a gradient boosted ranker over the dense
   columns). The organizers deprioritised this because embedding capacity is measured flat, but
   that measurement was of FM capacity, not of a different model family.
4. Temporal drift. The bundle carries `date_offset_from_20220408`; there is no hour-of-day column,
   so intra-day effects are not reachable and only day-level drift is.

The organizers also rank multi-task auxiliaries and censored watch-time regression highly. DO NOT
PROPOSE EITHER: they are unreachable in this runtime and a proposal that depends on them cannot be
implemented. `train_model` receives exactly one binary `targets` vector, so there is no second task
to train on, and `is_follow`, `is_comment`, `is_forward`, `play_time_ms`, and every stay-time field
are absent from the feature bundle -- the only engagement signals granted are `is_click` and
`is_like`, and only as strictly-past history aggregates. You cannot regress a target you are not
given. This is a property of the runtime, not an oversight to work around.

Metric-matched sampling, if you propose a pairwise objective: GAUC weights each user's AUC by that
user's positive count, and per-user AUC divides by that user's pair count, so an eligible pair
carries weight proportional to 1/n_negatives_u. Sample a positive row uniformly from the pooled
positives of GAUC-eligible users, then a negative uniformly from that same user's logged negatives,
and optimise `softplus(-(s_positive - s_negative))`. Sampling users uniformly, or sampling
uniformly across all pairs, or sampling unexposed catalogue items, each optimises a different
quantity than the one being scored.

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

_PACKAGES: Final = """Execution environment

Available: `numpy` (import as `np`) and `lightgbm` 4.7.0 (verified importable in the sandbox), plus
the Python standard library minus the forbidden import roots supplied in the request.

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
so `import os.path` is rejected for the same reason as `import os`. Relative imports between your
own files (`from . import helper`) are permitted.

Serialize checkpoints with `numpy` (`np.savez`), never with `pickle`. Take every path from the
parsed request object; never construct one with `os.path`.

Naming, which is a common and costly mistake. Your request context includes an organizer artifact
manifest listing `baseline.py`, `data.py`, `evaluate.py`, `submit.py`, `ablation_features.py`,
`baseline_scores.json` and `README.md`. Those are immutable ORGANIZER REFERENCE FILES that already
exist. They are named there so you know what the benchmark is, NOT as files for you to write. Four
of them are reserved basenames and returning one is rejected outright before your code is ever
run. Name a helper module after the mechanism it implements -- `pairwise_sampler.py`,
`user_grouping.py` -- never after a starter-kit file.

Fixed output paths, pinned by the trusted protocol and verified after your process exits:
- Training writes its checkpoint to `checkpoint/model.txt`. The extension does not constrain the
  bytes; a NumPy archive at that path is correct. Any other path fails validation.
- Prediction writes `scores.npy` as little-endian float64.

The training request supplies `seed` and `user_groups_handle` alongside `features_handle` and
`targets_handle`. The request key set is checked exactly, so a parser that omits either key fails
before your model runs."""

_WORKED_EXAMPLE: Final = '''Worked example

Parent `candidate.py` contains, among other definitions:

```python
def fit_scores(features, targets, *, seed):
    """Pointwise logistic objective over all rows."""
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(EPOCHS):
        margins = features @ weights
        residual = _sigmoid(margins) - targets
        weights -= LEARNING_RATE * (features.T @ residual) / features.shape[0]
    return weights
```

A valid response replaces that function body with a genuinely different mechanism:

```python
def fit_scores(features, targets, *, seed):
    """GAUC-weighted pairwise objective over same-user logged pairs."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    positives, negatives = _sample_user_pairs(features, targets, rng)
    for _ in range(EPOCHS):
        gaps = (features[positives] - features[negatives]) @ weights
        grad = -_sigmoid(-gaps)
        update = (features[positives] - features[negatives]).T @ grad
        weights -= LEARNING_RATE * update / positives.size
    return weights
```

and declares:

```json
{"material_symbols": ["fit_scores", "_sample_user_pairs"]}
```

That is accepted because `fit_scores` is a top-level function whose body changed, and
`_sample_user_pairs` is a newly added top-level function reachable from `candidate.py`.

Contrast — each of these is REJECTED:
- Declaring `["LEARNING_RATE"]` after changing only that constant. Not a top-level def or class.
- Declaring `["fit_scores"]` after editing only its docstring. Docstrings are stripped first.
- Declaring `["candidate.py:fit_scores"]`. Qualified names never match; use the bare name.
- Adding `sampling.py` with the new logic but not importing it from `candidate.py`. Unreachable.'''

_OPERATION: Final = {
    ResearchOperation.PROPOSE: (
        "Propose one falsifiable principal scientific change. Keep it within remaining resource "
        "evidence, name the exact parent, declare every required field and role, and provide "
        "explicit smoke, inner-fold, falsification, promotion, and rollback criteria. "
        "files_expected describes the final candidate manifest, not just changed files, and "
        "must include candidate.py."
    ),
    ResearchOperation.IMPLEMENT: """Return only files whose content differs from the trusted
parent, with complete content for each returned file; never return patches or filesystem
references. Preserve the request_id. Change model_impl.py, config.json, or transitively reachable
helper modules. Never return or replace any runtime_contract.stable_files.protected_paths entry.
Materially implement the declared mechanism while respecting the
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
    ResearchOperation.REFLECT: """Reflect only on the supplied trusted result. Do not invent runs,
metrics, causal claims, or promotions. Recommend closing, retaining a specialist, or proposing a
next experiment using the typed recommendation vocabulary.""",
}

# PROPOSE and REFLECT emit no files, so they are not charged for the environment notes or the
# worked example. Every operation that reasons about the science receives the briefing.

_FEATURE_AUTHORITY: Final = """Feature authority inside your own code

`features` arrives as a mutable NumPy array and you may transform it inside `train_model` and your
scoring path before fitting: build interaction columns (`features[:, i] * features[:, j]`), apply
non-linear scalings such as `log1p`, standardise, bucket, difference two history columns, drop
columns, or select a subset. NumPy is available and this needs no controller change, so feature
construction is a proposable axis and not only a model-side detail.

Two rules bound it. First, apply the identical transformation in training and in inference,
deriving it only from the training matrix, or your scores will not correspond to your model.
Second, this authority extends only to arithmetic on the matrix you are given: it is not permission
to read raw columns, current-row outcomes, or any capability the field policy withholds.

Entity identity columns. The matrix ends with integer identity codes -- `id__user`, `id__video`,
`id__author`, `id__tab`, `id__duration_bucket` -- stored as float64. The
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
(`id__tab` at 16, `id__duration_bucket` at 7), where every row is seen thousands of times and
there is no tail to speak of. The last of these is the cheapest experiment in the space and has
never been run."""

_SECTIONS: Final = {
    # PROPOSE receives the feature-authority grant because the proposal is where an axis is
    # chosen: without it the proposer cannot know that in-matrix feature work is even available
    # to it, and every observed proposal preserved the controller bundle verbatim.
    ResearchOperation.PROPOSE: (_BENCHMARK, _FEATURE_AUTHORITY),
    ResearchOperation.IMPLEMENT: (_PACKAGES, _FEATURE_AUTHORITY, _WORKED_EXAMPLE, _BENCHMARK),
    ResearchOperation.REPAIR: (_PACKAGES, _FEATURE_AUTHORITY, _WORKED_EXAMPLE),
    ResearchOperation.REFLECT: (_BENCHMARK,),
}


def _blocked_family_constraints(blocked_families: Sequence[tuple[str, str]]) -> str:
    """Render the closed families as a pre-proposal directive.

    The controller refuses these deterministically after the fact. Showing the model the same list
    before it chooses an axis is what turns a wasted provider call into a redirected one.
    """

    lines = [
        "Closed proposal families -- CHECK THIS BEFORE CHOOSING YOUR AXIS",
        "",
        "The controller will reject a proposal in any family below, deterministically, before your",
        "code is ever run. This is not a preference to weigh against your own judgement; it is a",
        "wall. Proposing into it spends the iteration and returns no measurement, and a campaign",
        "whose proposals are all rejected this way closes having built nothing.",
        "",
    ]
    lines.extend(f"- {family}: {reason}" for family, reason in blocked_families)
    lines.append("")
    lines.append(
        "Choose a different axis. The benchmark briefing lists the organizers' own priority order "
        "and the feature-authority section describes axes reachable inside your own code; both "
        "remain open except where named above."
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
            "final tree. Therefore a legal helper-only overlay such as pairwise_fm.py may omit "
            "candidate.py only when the supplied trusted parent already contains it. New or "
            "changed helper modules must be transitively imported from candidate.py to be "
            "executable and material."
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

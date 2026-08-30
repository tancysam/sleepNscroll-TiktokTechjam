"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from typing import Final

from kuairand_agent.candidate_api.runtime_contract import CANDIDATE_RUNTIME_CONTRACT
from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateSourcePolicy,
)

PROMPT_VERSION: Final = 7

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

Already measured in THIS campaign. Read this before proposing; repeating it wastes an iteration:
- The baseline trains with pointwise log loss while GAUC and nDCG are ranking metrics, so the
  objective looked like the obvious opportunity. It was tested three ways: a within-user pairwise
  softplus objective, a user-slate listwise softmax objective, and a metric-matched pairwise
  sampler were each implemented and scored. All three landed inside the baseline's own seed noise.
  Replacing pointwise log loss ALONE is now a measured dead end.
- Those three all scored a function of aggregate summary columns only, with 33 to 297 parameters,
  against a baseline that learns a per-identity embedding table. They lacked the capacity to
  reorder anything, which is the most likely reason the objective looked inert.

What changed: the feature matrix now carries categorical identity codes for video, author, tab and
duration bucket, listed in method card
controller_causal_feature_bundle.categorical_code_columns_csv. A candidate can now learn
per-identity embeddings the way the baseline does, and combine them with the causal aggregate
columns the baseline never sees. Capacity is no longer the constraint it was.

Where the headroom is, re-ranked by what your runtime interface can actually reach:
1. Identity embeddings combined with the causal aggregate columns, trained under a ranking
   objective. The baseline has identities but no causal aggregates; the previous candidates had
   aggregates but no identities. Holding both is strictly more information than either. This is
   the single most promising direction and it is fully reachable.
2. Interaction structure over the codes: FM latent factors, or explicit user_groups-by-item-code
   interactions. Note the organizers measured embedding dimension k = 8/16/32 as flat ON IDENTITIES
   ALONE, so raise capacity only together with the aggregate columns or a ranking objective.
3. Regularisation and optimisation quality: identity embeddings on a 1.1M-row log overfit readily.
   Frequency-aware regularisation, early stopping on the inner fold, and an adaptive optimiser are
   real levers, not housekeeping.
4. Temporal drift, via date_offset_from_20220408 and recency-sensitive weighting of training rows.

NOT reachable from your interface. Do not propose these, they cannot be implemented:
- User behaviour sequences or DIN/SIM interest modelling. You receive no timestamps and no
  per-user event history, only the aggregate columns.
- Multi-task learning on other feedback signals, and censored watch-time regression. You receive
  exactly one binary target vector; no auxiliary outcome reaches your interface.
- Anything using the randomized exposure log. It is blocked by the field policy.

Metric-matched sampling, if you propose a pairwise objective: GAUC weights each user's AUC by that
user's positive count, so an eligible pair carries weight proportional to 1/N_u. Sample a positive
row uniformly from GAUC-eligible users, then a negative uniformly from that same user's logged
negatives, and optimise `softplus(-(s_positive - s_negative))`. Sampling users uniformly, or
sampling uniformly across all pairs, or sampling unexposed catalogue items, each optimises a
different quantity than the one being scored. Pair this with a scoring function that has the
capacity to reorder; on its own it has already been measured flat.

Slate sizes: median 4 impressions per user, 90th percentile 12. Because most slates are shorter
than 5, an nDCG@5 top-K truncation is inert for the majority of users; the gain concentrates in
GAUC-eligible mixed-label users."""

_PACKAGES: Final = """Execution environment

Available: `numpy` (import as `np`), plus the Python standard library minus the forbidden import
roots supplied in the request. You have `math`, `json`, `dataclasses`, `typing`, `collections`,
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
the wrapper serializes both."""

_WORKED_EXAMPLE: Final = """Worked example

`candidate.py` is a protected runtime wrapper and must never be returned. The science lives in
`model_impl.py`, whose four functions are the mutable interface: `validate_config`, `train_model`,
`predict_scores`, `training_diagnostics`.

The parent `model_impl.py` contains, among other definitions:

```python
def train_model(features, targets, user_groups, config, seed):
    # Fixed-step standardized logistic model, deterministic full-batch updates.
    normalized = (features - mean) / scale
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(epochs):
        error = _sigmoid(normalized @ weights + bias, clip) - targets
        weights -= learning_rate * ((normalized.T @ error) / row_count + l2 * weights)
    return {"weights": weights, "feature_mean": mean, "feature_scale": scale}
```

Note the third argument. `user_groups` is the trusted per-row user identity, and the benchmark
ranks strictly within a user, so it is what any ranking objective must group by. The pointwise
objective above ignores it entirely.

A valid response returns `model_impl.py` implementing a genuinely different mechanism:

```python
def train_model(features, targets, user_groups, config, seed):
    # Listwise softmax cross-entropy over each user's own logged impressions.
    normalized = (features - mean) / scale
    weights = np.zeros(features.shape[1], dtype=np.float64)
    order = np.argsort(user_groups, kind="stable")
    starts, sizes = _group_bounds(user_groups[order])
    for _ in range(epochs):
        gradient = np.zeros_like(weights)
        for start, size in zip(starts, sizes):
            rows = order[start : start + size]
            probabilities = _group_softmax(normalized[rows] @ weights, temperature)
            residual = probabilities * targets[rows].sum() - targets[rows]
            gradient += normalized[rows].T @ residual / temperature
        weights -= learning_rate * (gradient / len(sizes) + l2 * weights)
    return {"weights": weights, "feature_mean": mean, "feature_scale": scale}
```

and declares:

```json
{"material_symbols": ["train_model", "_group_bounds", "_group_softmax"]}
```

Accepted because `train_model` is a top-level function whose body changed, and the two helpers are
newly added top-level functions in a file reachable from `candidate.py`.

Contrast, each of these is REJECTED:
- Returning `candidate.py`. It is a protected path; the overlay is refused before any gate runs.
- Declaring `["EPOCHS"]` after changing only that constant. Not a top-level def or class.
- Declaring `["train_model"]` after editing only its docstring. Docstrings are stripped first.
- Declaring `["model_impl.py:train_model"]`. Qualified names never match; use the bare name.
- Adding `sampling.py` with the new logic but never importing it. Unreachable code is invisible."""


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
_SECTIONS: Final = {
    ResearchOperation.PROPOSE: (_BENCHMARK,),
    ResearchOperation.IMPLEMENT: (_PACKAGES, _WORKED_EXAMPLE, _BENCHMARK),
    ResearchOperation.REPAIR: (_PACKAGES, _WORKED_EXAMPLE),
    ResearchOperation.REFLECT: (_BENCHMARK,),
}


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
                "CATEGORICAL CODES, not magnitudes. They identify the video, author, tab and "
                "duration bucket. Embed them (an embedding table or FM latent factors indexed by "
                "the code); using them as continuous numbers is meaningless and will not rank. "
                "Every other column is a genuine continuous feature."
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
) -> str:
    """Return deterministic operation-specific instructions for one provider attempt."""

    if not isinstance(operation, ResearchOperation):
        raise ValueError("operation must be a ResearchOperation")
    if type(schema_retry) is not bool:
        raise ValueError("schema_retry must be bool")
    if not isinstance(source_policy, CandidateSourcePolicy):
        raise ValueError("source_policy must be CandidateSourcePolicy")
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
    sections = "".join(f"\n\n{section}" for section in _SECTIONS[operation])
    return f"{_COMMON}{sections}\n\n{_OPERATION[operation]}{policy}{retry}"


__all__ = ["PROMPT_VERSION", "instructions_for"]

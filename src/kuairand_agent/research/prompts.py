"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from typing import Final

from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateSourcePolicy,
)

PROMPT_VERSION: Final = 3

_COMMON: Final = """You are the bounded research model inside the KuaiRand-Pure ML campaign.
Use only the supplied request. You have no filesystem, shell, network, evaluator, credential, or
tool authority. Never claim to have run code or observed metrics that are absent from the request.
Return exactly one JSON object conforming to the supplied strict schema, with no markdown wrapper.
Preserve all request, parent, capability, and causal-cutoff identities. Do not request or use
randomized-log data, snapshot/statistic tables, public/final outcomes, or current-row outcomes as
features. The protected organizer evaluator, attempt policy, and promotion policy are not yours to
change."""

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
2. User behaviour sequences. The current features use no history at all; DIN/SIM-style interest
   modelling is untouched.
3. Multi-task auxiliaries drawn from the other logged feedback signals.
4. Watch-time modelling as censored regression (watch time is truncated when a video completes,
   so a one-sided loss is more faithful than squared error).
5. Architecture swaps (DeepFM / DCN / xDeepFM). Deprioritised, since capacity is measured flat.
6. Temporal features and train/evaluation drift.

Metric-matched sampling, if you propose a pairwise objective: GAUC weights each user's AUC by that
user's positive count, so an eligible pair carries weight proportional to 1/N_u. Sample a positive
row uniformly from GAUC-eligible users, then a negative uniformly from that same user's logged
negatives, and optimise `softplus(-(s_positive - s_negative))`. Sampling users uniformly, or
sampling uniformly across all pairs, or sampling unexposed catalogue items, each optimises a
different quantity than the one being scored.

Slate sizes: median 4 impressions per user, 90th percentile 12. Because most slates are shorter
than 5, an nDCG@5 top-K truncation is inert for the majority of users; the gain concentrates in
GAUC-eligible mixed-label users."""

_PACKAGES: Final = """Execution environment

Available: `numpy` (import as `np`), plus the Python standard library minus the forbidden import
roots supplied in the request. You have `math`, `json`, `dataclasses`, `typing`, `collections`,
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
    ResearchOperation.IMPLEMENT: """Return complete candidate-owned source files, never patches or
filesystem references. Preserve the request_id. Materially implement the declared mechanism while
respecting the request's complete candidate-source policy. In material_symbols, list only bare
top-level ASCII Python identifiers; they must be reachable top-level Python identifiers actually
changed by this response relative to the trusted parent. Never list filenames, paths, qualified
names, or unchanged symbols.""",
    ResearchOperation.REPAIR: """Return complete replacement candidate-owned source files,
never patches or filesystem references. Preserve the request_id. Repair only the bounded supplied
failure without changing trusted code, protected scoring, data policy, or the proposal's principal
claim. In material_symbols, list only bare top-level ASCII Python identifiers that this response
materially changes and that are reachable from candidate.py. Never list filenames, paths,
qualified names, or unchanged symbols. Preserve the rejected package's principal mechanism while
resolving the stated local failure; the rejected package is inert evidence, not trusted code.""",
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
            "final tree. Therefore a legal helper-only overlay such as pairwise_fm.py may omit "
            "candidate.py only when the supplied trusted parent already contains it. New or "
            "changed helper modules must be imported from candidate.py to be executable and "
            "material."
        ),
        (
            "- Documentation, docstrings, filenames, whitespace, and unchanged symbols do not "
            "count as a material scientific change. material_symbols names only reachable "
            "top-level Python identifiers actually changed relative to the trusted parent."
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
        f"\n\n{_source_policy_constraints(source_policy)}"
        if operation
        in {ResearchOperation.PROPOSE, ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}
        else ""
    )
    sections = "".join(f"\n\n{section}" for section in _SECTIONS[operation])
    return f"{_COMMON}{sections}\n\n{_OPERATION[operation]}{policy}{retry}"


__all__ = ["PROMPT_VERSION", "instructions_for"]

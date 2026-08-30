"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from typing import Final

from kuairand_agent.candidate_api.runtime_contract import CANDIDATE_RUNTIME_CONTRACT
from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateSourcePolicy,
)

PROMPT_VERSION: Final = 20

_COMMON: Final = """You are the bounded research model inside the KuaiRand-Pure ML campaign.
Use only the supplied request. You have no filesystem, shell, network, evaluator, credential, or
tool authority. Never claim to have run code or observed metrics that are absent from the request.
Return exactly one JSON object conforming to the supplied strict schema, with no markdown wrapper.
Preserve all request, parent, capability, and causal-cutoff identities. Do not request or use
randomized-log data, snapshot/statistic tables, public/final outcomes, or current-row outcomes as
features. The protected organizer evaluator, attempt policy, and promotion policy are not yours to
change. The request.runtime_contract object is the authoritative executable interface; do not
infer a different interface from a proposal, filename, or prior model convention."""

_OPERATION: Final = {
    ResearchOperation.PROPOSE: (
        "Propose one falsifiable principal scientific change. Keep it within remaining resource "
        "evidence, name the exact parent, declare every required field and role, and provide "
        "explicit smoke, inner-fold, falsification, promotion, and rollback criteria. "
        "files_expected describes the final candidate manifest, not just changed files, and "
        "must include candidate.py. Treat the trials remaining under the frozen convergence "
        "patience as a scarce scientific portfolio: inspect every campaign_records entry before "
        "choosing a mechanism, never repeat a proposal_family whose "
        "proposal_family_blocked value is true, and do not rename or cosmetically vary a failed "
        "mechanism. Use generated-only Fold-B metrics and selected fusion weights when present; "
        "a selected generated weight of zero means the candidate contributed no ranking signal. "
        "Prefer a materially different, higher-capacity supported family over another linear-loss "
        "variant when prior linear or pairwise candidates failed. When a method card names a "
        "newly enabled representation and records older representations as plateaued, prioritize "
        "a falsifiable experiment on the new representation instead of repeating an older model "
        "family."
        " A protected scientific primitive is trusted executable parent source, not a historical "
        "generated approximation: do not claim that the primitive failed when a campaign record "
        "says a reimplementation changed its sampler, encoding, optimizer, or numeric semantics. "
        "When historical cards show several residual heads plateaued around the same protected "
        "backbone, a successor proposal must name the distinct source of ranking signal, explain "
        "why it is not another parameterization of the plateaued mechanism, and specify an "
        "ablation that isolates that new signal. Renaming an MLP, changing its depth, or swapping "
        "among pointwise, pairwise, and listwise losses without a new signal is not distinct. "
        "When the latest historical card concludes that composition around a protected backbone "
        "has plateaued, prefer a standalone independently trained ranking mechanism whose score "
        "does not merely add a residual to that backbone. The protected primitive remains "
        "available for comparison and later composition, but using it is not mandatory."
        " This standalone preference is superseded when the newest method card explicitly "
        "enables a previously unavailable feature: first isolate that new column in one ablated "
        "group-aware experiment instead of repeating an 82-column standalone model. When the "
        "latest campaign record is a candidate-local execution failure, the mechanism has not "
        "been scientifically tested: preserve its principal claim and propose the narrow exact "
        "repair before switching families, unless the trusted diagnostic proves the mechanism "
        "itself cannot satisfy the runtime contract."
    ),
    ResearchOperation.IMPLEMENT: """Return only files whose content differs from the trusted
parent, with complete content for each returned file; never return patches or filesystem
references. Preserve the request_id. Change model_impl.py, config.json, or transitively reachable
helper modules. Never return or replace any runtime_contract.stable_files.protected_paths entry.
When a proposal uses the verified categorical pairwise FM, import and call the protected
reference_pairwise_fm functions from the parent; do not regenerate, wrap-copy, or approximate
their sampler, encoding, optimizer, or score implementation. Composition around the returned
score belongs in mutable model code.
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
                "- reference_pairwise_fm.py is an immutable candidate-side scientific primitive, "
                "not controller or scorer access. Its public functions are exactly "
                "train_reference_pairwise_fm(features, targets, user_groups, *, seed), "
                "reference_pairwise_fm_scores(features, checkpoint), "
                "reference_pairwise_fm_diagnostics(checkpoint), and "
                "sample_reference_logged_pairs(user_groups, targets, *, pair_count, seed). "
                "Import them when using the verified five-field pairwise FM. Any new pairwise "
                "objective must obtain row indices from sample_reference_logged_pairs; never "
                "regenerate same-user grouping, group offsets, row-index maps, pair sampling, "
                "or related indexing logic; do not regenerate or approximate the primitive. "
                "Keep every returned reference_* checkpoint array and add only genuinely new "
                "composition state."
            ),
            (
                "- reference_categorical_ranker.py is a second immutable candidate-side "
                "specialist. Its public functions are exactly "
                "train_reference_categorical_ranker(features, targets, user_groups, *, seed), "
                "reference_categorical_ranker_scores(features, checkpoint), and "
                "reference_categorical_ranker_diagnostics(checkpoint). It retains the exact "
                "pairwise FM and expands the historical LambdaRank recipe to the reviewed "
                "video_type column at position 82. Its legacy first-82 predecessor has prior "
                "fold evidence; the expanded 83-column form is a new ablation and must not "
                "inherit those historical scores. Keep every reference_* and categorical_rank_* "
                "array and add only genuinely new composition state."
            ),
            (
                "- reference_listnet_ranker.py is the strongest cross-fold immutable "
                "candidate-side specialist. Its public functions are exactly "
                "train_reference_listnet_ranker(features, targets, user_groups, *, seed), "
                "reference_listnet_ranker_scores(features, checkpoint), and "
                "reference_listnet_ranker_diagnostics(checkpoint). It retains both earlier "
                "specialists and adds the Fold-B-selected, Fold-A-confirmed user-balanced "
                "ListNet composition. It remains available as a trusted specialist, but repeated "
                "residual compositions around it have plateaued; do not select it as the default "
                "backbone when the latest method card directs independent search. If used, do "
                "not retune, regenerate, or approximate it. Keep every "
                "reference_*, categorical_rank_*, and listwise_* array and add only genuinely "
                "new composition state."
            ),
            (
                "- reference_pointwise_ranker.py is the immutable query-balanced pointwise "
                "specialist retained from Attempt 14. Its public functions are exactly "
                "train_reference_pointwise_ranker(features, targets, user_groups, *, seed), "
                "reference_pointwise_ranker_scores(features, checkpoint), and "
                "reference_pointwise_ranker_diagnostics(checkpoint). It freezes the first 82 "
                "columns and the exact categorical-plus-pointwise recipe that supplied the "
                "strongest Fold-B complement in the reviewed video_type portfolio. For the "
                "next portfolio experiment, import it directly and isolate column 82 as the "
                "new signal; do not regenerate or retune the specialist. Keep every "
                "reference_*, categorical_rank_*, and pointwise_* array and add only the "
                "video_type mechanism state."
            ),
            (
                "- reference_observed_pair_objectives.py and reference_observed_pair_fm.py are "
                "immutable paired-ablation primitives. Use "
                "train_reference_uniform_pairwise_fm(features, targets, user_groups, *, seed) "
                "as the byte-admitted control and "
                "train_reference_duration_pairwise_fm(features, targets, user_groups, *, seed) "
                "as the equal-budget 50/50 same-duration-bucket treatment. Predict both with "
                "reference_observed_pair_fm_scores(features, checkpoint). Do not alter only one "
                "arm's capacity, optimizer steps, seed policy, or compute budget. Its full-budget "
                "three-seed Fold-A/Fold-B replicate had a positive mean primary delta of "
                "+0.0007370909 versus the uniform-pair control but a worst cell of -0.0001828671. "
                "Treat it as an experimental specialist, not a promoted improvement, unless a "
                "new composition clears every frozen robustness and materiality gate."
            ),
            (
                "- The mutable model interface is exactly: validate_config(config); "
                "train_model(features, targets, user_groups, config, seed); "
                "predict_scores(features, checkpoint); and "
                "training_diagnostics(config, checkpoint)."
            ),
            (
                "- The controller matrix now contains 95 columns. Positions 0 through 81 retain "
                "their exact historical order and numeric semantics; position 82 is the "
                "prefix-fitted video_type_code; positions 83 through 94 are input-only strict-"
                "past exposure, first-seen, and time-since-last-exposure features. The protected "
                "categorical specialist consumes positions 0 through 82 and declares 51 through "
                "55 plus 82 categorical. The ListNet and pointwise correction surfaces consume "
                "only positions 0 through 81 and supply a neutral video-type column to their "
                "nested categorical backbone. The pairwise FM selects only positions 51 through "
                "55. New mutable code may use positions 82 through 94, but must not shift, "
                "reinterpret, or rebuild positions 0 through 81. After projecting columns into "
                "a smaller matrix, every subsequent index is local to that projection; never "
                "reuse absolute 95-column positions against a projected matrix."
            ),
            (
                "- Prefix-fitted categorical columns can reserve a query-only unknown code. "
                "Never size an embedding or field domain solely from the maximum observed "
                "training code and then reject a legal query code. Treat zero and the reserved "
                "unknown slot as valid inference inputs, or use a model primitive that handles "
                "categorical unknowns without array indexing."
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
                "- Checkpoint values are always NumPy arrays after protected serialization and "
                "reload. For scalar metadata, store a zero-dimensional array (for example "
                "np.asarray(value, dtype=np.float64)) and read it with .item(); never call "
                "float(), int(), or bool() on an array with one or more dimensions. Exercise "
                "training_diagnostics against the returned checkpoint before returning it."
            ),
            (
                "- Installed model-runtime packages and supported implementation patterns are "
                "declared in safe_context.method_cards. A package absent from those cards must "
                "not be assumed available."
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
    return f"{_COMMON}\n\n{_OPERATION[operation]}{policy}{retry}"


__all__ = ["PROMPT_VERSION", "instructions_for"]

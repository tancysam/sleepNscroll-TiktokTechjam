"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from typing import Final

from kuairand_agent.candidate_api.runtime_contract import CANDIDATE_RUNTIME_CONTRACT
from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateSourcePolicy,
)

PROMPT_VERSION: Final = 6

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
        f"\n\n{_runtime_contract_constraints()}"
        if operation
        in {ResearchOperation.PROPOSE, ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}
        else ""
    )
    return f"{_COMMON}\n\n{_OPERATION[operation]}{policy}{retry}"


__all__ = ["PROMPT_VERSION", "instructions_for"]

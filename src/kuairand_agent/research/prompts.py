"""Versioned, authority-free instructions for the real research-model adapter."""

from __future__ import annotations

from typing import Final

from kuairand_agent.research.schemas import ResearchOperation

PROMPT_VERSION: Final = 1

_COMMON: Final = """You are the bounded research model inside the KuaiRand-Pure ML campaign.
Use only the supplied request. You have no filesystem, shell, network, evaluator, credential, or
tool authority. Never claim to have run code or observed metrics that are absent from the request.
Return exactly one JSON object conforming to the supplied strict schema, with no markdown wrapper.
Preserve all request, parent, capability, and causal-cutoff identities. Do not request or use
randomized-log data, snapshot/statistic tables, public/final outcomes, or current-row outcomes as
features. The protected organizer evaluator, attempt policy, and promotion policy are not yours to
change."""

_OPERATION: Final = {
    ResearchOperation.PROPOSE: """Propose one falsifiable principal scientific change. Keep it
within remaining resource evidence, name the exact parent, declare every required field and role,
and provide explicit smoke, inner-fold, falsification, promotion, and rollback criteria.""",
    ResearchOperation.IMPLEMENT: """Return complete candidate-owned source files, never patches or
filesystem references. Preserve the request_id. Materially implement the declared mechanism while
respecting the request's file-count, byte, and suffix limits.""",
    ResearchOperation.REPAIR: """Return complete replacement candidate-owned source files,
never patches or filesystem references. Preserve the request_id. Repair only the bounded supplied
failure without changing trusted code, protected scoring, data policy, or the proposal's principal
claim.""",
    ResearchOperation.REFLECT: """Reflect only on the supplied trusted result. Do not invent runs,
metrics, causal claims, or promotions. Recommend closing, retaining a specialist, or proposing a
next experiment using the typed recommendation vocabulary.""",
}


def instructions_for(operation: ResearchOperation, *, schema_retry: bool = False) -> str:
    """Return deterministic operation-specific instructions for one provider attempt."""

    if not isinstance(operation, ResearchOperation):
        raise ValueError("operation must be a ResearchOperation")
    if type(schema_retry) is not bool:
        raise ValueError("schema_retry must be bool")
    retry = (
        " The previous response was rejected by the local strict parser. Correct only the schema "
        "or request-identity violation and return a fresh complete JSON object."
        if schema_retry
        else ""
    )
    return f"{_COMMON}\n\n{_OPERATION[operation]}{retry}"


__all__ = ["PROMPT_VERSION", "instructions_for"]

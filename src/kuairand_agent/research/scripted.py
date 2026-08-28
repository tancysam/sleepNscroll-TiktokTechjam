"""Deterministic research-model adapter for integration and failure-recovery tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kuairand_agent.research.interface import ResearchModelError
from kuairand_agent.research.schemas import (
    GeneratedPackage,
    ImplementationRequest,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RepairRequest,
    ResearchOperation,
)

ScriptedPayload = Proposal | GeneratedPackage | Reflection


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    operation: ResearchOperation
    payload: ScriptedPayload

    def __post_init__(self) -> None:
        if not isinstance(self.operation, ResearchOperation):
            raise ResearchModelError("scripted response operation is invalid")
        expected_type: type[Proposal] | type[GeneratedPackage] | type[Reflection]
        if self.operation is ResearchOperation.PROPOSE:
            expected_type = Proposal
        elif self.operation in {ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}:
            expected_type = GeneratedPackage
        else:
            expected_type = Reflection
        if not isinstance(self.payload, expected_type):
            raise ResearchModelError(
                f"scripted {self.operation.value} response has the wrong payload type"
            )


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    operation: ResearchOperation
    request_digest: str
    response_digest: str


class ScriptedResearchModel:
    """Consume an immutable operation-ordered script and retain deterministic call evidence."""

    def __init__(self, responses: Sequence[ScriptedResponse]) -> None:
        if not responses:
            raise ResearchModelError("scripted research model requires at least one response")
        if any(not isinstance(response, ScriptedResponse) for response in responses):
            raise ResearchModelError("scripted research model received an invalid response")
        self._responses = tuple(responses)
        self._next_index = 0
        self._calls: list[ScriptedCall] = []

    @property
    def calls(self) -> tuple[ScriptedCall, ...]:
        return tuple(self._calls)

    @property
    def remaining_responses(self) -> int:
        return len(self._responses) - self._next_index

    def _consume(
        self,
        operation: ResearchOperation,
        request: ProposalRequest | ImplementationRequest | RepairRequest | ReflectionRequest,
    ) -> ScriptedPayload:
        if self._next_index >= len(self._responses):
            raise ResearchModelError(
                f"scripted research model has no {operation.value} response remaining"
            )
        scripted = self._responses[self._next_index]
        if scripted.operation is not operation:
            raise ResearchModelError(
                f"scripted research model expected {scripted.operation.value}, "
                f"not {operation.value}"
            )
        payload = scripted.payload
        if operation is ResearchOperation.PROPOSE:
            assert isinstance(request, ProposalRequest)
            assert isinstance(payload, Proposal)
            if payload.parent_candidate_id != request.parent_candidate_id:
                raise ResearchModelError("scripted proposal names the wrong parent candidate")
        elif operation in {ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}:
            assert isinstance(payload, GeneratedPackage)
            if payload.request_id != request.request_id:
                raise ResearchModelError(
                    f"scripted {operation.value} response names the wrong request"
                )
        self._next_index += 1
        self._calls.append(
            ScriptedCall(
                operation=operation,
                request_digest=request.digest,
                response_digest=payload.digest,
            )
        )
        return payload

    def propose(self, request: ProposalRequest) -> Proposal:
        if not isinstance(request, ProposalRequest):
            raise ResearchModelError("propose requires ProposalRequest")
        payload = self._consume(ResearchOperation.PROPOSE, request)
        assert isinstance(payload, Proposal)
        return payload

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        if not isinstance(request, ImplementationRequest):
            raise ResearchModelError("implement requires ImplementationRequest")
        payload = self._consume(ResearchOperation.IMPLEMENT, request)
        assert isinstance(payload, GeneratedPackage)
        return payload

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        if not isinstance(request, RepairRequest):
            raise ResearchModelError("repair requires RepairRequest")
        payload = self._consume(ResearchOperation.REPAIR, request)
        assert isinstance(payload, GeneratedPackage)
        return payload

    def reflect(self, request: ReflectionRequest) -> Reflection:
        if not isinstance(request, ReflectionRequest):
            raise ResearchModelError("reflect requires ReflectionRequest")
        payload = self._consume(ResearchOperation.REFLECT, request)
        assert isinstance(payload, Reflection)
        return payload

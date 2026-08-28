"""Small typed seam implemented by scripted and real research-model adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kuairand_agent.research.schemas import (
    GeneratedPackage,
    ImplementationRequest,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RepairRequest,
)


class ResearchModelError(RuntimeError):
    """A research adapter failed without granting it authority over campaign policy."""


@runtime_checkable
class ResearchModel(Protocol):
    """Typed propose, implement, repair, and reflect interface for the controller."""

    def propose(self, request: ProposalRequest) -> Proposal: ...

    def implement(self, request: ImplementationRequest) -> GeneratedPackage: ...

    def repair(self, request: RepairRequest) -> GeneratedPackage: ...

    def reflect(self, request: ReflectionRequest) -> Reflection: ...

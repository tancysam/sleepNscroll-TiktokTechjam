"""Optional proposal adapters constrained to the canonical experiment language."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from kuairand_agent.domain.experiment import ExperimentSpec, MechanismMetadata
from kuairand_agent.domain.identity import ContractId, FamilyId
from kuairand_agent.research.schemas import Proposal as LegacyProposal


class ProposalFailureKind(StrEnum):
    UNAVAILABLE = "UNAVAILABLE"
    DEADLINE = "DEADLINE"
    TRANSPORT = "TRANSPORT"
    MALFORMED = "MALFORMED"
    OUTSIDE_ALLOWLIST = "OUTSIDE_ALLOWLIST"
    INTERNAL = "INTERNAL"


class ProposalAdapterError(RuntimeError):
    """An optional adapter failed without changing deterministic campaign science."""

    def __init__(self, kind: ProposalFailureKind, diagnostic: str) -> None:
        if not isinstance(kind, ProposalFailureKind):
            raise TypeError("kind must be ProposalFailureKind")
        normalized = " ".join(diagnostic.split())[:1000]
        super().__init__(normalized or kind.value)
        self.kind = kind
        self.diagnostic = normalized or kind.value


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Label- and score-free context made available to proposal adapters."""

    contract_id: ContractId
    parent_experiment_ref: str | None
    allowed_proposal_keys: tuple[str, ...]
    completed_family_ids: tuple[FamilyId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.contract_id, ContractId):
            raise ValueError("proposal context requires one ContractId")
        if self.parent_experiment_ref is not None and not self.parent_experiment_ref:
            raise ValueError("parent experiment reference cannot be empty")
        if not self.allowed_proposal_keys:
            raise ValueError("proposal context requires an allowlisted next proposal")
        if len(self.allowed_proposal_keys) != len(set(self.allowed_proposal_keys)):
            raise ValueError("allowed proposal keys contain duplicates")
        if any(not isinstance(value, FamilyId) for value in self.completed_family_ids):
            raise ValueError("completed family identities must be FamilyId values")
        if len(self.completed_family_ids) != len(set(self.completed_family_ids)):
            raise ValueError("completed family identities contain duplicates")


@runtime_checkable
class ProposalAdapter(Protocol):
    """One small optional seam: return a valid spec or a typed adapter failure."""

    def propose(self, context: ProposalContext) -> ExperimentSpec: ...


class LegacyProposalProvider(Protocol):
    """The proposal-only subset of the existing research-model interface."""

    def propose(self, request: object) -> LegacyProposal: ...


@dataclass(frozen=True, slots=True)
class ExistingProviderProposalAdapter:
    """Translate existing provider output by selecting a trusted registered experiment.

    Provider fields never become trainer parameters, feature identifiers, targets, source code, or
    parent-selection fitness.  The existing ``proposal_id`` selects one catalog entry and its prose
    is retained only as non-semantic mechanism metadata.
    """

    provider: LegacyProposalProvider = field(repr=False)
    request_factory: Callable[[ProposalContext], object] = field(repr=False)
    catalog: Mapping[str, ExperimentSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, Mapping) or not self.catalog:
            raise ValueError("existing-provider adapter requires a non-empty trusted catalog")
        normalized: dict[str, ExperimentSpec] = {}
        for key, spec in self.catalog.items():
            if type(key) is not str or not isinstance(spec, ExperimentSpec):
                raise ValueError("existing-provider catalog entries must be ExperimentSpec values")
            if key != spec.proposal_key:
                raise ValueError("catalog key must equal ExperimentSpec.proposal_key")
            normalized[key] = spec
        object.__setattr__(self, "catalog", MappingProxyType(dict(sorted(normalized.items()))))

    def propose(self, context: ProposalContext) -> ExperimentSpec:
        try:
            legacy = self.provider.propose(self.request_factory(context))
        except TimeoutError as exc:
            raise ProposalAdapterError(ProposalFailureKind.DEADLINE, str(exc)) from exc
        except (ConnectionError, OSError) as exc:
            raise ProposalAdapterError(ProposalFailureKind.TRANSPORT, str(exc)) from exc
        except ProposalAdapterError:
            raise
        except Exception as exc:
            raise ProposalAdapterError(ProposalFailureKind.INTERNAL, str(exc)) from exc
        if not isinstance(legacy, LegacyProposal):
            raise ProposalAdapterError(
                ProposalFailureKind.MALFORMED,
                "existing proposal provider returned a value outside its schema",
            )
        return translate_existing_proposal(legacy, context=context, catalog=self.catalog)


def translate_existing_proposal(
    legacy: LegacyProposal,
    *,
    context: ProposalContext,
    catalog: Mapping[str, ExperimentSpec],
) -> ExperimentSpec:
    """Translate old provider prose without granting it executable authority."""

    if legacy.proposal_id not in context.allowed_proposal_keys:
        raise ProposalAdapterError(
            ProposalFailureKind.OUTSIDE_ALLOWLIST,
            f"provider selected non-admitted proposal key {legacy.proposal_id!r}",
        )
    try:
        selected = catalog[legacy.proposal_id]
    except KeyError as exc:
        raise ProposalAdapterError(
            ProposalFailureKind.OUTSIDE_ALLOWLIST,
            f"provider selected unknown proposal key {legacy.proposal_id!r}",
        ) from exc
    if (
        context.parent_experiment_ref is not None
        and selected.parent_experiment_refs
        and selected.parent_experiment_refs[0] != context.parent_experiment_ref
    ):
        raise ProposalAdapterError(
            ProposalFailureKind.OUTSIDE_ALLOWLIST,
            "trusted proposal parent does not match the current deterministic parent",
        )
    metadata = MechanismMetadata(
        mechanism=legacy.mechanism,
        falsifiable_hypothesis=legacy.hypothesis,
        expected_metric_effect=("Expected effect on " + ", ".join(legacy.expected_metric_effects)),
        leakage_argument=f"Causal cutoff declared by legacy proposal: {legacy.causal_cutoff}",
        rejection_criterion=legacy.falsification_criteria,
        attributions=legacy.attributions,
    )
    return selected.with_metadata(metadata)

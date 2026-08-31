from __future__ import annotations

from dataclasses import dataclass

import pytest

from kuairand_agent.domain.experiment import ExperimentSpec, ModelFamily, RequiredAblation
from kuairand_agent.domain.identity import ContractId
from kuairand_agent.proposal.deterministic import DeterministicProposalLadder
from kuairand_agent.proposal.protocol import (
    ExistingProviderProposalAdapter,
    ProposalAdapterError,
    ProposalContext,
    ProposalFailureKind,
)
from kuairand_agent.research.schemas import Proposal, RequiredField


def _legacy(proposal_id: str) -> Proposal:
    return Proposal(
        proposal_id=proposal_id,
        hypothesis="Provider prose for a bounded registered experiment.",
        mechanism="Provider mechanism prose.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="official-fm",
        principal_change="Choose one registered experiment.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(source_field="user_id", role="input", purpose="organizer grouping"),
        ),
        objective="Provider does not control this executable field.",
        sampling="Provider does not control sampling.",
        grouping="Provider does not control grouping.",
        weighting="Provider does not control weights.",
        causal_cutoff="strict past",
        estimated_runtime_seconds=10,
        estimated_memory_mb=10,
        smoke_plan="Run the registered deterministic smoke test.",
        inner_fold_plan="Use the registered folds.",
        falsification_criteria="Reject on the registered matched control.",
        promotion_criteria="Use the frozen policy.",
        maximum_repairs=0,
        rollback_parent_id="official-fm",
        attributions=("advisory source",),
    )


def test_mandatory_ladder_is_complete_ordered_and_provider_free() -> None:
    proposals = DeterministicProposalLadder().proposals()
    assert tuple(value.proposal_key for value in proposals) == (
        "official-fm",
        "lambdarank-base",
        "pointwise-base-control",
        "lambdarank-strict-past",
        "pointwise-strict-past-control",
        "exact-rank-fusion",
    )
    assert all(isinstance(value, ExperimentSpec) for value in proposals)
    assert proposals[0].model_family is ModelFamily.OFFICIAL_FM
    assert proposals[1].model_family is ModelFamily.LIGHTGBM_LAMBDARANK
    assert proposals[2].model_family is ModelFamily.LIGHTGBM_POINTWISE
    assert proposals[3].strict_past.enabled
    assert RequiredAblation.NO_STRICT_PAST_FEATURES in proposals[3].required_ablations
    assert proposals[-1].model_family is ModelFamily.EXACT_RANK_FUSION
    assert proposals[-1].rank_fusion is not None
    assert sum(value.weight for value in proposals[-1].rank_fusion.members) == 1.0


@dataclass
class _Provider:
    output: Proposal
    calls: int = 0

    def propose(self, request: object) -> Proposal:
        self.calls += 1
        return self.output


def test_existing_provider_output_only_selects_registered_semantics() -> None:
    ladder = DeterministicProposalLadder()
    catalog = ladder.catalog()
    provider = _Provider(_legacy("lambdarank-base"))
    adapter = ExistingProviderProposalAdapter(
        provider=provider,
        request_factory=lambda context: context,
        catalog=catalog,
    )
    context = ProposalContext(
        contract_id=ContractId("c" * 64),
        parent_experiment_ref="official-fm",
        allowed_proposal_keys=("lambdarank-base",),
    )
    selected = adapter.propose(context)

    assert provider.calls == 1
    assert selected.semantic_digest == catalog["lambdarank-base"].semantic_digest
    assert selected.metadata.mechanism == "Provider mechanism prose."
    assert selected.hyperparameters == catalog["lambdarank-base"].hyperparameters


def test_provider_cannot_select_an_unregistered_or_unadmitted_experiment() -> None:
    catalog = DeterministicProposalLadder().catalog()
    adapter = ExistingProviderProposalAdapter(
        provider=_Provider(_legacy("lambdarank-base")),
        request_factory=lambda context: context,
        catalog=catalog,
    )
    context = ProposalContext(
        contract_id=ContractId("c" * 64),
        parent_experiment_ref="official-fm",
        allowed_proposal_keys=("pointwise-base-control",),
    )
    with pytest.raises(ProposalAdapterError) as failure:
        adapter.propose(context)
    assert failure.value.kind is ProposalFailureKind.OUTSIDE_ALLOWLIST

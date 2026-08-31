from __future__ import annotations

from kuairand_agent.domain.experiment import MechanismMetadata
from kuairand_agent.domain.identity import ContractId
from kuairand_agent.proposal.deterministic import deterministic_proposals
from kuairand_agent.search.family_ledger import (
    BranchResult,
    FamilyLedger,
    FamilyLedgerKey,
)


def test_family_key_is_contract_and_semantic_family_not_campaign_or_prose() -> None:
    contract = ContractId("a" * 64)
    spec = deterministic_proposals()[1]
    reworded = spec.with_metadata(
        MechanismMetadata(
            mechanism="Reworded mechanism.",
            falsifiable_hypothesis="Reworded hypothesis.",
            expected_metric_effect="Reworded expectation.",
            leakage_argument="Reworded leakage argument.",
            rejection_criterion="Reworded rejection rule.",
        )
    )
    assert FamilyLedgerKey.for_spec(contract, spec) == FamilyLedgerKey.for_spec(contract, reworded)
    assert FamilyLedgerKey.for_spec(ContractId("b" * 64), spec) != FamilyLedgerKey.for_spec(
        contract, spec
    )


def test_infrastructure_failure_does_not_close_family_but_negative_science_does() -> None:
    contract = ContractId("a" * 64)
    spec = deterministic_proposals()[1]
    ledger = FamilyLedger(max_negative_results=1)

    failed = ledger.record(
        contract_id=contract,
        spec=spec,
        result=BranchResult.INFRASTRUCTURE_FAILURE,
    )
    assert not failed.closed
    assert not ledger.is_closed(contract_id=contract, spec=spec)

    negative = ledger.record(
        contract_id=contract,
        spec=spec,
        result=BranchResult.NO_IMPROVEMENT,
    )
    assert negative.closed
    assert ledger.is_closed(contract_id=contract, spec=spec)


def test_replaying_same_branch_result_is_idempotent() -> None:
    contract = ContractId("a" * 64)
    spec = deterministic_proposals()[3]
    ledger = FamilyLedger(max_negative_results=2)
    first = ledger.record(
        contract_id=contract,
        spec=spec,
        result=BranchResult.NO_IMPROVEMENT,
    )
    replay = ledger.record(
        contract_id=contract,
        spec=spec,
        result=BranchResult.NO_IMPROVEMENT,
    )
    assert first == replay
    assert replay.scientific_negative_count == 1
    assert not replay.closed


def test_fingerprint_manifest_contains_the_exact_branch_closure_axes() -> None:
    entry = FamilyLedger().record(
        contract_id=ContractId("a" * 64),
        spec=deterministic_proposals()[-1],
        result=BranchResult.IMPROVED,
    )
    fingerprint = entry.fingerprints[0].manifest()
    assert set(fingerprint) == {
        "representation",
        "model_family",
        "objective",
        "temporal_policy",
        "fusion_member",
        "result",
        "fingerprint",
    }
    assert entry.closed

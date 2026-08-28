from __future__ import annotations

import pytest

from kuairand_agent.research.interface import ResearchModel, ResearchModelError
from kuairand_agent.research.schemas import (
    ExperimentResultSummary,
    GeneratedFile,
    GeneratedPackage,
    ImplementationRequest,
    ParentSnapshot,
    ParentSourceFile,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RepairRequest,
    RequiredField,
    ResearchOperation,
)
from kuairand_agent.research.scripted import ScriptedResearchModel, ScriptedResponse


def parent() -> ParentSnapshot:
    return ParentSnapshot(
        candidate_id="fm-seed",
        files=(ParentSourceFile.create("candidate.py", "def score(x):\n    return x\n"),),
    )


def proposal_fixture() -> Proposal:
    return Proposal(
        proposal_id="tab-bias-v1",
        hypothesis="A learned tab offset improves within-user ordering.",
        mechanism="Add a train-fitted tab bias to the candidate score.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="fm-seed",
        principal_change="One tab-bias scoring term.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:tab",
                "inference_input",
                "Fit and apply a categorical tab bias.",
            ),
        ),
        objective="binary cross entropy",
        sampling="all logged impressions",
        grouping="user impression groups",
        weighting="uniform rows",
        causal_cutoff="No response-derived feature is used.",
        estimated_runtime_seconds=30,
        estimated_memory_mb=256,
        smoke_plan="Fit two tabs and predict four rows.",
        inner_fold_plan="Screen seed 0 on Fold B.",
        falsification_criteria="Reject without a Fold B improvement.",
        promotion_criteria="Require positive mean across A and B.",
        maximum_repairs=1,
        rollback_parent_id="fm-seed",
        attributions=("local scripted vertical-slice fixture",),
    )


def generated(request_id: str, response_id: str, source: str) -> GeneratedPackage:
    return GeneratedPackage(
        request_id=request_id,
        response_id=response_id,
        files=(GeneratedFile("candidate.py", source),),
        material_change_summary="Change score.",
        material_symbols=("score",),
    )


def test_scripted_model_implements_the_typed_four_operation_interface() -> None:
    proposal = proposal_fixture()
    bad = generated("implementation-request", "bad", "def score(:\n")
    fixed = generated("repair-request", "fixed", "def score(x):\n    return x + 0.125\n")
    reflection = Reflection(
        response_id="reflection-1",
        summary="The tab-bias fixture completed.",
        recommendation="propose_next",
        lessons=("Retain the material source lineage.",),
    )
    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, proposal),
            ScriptedResponse(ResearchOperation.IMPLEMENT, bad),
            ScriptedResponse(ResearchOperation.REPAIR, fixed),
            ScriptedResponse(ResearchOperation.REFLECT, reflection),
        )
    )
    assert isinstance(model, ResearchModel)
    safe_context = {"schema_version": 1, "budgets": {"remaining_attempts": 1}}
    proposal_request = ProposalRequest.create(
        request_id="proposal-request",
        campaign_id="campaign-1",
        scientific_iteration=1,
        parent_candidate_id="fm-seed",
        safe_context=safe_context,
    )
    implementation_request = ImplementationRequest.create(
        request_id="implementation-request",
        proposal=proposal,
        parent=parent(),
        safe_context=safe_context,
    )
    repair_request = RepairRequest.create(
        request_id="repair-request",
        proposal_id=proposal.proposal_id,
        failed_candidate_id="bad-child",
        failed_child=parent(),
        failure_category="syntax_error",
        diagnostics="candidate.py:1 invalid syntax",
        remaining_repairs=1,
        safe_context=safe_context,
    )
    result = ExperimentResultSummary(
        tier="fixture",
        status="passed",
        gauc=0.6,
        ndcg_at_5=0.5,
        primary=0.55,
        runtime_seconds=0.1,
        peak_memory_mb=20.0,
    )
    reflection_request = ReflectionRequest.create(
        request_id="reflection-request",
        proposal_id=proposal.proposal_id,
        candidate_id="fixed-child",
        source_digest="a" * 64,
        diff_digest="b" * 64,
        result=result,
        safe_context=safe_context,
    )

    assert model.propose(proposal_request) is proposal
    assert model.implement(implementation_request) is bad
    assert model.repair(repair_request) is fixed
    assert model.reflect(reflection_request) is reflection
    assert [call.operation for call in model.calls] == list(ResearchOperation)
    assert all(len(call.request_digest) == 64 for call in model.calls)
    assert all(len(call.response_digest) == 64 for call in model.calls)


def test_scripted_model_fails_closed_on_operation_order() -> None:
    proposal = proposal_fixture()
    model = ScriptedResearchModel((ScriptedResponse(ResearchOperation.PROPOSE, proposal),))
    request = ImplementationRequest.create(
        request_id="implementation-request",
        proposal=proposal,
        parent=parent(),
        safe_context={"schema_version": 1},
    )

    with pytest.raises(ResearchModelError, match="expected propose"):
        model.implement(request)

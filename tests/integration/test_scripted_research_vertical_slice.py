from __future__ import annotations

from pathlib import Path

import pytest

from kuairand_agent.research.context import ResearchBudgetContext, build_safe_research_context
from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.materialize import (
    CandidateStaticError,
    materialize_candidate,
    require_material_executable_change,
    snapshot_materialized_candidate,
    validate_candidate_static,
)
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
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

ROOT = Path(__file__).parents[2]
BASE = """\
def train(duration_ms, labels):
    return 0.0


def predict(duration_ms, tab_bias):
    return [0.0 for _ in duration_ms]
"""
FIXED = """\
def train(duration_ms, labels):
    return 0.125


def predict(duration_ms, tab_bias):
    return [float(value) + tab_bias for value in duration_ms]
"""


def proposal_fixture() -> Proposal:
    return Proposal(
        proposal_id="duration-tab-bias-v1",
        hypothesis="A legal duration signal plus tab bias improves the fixture ranking.",
        mechanism="Fit one tab offset and add it to duration-derived scores.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="fm-seed",
        principal_change="Replace constant prediction with a duration and tab-bias score.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:duration_ms",
                "inference_input",
                "Legal current-row duration signal.",
            ),
        ),
        objective="fixture binary ranking",
        sampling="all fixture impressions",
        grouping="user impression groups",
        weighting="uniform rows",
        causal_cutoff="No response history is used.",
        estimated_runtime_seconds=1,
        estimated_memory_mb=64,
        smoke_plan="Train and predict four synthetic rows.",
        inner_fold_plan="Use the fixture as a structural proxy for Fold B.",
        falsification_criteria="Reject a non-finite or non-improving fixture result.",
        promotion_criteria="Require all structural and scorer gates.",
        maximum_repairs=1,
        rollback_parent_id="fm-seed",
        attributions=("scripted deterministic test mechanism",),
    )


def execute_candidate(source: str) -> list[float]:
    namespace: dict[str, object] = {}
    exec(compile(source, "candidate.py", "exec"), namespace)
    train = namespace["train"]
    predict = namespace["predict"]
    assert callable(train)
    assert callable(predict)
    duration_ms = [0.1, 0.9, 0.8, 0.2]
    labels = [0, 1, 1, 0]
    bias = train(duration_ms, labels)
    result = predict(duration_ms, bias)
    assert isinstance(result, list)
    return [float(value) for value in result]


def test_scripted_syntax_failure_repairs_and_completes_protected_score_loop(
    tmp_path: Path,
) -> None:
    proposal = proposal_fixture()
    parent = ParentSnapshot(
        candidate_id="fm-seed",
        files=(ParentSourceFile.create("candidate.py", BASE),),
    )
    bad_package = GeneratedPackage(
        request_id="implement-1",
        response_id="bad-syntax",
        files=(GeneratedFile("candidate.py", "def train(:\n"),),
        material_change_summary="Attempt the duration and tab-bias implementation.",
        material_symbols=("train", "predict"),
    )
    fixed_package = GeneratedPackage(
        request_id="repair-1",
        response_id="fixed-syntax",
        files=(GeneratedFile("candidate.py", FIXED),),
        material_change_summary="Repair syntax while retaining the claimed mechanism.",
        material_symbols=("train", "predict"),
    )
    reflection = Reflection(
        response_id="reflect-1",
        summary="The repaired child passed fixture scoring.",
        recommendation="propose_next",
        lessons=("A failed child did not alter the parent or incumbent.",),
    )
    model: ResearchModel = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, proposal),
            ScriptedResponse(ResearchOperation.IMPLEMENT, bad_package),
            ScriptedResponse(ResearchOperation.REPAIR, fixed_package),
            ScriptedResponse(ResearchOperation.REFLECT, reflection),
        )
    )
    context = build_safe_research_context(
        starter_manifest_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(44, 18_000, 6, 0),
    ).to_wire()

    proposed = model.propose(
        ProposalRequest.create(
            request_id="propose-1",
            campaign_id="campaign-1",
            scientific_iteration=1,
            parent_candidate_id="fm-seed",
            safe_context=context,
        )
    )
    implementation_request = ImplementationRequest.create(
        request_id="implement-1",
        proposal=proposed,
        parent=parent,
        safe_context=context,
    )
    implementation = model.implement(implementation_request)
    bad_child = materialize_candidate(parent, implementation, tmp_path / "bad-child")
    with pytest.raises(CandidateStaticError, match="invalid Python") as failure:
        validate_candidate_static(bad_child)

    failed_snapshot = snapshot_materialized_candidate(bad_child, candidate_id="bad-child")
    repair_request = RepairRequest.create(
        request_id="repair-1",
        proposal_id=proposal.proposal_id,
        failed_candidate_id="bad-child",
        failed_child=failed_snapshot,
        failure_category="syntax_error",
        diagnostics=str(failure.value),
        remaining_repairs=1,
        safe_context=context,
    )
    repaired = model.repair(repair_request)
    fixed_child = materialize_candidate(failed_snapshot, repaired, tmp_path / "fixed-child")
    validate_candidate_static(fixed_child)
    evidence = require_material_executable_change(parent, fixed_child)
    assert evidence.changed_symbols == ("candidate.py:predict", "candidate.py:train")

    # Exact parent request/response reproduces the logical child and controller diff in a fresh
    # location.  No model call or human edit participates in this replay.
    replayed = materialize_candidate(failed_snapshot, repaired, tmp_path / "replayed-child")
    assert replayed.source_digest == fixed_child.source_digest
    assert replayed.diff_digest == fixed_child.diff_digest
    assert replayed.unified_diff == fixed_child.unified_diff

    scores = execute_candidate(fixed_child.file("candidate.py").content)
    split = SplitIdentity("fixture", "fixture-token", 4)
    alignment = Alignment.from_ids(
        split=split,
        user_ids=("u1", "u1", "u2", "u2"),
        video_ids=("v1", "v2", "v3", "v4"),
    )
    scored = ProtectedScorer(
        starter_dir=ROOT / "kuairand-starter-kit", trusted_alignment=alignment
    ).score(
        alignment=alignment,
        split=split,
        labels=(0, 1, 1, 0),
        scores=scores,
    )
    assert scored.gauc == 1.0
    assert scored.ndcg_at_5 == 1.0
    result = ExperimentResultSummary(
        tier="fixture",
        status="passed",
        gauc=scored.gauc,
        ndcg_at_5=scored.ndcg_at_5,
        primary=scored.primary,
        runtime_seconds=scored.runtime_seconds,
        peak_memory_mb=1.0,
    )
    reflected = model.reflect(
        ReflectionRequest.create(
            request_id="reflect-request",
            proposal_id=proposal.proposal_id,
            candidate_id="fixed-child",
            source_digest=fixed_child.source_digest,
            diff_digest=fixed_child.diff_digest,
            result=result,
            safe_context=context,
        )
    )
    assert reflected is reflection
    assert parent.file("candidate.py").content == BASE

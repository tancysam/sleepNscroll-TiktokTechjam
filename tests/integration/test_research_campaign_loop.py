from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.candidate_api.protocol import (
    PredictionExpectation,
    validate_prediction_outputs,
)
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.policy import (
    ApprovedInput,
    CandidateInputRole,
    SplitRole,
    WorkspacePolicy,
)
from kuairand_agent.execution.runner import ExecutionResult, Runner
from kuairand_agent.execution.workspace import CandidateWorkspace, WorkspaceMaterializer
from kuairand_agent.research.context import ResearchBudgetContext, build_safe_research_context
from kuairand_agent.research.loop import (
    CampaignStoreResearchLedger,
    CandidateEvidence,
    LocalExecutionTemplate,
    ResearchCampaignLoop,
    SelectionDecision,
    TrustedEvaluation,
    artifact_spec,
)
from kuairand_agent.research.materialize import MaterializedCandidate
from kuairand_agent.research.schemas import (
    ExperimentResultSummary,
    GeneratedFile,
    GeneratedPackage,
    ParentSnapshot,
    ParentSourceFile,
    Proposal,
    Reflection,
    RequiredField,
    ResearchOperation,
)
from kuairand_agent.research.scripted import ScriptedResearchModel, ScriptedResponse
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

ROOT = Path(__file__).parents[2]
SHA = "a" * 64


def _candidate_source(expression: str) -> str:
    return f"""\
import hashlib
import json
from pathlib import Path

import numpy as np


def score(rows):
    {expression}


def main():
    request = json.loads(Path("request.json").read_text(encoding="utf-8"))
    input_path = request["approved_inputs"][0]["workspace_path"]
    rows = json.loads(Path(input_path).read_text(encoding="utf-8"))
    predictions = np.asarray(score(rows), dtype="<f8")
    scores_path = Path("output/scores.npy")
    with scores_path.open("xb") as handle:
        np.save(handle, predictions, allow_pickle=False)
    identity = request["request"]
    result = {{
        "schema_version": 1,
        "kind": "prediction",
        "source_digest": identity["source_digest"],
        "config_digest": identity["config_digest"],
        "data_digest": identity["data_digest"],
        "split_token": identity["split_token"],
        "checkpoint_digest": identity["checkpoint_digest"],
        "expected_count": len(predictions),
        "dtype": "<f8",
        "scores_path": "scores.npy",
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "diagnostics": {{}},
    }}
    Path("output/prediction_result.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
"""


BASE_SOURCE = _candidate_source("return [0.0 for _ in rows]")
COARSE_SOURCE = _candidate_source('return [float(row["coarse"]) for row in rows]')
FINE_SOURCE = _candidate_source('return [float(row["fine"]) for row in rows]')
RUNTIME_FAILURE_SOURCE = _candidate_source('raise RuntimeError("deliberate runtime child")')


def _proposal(iteration: int, parent: str) -> Proposal:
    return Proposal(
        proposal_id=f"proposal-{iteration}",
        hypothesis=f"Legal fixture signal {iteration} can improve logged-impression ranking.",
        mechanism="Replace the current score function with one legal input-only transform.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id=parent,
        principal_change="Change only the executable score mechanism.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:duration_ms",
                "inference_input",
                "Legal current-row input used only as a fixture analogue.",
            ),
        ),
        objective="long_view ranking over logged impressions",
        sampling="all fixture impressions",
        grouping="user impression groups",
        weighting="uniform impressions",
        causal_cutoff="No current or future outcome is consumed.",
        estimated_runtime_seconds=1,
        estimated_memory_mb=64,
        smoke_plan="Execute the generated entry point on four input-only fixture rows.",
        inner_fold_plan="Use the deterministic fixture as a structural fold proxy.",
        falsification_criteria="Reject invalid source, process failure, or non-improvement.",
        promotion_criteria="Require trusted scoring and strict primary improvement.",
        maximum_repairs=1,
        rollback_parent_id=parent,
        attributions=("scripted vertical-slice mechanism",),
    )


class FixtureEvaluator:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts
        split = SplitIdentity("fixture", "fixture-token", 4)
        self.split = split
        self.alignment = Alignment.from_ids(
            split=split,
            user_ids=("u1", "u1", "u2", "u2"),
            video_ids=("v1", "v2", "v3", "v4"),
        )
        self.scorer = ProtectedScorer(
            starter_dir=ROOT / "kuairand-starter-kit", trusted_alignment=self.alignment
        )
        self.calls: list[str] = []

    def evaluate(
        self,
        *,
        candidate: MaterializedCandidate,
        workspace: CandidateWorkspace,
        execution: ExecutionResult,
    ) -> TrustedEvaluation:
        assert execution.candidate_metrics_accepted is False
        assert "metrics" not in execution.manifest()
        self.calls.append(candidate.source_digest)
        validated = validate_prediction_outputs(
            workspace.output_dir,
            PredictionExpectation(
                source_digest=candidate.source_digest,
                config_digest="b" * 64,
                data_digest="c" * 64,
                split_token="fixture-token",
                checkpoint_digest="d" * 64,
                expected_count=4,
            ),
        )
        scored = self.scorer.score(
            alignment=self.alignment,
            split=self.split,
            labels=(0, 1, 1, 0),
            scores=validated.scores,
        )
        prediction_ref = self.artifacts.put_file(
            validated.scores_path, kind=ArtifactKind.PREDICTION
        )
        closure_digest = hashlib.sha256(
            f"closure:{prediction_ref.sha256}:{'d' * 64}".encode()
        ).hexdigest()
        return TrustedEvaluation(
            summary=ExperimentResultSummary(
                tier="outer",
                status="passed",
                gauc=scored.gauc,
                ndcg_at_5=scored.ndcg_at_5,
                primary=scored.primary,
                runtime_seconds=execution.wall_seconds,
                peak_memory_mb=execution.peak_tree_rss_bytes / (1024 * 1024),
            ),
            scorer_digest=self.scorer.scorer_digest,
            prediction_digest=prediction_ref.sha256,
            eligible=True,
            replay_verified=True,
            checkpoint_digest="d" * 64,
            artifact_closure_digest=closure_digest,
            artifacts=(("trusted_predictions", artifact_spec(prediction_ref)),),
        )


class StrictFixtureSelector:
    def decide(
        self, incumbent_primary: float | None, candidate: CandidateEvidence
    ) -> SelectionDecision:
        candidate_primary = candidate.evaluation.summary.primary
        promote = (
            candidate.evaluation.eligible
            and candidate.evaluation.replay_verified
            and candidate_primary is not None
            and (incumbent_primary is None or candidate_primary > incumbent_primary)
        )
        return SelectionDecision(
            promote=promote,
            eligibility="fixture_outer_eligible",
            reason="strict trusted fixture improvement" if promote else "retain incumbent",
        )


def _store(tmp_path: Path) -> CampaignStore:
    store = CampaignStore.create(
        tmp_path / "campaign.sqlite3",
        campaign_id="scripted-campaign",
        config_digest="1" * 64,
        benchmark_digest="2" * 64,
        starter_digest="3" * 64,
        dataset_digest="4" * 64,
        environment_digest="5" * 64,
        hard_deadline_utc="2030-01-01T00:00:00+00:00",
        initial_convergence=ConvergenceState.initial(0.5).manifest(),
        max_launches=6,
    )
    store.record_incumbent(
        incumbent_id="fm-fallback",
        eligibility="official_fm_qualified",
        source_digest="6" * 64,
        checkpoint_digest="7" * 64,
        artifact_closure_digest="8" * 64,
        replay_verified=True,
        is_fallback=True,
        expected_revision=store.snapshot().revision,
        reason="immutable qualified fallback fixture",
        outer_primary_mean=0.5,
    )
    return store


def test_three_iteration_scripted_campaign_repairs_selects_and_retains_evidence(
    tmp_path: Path,
) -> None:
    parent = ParentSnapshot(
        candidate_id="fm-fallback",
        files=(ParentSourceFile.create("candidate.py", BASE_SOURCE),),
    )
    responses = (
        ScriptedResponse(ResearchOperation.PROPOSE, _proposal(1, "fm-fallback")),
        ScriptedResponse(
            ResearchOperation.IMPLEMENT,
            GeneratedPackage(
                request_id="iteration-01-implement",
                response_id="iteration-01-source",
                files=(GeneratedFile("candidate.py", COARSE_SOURCE),),
                material_change_summary="Use the coarse legal fixture signal.",
                material_symbols=("score",),
            ),
        ),
        ScriptedResponse(
            ResearchOperation.REFLECT,
            Reflection(
                "iteration-01-reflection",
                "The material coarse signal passed trusted scoring.",
                "propose_next",
                ("A legal input-only transform improved the fixture.",),
            ),
        ),
        ScriptedResponse(
            ResearchOperation.PROPOSE,
            _proposal(2, "iteration-01-child-0"),
        ),
        ScriptedResponse(
            ResearchOperation.IMPLEMENT,
            GeneratedPackage(
                request_id="iteration-02-implement",
                response_id="iteration-02-bad-syntax",
                files=(GeneratedFile("candidate.py", "def score(:\n"),),
                material_change_summary="Attempt the fine fixture signal.",
                material_symbols=("score",),
            ),
        ),
        ScriptedResponse(
            ResearchOperation.REPAIR,
            GeneratedPackage(
                request_id="iteration-02-repair-1",
                response_id="iteration-02-fixed-syntax",
                files=(GeneratedFile("candidate.py", FINE_SOURCE),),
                material_change_summary="Repair syntax and retain the fine signal mechanism.",
                material_symbols=("score",),
            ),
        ),
        ScriptedResponse(
            ResearchOperation.REFLECT,
            Reflection(
                "iteration-02-reflection",
                "The exact failed child was repaired and improved trusted metrics.",
                "propose_next",
                ("One bounded syntax repair preserved the principal change.",),
            ),
        ),
        ScriptedResponse(
            ResearchOperation.PROPOSE,
            _proposal(3, "iteration-02-repair-1"),
        ),
        ScriptedResponse(
            ResearchOperation.IMPLEMENT,
            GeneratedPackage(
                request_id="iteration-03-implement",
                response_id="iteration-03-runtime-failure",
                files=(GeneratedFile("candidate.py", RUNTIME_FAILURE_SOURCE),),
                material_change_summary="Exercise a material runtime-failing branch.",
                material_symbols=("score",),
            ),
        ),
    )
    model = ScriptedResearchModel(responses)
    safe_context = build_safe_research_context(
        starter_manifest_sha256="9" * 64,
        dataset_manifest_sha256="a" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(47, 18_000, 6, 0),
    )
    artifacts = ArtifactStore(tmp_path / "artifacts", max_object_bytes=1024 * 1024)
    rows = [
        {"coarse": 0, "fine": 0},
        {"coarse": 1, "fine": 1},
        {"coarse": 0, "fine": 1},
        {"coarse": 0, "fine": 0},
    ]
    input_ref = artifacts.put_bytes(json.dumps(rows).encode(), kind=ArtifactKind.INPUT)
    policy = WorkspacePolicy(
        max_input_file_bytes=4096,
        max_input_total_bytes=4096,
        max_output_file_bytes=4096,
        max_output_total_bytes=8192,
        max_temp_bytes=4096,
    )
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces", artifact_store=artifacts, policy=policy
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    execution = LocalExecutionTemplate(
        approved_inputs=(
            ApprovedInput("fixture", CandidateInputRole.OUTER_VALID_INPUTS, input_ref),
        ),
        split_role=SplitRole.OUTER_VALID,
        request_payload={"seed": 0, "mode": "fixture", "split_token": "fixture-token"},
        interpreter=Path(sys.executable),
        arguments=("source/candidate.py",),
        control_root=controls,
        config_digest="b" * 64,
        environment_digest="e" * 64,
        data_digest="c" * 64,
        checkpoint_digest="d" * 64,
        timeout_seconds=3,
        memory_limit_bytes=512 * 1024 * 1024,
        workspace_disk_limit_bytes=1024 * 1024,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
        output_limit_bytes=8192,
        temp_limit_bytes=4096,
        threads=1,
    )
    store = _store(tmp_path)
    evaluator = FixtureEvaluator(artifacts)
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=safe_context,
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=artifacts,
        generated_root=tmp_path / "generated",
        workspace_materializer=workspaces,
        runner=Runner(),
        execution=execution,
        evaluator=evaluator,
        selector=StrictFixtureSelector(),
    )

    result = loop.run(parent=parent, max_iterations=3)

    assert [item.status for item in result.iterations] == ["promoted", "promoted", "failed"]
    assert result.selected_candidate_id == "iteration-02-repair-1"
    assert result.fallback_candidate_id == "fm-fallback"
    assert result.iterations[1].repairs == 1
    assert result.iterations[2].trusted_evaluation is None
    assert len(evaluator.calls) == 2
    assert model.remaining_responses == 0
    incumbent = store.current_incumbent()
    assert incumbent is not None
    assert incumbent.incumbent_id == "iteration-02-repair-1"
    assert incumbent.outer_primary_mean == 1.0
    snapshot = store.snapshot()
    assert snapshot.launches_used == 3
    convergence = ConvergenceState.from_manifest(snapshot.convergence_state)
    assert convergence.completed_iterations == 2
    assert convergence.best_primary == 1.0
    assert parent.file("candidate.py").content == BASE_SOURCE

    with sqlite3.connect(store.path) as connection:
        operations = {
            row[0]
            for row in connection.execute(
                "SELECT json_extract(metadata_json, '$.operation') FROM transitions "
                "WHERE json_extract(metadata_json, '$.operation') IS NOT NULL"
            )
        }
        artifact_roles = {row[0] for row in connection.execute("SELECT role FROM artifact_links")}
        stored_metrics = connection.execute(
            "SELECT gauc, ndcg_at_5, primary_value FROM metrics ORDER BY metric_id"
        ).fetchall()
        failure_categories = {row[0] for row in connection.execute("SELECT category FROM failures")}
    assert {"propose", "implement", "repair", "run", "evaluate", "reflect"} <= operations
    assert {
        "proposal_transcript",
        "implementation_transcript",
        "repair_transcript",
    } <= artifact_roles
    assert {"source_diff", "source_manifest", "reflection_transcript"} <= artifact_roles
    assert all(0 <= value <= 1 for row in stored_metrics for value in row)
    assert "999" not in repr(stored_metrics)
    assert {"syntax_error", "runtime_error"} <= failure_categories
    assert not any((tmp_path / "workspaces").glob("iteration-*"))
    executions = store.executions()
    assert len(executions) == 3
    assert all(record.environment_digest == "e" * 64 for record in executions)
    assert all(record.process_environment_digest is not None for record in executions)
    assert all(
        record.process_environment_digest != record.environment_digest for record in executions
    )
    store.close()

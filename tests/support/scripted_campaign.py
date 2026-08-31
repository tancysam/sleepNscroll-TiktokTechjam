from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
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


_BASE_SOURCE = _candidate_source("return [0.0 for _ in rows]")
_COARSE_SOURCE = _candidate_source('return [float(row["coarse"]) for row in rows]')
_FINE_SOURCE = _candidate_source('return [float(row["fine"]) for row in rows]')
_RUNTIME_FAILURE_SOURCE = _candidate_source(
    'raise RuntimeError("deliberate material runtime child")'
)


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
                "acceptance_fixture:coarse",
                "inference_input",
                "Synthetic input-only coarse signal used by the controller fixture.",
            ),
            RequiredField(
                "acceptance_fixture:fine",
                "inference_input",
                "Synthetic input-only fine signal used by the controller fixture.",
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
        attributions=("scripted full-data acceptance mechanism",),
    )


class _FixtureEvaluator:
    def __init__(
        self,
        artifacts: ArtifactStore,
        starter_dir: Path,
        *,
        fixture_data_digest: str,
    ) -> None:
        self.artifacts = artifacts
        self.fixture_data_digest = fixture_data_digest
        self.split = SplitIdentity("fixture", "fixture-token", 4)
        self.alignment = Alignment.from_ids(
            split=self.split,
            user_ids=("u1", "u1", "u2", "u2"),
            video_ids=("v1", "v2", "v3", "v4"),
        )
        self.scorer = ProtectedScorer(
            starter_dir=starter_dir,
            trusted_alignment=self.alignment,
        )
        self.calls: list[str] = []

    def evaluate(
        self,
        *,
        candidate: MaterializedCandidate,
        workspace: CandidateWorkspace,
        execution: ExecutionResult,
    ) -> TrustedEvaluation:
        if execution.candidate_metrics_accepted or "metrics" in execution.manifest():
            raise AssertionError("candidate-authored metrics crossed the trusted boundary")
        self.calls.append(candidate.source_digest)
        validated = validate_prediction_outputs(
            workspace.output_dir,
            PredictionExpectation(
                source_digest=candidate.source_digest,
                config_digest="b" * 64,
                data_digest=self.fixture_data_digest,
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
        prediction = self.artifacts.put_file(
            validated.scores_path,
            kind=ArtifactKind.PREDICTION,
        )
        closure = hashlib.sha256(
            f"closure:{prediction.sha256}:{'d' * 64}".encode("ascii")
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
            prediction_digest=prediction.sha256,
            eligible=True,
            replay_verified=True,
            checkpoint_digest="d" * 64,
            artifact_closure_digest=closure,
            artifacts=(("trusted_predictions", artifact_spec(prediction)),),
        )


class _StrictSelector:
    def decide(
        self,
        incumbent_primary: float | None,
        candidate: CandidateEvidence,
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


def _store(
    root: Path,
    *,
    benchmark_digest: str,
    starter_digest: str,
    dataset_digest: str,
) -> CampaignStore:
    store = CampaignStore.create(
        root / "campaign.sqlite3",
        campaign_id="scripted-full-data-acceptance",
        config_digest="1" * 64,
        benchmark_digest=benchmark_digest,
        starter_digest=starter_digest,
        dataset_digest=dataset_digest,
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


def _responses() -> tuple[ScriptedResponse, ...]:
    return (
        ScriptedResponse(ResearchOperation.PROPOSE, _proposal(1, "fm-fallback")),
        ScriptedResponse(
            ResearchOperation.IMPLEMENT,
            GeneratedPackage(
                request_id="iteration-01-implement",
                response_id="iteration-01-source",
                files=(GeneratedFile("candidate.py", _COARSE_SOURCE),),
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
                files=(GeneratedFile("candidate.py", _FINE_SOURCE),),
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
                files=(GeneratedFile("candidate.py", _RUNTIME_FAILURE_SOURCE),),
                material_change_summary="Exercise a material runtime-failing branch.",
                material_symbols=("score",),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ScriptedCampaignAcceptance:
    statuses: tuple[str, ...]
    selected_candidate_id: str
    fallback_candidate_id: str
    repairs: tuple[int, ...]
    trusted_evaluation_present: tuple[bool, ...]
    evaluator_calls: int
    model_responses_remaining: int
    incumbent_id: str
    incumbent_primary: float
    launches_used: int
    completed_iterations: int
    best_primary: float
    operations: frozenset[str]
    artifact_roles: frozenset[str]
    failure_categories: frozenset[str]
    stored_metrics: tuple[tuple[float, float, float], ...]
    workspaces_cleaned: bool
    execution_count: int
    environment_receipts_complete: bool
    parent_source_unchanged: bool
    campaign_benchmark_digest: str
    campaign_starter_digest: str
    campaign_dataset_digest: str


def run_scripted_three_iteration_acceptance(
    root: Path,
    *,
    starter_dir: Path,
    verified_audit_digest: str,
    verified_starter_digest: str,
    verified_dataset_digest: str,
) -> ScriptedCampaignAcceptance:
    """Run three autonomous iterations with one repair and one closed failed branch."""

    if not isinstance(root, Path) or not isinstance(starter_dir, Path):
        raise TypeError("acceptance root and starter_dir must be pathlib.Path values")
    root.mkdir(parents=True, exist_ok=False)
    parent = ParentSnapshot(
        candidate_id="fm-fallback",
        files=(ParentSourceFile.create("candidate.py", _BASE_SOURCE),),
    )
    model = ScriptedResearchModel(_responses())
    context = build_safe_research_context(
        starter_manifest_sha256=verified_starter_digest,
        dataset_manifest_sha256=verified_audit_digest,
        capability_manifests=(),
        budgets=ResearchBudgetContext(47, 18_000, 6, 0),
    )
    artifacts = ArtifactStore(root / "artifacts", max_object_bytes=1024 * 1024)
    rows = [
        {"coarse": 0, "fine": 0},
        {"coarse": 1, "fine": 1},
        {"coarse": 0, "fine": 1},
        {"coarse": 0, "fine": 0},
    ]
    input_ref = artifacts.put_bytes(json.dumps(rows).encode(), kind=ArtifactKind.INPUT)
    workspaces = WorkspaceMaterializer(
        root / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(
            max_input_file_bytes=4096,
            max_input_total_bytes=4096,
            max_output_file_bytes=4096,
            max_output_total_bytes=8192,
            max_temp_bytes=4096,
        ),
    )
    controls = root / "controls"
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
        data_digest=input_ref.sha256,
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
    evaluator = _FixtureEvaluator(
        artifacts,
        starter_dir,
        fixture_data_digest=input_ref.sha256,
    )
    with _store(
        root,
        benchmark_digest=verified_audit_digest,
        starter_digest=verified_starter_digest,
        dataset_digest=verified_dataset_digest,
    ) as store:
        loop = ResearchCampaignLoop(
            model=model,
            safe_context=context,
            ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
            artifacts=artifacts,
            generated_root=root / "generated",
            workspace_materializer=workspaces,
            runner=Runner(),
            execution=execution,
            evaluator=evaluator,
            selector=_StrictSelector(),
        )
        result = loop.run(parent=parent, max_iterations=3)
        incumbent = store.current_incumbent()
        if incumbent is None or incumbent.outer_primary_mean is None:
            raise AssertionError("scripted campaign lost its eligible incumbent")
        snapshot = store.snapshot()
        convergence = ConvergenceState.from_manifest(snapshot.convergence_state)
        with contextlib.closing(sqlite3.connect(store.path)) as connection:
            operations = frozenset(
                str(row[0])
                for row in connection.execute(
                    "SELECT json_extract(metadata_json, '$.operation') FROM transitions "
                    "WHERE json_extract(metadata_json, '$.operation') IS NOT NULL"
                )
            )
            artifact_roles = frozenset(
                str(row[0]) for row in connection.execute("SELECT role FROM artifact_links")
            )
            stored_metrics = tuple(
                (float(row[0]), float(row[1]), float(row[2]))
                for row in connection.execute(
                    "SELECT gauc, ndcg_at_5, primary_value FROM metrics ORDER BY metric_id"
                )
            )
            failure_categories = frozenset(
                str(row[0]) for row in connection.execute("SELECT category FROM failures")
            )
            identity_row = connection.execute(
                "SELECT benchmark_digest, starter_digest, dataset_digest FROM campaigns"
            ).fetchone()
            if identity_row is None:
                raise AssertionError("scripted campaign identity row is missing")
            campaign_benchmark_digest = str(identity_row[0])
            campaign_starter_digest = str(identity_row[1])
            campaign_dataset_digest = str(identity_row[2])
        executions = store.executions()
        return ScriptedCampaignAcceptance(
            statuses=tuple(item.status for item in result.iterations),
            selected_candidate_id=result.selected_candidate_id,
            fallback_candidate_id=result.fallback_candidate_id,
            repairs=tuple(item.repairs for item in result.iterations),
            trusted_evaluation_present=tuple(
                item.trusted_evaluation is not None for item in result.iterations
            ),
            evaluator_calls=len(evaluator.calls),
            model_responses_remaining=model.remaining_responses,
            incumbent_id=incumbent.incumbent_id,
            incumbent_primary=incumbent.outer_primary_mean,
            launches_used=snapshot.launches_used,
            completed_iterations=convergence.completed_iterations,
            best_primary=convergence.best_primary,
            operations=operations,
            artifact_roles=artifact_roles,
            failure_categories=failure_categories,
            stored_metrics=stored_metrics,
            workspaces_cleaned=not any((root / "workspaces").glob("iteration-*")),
            execution_count=len(executions),
            environment_receipts_complete=bool(executions)
            and all(
                item.environment_digest == "e" * 64
                and item.process_environment_digest is not None
                and item.process_environment_digest != item.environment_digest
                for item in executions
            ),
            parent_source_unchanged=parent.file("candidate.py").content == _BASE_SOURCE,
            campaign_benchmark_digest=campaign_benchmark_digest,
            campaign_starter_digest=campaign_starter_digest,
            campaign_dataset_digest=campaign_dataset_digest,
        )


__all__ = ["ScriptedCampaignAcceptance", "run_scripted_three_iteration_acceptance"]

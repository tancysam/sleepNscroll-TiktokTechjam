from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.execution.policy import SplitRole, WorkspacePolicy
from kuairand_agent.execution.runner import Runner
from kuairand_agent.execution.workspace import WorkspaceMaterializer
from kuairand_agent.research.context import (
    ResearchBudgetContext,
    SafeResearchContext,
    build_safe_research_context,
)
from kuairand_agent.research.loop import (
    CampaignStoreResearchLedger,
    CandidateEvidence,
    LocalExecutionTemplate,
    ResearchCampaignLoop,
    SelectionDecision,
    TrustedEvaluation,
)
from kuairand_agent.research.provider import (
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    TokenPricing,
    TransportRequest,
    TransportResponse,
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

BASE = """\
def score(rows):
    return [0.0 for _ in rows]
"""

FAKE_METRIC_ONLY = """\
import json
from pathlib import Path


def score(rows):
    return [1.0 for _ in rows]


def main():
    Path("output/metrics.json").write_text(
        json.dumps({"primary": 999, "GAUC": 999, "nDCG@5": 999}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
"""


def _proposal(*, repairs: int, proposal_id: str = "fault-proposal") -> Proposal:
    return Proposal(
        proposal_id=proposal_id,
        hypothesis="A legal input-only score function may improve fixture ranking.",
        mechanism="Replace the executable score function.",
        expected_metric_effects=("GAUC",),
        parent_candidate_id="fm-fallback",
        principal_change="Change one executable score function.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:duration_ms",
                "inference_input",
                "Legal current-row input.",
            ),
        ),
        objective="long_view ranking",
        sampling="all fixture rows",
        grouping="user impression groups",
        weighting="uniform impressions",
        causal_cutoff="No outcome history is used.",
        estimated_runtime_seconds=1,
        estimated_memory_mb=64,
        smoke_plan="Execute a four-row fixture.",
        inner_fold_plan="Use a structural fixture fold.",
        falsification_criteria="Reject any failed trusted gate.",
        promotion_criteria="Require trusted protected scoring.",
        maximum_repairs=repairs,
        rollback_parent_id="fm-fallback",
        attributions=("fault-injection fixture",),
    )


def _store(tmp_path: Path, name: str) -> CampaignStore:
    store = CampaignStore.create(
        tmp_path / f"{name}.sqlite3",
        campaign_id=name,
        config_digest="1" * 64,
        benchmark_digest="2" * 64,
        starter_digest="3" * 64,
        dataset_digest="4" * 64,
        environment_digest="5" * 64,
        hard_deadline_utc="2030-01-01T00:00:00+00:00",
        initial_convergence=ConvergenceState.initial(0.6).manifest(),
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
        reason="qualified fallback",
        outer_primary_mean=0.6,
    )
    return store


def _context() -> SafeResearchContext:
    return build_safe_research_context(
        starter_manifest_sha256="9" * 64,
        dataset_manifest_sha256="a" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(49, 18_000, 6, 0),
    )


def _execution(tmp_path: Path) -> LocalExecutionTemplate:
    controls = tmp_path / "controls"
    controls.mkdir(exist_ok=True)
    return LocalExecutionTemplate(
        approved_inputs=(),
        split_role=SplitRole.OUTER_VALID,
        request_payload={"mode": "fault_fixture"},
        interpreter=Path(sys.executable),
        arguments=("source/candidate.py",),
        control_root=controls,
        config_digest="b" * 64,
        environment_digest="e" * 64,
        data_digest="c" * 64,
        checkpoint_digest="d" * 64,
        timeout_seconds=2,
        memory_limit_bytes=256 * 1024 * 1024,
        workspace_disk_limit_bytes=1024 * 1024,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
        output_limit_bytes=4096,
        temp_limit_bytes=4096,
        threads=1,
    )


class NeverCalled:
    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        raise AssertionError(f"failure path unexpectedly called {name}")


class CaptureRejectedPackageModel:
    def __init__(self, proposal: Proposal) -> None:
        self.proposal = proposal
        self.implementation: GeneratedPackage | None = None
        self.repair_request: RepairRequest | None = None

    def propose(self, _request: ProposalRequest) -> Proposal:
        return self.proposal

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        package = GeneratedPackage(
            request_id=request.request_id,
            response_id="forbidden-implementation",
            files=(
                GeneratedFile("baseline.py", "def forbidden_helper():\n    return 1.0\n"),
                GeneratedFile(
                    "candidate.py",
                    "def score(rows):\n    return [1.0 for _ in rows]\n",
                ),
            ),
            material_change_summary="Use a generated scoring function.",
            material_symbols=("score",),
        )
        self.implementation = package
        return package

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        self.repair_request = request
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="legal-replacement",
            files=(
                GeneratedFile(
                    "candidate.py",
                    "def score(rows):\n    return [1.0 for _ in rows]\n",
                ),
            ),
            material_change_summary="Preserve the generated score in the legal entrypoint.",
            material_symbols=("score",),
        )

    def reflect(self, _request: ReflectionRequest) -> Reflection:
        raise AssertionError("failed execution must not be reflected")


class RejectCandidateMetrics:
    calls = 0

    def evaluate(self, **_kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise ValueError("prediction_result.json is absent; candidate metrics are not authority")


class AcceptRecoveredCandidate:
    calls = 0

    def evaluate(self, **_kwargs: object) -> TrustedEvaluation:
        self.calls += 1
        return TrustedEvaluation(
            summary=ExperimentResultSummary(
                tier="outer",
                status="passed",
                gauc=0.7,
                ndcg_at_5=0.5,
                primary=0.6,
                runtime_seconds=0.01,
                peak_memory_mb=1.0,
            ),
            scorer_digest="1" * 64,
            prediction_digest="2" * 64,
            eligible=True,
            replay_verified=True,
            checkpoint_digest="3" * 64,
            artifact_closure_digest="4" * 64,
        )


class RetainFallback:
    def decide(
        self, _incumbent_primary: float | None, _candidate: CandidateEvidence
    ) -> SelectionDecision:
        return SelectionDecision(False, "fixture_eligible", "retain qualified fallback")


class ProviderQueue:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = responses
        self.requests: list[TransportRequest] = []

    def send(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _provider_response(output_text: str) -> bytes:
    return json.dumps(
        {
            "id": "fault-provider-response",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                        "refusal": None,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_invalid_proposal_manifest_never_reaches_implementation(
    tmp_path: Path,
) -> None:
    invalid = replace(
        _proposal(repairs=0),
        files_expected=("candidate.py", "baseline.py", "submission.csv"),
    )
    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, invalid),
            ScriptedResponse(
                ResearchOperation.IMPLEMENT,
                GeneratedPackage(
                    request_id="iteration-01-implement",
                    response_id="must-not-be-consumed",
                    files=(GeneratedFile("candidate.py", "def score(rows):\n    return [1.0]\n"),),
                    material_change_summary="This implementation must remain unreachable.",
                    material_symbols=("score",),
                ),
            ),
        )
    )
    store = _store(tmp_path, "proposal-policy-admission")
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        generated_root=tmp_path / "generated",
        workspace_materializer=NeverCalled(),
        runner=NeverCalled(),
        execution=_execution(tmp_path),
        evaluator=NeverCalled(),
        selector=NeverCalled(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=1,
    )

    assert result.iterations[0].status == "failed"
    assert "baseline.py" in (result.iterations[0].diagnostic or "")
    assert [call.operation for call in model.calls] == [ResearchOperation.PROPOSE]
    assert model.remaining_responses == 1
    assert store.snapshot().launches_used == 0
    assert store.current_incumbent() is not None
    assert store.current_incumbent().incumbent_id == "fm-fallback"  # type: ignore[union-attr]
    store.close()


def test_pre_materialization_repair_receives_exact_rejected_package(
    tmp_path: Path,
) -> None:
    model = CaptureRejectedPackageModel(_proposal(repairs=1))
    store = _store(tmp_path, "rejected-package-repair")
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="capturing"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        generated_root=tmp_path / "generated",
        workspace_materializer=NeverCalled(),
        runner=NeverCalled(),
        execution=_execution(tmp_path),
        evaluator=NeverCalled(),
        selector=NeverCalled(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=1,
    )

    assert model.implementation is not None
    assert model.repair_request is not None
    assert model.repair_request.rejected_package is not None
    assert model.repair_request.rejected_package.package_digest == model.implementation.digest
    assert model.repair_request.failed_child.candidate_id == "fm-fallback"
    assert result.iterations[0].candidate_id == "iteration-01-repair-1"
    assert not (tmp_path / "generated" / "iteration-01-child-0").exists()
    assert (tmp_path / "generated" / "iteration-01-repair-1").is_dir()
    assert not (tmp_path / "generated" / "iteration-01-repair-1" / "baseline.py").exists()
    store.close()


def test_semantic_duplicate_with_unchanged_evidence_never_reaches_implementation(
    tmp_path: Path,
) -> None:
    first = _proposal(repairs=0, proposal_id="pairwise-first")
    duplicate = replace(
        first,
        proposal_id="pairwise-paraphrase",
        hypothesis="Different prose around the same proposed scientific operation.",
        attributions=("different prose attribution",),
    )
    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, first),
            ScriptedResponse(
                ResearchOperation.IMPLEMENT,
                GeneratedPackage(
                    request_id="iteration-01-implement",
                    response_id="first-static-failure",
                    files=(GeneratedFile("candidate.py", "def score(:\n"),),
                    material_change_summary="A bounded pre-training implementation failure.",
                    material_symbols=("score",),
                ),
            ),
            ScriptedResponse(ResearchOperation.PROPOSE, duplicate),
            ScriptedResponse(
                ResearchOperation.IMPLEMENT,
                GeneratedPackage(
                    request_id="iteration-02-implement",
                    response_id="duplicate-must-not-be-consumed",
                    files=(GeneratedFile("candidate.py", "def score(:\n"),),
                    material_change_summary="The duplicate must not reach implementation.",
                    material_symbols=("score",),
                ),
            ),
        )
    )
    store = _store(tmp_path, "proposal-novelty-duplicate")
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        generated_root=tmp_path / "generated",
        workspace_materializer=NeverCalled(),
        runner=NeverCalled(),
        execution=_execution(tmp_path),
        evaluator=NeverCalled(),
        selector=NeverCalled(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=2,
    )

    assert [item.status for item in result.iterations] == ["failed", "failed"]
    assert "duplicate" in (result.iterations[1].diagnostic or "")
    assert [call.operation for call in model.calls] == [
        ResearchOperation.PROPOSE,
        ResearchOperation.IMPLEMENT,
        ResearchOperation.PROPOSE,
    ]
    assert model.remaining_responses == 1
    assert store.snapshot().launches_used == 0
    store.close()


def test_scientific_family_has_at_most_two_implementation_admissions_before_training(
    tmp_path: Path,
) -> None:
    proposals = tuple(
        replace(
            _proposal(repairs=0, proposal_id=f"pairwise-{ordinal}"),
            mechanism=f"Pairwise BPR mechanism variant {ordinal}.",
            principal_change=f"Change pairwise executable loss variant {ordinal}.",
        )
        for ordinal in range(1, 4)
    )
    responses: list[ScriptedResponse] = []
    for ordinal, proposal in enumerate(proposals, start=1):
        responses.append(ScriptedResponse(ResearchOperation.PROPOSE, proposal))
        if ordinal < 3:
            responses.append(
                ScriptedResponse(
                    ResearchOperation.IMPLEMENT,
                    GeneratedPackage(
                        request_id=f"iteration-{ordinal:02d}-implement",
                        response_id=f"pairwise-{ordinal}-static-failure",
                        files=(GeneratedFile("candidate.py", "def score(:\n"),),
                        material_change_summary="Bounded failure before trusted training.",
                        material_symbols=("score",),
                    ),
                )
            )
    model = ScriptedResearchModel(tuple(responses))
    store = _store(tmp_path, "proposal-family-pretraining-limit")
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        generated_root=tmp_path / "generated",
        workspace_materializer=NeverCalled(),
        runner=NeverCalled(),
        execution=_execution(tmp_path),
        evaluator=NeverCalled(),
        selector=NeverCalled(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=3,
    )

    assert [call.operation for call in model.calls] == [
        ResearchOperation.PROPOSE,
        ResearchOperation.IMPLEMENT,
        ResearchOperation.PROPOSE,
        ResearchOperation.IMPLEMENT,
        ResearchOperation.PROPOSE,
    ]
    assert "family_pretraining_limit:pairwise" in (result.iterations[2].diagnostic or "")
    assert store.snapshot().launches_used == 0
    store.close()


def test_exhausted_syntax_repair_never_launches_or_changes_incumbent(
    tmp_path: Path,
) -> None:
    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, _proposal(repairs=1)),
            ScriptedResponse(
                ResearchOperation.IMPLEMENT,
                GeneratedPackage(
                    request_id="iteration-01-implement",
                    response_id="bad-one",
                    files=(GeneratedFile("candidate.py", "def score(:\n"),),
                    material_change_summary="First invalid source child.",
                    material_symbols=("score",),
                ),
            ),
            ScriptedResponse(
                ResearchOperation.REPAIR,
                GeneratedPackage(
                    request_id="iteration-01-repair-1",
                    response_id="bad-two",
                    files=(GeneratedFile("candidate.py", "def score(\n"),),
                    material_change_summary="Deliberately invalid bounded repair.",
                    material_symbols=("score",),
                ),
            ),
        )
    )
    store = _store(tmp_path, "syntax-exhaustion")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    workspaces = NeverCalled()
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=artifacts,
        generated_root=tmp_path / "generated",
        workspace_materializer=workspaces,
        runner=NeverCalled(),
        execution=_execution(tmp_path),
        evaluator=NeverCalled(),
        selector=NeverCalled(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=1,
    )

    assert result.iterations[0].status == "failed"
    assert result.iterations[0].repairs == 1
    assert model.remaining_responses == 0
    assert store.snapshot().launches_used == 0
    assert store.current_incumbent() is not None
    assert store.current_incumbent().incumbent_id == "fm-fallback"  # type: ignore[union-attr]
    convergence = ConvergenceState.from_manifest(store.snapshot().convergence_state)
    assert convergence.completed_iterations == 0
    assert convergence.best_primary == 0.6
    store.close()


def test_candidate_metric_spoof_is_ignored_when_trusted_evaluator_rejects_outputs(
    tmp_path: Path,
) -> None:
    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, _proposal(repairs=0)),
            ScriptedResponse(
                ResearchOperation.IMPLEMENT,
                GeneratedPackage(
                    request_id="iteration-01-implement",
                    response_id="fake-metric-source",
                    files=(GeneratedFile("candidate.py", FAKE_METRIC_ONLY),),
                    material_change_summary="Emit a candidate-declared fake metric only.",
                    material_symbols=("score",),
                ),
            ),
        )
    )
    store = _store(tmp_path, "metric-spoof")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    policy = WorkspacePolicy(
        max_output_file_bytes=4096,
        max_output_total_bytes=4096,
        max_temp_bytes=4096,
    )
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces", artifact_store=artifacts, policy=policy
    )
    evaluator = RejectCandidateMetrics()
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=artifacts,
        generated_root=tmp_path / "generated",
        workspace_materializer=workspaces,
        runner=Runner(),
        execution=_execution(tmp_path),
        evaluator=evaluator,
        selector=NeverCalled(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=1,
    )

    assert result.iterations[0].status == "failed"
    assert result.iterations[0].trusted_evaluation is None
    assert evaluator.calls == 1
    assert store.current_incumbent() is not None
    assert store.current_incumbent().incumbent_id == "fm-fallback"  # type: ignore[union-attr]
    assert store.snapshot().launches_used == 1
    assert not any((tmp_path / "workspaces").glob("iteration-*"))
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0
        failure = connection.execute(
            "SELECT category, metadata_json FROM failures ORDER BY failure_id DESC LIMIT 1"
        ).fetchone()
    assert failure[0] == "output_contract"
    assert "999" not in failure[1]
    convergence = ConvergenceState.from_manifest(store.snapshot().convergence_state)
    assert convergence.completed_iterations == 0
    store.close()


def test_generated_import_failure_is_repaired_from_exact_child_and_charged_once_per_release(
    tmp_path: Path,
) -> None:
    import_failure = """\
import definitely_missing_kuairand_fault_fixture


def score(rows):
    return [1.0 for _ in rows]
"""
    repaired = """\
def score(rows):
    return [1.0 for _ in rows]
"""
    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, _proposal(repairs=1)),
            ScriptedResponse(
                ResearchOperation.IMPLEMENT,
                GeneratedPackage(
                    request_id="iteration-01-implement",
                    response_id="import-failure-child",
                    files=(GeneratedFile("candidate.py", import_failure),),
                    material_change_summary="Exercise a missing optional dependency.",
                    material_symbols=("score",),
                ),
            ),
            ScriptedResponse(
                ResearchOperation.REPAIR,
                GeneratedPackage(
                    request_id="iteration-01-repair-1",
                    response_id="import-recovery-child",
                    files=(GeneratedFile("candidate.py", repaired),),
                    material_change_summary="Remove the unavailable dependency.",
                    material_symbols=("score",),
                ),
            ),
            ScriptedResponse(
                ResearchOperation.REFLECT,
                Reflection(
                    "import-recovery-reflection",
                    "The bounded import repair passed trusted execution.",
                    "close_branch",
                    ("Do not depend on unavailable local modules.",),
                ),
            ),
        )
    )
    store = _store(tmp_path, "import-recovery")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(
            max_output_file_bytes=4096,
            max_output_total_bytes=4096,
            max_temp_bytes=4096,
        ),
    )
    evaluator = AcceptRecoveredCandidate()
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="scripted"),
        artifacts=artifacts,
        generated_root=tmp_path / "generated",
        workspace_materializer=workspaces,
        runner=Runner(),
        execution=_execution(tmp_path),
        evaluator=evaluator,
        selector=RetainFallback(),
    )

    result = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=1,
    )

    assert result.iterations[0].status == "retained"
    assert result.iterations[0].candidate_id == "iteration-01-repair-1"
    assert result.iterations[0].repairs == 1
    assert evaluator.calls == 1
    assert [call.operation for call in model.calls] == [
        ResearchOperation.PROPOSE,
        ResearchOperation.IMPLEMENT,
        ResearchOperation.REPAIR,
        ResearchOperation.REFLECT,
    ]
    assert store.snapshot().launches_used == 2
    assert [launch.state for launch in store.launches()] == ["FINISHED", "FINISHED"]
    incumbent = store.current_incumbent()
    assert incumbent is not None and incumbent.incumbent_id == "fm-fallback"
    with sqlite3.connect(store.path) as connection:
        failures = connection.execute(
            "SELECT category, repair_action, metadata_json FROM failures ORDER BY created_at"
        ).fetchall()
    assert len(failures) == 1
    assert failures[0][0] == "import_error"
    assert failures[0][1] == "request bounded exact-child repair"
    assert "ModuleNotFoundError" in failures[0][2]
    assert not any((tmp_path / "workspaces").glob("iteration-*"))
    store.close()


def test_malformed_provider_json_is_bounded_persisted_and_resumable_without_a_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_FAULT_PROVIDER_KEY", "sk-fault-provider-secret")
    malformed = json.dumps({"schema_version": 1, "unexpected": True})
    transport = ProviderQueue(
        [
            TransportResponse(
                200,
                _provider_response(
                    json.dumps(_proposal(repairs=0).to_wire(), separators=(",", ":"))
                ),
            ),
            TransportResponse(200, _provider_response(malformed)),
            TransportResponse(200, _provider_response(malformed)),
        ]
    )
    model = OpenAIResponsesModel(
        OpenAIResponsesConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            reasoning_effort="high",
            pricing=TokenPricing("1", "0.5", "4"),
            api_key_env="KUAIRAND_FAULT_PROVIDER_KEY",
            timeout_seconds=1.0,
            max_response_bytes=1024 * 1024,
            max_output_tokens=4096,
            max_malformed_retries=1,
        ),
        transport=transport,
    )
    name = "malformed-provider"
    store = _store(tmp_path, name)
    store_path = store.path
    artifacts = ArtifactStore(tmp_path / "artifacts")
    loop = ResearchCampaignLoop(
        model=model,
        safe_context=_context(),
        ledger=CampaignStoreResearchLedger(store, provider_name="openai"),
        artifacts=artifacts,
        generated_root=tmp_path / "generated",
        workspace_materializer=NeverCalled(),
        runner=NeverCalled(),
        execution=_execution(tmp_path),
        evaluator=NeverCalled(),
        selector=NeverCalled(),
    )

    first = loop.run(
        parent=ParentSnapshot(
            candidate_id="fm-fallback",
            files=(ParentSourceFile.create("candidate.py", BASE),),
        ),
        max_iterations=1,
    )

    assert first.iterations[0].status == "failed"
    assert len(transport.requests) == 3
    assert [item.outcome for item in model.transcripts] == [
        "accepted",
        "malformed",
        "malformed",
    ]
    assert store.snapshot().launches_used == 0
    assert store.current_incumbent() is not None
    assert store.current_incumbent().incumbent_id == "fm-fallback"  # type: ignore[union-attr]
    failure = store.failure("iteration-01-failure-0-provider_error")
    assert failure is not None and failure.category == "provider_error"
    assert store.experiment("iteration-01").status == "CLOSED"  # type: ignore[union-attr]
    store.close()

    with CampaignStore.open(store_path, campaign_id=name) as resumed_store:
        resumed_model = ScriptedResearchModel(
            (
                ScriptedResponse(
                    ResearchOperation.PROPOSE,
                    _proposal(repairs=0, proposal_id="resume-proposal"),
                ),
                ScriptedResponse(
                    ResearchOperation.IMPLEMENT,
                    GeneratedPackage(
                        request_id="iteration-02-implement",
                        response_id="resume-bounded-failure",
                        files=(GeneratedFile("candidate.py", "def score(:\n"),),
                        material_change_summary="Exercise a bounded resumed failure.",
                        material_symbols=("score",),
                    ),
                ),
            )
        )
        resumed = ResearchCampaignLoop(
            model=resumed_model,
            safe_context=_context(),
            ledger=CampaignStoreResearchLedger(resumed_store, provider_name="scripted"),
            artifacts=artifacts,
            generated_root=tmp_path / "generated",
            workspace_materializer=NeverCalled(),
            runner=NeverCalled(),
            execution=_execution(tmp_path),
            evaluator=NeverCalled(),
            selector=NeverCalled(),
        ).run(
            parent=ParentSnapshot(
                candidate_id="fm-fallback",
                files=(ParentSourceFile.create("candidate.py", BASE),),
            ),
            max_iterations=1,
        )
        assert resumed.iterations[0].iteration == 2
        assert resumed.iterations[0].status == "failed"
        assert resumed_store.snapshot().launches_used == 0
        assert resumed_store.current_incumbent() is not None
        assert resumed_store.current_incumbent().incumbent_id == "fm-fallback"  # type: ignore[union-attr]

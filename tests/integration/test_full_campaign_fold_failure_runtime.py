from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.baselines.fold_control_runner import SupervisedFoldFMExecutionError
from kuairand_agent.campaign import full_campaign_runtime as runtime
from kuairand_agent.campaign.budgets import LaunchCategory
from kuairand_agent.campaign.controller import CAMPAIGN_DATABASE_NAME
from kuairand_agent.campaign.full_campaign import (
    FullCampaignCancelled,
    FullCampaignError,
    FullCampaignOutcomeRepository,
    FullCampaignStage,
)
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
)
from kuairand_agent.execution.candidate_executor import CandidateExecutionArtifacts
from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionResult,
    LogEvidence,
)
from kuairand_agent.research.schemas import canonical_json_bytes
from tests.integration.test_full_campaign_runtime_resilience import (
    _fold,
    _Harness,
    _install_harness,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finish_charged_launch(
    campaign_store: CampaignStore,
    *,
    fold_name: str,
    category: LaunchCategory,
    outcome: ExecutionOutcome,
) -> str:
    launch_id = f"synthetic-fold-{fold_name}-{outcome.value}"
    campaign_store.reserve_launch(
        launch_id=launch_id,
        reservation_key=f"synthetic-fold:{fold_name}:{outcome.value}",
        category=category.value,
        purpose=f"trusted synthetic Fold {fold_name} {outcome.value} execution",
        expected_revision=campaign_store.snapshot().revision,
        seed=0,
        metadata={"fold_name": fold_name, "outcome": outcome.value},
    )
    start_receipt = _sha256(f"{launch_id}:started".encode("ascii"))
    campaign_store.transition_launch(
        launch_id,
        to_state="STARTED",
        expected_revision=campaign_store.snapshot().revision,
        start_receipt_digest=start_receipt,
        metadata={"fold_name": fold_name, "phase": "started"},
    )
    campaign_store.transition_launch(
        launch_id,
        to_state="FINISHED",
        expected_revision=campaign_store.snapshot().revision,
        metadata={"fold_name": fold_name, "phase": "finished"},
    )
    return launch_id


def _trusted_execution_error(
    *,
    fold_name: str,
    outcome: ExecutionOutcome,
    campaign_store: CampaignStore,
    artifacts: ArtifactStore,
    cleanup_verified: bool = True,
) -> SupervisedFoldFMExecutionError:
    category = (
        LaunchCategory.DIVERSE_INNER_SCREEN
        if fold_name == "B"
        else LaunchCategory.TEMPORAL_FOLD_CONFIRMATION
    )
    launch_id = _finish_charged_launch(
        campaign_store,
        fold_name=fold_name,
        category=category,
        outcome=outcome,
    )
    stdout = b""
    diagnostic = f"synthetic Fold {fold_name} {outcome.value} diagnostic"
    stderr = diagnostic.encode("ascii")
    succeeded = outcome is ExecutionOutcome.SUCCEEDED
    result = ExecutionResult(
        execution_id=launch_id,
        outcome=outcome,
        process=None,
        candidate_released=True,
        exit_code=0 if succeeded else 17 if outcome is ExecutionOutcome.EXIT_NONZERO else None,
        terminating_signal=(
            None if outcome in {ExecutionOutcome.SUCCEEDED, ExecutionOutcome.EXIT_NONZERO} else 15
        ),
        started_at_utc="2030-01-01T00:00:00+00:00",
        ended_at_utc="2030-01-01T00:00:01+00:00",
        wall_seconds=1.0,
        peak_tree_rss_bytes=64 * 1024 * 1024,
        peak_workspace_bytes=8 * 1024 * 1024,
        peak_process_count=1,
        stdout=LogEvidence(
            artifacts.root / f"{launch_id}-stdout.log",
            len(stdout),
            len(stdout),
            False,
            _sha256(stdout),
        ),
        stderr=LogEvidence(
            artifacts.root / f"{launch_id}-stderr.log",
            len(stderr),
            len(stderr),
            False,
            _sha256(stderr),
        ),
        cleanup_verified=cleanup_verified,
        device="cpu",
        threads=1,
        detail=diagnostic,
    )
    manifest = artifacts.put_bytes(
        canonical_json_bytes(result.manifest()),
        kind=ArtifactKind.MANIFEST,
    )
    stdout_ref = artifacts.put_bytes(stdout, kind=ArtifactKind.LOG)
    stderr_ref = artifacts.put_bytes(stderr, kind=ArtifactKind.LOG)
    cleanup = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "execution_id": launch_id,
                "workspace_removed": cleanup_verified,
                "error_type": None if cleanup_verified else "OSError",
            }
        ),
        kind=ArtifactKind.MANIFEST,
    )
    diagnostic_ref = artifacts.put_bytes(stderr, kind=ArtifactKind.LOG)
    closure = CandidateExecutionArtifacts(
        (
            ("execution_manifest", manifest),
            ("failure_diagnostic", diagnostic_ref),
            ("stderr", stderr_ref),
            ("stdout", stdout_ref),
            ("workspace_cleanup", cleanup),
        ),
        output_validated=False,
        diagnostic=diagnostic,
    )
    return SupervisedFoldFMExecutionError(
        diagnostic,
        result=result,
        artifacts=closure,
    )


def _artifact_store(harness: _Harness) -> ArtifactStore:
    return ArtifactStore(harness.request.run_dir / "artifacts")


def _assert_no_outcome(harness: _Harness) -> None:
    checkpoints = harness.progress().checkpoints()
    assert checkpoints[-1].stage is FullCampaignStage.QUALIFICATION_VERIFIED
    assert all(
        checkpoint.stage is not FullCampaignStage.FINALIZATION_REQUIRED
        for checkpoint in checkpoints
    )
    assert not harness.engine.status(harness.request.run_dir).finalization_required
    with pytest.raises(FullCampaignError, match="not durably finalization-ready"):
        FullCampaignOutcomeRepository(
            run_dir=harness.request.run_dir,
            artifact_store=_artifact_store(harness),
            progress=harness.progress(),
        ).load(request_digest=harness.request.digest)


def _diagnostic_payload(harness: _Harness) -> tuple[Mapping[str, object], ArtifactRef]:
    checkpoint = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.FOLD_CONTROLS_READY
    )
    raw_reference = checkpoint.evidence["diagnostic_artifact"]
    assert isinstance(raw_reference, Mapping)
    reference = ArtifactRef.from_manifest(raw_reference)
    store = _artifact_store(harness)
    store.verify(reference)
    payload = json.loads(store.read_bytes(reference, max_bytes=reference.size_bytes))
    assert isinstance(payload, dict)
    assert canonical_json_bytes(payload) == store.read_bytes(
        reference,
        max_bytes=reference.size_bytes,
    )
    return cast(Mapping[str, object], payload), reference


def _assert_exact_failure_diagnostic(
    harness: _Harness,
    *,
    reason: str,
    outcome: ExecutionOutcome,
    closure: CandidateExecutionArtifacts,
) -> None:
    checkpoint = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.FOLD_CONTROLS_READY
    )
    payload, reference = _diagnostic_payload(harness)
    assert reference.kind is ArtifactKind.LOG
    assert payload == {
        "schema_version": 1,
        "category": "fold_control_failed",
        "reason": reason,
        "fallback_receipt_digest": checkpoint.evidence["fallback_receipt_digest"],
        "fallback_preserved": True,
        "final_outcomes_scored": 0,
        "details": {
            "error_type": "SupervisedFoldFMExecutionError",
            "admission_reason": None,
            "execution_outcome": outcome.value,
            "journal_closure_digest": closure.closure_digest,
            "candidate_released": True,
            "cleanup_verified": True,
            "launch_charged": True,
        },
    }
    store = _artifact_store(harness)
    manifest = closure.artifact("execution_manifest")
    store.verify(manifest)
    decoded_manifest = json.loads(store.read_bytes(manifest, max_bytes=manifest.size_bytes))
    assert decoded_manifest["outcome"] == outcome.value
    assert decoded_manifest["cleanup_verified"] is True
    assert decoded_manifest["candidate_metrics_accepted"] is False


def test_fold_b_timeout_closes_on_finalizable_qualified_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(tmp_path, monkeypatch)
    failures: list[SupervisedFoldFMExecutionError] = []

    def fold_control(
        *,
        fold_name: str,
        campaign_store: CampaignStore,
        artifacts: ArtifactStore,
        **_kwargs: object,
    ) -> object:
        assert fold_name == "B"
        error = _trusted_execution_error(
            fold_name=fold_name,
            outcome=ExecutionOutcome.TIMED_OUT,
            campaign_store=campaign_store,
            artifacts=artifacts,
        )
        failures.append(error)
        raise error

    monkeypatch.setattr(runtime, "_fold_control", fold_control)

    outcome = harness.run()

    assert harness.science_configs == []
    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.selection is None
    assert outcome.scientific_result_digest is None
    assert outcome.launches_used == 7
    checkpoint = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.FOLD_CONTROLS_READY
    )
    assert checkpoint.evidence["reason"] == "fold_B_control_branch_failed"
    assert checkpoint.evidence["fold_b_status"] == "failed"
    assert checkpoint.evidence["fold_a_status"] == "not_started"
    closure = cast(CandidateExecutionArtifacts, failures[0].artifacts)
    _assert_exact_failure_diagnostic(
        harness,
        reason="fold_B_control_branch_failed",
        outcome=ExecutionOutcome.TIMED_OUT,
        closure=closure,
    )
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        launches = store.launches()
        assert store.snapshot().launches_used == 7
        assert len(launches) == 7
        assert launches[-1].launch_number == 7
        assert launches[-1].category == LaunchCategory.DIVERSE_INNER_SCREEN.value
        assert launches[-1].state == "FINISHED"
        assert launches[-1].charged is True


def test_fold_a_memory_limit_after_successful_fold_b_closes_on_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(tmp_path, monkeypatch)
    failures: list[SupervisedFoldFMExecutionError] = []

    def fold_control(
        *,
        fold_name: str,
        campaign_store: CampaignStore,
        artifacts: ArtifactStore,
        **_kwargs: object,
    ) -> object:
        if fold_name == "B":
            _finish_charged_launch(
                campaign_store,
                fold_name="B",
                category=LaunchCategory.DIVERSE_INNER_SCREEN,
                outcome=ExecutionOutcome.SUCCEEDED,
            )
            return _fold("B")
        assert fold_name == "A"
        error = _trusted_execution_error(
            fold_name=fold_name,
            outcome=ExecutionOutcome.MEMORY_LIMIT,
            campaign_store=campaign_store,
            artifacts=artifacts,
        )
        failures.append(error)
        raise error

    monkeypatch.setattr(runtime, "_fold_control", fold_control)

    outcome = harness.run()

    assert harness.science_configs == []
    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.selection is None
    assert outcome.scientific_result_digest is None
    assert outcome.launches_used == 8
    checkpoint = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.FOLD_CONTROLS_READY
    )
    assert checkpoint.evidence["reason"] == "fold_A_control_branch_failed"
    assert checkpoint.evidence["fold_b_status"] == "completed"
    assert checkpoint.evidence["fold_a_status"] == "failed"
    assert checkpoint.evidence["fold_b_evidence_digest"] == _sha256(b"fold-B")
    closure = cast(CandidateExecutionArtifacts, failures[0].artifacts)
    _assert_exact_failure_diagnostic(
        harness,
        reason="fold_A_control_branch_failed",
        outcome=ExecutionOutcome.MEMORY_LIMIT,
        closure=closure,
    )
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        launches = store.launches()
        assert store.snapshot().launches_used == 8
        assert len(launches) == 8
        assert tuple(item.launch_number for item in launches[-2:]) == (7, 8)
        assert tuple(item.category for item in launches[-2:]) == (
            LaunchCategory.DIVERSE_INNER_SCREEN.value,
            LaunchCategory.TEMPORAL_FOLD_CONFIRMATION.value,
        )
        assert tuple(item.state for item in launches[-2:]) == ("FINISHED", "FINISHED")
        assert all(item.charged for item in launches[-2:])


def test_cancelled_fold_execution_propagates_without_fallback_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(tmp_path, monkeypatch)

    def fold_control(
        *,
        fold_name: str,
        campaign_store: CampaignStore,
        artifacts: ArtifactStore,
        **_kwargs: object,
    ) -> object:
        raise _trusted_execution_error(
            fold_name=fold_name,
            outcome=ExecutionOutcome.CANCELLED,
            campaign_store=campaign_store,
            artifacts=artifacts,
        )

    monkeypatch.setattr(runtime, "_fold_control", fold_control)

    with pytest.raises(FullCampaignCancelled, match="cancelled"):
        harness.run()

    assert harness.science_configs == []
    _assert_no_outcome(harness)
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        launches = store.launches()
        assert store.snapshot().launches_used == 7
        assert len(launches) == 7
        assert launches[-1].state == "FINISHED"
        assert launches[-1].charged is True


@pytest.mark.parametrize(
    "execution_outcome",
    (
        pytest.param(ExecutionOutcome.EXIT_NONZERO, id="exit-nonzero"),
        pytest.param(ExecutionOutcome.SUCCEEDED, id="succeeded-but-output-invalid"),
    ),
)
def test_nonrecoverable_fold_execution_propagates_without_fallback_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_outcome: ExecutionOutcome,
) -> None:
    harness = _install_harness(tmp_path, monkeypatch)
    failures: list[SupervisedFoldFMExecutionError] = []

    def fold_control(
        *,
        fold_name: str,
        campaign_store: CampaignStore,
        artifacts: ArtifactStore,
        **_kwargs: object,
    ) -> object:
        error = _trusted_execution_error(
            fold_name=fold_name,
            outcome=execution_outcome,
            campaign_store=campaign_store,
            artifacts=artifacts,
        )
        failures.append(error)
        raise error

    monkeypatch.setattr(runtime, "_fold_control", fold_control)

    with pytest.raises(SupervisedFoldFMExecutionError) as raised:
        harness.run()

    assert raised.value is failures[0]
    assert harness.science_configs == []
    _assert_no_outcome(harness)
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        launches = store.launches()
        assert store.snapshot().launches_used == 7
        assert len(launches) == 7
        assert launches[-1].state == "FINISHED"
        assert launches[-1].charged is True

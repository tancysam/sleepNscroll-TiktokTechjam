from __future__ import annotations

import hashlib
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from kuairand_agent.campaign.generated_scientific_runner import (
    DurableGeneratedScientificRunner,
    FileScientificRunEvidenceRepository,
    FoldBFusionSelector,
    GeneratedScientificRunnerCancelledError,
    GeneratedScientificRunRecord,
    ProtectedScoringCapability,
    ScientificRunEvidenceRepository,
    ScientificTierCapabilities,
    TrustedFMRankFusion,
)
from kuairand_agent.campaign.scientific import (
    ScientificRunRequest,
    ScientificTier,
)
from kuairand_agent.candidates.fusion import normalize_within_user_percentiles
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.candidate_executor import (
    CandidateAction,
    CandidateExecutionArtifacts,
    CandidateExecutionJournal,
    GeneratedCandidateExecutor,
    GeneratedCandidateIdentity,
    LocalCandidateLimits,
    put_numpy_capability,
)
from kuairand_agent.execution.policy import WorkspacePolicy
from kuairand_agent.execution.runner import ExecutionResult, ExecutionSpec, ProcessRecord
from kuairand_agent.execution.workspace import CandidateWorkspace, WorkspaceMaterializer
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, ScoreResult, SplitIdentity
from kuairand_agent.scoring.submission import prediction_digest

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
TEMPLATE = ROOT / "candidate_templates" / "lambdarank"
_DIGESTS = tuple(
    hashlib.sha256(f"generated-science-{index}".encode()).hexdigest() for index in range(20)
)


class _Repository(ScientificRunEvidenceRepository):
    def __init__(self) -> None:
        self.values: dict[str, GeneratedScientificRunRecord] = {}
        self.commits = 0

    def load(self, request_digest: str) -> GeneratedScientificRunRecord | None:
        return self.values.get(request_digest)

    def commit(self, record: GeneratedScientificRunRecord) -> None:
        prior = self.values.get(record.request_digest)
        if prior is not None and prior.digest != record.digest:
            raise AssertionError("contradictory scientific evidence")
        self.values[record.request_digest] = record
        self.commits += 1


class _Journal(CandidateExecutionJournal):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.candidate_requests: list[dict[str, object]] = []

    def prepare(
        self,
        *,
        action: CandidateAction,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
    ) -> None:
        self.events.append(f"prepare:{action.value}:{spec.execution_id}")
        self.candidate_requests.append(json.loads(workspace.request_path.read_text()))

    def commit(self, process: ProcessRecord) -> None:
        self.events.append(f"commit:{process.execution_id}")

    def finish(
        self,
        *,
        action: CandidateAction,
        result: ExecutionResult,
        artifacts: CandidateExecutionArtifacts,
    ) -> None:
        assert artifacts.output_validated
        self.events.append(f"finish:{action.value}:{result.execution_id}")


class _JournalFactory:
    def __init__(self, journal: _Journal) -> None:
        self.journal = journal

    def __call__(
        self,
        *,
        request: ScientificRunRequest,
        action: CandidateAction,
        execution_id: str,
    ) -> CandidateExecutionJournal:
        assert request.digest.startswith(execution_id.rsplit("-", 1)[-1])
        expected_labels = {"predict", "replay"} if action is CandidateAction.PREDICT else {"train"}
        assert any(label in execution_id for label in expected_labels)
        return self.journal


def _approved_inputs(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document["approved_inputs"])


def _identity(store: ArtifactStore) -> GeneratedCandidateIdentity:
    source = store.put_directory(TEMPLATE, kind=ArtifactKind.SOURCE)
    return GeneratedCandidateIdentity(
        source_snapshot=source,
        source_digest=_DIGESTS[0],
        config_digest=hashlib.sha256((TEMPLATE / "config.json").read_bytes()).hexdigest(),
    )


def _request(
    identity: GeneratedCandidateIdentity,
    *,
    scorer_digest: str = _DIGESTS[8],
) -> ScientificRunRequest:
    return ScientificRunRequest(
        campaign_digest=_DIGESTS[1],
        candidate_id="generated-lambdarank-v1",
        reference_candidate_id="official-fm",
        family="lambdarank",
        tier=ScientificTier.FOLD_B_SCREEN,
        fold_id="B",
        seed=7,
        source_digest=identity.source_digest,
        parent_source_digest=_DIGESTS[2],
        executable_diff_digest=_DIGESTS[3],
        material_change_digest=_DIGESTS[4],
        controller_attestation_digest=_DIGESTS[5],
        config_digest=identity.config_digest,
        training_policy_digest=_DIGESTS[6],
        data_digest=_DIGESTS[7],
        scorer_digest=scorer_digest,
        environment_digest=_DIGESTS[9],
    )


def _executor(tmp_path: Path, store: ArtifactStore) -> GeneratedCandidateExecutor:
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces",
        artifact_store=store,
        policy=WorkspacePolicy(max_source_files=8),
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    return GeneratedCandidateExecutor(
        artifact_store=store,
        workspace_materializer=workspaces,
        control_root=controls,
        interpreter=Path(sys.executable),
        limits=LocalCandidateLimits(
            timeout_seconds=60,
            memory_limit_bytes=2 * 1024**3,
            workspace_disk_limit_bytes=512 * 1024**2,
            output_limit_bytes=128 * 1024**2,
            temp_limit_bytes=128 * 1024**2,
            threads=2,
        ),
    )


def test_real_generated_runner_replays_scores_hides_labels_and_caches_exact_request(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    identity = _identity(store)
    journal = _Journal()
    repository = _Repository()

    steps = np.repeat(np.arange(8, dtype=np.float64), 6)
    users = np.tile(np.asarray([20, 10, 30, 40, 50, 60], dtype=np.int64), 8)
    videos = np.arange(steps.size, dtype=np.int64)
    targets = (steps >= 4).astype(np.int8)
    features = np.column_stack((steps, np.sin(steps), users % 7)).astype("<f8")
    trusted_control = np.cos(np.arange(steps.size, dtype=np.float64))
    user_ids = tuple(int(value) for value in users)
    video_ids = tuple(int(value) for value in videos)
    split = SplitIdentity("inner_fold_B", "fold-b-valid", features.shape[0])
    alignment = Alignment.from_ids(split=split, user_ids=user_ids, video_ids=video_ids)
    protected = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)
    request = _request(identity, scorer_digest=protected.scorer_digest)
    capabilities = ScientificTierCapabilities(
        tier=ScientificTier.FOLD_B_SCREEN,
        fold_id="B",
        scientific_data_digest=request.data_digest,
        training_data_digest=_DIGESTS[10],
        prediction_data_digest=_DIGESTS[11],
        training_split_token="fold-b-train",
        prediction_split_token="fold-b-valid",
        training_features=put_numpy_capability(store, features),
        training_targets=put_numpy_capability(store, targets),
        training_user_groups=put_numpy_capability(store, users),
        prediction_features=put_numpy_capability(store, features),
        prediction_row_count=features.shape[0],
        fusion=TrustedFMRankFusion(
            phase=DataPhase.INNER_VALID,
            weights=(0.0, 1.0),
            user_ids=user_ids,
            video_ids=video_ids,
            control_scores=trusted_control,
        ),
    )

    def score(values: npt.NDArray[np.float64]) -> ScoreResult:
        return protected.score_with_encoded_labels(
            alignment=alignment,
            split=split,
            labels=targets,
            scores=values,
        )

    scoring = ProtectedScoringCapability(
        tier=ScientificTier.FOLD_B_SCREEN,
        fold_id="B",
        scientific_data_digest=request.data_digest,
        scorer_digest=protected.scorer_digest,
        alignment_digest=alignment.digest,
        row_count=features.shape[0],
        callback=score,
    )
    runner = DurableGeneratedScientificRunner(
        executor=_executor(tmp_path, store),
        artifact_store=store,
        identity=identity,
        capabilities={ScientificTier.FOLD_B_SCREEN: capabilities},
        scoring_callbacks={ScientificTier.FOLD_B_SCREEN: scoring},
        journal_factory=_JournalFactory(journal),
        evidence_repository=repository,
    )

    first = runner(request)
    retry = runner(request)

    expected = normalize_within_user_percentiles(
        user_ids,
        video_ids,
        trusted_control,
        phase=DataPhase.INNER_VALID,
    ).scores
    assert first.digest == retry.digest
    assert first.request_digest == request.digest
    assert first.metrics is not None
    assert first.replay_verified
    assert first.gates.failures == ()
    assert first.identities.prediction_digest == prediction_digest(expected)
    assert first.identities.scorer_digest == protected.scorer_digest
    prediction_request = cast(dict[str, object], journal.candidate_requests[1]["request"])
    assert first.identities.checkpoint_digest == prediction_request["checkpoint_digest"]
    assert first.resources.wall_seconds > 0.0
    assert first.resources.peak_rss_bytes > 0
    assert first.resources.disk_bytes > 0
    assert repository.commits == 1
    assert len(journal.events) == 9
    record = runner.load_record(request.digest)
    assert record is repository.values[request.digest]
    assert record is not None
    assert record.evidence.digest == first.digest
    assert record.checkpoint.sha256 == first.identities.checkpoint_digest
    assert record.raw_prediction.sha256 == record.replay_prediction.sha256
    assert record.scored_prediction_digest == first.identities.prediction_digest
    assert store.verify(record.checkpoint).is_file()
    assert store.verify(record.raw_prediction).is_file()
    assert store.verify(record.scored_prediction).is_file()
    assert record.generated_replay_exact

    persistent = FileScientificRunEvidenceRepository(tmp_path / "scientific-records")
    persistent.commit(record)
    persistent.commit(record)
    restored = FileScientificRunEvidenceRepository(tmp_path / "scientific-records").load(
        request.digest
    )
    assert restored is not None
    assert restored.digest == record.digest
    assert restored.manifest() == record.manifest()
    try:
        persistent.commit(replace(record, source_snapshot_digest=_DIGESTS[12]))
    except ValueError as exc:
        assert "contradictory" in str(exc)
    else:  # pragma: no cover - explicit append-only repository assertion.
        raise AssertionError("contradictory durable record replaced an existing request")
    record_path = persistent.root / f"{request.digest}.json"
    record_path.write_bytes(record_path.read_bytes() + b"\n")
    try:
        persistent.load(request.digest)
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:  # pragma: no cover - explicit durable corruption assertion.
        raise AssertionError("non-canonical durable record was accepted")

    train_inputs = _approved_inputs(journal.candidate_requests[0])
    predict_inputs = _approved_inputs(journal.candidate_requests[1])
    replay_inputs = _approved_inputs(journal.candidate_requests[2])
    assert {item["name"] for item in train_inputs} == {"features", "targets", "user_groups"}
    assert {item["name"] for item in predict_inputs} == {"features"}
    assert {item["name"] for item in replay_inputs} == {"features"}
    assert all("target" not in json.dumps(item).lower() for item in predict_inputs + replay_inputs)

    repository.values[request.digest] = replace(
        record,
        scored_prediction=record.raw_prediction,
    )
    try:
        runner(request)
    except ValueError as exc:
        assert "scored prediction" in str(exc)
    else:  # pragma: no cover - explicit durable corruption assertion.
        raise AssertionError("mismatched scored prediction artifact was accepted from cache")


def test_fold_b_grid_scores_every_fixed_weight_once_and_persists_frozen_winner(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    identity = _identity(store)
    journal = _Journal()
    repository = _Repository()
    steps = np.repeat(np.arange(8, dtype=np.float64), 6)
    users = np.tile(np.asarray([20, 10, 30, 40, 50, 60], dtype=np.int64), 8)
    videos = np.arange(steps.size, dtype=np.int64)
    targets = (steps >= 4).astype(np.int8)
    features = np.column_stack((steps, np.sin(steps), users % 7)).astype("<f8")
    trusted_control = np.cos(np.arange(steps.size, dtype=np.float64))
    user_ids = tuple(int(value) for value in users)
    video_ids = tuple(int(value) for value in videos)
    split = SplitIdentity("inner_fold_B", "fold-b-valid", features.shape[0])
    alignment = Alignment.from_ids(split=split, user_ids=user_ids, video_ids=video_ids)
    protected = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)
    request = _request(identity, scorer_digest=protected.scorer_digest)

    def score(values: npt.NDArray[np.float64]) -> ScoreResult:
        return protected.score_with_encoded_labels(
            alignment=alignment,
            split=split,
            labels=targets,
            scores=values,
        )

    capabilities = ScientificTierCapabilities(
        tier=ScientificTier.FOLD_B_SCREEN,
        fold_id="B",
        scientific_data_digest=request.data_digest,
        training_data_digest=_DIGESTS[10],
        prediction_data_digest=_DIGESTS[11],
        training_split_token="fold-b-grid-train",
        prediction_split_token="fold-b-grid-valid",
        training_features=put_numpy_capability(store, features),
        training_targets=put_numpy_capability(store, targets),
        training_user_groups=put_numpy_capability(store, users),
        prediction_features=put_numpy_capability(store, features),
        prediction_row_count=features.shape[0],
        fusion=FoldBFusionSelector(
            user_ids=user_ids,
            video_ids=video_ids,
            control_scores=trusted_control,
        ),
    )
    runner = DurableGeneratedScientificRunner(
        executor=_executor(tmp_path, store),
        artifact_store=store,
        identity=identity,
        capabilities={ScientificTier.FOLD_B_SCREEN: capabilities},
        scoring_callbacks={
            ScientificTier.FOLD_B_SCREEN: ProtectedScoringCapability(
                tier=ScientificTier.FOLD_B_SCREEN,
                fold_id="B",
                scientific_data_digest=request.data_digest,
                scorer_digest=protected.scorer_digest,
                alignment_digest=alignment.digest,
                row_count=features.shape[0],
                callback=score,
            )
        },
        journal_factory=_JournalFactory(journal),
        evidence_repository=repository,
    )

    evidence = runner(request)
    record = runner.load_record(request.digest)

    assert record is not None
    assert record.fusion_selection is not None
    assert tuple(point.weights for point in record.fusion_selection.points) == (
        (1.0, 0.0),
        (0.75, 0.25),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.0, 1.0),
    )
    expected_winner = max(
        enumerate(record.fusion_selection.points),
        key=lambda item: (item[1].metrics.primary_decimal, -item[0]),
    )[1]
    assert record.fusion_selection.selected_weights == expected_winner.weights
    assert record.fusion_selection.selected_prediction_digest == expected_winner.prediction_digest
    assert evidence.identities.prediction_digest == expected_winner.prediction_digest
    assert len(journal.events) == 9
    persistent = FileScientificRunEvidenceRepository(tmp_path / "grid-records")
    persistent.commit(record)
    restored = persistent.load(request.digest)
    assert restored is not None
    assert restored.fusion_selection is not None
    assert restored.fusion_selection.digest == record.fusion_selection.digest

    cancellation = threading.Event()

    def cancel_during_grid(values: npt.NDArray[np.float64]) -> ScoreResult:
        result = score(values)
        cancellation.set()
        return result

    cancelled_repository = _Repository()
    cancelled = DurableGeneratedScientificRunner(
        executor=_executor(tmp_path / "cancelled-grid", store),
        artifact_store=store,
        identity=identity,
        capabilities={ScientificTier.FOLD_B_SCREEN: capabilities},
        scoring_callbacks={
            ScientificTier.FOLD_B_SCREEN: ProtectedScoringCapability(
                tier=ScientificTier.FOLD_B_SCREEN,
                fold_id="B",
                scientific_data_digest=request.data_digest,
                scorer_digest=protected.scorer_digest,
                alignment_digest=alignment.digest,
                row_count=features.shape[0],
                callback=cancel_during_grid,
            )
        },
        journal_factory=_JournalFactory(_Journal()),
        evidence_repository=cancelled_repository,
        cancel_event=cancellation,
    )

    with pytest.raises(GeneratedScientificRunnerCancelledError):
        cancelled(replace(request, seed=8))

    assert cancelled_repository.commits == 0


def test_runner_rejects_unprepared_tier_without_calling_generated_code(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    identity = _identity(store)
    request = _request(identity)
    journal = _Journal()
    runner = DurableGeneratedScientificRunner(
        executor=_executor(tmp_path, store),
        artifact_store=store,
        identity=identity,
        capabilities={},
        scoring_callbacks={},
        journal_factory=_JournalFactory(journal),
        evidence_repository=_Repository(),
    )

    try:
        runner(request)
    except ValueError as exc:
        assert "prepared" in str(exc)
    else:  # pragma: no cover - explicit fail-closed assertion.
        raise AssertionError("unprepared scientific tier reached generated execution")
    assert journal.events == []


def test_cancelled_generated_scientific_runner_admits_no_action(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    identity = _identity(store)
    request = _request(identity)
    journal = _Journal()
    cancellation = threading.Event()
    cancellation.set()
    runner = DurableGeneratedScientificRunner(
        executor=_executor(tmp_path, store),
        artifact_store=store,
        identity=identity,
        capabilities={},
        scoring_callbacks={},
        journal_factory=_JournalFactory(journal),
        evidence_repository=_Repository(),
        cancel_event=cancellation,
    )

    with pytest.raises(GeneratedScientificRunnerCancelledError):
        runner(request)

    assert journal.events == []

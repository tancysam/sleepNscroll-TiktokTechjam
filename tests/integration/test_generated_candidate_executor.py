from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.candidate_executor import (
    CandidateAction,
    CandidateExecutionArtifacts,
    CandidateExecutionError,
    CandidateExecutionJournal,
    GeneratedCandidateExecutor,
    GeneratedCandidateIdentity,
    GeneratedPredictRequest,
    GeneratedTrainRequest,
    LocalCandidateLimits,
    put_numpy_capability,
)
from kuairand_agent.execution.policy import SplitRole, WorkspacePolicy
from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionResult,
    ExecutionSpec,
    ProcessRecord,
)
from kuairand_agent.execution.workspace import CandidateWorkspace, WorkspaceMaterializer

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "candidate_templates" / "lambdarank"
SEED = ROOT / "candidate_seed"


class _Journal(CandidateExecutionJournal):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.prepared: dict[str, ExecutionSpec] = {}
        self.finished: dict[str, tuple[ExecutionResult, CandidateExecutionArtifacts]] = {}

    def prepare(
        self,
        *,
        action: CandidateAction,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
    ) -> None:
        assert workspace.execution_id == spec.execution_id
        self.events.append(f"prepare:{action.value}:{spec.execution_id}")
        self.prepared[spec.execution_id] = spec

    def commit(self, process: ProcessRecord) -> None:
        assert process.execution_id in self.prepared
        self.events.append(f"commit:{process.execution_id}")

    def finish(
        self,
        *,
        action: CandidateAction,
        result: ExecutionResult,
        artifacts: CandidateExecutionArtifacts,
    ) -> None:
        self.events.append(f"finish:{action.value}:{result.execution_id}")
        self.finished[result.execution_id] = (result, artifacts)


def _identity(
    store: ArtifactStore,
    source_dir: Path = TEMPLATE,
) -> GeneratedCandidateIdentity:
    source = store.put_directory(source_dir, kind=ArtifactKind.SOURCE)
    config_digest = hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest()
    return GeneratedCandidateIdentity(
        source_snapshot=source,
        source_digest="1" * 64,
        config_digest=config_digest,
    )


def test_runner_workspace_executor_trains_predicts_and_exactly_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(max_source_files=8),
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    executor = GeneratedCandidateExecutor(
        artifact_store=artifacts,
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
    journal = _Journal()
    identity = _identity(artifacts)

    steps = np.repeat(np.arange(8, dtype=np.float64), 6)
    groups = np.tile(np.asarray([20, 10, 30, 40, 50, 60], dtype=np.int64), 8)
    targets = (steps >= 4).astype(np.int8)
    features = np.column_stack((steps, np.sin(steps), groups % 7)).astype("<f8")
    feature_ref = put_numpy_capability(artifacts, features)
    target_ref = put_numpy_capability(artifacts, targets)
    group_ref = put_numpy_capability(artifacts, groups)

    trained = executor.train(
        GeneratedTrainRequest(
            execution_id="generated-tree-train",
            identity=identity,
            split_role=SplitRole.INNER_TRAIN,
            data_digest="2" * 64,
            split_token="fold-b-train",
            seed=7,
            features=feature_ref,
            targets=target_ref,
            user_groups=group_ref,
        ),
        journal=journal,
    )
    first = executor.predict(
        GeneratedPredictRequest(
            execution_id="generated-tree-predict-a",
            identity=identity,
            split_role=SplitRole.INNER_VALID,
            data_digest="3" * 64,
            split_token="fold-b-valid",
            expected_count=features.shape[0],
            features=feature_ref,
            checkpoint=trained.checkpoint,
        ),
        journal=journal,
    )
    replay = executor.predict(
        GeneratedPredictRequest(
            execution_id="generated-tree-predict-b",
            identity=identity,
            split_role=SplitRole.INNER_VALID,
            data_digest="3" * 64,
            split_token="fold-b-valid",
            expected_count=features.shape[0],
            features=feature_ref,
            checkpoint=trained.checkpoint,
        ),
        journal=journal,
    )

    assert artifacts.verify(trained.checkpoint).is_file()
    assert first.scores.tobytes() == replay.scores.tobytes()
    assert first.prediction.sha256 == replay.prediction.sha256
    assert float(first.scores[24:].mean()) > float(first.scores[:24].mean())
    assert trained.seed == 7
    assert trained.artifacts.output_validated
    assert first.artifacts.output_validated
    assert all(result.succeeded for result, _ in journal.finished.values())
    assert journal.events == [
        "prepare:train:generated-tree-train",
        "commit:generated-tree-train",
        "finish:train:generated-tree-train",
        "prepare:predict:generated-tree-predict-a",
        "commit:generated-tree-predict-a",
        "finish:predict:generated-tree-predict-a",
        "prepare:predict:generated-tree-predict-b",
        "commit:generated-tree-predict-b",
        "finish:predict:generated-tree-predict-b",
    ]
    assert not (tmp_path / "workspaces" / "generated-tree-train").exists()
    assert not (tmp_path / "workspaces" / "generated-tree-predict-a").exists()
    assert all(
        artifacts.verify(evidence.artifact("workspace_cleanup")).is_file()
        for _result, evidence in journal.finished.values()
    )

    with monkeypatch.context() as validation_patch:

        def fail_train_validation(*_args: object, **_kwargs: object) -> None:
            raise ValueError("checkpoint manifest names the wrong path")

        validation_patch.setattr(
            "kuairand_agent.execution.candidate_executor.validate_train_outputs",
            fail_train_validation,
        )
        with pytest.raises(CandidateExecutionError) as validation_failure:
            executor.train(
                GeneratedTrainRequest(
                    execution_id="generated-tree-train-invalid-output",
                    identity=identity,
                    split_role=SplitRole.INNER_TRAIN,
                    data_digest="2" * 64,
                    split_token="fold-b-train",
                    seed=9,
                    features=feature_ref,
                    targets=target_ref,
                    user_groups=group_ref,
                ),
                journal=journal,
            )

        assert "ValueError: checkpoint manifest names the wrong path" in str(
            validation_failure.value
        )

    def fail_cleanup(_workspace: CandidateWorkspace) -> None:
        raise OSError("synthetic success cleanup failure")

    monkeypatch.setattr(workspaces, "cleanup", fail_cleanup)
    with pytest.raises(CandidateExecutionError) as cleanup_failure:
        executor.predict(
            GeneratedPredictRequest(
                execution_id="generated-tree-predict-cleanup-failure",
                identity=identity,
                split_role=SplitRole.INNER_VALID,
                data_digest="3" * 64,
                split_token="fold-b-valid",
                expected_count=features.shape[0],
                features=feature_ref,
                checkpoint=trained.checkpoint,
            ),
            journal=journal,
        )

    assert cleanup_failure.value.result is not None
    assert cleanup_failure.value.result.succeeded
    cleanup_artifacts = cleanup_failure.value.artifacts
    assert cleanup_artifacts is not None and not cleanup_artifacts.output_validated
    cleanup_receipt = json.loads(
        artifacts.read_bytes(
            cleanup_artifacts.artifact("workspace_cleanup"),
            max_bytes=4096,
        )
    )
    assert cleanup_receipt == {
        "error_type": "OSError",
        "execution_id": "generated-tree-predict-cleanup-failure",
        "schema_version": 1,
        "workspace_removed": False,
    }
    assert "workspace cleanup failed: OSError" in artifacts.read_bytes(
        cleanup_artifacts.artifact("failure_diagnostic"),
        max_bytes=4096,
    ).decode("utf-8")

    with pytest.raises(CandidateExecutionError) as train_cleanup_failure:
        executor.train(
            GeneratedTrainRequest(
                execution_id="generated-tree-train-cleanup-failure",
                identity=identity,
                split_role=SplitRole.INNER_TRAIN,
                data_digest="2" * 64,
                split_token="fold-b-train",
                seed=8,
                features=feature_ref,
                targets=target_ref,
                user_groups=group_ref,
            ),
            journal=journal,
        )

    assert train_cleanup_failure.value.result is not None
    assert train_cleanup_failure.value.result.succeeded
    train_failure_artifacts = train_cleanup_failure.value.artifacts
    assert train_failure_artifacts is not None
    train_receipt = json.loads(
        artifacts.read_bytes(
            train_failure_artifacts.artifact("workspace_cleanup"),
            max_bytes=4096,
        )
    )
    assert train_receipt["execution_id"] == "generated-tree-train-cleanup-failure"
    assert train_receipt["workspace_removed"] is False
    assert train_receipt["error_type"] == "OSError"
    assert "workspace cleanup failed: OSError" in artifacts.read_bytes(
        train_failure_artifacts.artifact("failure_diagnostic"),
        max_bytes=4096,
    ).decode("utf-8")


def test_candidate_seed_wrapper_accepts_model_specific_checkpoint_through_real_executor(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controls = tmp_path / "controls"
    controls.mkdir()
    seed_source = tmp_path / "seed-source"
    seed_source.mkdir()
    for name in ("README.md", "candidate.py", "config.json", "model_impl.py"):
        shutil.copy2(SEED / name, seed_source / name)
    (seed_source / "config.json").write_text('{"candidate_family":"contract-probe"}\n')
    (seed_source / "model_impl.py").write_text(
        """\
import numpy as np

def validate_config(config):
    if config != {"candidate_family": "contract-probe"}:
        raise ValueError("unexpected config")

def train_model(features, targets, user_groups, config, seed):
    del user_groups, config, seed
    return {
        "intercept_only": np.asarray(targets.mean(), dtype=np.float64),
        "projection_vector": np.arange(1, features.shape[1] + 1, dtype=np.float64),
    }

def predict_scores(features, checkpoint):
    return features @ checkpoint["projection_vector"] + checkpoint["intercept_only"]

def training_diagnostics(config, checkpoint):
    del config
    return {"learned_intercept": float(checkpoint["intercept_only"])}
""",
        encoding="utf-8",
    )
    executor = GeneratedCandidateExecutor(
        artifact_store=artifacts,
        workspace_materializer=WorkspaceMaterializer(
            tmp_path / "workspaces",
            artifact_store=artifacts,
            policy=WorkspacePolicy(max_source_files=8),
        ),
        control_root=controls,
        interpreter=Path(sys.executable),
        limits=LocalCandidateLimits(
            timeout_seconds=30,
            memory_limit_bytes=1024**3,
            workspace_disk_limit_bytes=128 * 1024**2,
            output_limit_bytes=64 * 1024**2,
            temp_limit_bytes=64 * 1024**2,
            threads=1,
        ),
    )
    journal = _Journal()
    identity = _identity(artifacts, seed_source)
    features = np.array(
        [[-2.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
        dtype="<f8",
    )
    targets = np.array([0, 1, 0, 1], dtype=np.int8)
    groups = np.array([10, 10, 20, 20], dtype=np.int64)

    trained = executor.train(
        GeneratedTrainRequest(
            execution_id="seed-real-executor-train",
            identity=identity,
            split_role=SplitRole.INNER_TRAIN,
            data_digest="2" * 64,
            split_token="seed-real-train",
            seed=7,
            features=put_numpy_capability(artifacts, features),
            targets=put_numpy_capability(artifacts, targets),
            user_groups=put_numpy_capability(artifacts, groups),
        ),
        journal=journal,
    )
    predicted = executor.predict(
        GeneratedPredictRequest(
            execution_id="seed-real-executor-predict",
            identity=identity,
            split_role=SplitRole.INNER_VALID,
            data_digest="3" * 64,
            split_token="seed-real-valid",
            expected_count=features.shape[0],
            features=put_numpy_capability(artifacts, features),
            checkpoint=trained.checkpoint,
        ),
        journal=journal,
    )

    assert trained.artifacts.output_validated
    assert predicted.artifacts.output_validated
    assert trained.checkpoint.sha256 == trained.checkpoint_digest
    assert predicted.scores.shape == (4,)
    with np.load(artifacts.verify(trained.checkpoint), allow_pickle=False) as checkpoint:
        assert set(checkpoint.files) == {"intercept_only", "projection_vector"}


def test_executor_identity_rejects_config_not_present_in_source_snapshot(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.put_directory(TEMPLATE, kind=ArtifactKind.SOURCE)

    try:
        GeneratedCandidateIdentity(
            source_snapshot=source,
            source_digest="1" * 64,
            config_digest="f" * 64,
        )
    except ValueError as exc:
        assert "config.json" in str(exc)
    else:  # pragma: no cover - explicit red/green contract assertion.
        raise AssertionError("mismatched generated config identity was accepted")


def test_executor_flushes_a_pre_admission_cancellation_without_starting_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(max_source_files=8),
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    executor = GeneratedCandidateExecutor(
        artifact_store=artifacts,
        workspace_materializer=workspaces,
        control_root=controls,
        interpreter=Path(sys.executable),
        limits=LocalCandidateLimits(
            timeout_seconds=60,
            memory_limit_bytes=512 * 1024**2,
            workspace_disk_limit_bytes=128 * 1024**2,
            output_limit_bytes=64 * 1024**2,
            temp_limit_bytes=64 * 1024**2,
            threads=1,
        ),
    )
    features = put_numpy_capability(artifacts, np.ones((4, 2), dtype=np.float64))
    targets = put_numpy_capability(artifacts, np.asarray([0, 1, 0, 1], dtype=np.int8))
    groups = put_numpy_capability(artifacts, np.asarray([1, 1, 2, 2], dtype=np.int64))
    cancellation = threading.Event()
    cancellation.set()
    journal = _Journal()

    def fail_cleanup(_workspace: CandidateWorkspace) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(workspaces, "cleanup", fail_cleanup)

    with pytest.raises(CandidateExecutionError) as raised:
        executor.train(
            GeneratedTrainRequest(
                execution_id="cancelled-generated-train",
                identity=_identity(artifacts),
                split_role=SplitRole.INNER_TRAIN,
                data_digest="2" * 64,
                split_token="fold-b-train",
                seed=7,
                features=features,
                targets=targets,
                user_groups=groups,
            ),
            journal=journal,
            cancel_event=cancellation,
        )

    assert raised.value.result is not None
    assert raised.value.result.outcome is ExecutionOutcome.CANCELLED
    assert raised.value.result.process is None
    assert journal.events == [
        "prepare:train:cancelled-generated-train",
        "finish:train:cancelled-generated-train",
    ]
    result, persisted = journal.finished["cancelled-generated-train"]
    assert result.outcome is ExecutionOutcome.CANCELLED
    diagnostic = artifacts.read_bytes(
        persisted.artifact("failure_diagnostic"),
        max_bytes=4096,
    ).decode("utf-8")
    assert diagnostic == persisted.diagnostic
    assert "workspace cleanup failed: OSError" in diagnostic
    assert (tmp_path / "workspaces" / "cancelled-generated-train").exists()

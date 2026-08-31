from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import StarterFMConfig
from kuairand_agent.candidates.grouping import build_user_grouping
from kuairand_agent.candidates.tree_ranker import (
    BackendFitRequest,
    BackendFitResult,
    LambdaRankConfig,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.data.causal_features import FeatureMatrix
from kuairand_agent.domain.identity import AttemptId, ExperimentId, TrialId
from kuairand_agent.execution.runner import ExecutionSpec
from kuairand_agent.training.lightgbm_cpu import (
    LightGBMCPUTrainer,
    LightGBMCPUTrialPayload,
)
from kuairand_agent.training.official_fm import OfficialFMTrainer, OfficialFMTrialPayload
from kuairand_agent.training.process_executor import ProcessExecutor
from kuairand_agent.training.protocol import (
    JsonValue,
    QualificationStatus,
    QualifiedTrainer,
    ResourceLimits,
    TrainerFailureCode,
    TrialRequest,
)
from kuairand_agent.training.qualification import qualify_trainer

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
_LOCK = "a" * 64
_DATA = "b" * 64
_FOLD = "c" * 64
_CODE = "d" * 64


def _request(
    trainer: QualifiedTrainer,
    payload: object,
    *,
    settings: Mapping[str, JsonValue],
    row_ids: tuple[int, ...],
    seed: int,
    threads: int,
) -> TrialRequest:
    experiment_id = ExperimentId.derive(
        experiment_spec={"model_family": trainer.identity.trainer_id, "objective": "fixture"},
        data_identities={"fixture": _DATA},
        fold_identities={"fixture": _FOLD},
        code_artifact_sha256=_CODE,
    )
    trial_id = TrialId.derive(
        experiment_id=experiment_id,
        trainer_id=trainer.identity.trainer_id,
        trainer_version=trainer.identity.trainer_version,
        backend=trainer.identity.backend,
        precision=trainer.identity.precision,
        dependency_lock_sha256=trainer.identity.dependency_lock_sha256,
        seed=seed,
        fold="fixture",
        fidelity={"rows": len(row_ids)},
        qualified_settings=settings,
    )
    return TrialRequest(
        experiment_id=experiment_id,
        trial_id=trial_id,
        attempt_id=AttemptId.derive(trial_id=trial_id, infrastructure_attempt=1),
        trainer_identity=trainer.identity,
        seed=seed,
        fold="fixture",
        fidelity={"rows": len(row_ids)},
        qualified_settings=settings,
        infrastructure_attempt=1,
        ordered_row_ids=row_ids,
        resource_limits=ResourceLimits(
            timeout_seconds=30.0,
            memory_limit_bytes=512 * 1024 * 1024,
            disk_limit_bytes=512 * 1024 * 1024,
            threads=threads,
        ),
        payload=payload,
    )


class _DeterministicBackend:
    identity = "fixture-lightgbm:cpu:v1"

    def fit(self, request: BackendFitRequest) -> BackendFitResult:
        assert request.params["device_type"] == "cpu"
        assert request.params["deterministic"] is True
        assert sum(request.train_group_sizes) == len(request.train_labels)
        return BackendFitResult(model_text="deterministic-fixture-model", best_iteration=2)

    def predict(
        self,
        *,
        model_text: str,
        features: npt.NDArray[np.float64],
        num_iteration: int,
    ) -> object:
        assert model_text == "deterministic-fixture-model"
        assert num_iteration == 2
        return features[:, 0] - (0.25 * features[:, 1])


def test_cpu_tree_adapter_qualifies_same_backend_without_importing_gpu_runtime() -> None:
    config = LambdaRankConfig(
        seed=7,
        num_threads=1,
        num_boost_round=2,
        early_stopping_rounds=1,
        min_data_in_leaf=1,
    )
    training = FeatureMatrix(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 2.0], [2.0, 0.5]],
        ("safe_rate", "safe_count"),
    )
    prediction = FeatureMatrix(
        [[0.25, 1.0], [1.5, 0.5], [0.75, 0.0]],
        training.feature_names,
    )
    grouping = build_user_grouping(
        ("u1", "u2", "u1", "u2"),
        ("v1", "v2", "v3", "v4"),
        phase=DataPhase.TRAIN,
    )
    trainer = LightGBMCPUTrainer(dependency_lock_sha256=_LOCK)
    payload = LightGBMCPUTrialPayload(
        training_features=training,
        training_labels=np.array([0, 1, 1, 0], dtype=np.int8),
        training_grouping=grouping,
        training_phase=DataPhase.TRAIN,
        prediction_features=prediction,
        prediction_phase=DataPhase.OUTER_VALID,
        config=config,
        backend=_DeterministicBackend(),
    )
    request = _request(
        trainer,
        payload,
        settings={"config_sha256": config.digest},
        row_ids=(100, 101, 102),
        seed=7,
        threads=1,
    )

    receipt = qualify_trainer(trainer, request)
    result = trainer.fit_predict(request)

    assert receipt.status is QualificationStatus.QUALIFIED
    assert receipt.same_backend_replay_verified
    assert result.environment.backend == "lightgbm-cpu"
    assert result.model.backend == "lightgbm-cpu"
    assert result.diagnostics["backend_identity"] == _DeterministicBackend.identity
    np.testing.assert_array_equal(result.predictions, np.array([0.0, 1.375, 0.75]))


def test_installed_cpu_lightgbm_build_qualifies_with_exact_same_backend_replay() -> None:
    config = LambdaRankConfig(
        seed=11,
        num_threads=1,
        num_boost_round=4,
        early_stopping_rounds=2,
        num_leaves=4,
        min_data_in_leaf=1,
    )
    training = FeatureMatrix(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 2.0],
            [2.0, 0.5],
            [0.25, 1.5],
            [1.5, 0.25],
        ],
        ("safe_rate", "safe_count"),
    )
    prediction = FeatureMatrix(
        [[0.25, 1.0], [1.5, 0.5], [0.75, 0.0]],
        training.feature_names,
    )
    grouping = build_user_grouping(
        ("u1", "u1", "u1", "u2", "u2", "u2"),
        tuple(f"v{index}" for index in range(6)),
        phase=DataPhase.TRAIN,
    )
    trainer = LightGBMCPUTrainer(dependency_lock_sha256=_LOCK)
    request = _request(
        trainer,
        LightGBMCPUTrialPayload(
            training_features=training,
            training_labels=np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
            training_grouping=grouping,
            training_phase=DataPhase.TRAIN,
            prediction_features=prediction,
            prediction_phase=DataPhase.OUTER_VALID,
            config=config,
        ),
        settings={"config_sha256": config.digest},
        row_ids=(110, 111, 112),
        seed=11,
        threads=1,
    )

    receipt = qualify_trainer(trainer, request)
    if receipt.failure_code is TrainerFailureCode.DEPENDENCY_ERROR:
        pytest.skip(receipt.detail or "native CPU LightGBM is unavailable")

    assert receipt.status is QualificationStatus.QUALIFIED
    assert receipt.first_prediction_sha256 == receipt.replay_prediction_sha256
    assert receipt.resource_receipts[0].device == "cpu"


def _inputs(prefix: str, rows: int, *, start_time: int = 0) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u{index % 3}" for index in range(rows)),
        video_id=tuple(f"{prefix}-v{index % 4}" for index in range(rows)),
        date=tuple(20220408 for _ in range(rows)),
        duration_ms=tuple(float(1000 + index * 137) for index in range(rows)),
        tab=tuple(str(index % 2) for index in range(rows)),
        author_id=tuple(f"a{index % 2}" for index in range(rows)),
        time_ms=tuple(start_time + index for index in range(rows)),
    )


@dataclass(frozen=True)
class _Targets:
    primary: npt.NDArray[np.int8]
    training_inputs_digest: str
    digest: str = "e" * 64

    @property
    def row_count(self) -> int:
        return int(self.primary.size)


@dataclass(frozen=True)
class _Scorer:
    validation_inputs_digest: str
    callback: Callable[[npt.NDArray[np.float64]], object]

    def __call__(self, scores: npt.NDArray[np.float64]) -> object:
        return self.callback(scores)


def test_official_fm_adapter_reuses_exact_organizer_training_and_replay() -> None:
    train = _inputs("train", 8)
    validation = _inputs("validation", 4, start_time=100)
    encoding = StarterEncoding.fit(train)
    targets = _Targets(
        np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
        training_inputs_digest=train.digest,
    )

    def constant_scorer(_scores: npt.NDArray[np.float64]) -> dict[str, float]:
        return {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5}

    config = StarterFMConfig(seed=7)
    trainer = OfficialFMTrainer(
        starter_dir=STARTER,
        dependency_lock_sha256=_LOCK,
        config=config,
    )
    payload = OfficialFMTrialPayload(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=validation,
        validation_scorer=_Scorer(validation.digest, constant_scorer),
        prediction_inputs=validation,
    )
    request = _request(
        trainer,
        payload,
        settings={
            "config_sha256": config.digest,
            "starter_manifest_sha256": trainer.starter_manifest_sha256,
        },
        row_ids=(200, 201, 202, 203),
        seed=7,
        threads=1,
    )

    receipt = qualify_trainer(trainer, request)
    result = trainer.fit_predict(request)

    assert receipt.status is QualificationStatus.QUALIFIED
    assert result.model.backend == "organizer-numpy-fm"
    assert result.model.selected_iteration == 1
    assert result.diagnostics["epochs_completed"] == 5
    assert "metrics" not in result.manifest()


def test_process_executor_delegates_to_runner_and_maps_timeout_without_stdout_parsing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = ExecutionSpec(
        execution_id="trainer-timeout",
        nonce="trainer-timeout-nonce-0123456789",
        interpreter=Path(sys.executable),
        arguments=("-c", "import time; print('OOM'); time.sleep(5)"),
        workspace=workspace,
        control_dir=tmp_path / "control",
        timeout_seconds=0.05,
        memory_limit_bytes=512 * 1024 * 1024,
        workspace_disk_limit_bytes=16 * 1024 * 1024,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
        threads=1,
        source_digest="1" * 64,
        config_digest="2" * 64,
        data_digest="3" * 64,
        checkpoint_digest="4" * 64,
        poll_interval_seconds=0.01,
        disk_poll_interval_seconds=0.02,
        termination_grace_seconds=0.1,
    )

    receipt = ProcessExecutor().execute(spec, commit_launch=lambda _record: None)

    assert receipt.failure_code is TrainerFailureCode.TIMEOUT
    assert not receipt.succeeded
    assert receipt.execution.cleanup_verified
    assert receipt.resources.peak_process_count >= 1
    assert not receipt.resources.cpu_seconds_measured
    assert receipt.execution.stdout.path.read_text(encoding="utf-8").strip() == "OOM"

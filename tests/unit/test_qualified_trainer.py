from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from kuairand_agent.domain.identity import AttemptId, ExperimentId, TrialId
from kuairand_agent.training.lightgbm_gpu import LightGBMGPUTrainer
from kuairand_agent.training.protocol import (
    JsonValue,
    QualificationStatus,
    QualifiedTrainer,
    ResourceLimits,
    TrainerContractError,
    TrainerError,
    TrainerFailureCode,
    TrialRequest,
)
from kuairand_agent.training.qualification import qualify_trainer
from kuairand_agent.training.scripted import ScriptedTrainer, ScriptedTrialPayload

_LOCK = "a" * 64
_DIGESTS = tuple(character * 64 for character in "bcdef0123456789")


def _experiment() -> ExperimentId:
    return ExperimentId.derive(
        experiment_spec={"model_family": "scripted", "objective": "fixture"},
        data_identities={"train": _DIGESTS[0], "predict": _DIGESTS[1]},
        fold_identities={"fixture": _DIGESTS[2]},
        code_artifact_sha256=_DIGESTS[3],
    )


def _request(
    trainer: QualifiedTrainer,
    payload: object,
    *,
    settings: Mapping[str, JsonValue] | None = None,
    attempt: int = 1,
) -> TrialRequest:
    experiment_id = _experiment()
    qualified_settings = {} if settings is None else settings
    trial_id = TrialId.derive(
        experiment_id=experiment_id,
        trainer_id=trainer.identity.trainer_id,
        trainer_version=trainer.identity.trainer_version,
        backend=trainer.identity.backend,
        precision=trainer.identity.precision,
        dependency_lock_sha256=trainer.identity.dependency_lock_sha256,
        seed=7,
        fold="fixture",
        fidelity={"rows": 3},
        qualified_settings=qualified_settings,
    )
    return TrialRequest(
        experiment_id=experiment_id,
        trial_id=trial_id,
        attempt_id=AttemptId.derive(
            trial_id=trial_id,
            infrastructure_attempt=attempt,
        ),
        trainer_identity=trainer.identity,
        seed=7,
        fold="fixture",
        fidelity={"rows": 3},
        qualified_settings=qualified_settings,
        infrastructure_attempt=attempt,
        ordered_row_ids=(10, 11, 12),
        resource_limits=ResourceLimits(
            timeout_seconds=30.0,
            memory_limit_bytes=128 * 1024 * 1024,
            disk_limit_bytes=128 * 1024 * 1024,
            threads=1,
        ),
        payload=payload,
    )


def _payload() -> ScriptedTrialPayload:
    return ScriptedTrialPayload(
        predictions=(0.25, -0.5, 1.25),
        training_data_sha256=_DIGESTS[0],
        prediction_data_sha256=_DIGESTS[1],
        training_feature_sha256=_DIGESTS[2],
        prediction_feature_sha256=_DIGESTS[3],
        feature_schema_sha256=_DIGESTS[4],
        config_sha256=_DIGESTS[5],
        model_sha256=_DIGESTS[6],
        training_rows=5,
        feature_count=2,
    )


def test_failure_taxonomy_is_exact_and_closed() -> None:
    assert {code.value for code in TrainerFailureCode} == {
        "UNSUPPORTED",
        "ADMISSION_REJECTED",
        "TIMEOUT",
        "OOM",
        "CANCELLED",
        "DEPENDENCY_ERROR",
        "NUMERICAL_ERROR",
        "INTERNAL_ERROR",
    }


def test_trial_request_recomputes_trial_and_attempt_identities() -> None:
    trainer = ScriptedTrainer(dependency_lock_sha256=_LOCK)
    valid = _request(trainer, _payload())
    wrong_trial = TrialId.derive(
        experiment_id=valid.experiment_id,
        trainer_id=trainer.identity.trainer_id,
        trainer_version=trainer.identity.trainer_version,
        backend="another-cpu",
        precision=trainer.identity.precision,
        dependency_lock_sha256=_LOCK,
        seed=valid.seed,
        fold=valid.fold,
        fidelity=valid.fidelity,
        qualified_settings=valid.qualified_settings,
    )
    with pytest.raises(TrainerContractError, match="trial_id"):
        TrialRequest(
            experiment_id=valid.experiment_id,
            trial_id=wrong_trial,
            attempt_id=valid.attempt_id,
            trainer_identity=valid.trainer_identity,
            seed=valid.seed,
            fold=valid.fold,
            fidelity=valid.fidelity,
            qualified_settings=valid.qualified_settings,
            infrastructure_attempt=valid.infrastructure_attempt,
            ordered_row_ids=valid.ordered_row_ids,
            resource_limits=valid.resource_limits,
            payload=valid.payload,
        )


def test_scripted_trainer_qualifies_only_after_distinct_exact_replay() -> None:
    trainer = ScriptedTrainer(dependency_lock_sha256=_LOCK)
    request = _request(trainer, _payload())

    receipt = qualify_trainer(trainer, request)

    assert isinstance(trainer, QualifiedTrainer)
    assert receipt.status is QualificationStatus.QUALIFIED
    assert receipt.campaign_admissible
    assert receipt.same_backend_replay_verified
    assert receipt.first_prediction_sha256 == receipt.replay_prediction_sha256
    assert len(receipt.resource_receipts) == 2
    assert receipt.p50_wall_seconds is not None
    assert receipt.p95_wall_seconds is not None
    result = trainer.fit_predict(request)
    np.testing.assert_array_equal(result.predictions, np.array([0.25, -0.5, 1.25]))
    assert not result.predictions.flags.writeable
    assert result.data.prediction_rows == 3
    assert result.environment.backend == result.model.backend == "scripted-cpu"


@pytest.mark.parametrize(
    "code",
    [
        TrainerFailureCode.TIMEOUT,
        TrainerFailureCode.OOM,
        TrainerFailureCode.CANCELLED,
        TrainerFailureCode.DEPENDENCY_ERROR,
        TrainerFailureCode.NUMERICAL_ERROR,
        TrainerFailureCode.INTERNAL_ERROR,
    ],
)
def test_qualification_preserves_typed_fit_failure_identity(code: TrainerFailureCode) -> None:
    trainer = ScriptedTrainer(dependency_lock_sha256=_LOCK, fit_failure=code)
    request = _request(trainer, _payload())

    receipt = qualify_trainer(trainer, request)

    assert receipt.status is QualificationStatus.FAILED
    assert receipt.failure_code is code
    assert not receipt.campaign_admissible


def test_gpu_preflight_is_explicitly_unsupported_and_never_runs_cpu() -> None:
    gpu = LightGBMGPUTrainer(dependency_lock_sha256=_LOCK)
    request = _request(gpu, object())

    receipt = gpu.preflight(request)

    assert receipt.status is QualificationStatus.UNSUPPORTED
    assert receipt.failure_code is TrainerFailureCode.UNSUPPORTED
    assert receipt.environment.backend == "lightgbm-gpu"
    assert receipt.environment.device == "gpu"
    assert not gpu.capability.gpu_build_qualified
    assert not gpu.capability.runner_gpu_visible
    with pytest.raises(TrainerError) as raised:
        gpu.fit_predict(request)
    assert raised.value.code is TrainerFailureCode.UNSUPPORTED
    assert raised.value.trial_id == request.trial_id
    assert raised.value.attempt_id == request.attempt_id

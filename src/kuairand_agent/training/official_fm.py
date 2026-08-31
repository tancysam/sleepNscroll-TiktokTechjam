"""Qualified official-FM adapter over the immutable organizer arithmetic."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from kuairand_agent.baselines.starter_fm import (
    STARTER_FIELDS,
    EncodingProtocol,
    StarterFMAdapter,
    StarterFMConfig,
    StarterFMError,
    TrainTargetsProtocol,
    ValidationScorer,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.domain.identity import PredictionId
from kuairand_agent.training.protocol import (
    DataReceipt,
    EnvironmentReceipt,
    FeatureReceipt,
    ModelReceipt,
    QualificationReceipt,
    QualificationStatus,
    ResourceReceipt,
    TimingReceipt,
    TrainerError,
    TrainerFailureCode,
    TrainerIdentity,
    TrialRequest,
    TrialResult,
    validate_request_for_trainer,
)

OFFICIAL_FM_TRAINER_VERSION = "v1"


def _digest_manifest(domain: bytes, value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OfficialFMTrialPayload:
    """Trusted FM capabilities; the validation scorer is an inner/qualification-only closure."""

    encoding: EncodingProtocol
    train_inputs: CanonicalInputs
    train_targets: TrainTargetsProtocol
    validation_inputs: CanonicalInputs
    validation_scorer: ValidationScorer
    prediction_inputs: CanonicalInputs


class OfficialFMTrainer:
    """Expose ``StarterFMAdapter`` through ``QualifiedTrainer`` without changing its loop."""

    def __init__(
        self,
        *,
        starter_dir: str | Path,
        dependency_lock_sha256: str,
        config: StarterFMConfig | None = None,
    ) -> None:
        self._config = StarterFMConfig() if config is None else config
        self._adapter = StarterFMAdapter(starter_dir=starter_dir, config=self._config)
        self._identity = TrainerIdentity(
            trainer_id="official-fm",
            trainer_version=OFFICIAL_FM_TRAINER_VERSION,
            backend="organizer-numpy-fm",
            device="cpu",
            precision="float32",
            dependency_lock_sha256=dependency_lock_sha256,
        )

    @property
    def identity(self) -> TrainerIdentity:
        return self._identity

    @property
    def config(self) -> StarterFMConfig:
        return self._config

    @property
    def starter_manifest_sha256(self) -> str:
        return self._adapter.starter_manifest_digest

    def preflight(self, request: TrialRequest) -> QualificationReceipt:
        validate_request_for_trainer(request, self.identity)
        environment = EnvironmentReceipt.capture(self.identity)
        payload = request.payload
        if not isinstance(payload, OfficialFMTrialPayload):
            return self._rejected(
                request,
                environment,
                "official FM requires OfficialFMTrialPayload",
                checks=("canonical-request", "official-fm-payload"),
            )
        checks = ["canonical-request", "official-fm-payload"]
        if self.config.seed != request.seed:
            return self._rejected(
                request,
                environment,
                "official FM config seed differs from the trial seed",
                checks=tuple([*checks, "seed-identity"]),
            )
        expected_settings = {
            "config_sha256": self.config.digest,
            "starter_manifest_sha256": self._adapter.starter_manifest_digest,
        }
        for name, expected in expected_settings.items():
            if request.qualified_settings.get(name) != expected:
                return self._rejected(
                    request,
                    environment,
                    f"qualified_settings.{name} differs from the official FM adapter",
                    checks=tuple([*checks, "qualified-settings"]),
                )
        checks.append("qualified-settings")
        if len(payload.prediction_inputs) != len(request.ordered_row_ids):
            return self._rejected(
                request,
                environment,
                "prediction input count differs from ordered row identities",
                checks=tuple([*checks, "prediction-row-count"]),
            )
        if getattr(payload.encoding, "training_inputs_digest", None) != payload.train_inputs.digest:
            return self._rejected(
                request,
                environment,
                "official FM encoding is not bound to the supplied training inputs",
                checks=tuple([*checks, "encoding-alignment"]),
            )
        if (
            getattr(payload.train_targets, "training_inputs_digest", None)
            != payload.train_inputs.digest
        ):
            return self._rejected(
                request,
                environment,
                "official FM targets are not bound to the supplied training inputs",
                checks=tuple([*checks, "target-alignment"]),
            )
        if (
            getattr(payload.validation_scorer, "validation_inputs_digest", None)
            != payload.validation_inputs.digest
        ):
            return self._rejected(
                request,
                environment,
                "official FM validation scorer is not bound to validation inputs",
                checks=tuple([*checks, "validation-alignment"]),
            )
        return QualificationReceipt(
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.PREFLIGHT_PASSED,
            checks=tuple(
                [
                    *checks,
                    "canonical-input-capabilities",
                    "prediction-row-count",
                    "encoding-alignment",
                    "target-alignment",
                    "validation-alignment",
                ]
            ),
            environment=environment,
        )

    def _rejected(
        self,
        request: TrialRequest,
        environment: EnvironmentReceipt,
        detail: str,
        *,
        checks: tuple[str, ...],
    ) -> QualificationReceipt:
        return QualificationReceipt(
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.ADMISSION_REJECTED,
            checks=checks,
            environment=environment,
            failure_code=TrainerFailureCode.ADMISSION_REJECTED,
            detail=detail,
        )

    def fit_predict(self, request: TrialRequest) -> TrialResult:
        validate_request_for_trainer(request, self.identity)
        preflight = self.preflight(request)
        if preflight.status is not QualificationStatus.PREFLIGHT_PASSED:
            raise TrainerError(
                preflight.failure_code or TrainerFailureCode.ADMISSION_REJECTED,
                preflight.detail or "official FM trial did not pass preflight",
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                attempt_id=request.attempt_id,
            )
        payload = request.payload
        assert isinstance(payload, OfficialFMTrialPayload)

        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        try:
            run = self._adapter.fit(
                encoding=payload.encoding,
                train_inputs=payload.train_inputs,
                train_targets=payload.train_targets,
                validation_inputs=payload.validation_inputs,
                validation_scorer=payload.validation_scorer,
            )
            if payload.prediction_inputs.digest == payload.validation_inputs.digest:
                predictions = run.validation_predictions
            else:
                predictions = self._adapter.predict(
                    checkpoint=run.checkpoint,
                    encoding=payload.encoding,
                    inputs=payload.prediction_inputs,
                )
        except MemoryError as exc:
            raise self._error(
                request,
                TrainerFailureCode.OOM,
                "official FM exhausted memory",
            ) from exc
        except StarterFMError as exc:
            detail = str(exc)
            lowered = detail.casefold()
            code = (
                TrainerFailureCode.NUMERICAL_ERROR
                if any(
                    word in lowered
                    for word in ("non-finite", "invalid loss", "invalid predictions")
                )
                else TrainerFailureCode.ADMISSION_REJECTED
            )
            raise self._error(request, code, f"official FM failed: {detail}") from exc
        except Exception as exc:  # pragma: no cover - defensive adapter boundary.
            raise self._error(
                request,
                TrainerFailureCode.INTERNAL_ERROR,
                f"unexpected official FM failure: {type(exc).__name__}",
            ) from exc
        ended = time.perf_counter_ns()
        cpu_seconds = (time.process_time_ns() - cpu_started) / 1_000_000_000
        wall_seconds = (ended - started) / 1_000_000_000

        prediction_id = PredictionId.from_trial(
            ordered_row_ids=request.ordered_row_ids,
            prediction_sha256=predictions.digest,
            trial_id=request.trial_id,
        )
        encoding_digest = str(payload.encoding.digest)
        training_feature_digest = _digest_manifest(
            b"kuairand-official-fm-training-features-v1\0",
            {"encoding": encoding_digest, "inputs": payload.train_inputs.digest},
        )
        prediction_feature_digest = _digest_manifest(
            b"kuairand-official-fm-prediction-features-v1\0",
            {"encoding": encoding_digest, "inputs": payload.prediction_inputs.digest},
        )
        training_data_digest = _digest_manifest(
            b"kuairand-official-fm-training-data-v1\0",
            {
                "inputs": payload.train_inputs.digest,
                "targets": run.training_targets_digest,
            },
        )
        return TrialResult(
            trial_id=request.trial_id,
            attempt_id=request.attempt_id,
            prediction_id=prediction_id,
            trainer_identity=self.identity,
            ordered_row_ids=request.ordered_row_ids,
            predictions=predictions.scores,
            prediction_sha256=predictions.digest,
            data=DataReceipt(
                training_data_sha256=training_data_digest,
                prediction_data_sha256=payload.prediction_inputs.digest,
                training_rows=len(payload.train_inputs),
                prediction_rows=len(payload.prediction_inputs),
            ),
            features=FeatureReceipt(
                training_feature_sha256=training_feature_digest,
                prediction_feature_sha256=prediction_feature_digest,
                feature_schema_sha256=_digest_manifest(
                    b"kuairand-official-fm-feature-schema-v1\0",
                    list(STARTER_FIELDS),
                ),
                feature_count=len(STARTER_FIELDS),
            ),
            model=ModelReceipt(
                model_sha256=run.checkpoint.digest,
                config_sha256=run.config_digest,
                backend=self.identity.backend,
                selected_iteration=run.checkpoint.best_epoch,
            ),
            environment=EnvironmentReceipt.capture(self.identity),
            resources=ResourceReceipt(
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_rss_bytes=run.resources.max_observed_rss_bytes,
                peak_disk_bytes=0,
                peak_process_count=1,
                threads=request.resource_limits.threads,
                device=self.identity.device,
            ),
            timing=TimingReceipt(
                started_monotonic_ns=started,
                ended_monotonic_ns=ended,
                wall_seconds=wall_seconds,
            ),
            diagnostics={
                "checkpoint_sha256": run.checkpoint.digest,
                "logical_run_sha256": run.logical_digest,
                "epochs_completed": run.checkpoint.epochs_completed,
                "optimizer_steps": run.checkpoint.optimizer_steps,
                "prediction_role": "inner_or_qualification_only",
            },
        )

    def _error(
        self,
        request: TrialRequest,
        code: TrainerFailureCode,
        detail: str,
    ) -> TrainerError:
        return TrainerError(
            code,
            detail,
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            attempt_id=request.attempt_id,
        )


__all__ = [
    "OFFICIAL_FM_TRAINER_VERSION",
    "OfficialFMTrainer",
    "OfficialFMTrialPayload",
]

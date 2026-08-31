"""Deterministic in-memory trainer for integration and fault-injection tests."""

from __future__ import annotations

import hashlib
import resource
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from kuairand_agent.domain.identity import PredictionId
from kuairand_agent.scoring.submission import prediction_digest
from kuairand_agent.training.protocol import (
    DataReceipt,
    EnvironmentReceipt,
    FeatureReceipt,
    JsonValue,
    ModelReceipt,
    QualificationReceipt,
    QualificationStatus,
    ResourceReceipt,
    TimingReceipt,
    TrainerContractError,
    TrainerError,
    TrainerFailureCode,
    TrainerIdentity,
    TrialRequest,
    TrialResult,
    validate_request_for_trainer,
)

SCRIPTED_TRAINER_VERSION = "v1"


def _observed_peak_rss_bytes() -> int:
    """Return the process high-water mark using this platform's documented unit."""

    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith("linux"):
        observed *= 1024
    return max(observed, 1)


@dataclass(frozen=True, slots=True)
class ScriptedTrialPayload:
    """Exact scripted evidence; no metric or hidden label crosses this seam."""

    predictions: tuple[float, ...]
    training_data_sha256: str
    prediction_data_sha256: str
    training_feature_sha256: str
    prediction_feature_sha256: str
    feature_schema_sha256: str
    config_sha256: str
    model_sha256: str
    training_rows: int
    feature_count: int
    selected_iteration: int = 1
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)


class ScriptedTrainer:
    """A contract-real trainer whose outputs and failures are fully predeclared."""

    def __init__(
        self,
        *,
        dependency_lock_sha256: str,
        preflight_failure: TrainerFailureCode | None = None,
        fit_failure: TrainerFailureCode | None = None,
    ) -> None:
        self._identity = TrainerIdentity(
            trainer_id="scripted",
            trainer_version=SCRIPTED_TRAINER_VERSION,
            backend="scripted-cpu",
            device="cpu",
            precision="float64",
            dependency_lock_sha256=dependency_lock_sha256,
        )
        self._preflight_failure = preflight_failure
        self._fit_failure = fit_failure

    @property
    def identity(self) -> TrainerIdentity:
        return self._identity

    def preflight(self, request: TrialRequest) -> QualificationReceipt:
        validate_request_for_trainer(request, self.identity)
        environment = EnvironmentReceipt.capture(self.identity)
        if self._preflight_failure is not None:
            if self._preflight_failure is TrainerFailureCode.UNSUPPORTED:
                status = QualificationStatus.UNSUPPORTED
            elif self._preflight_failure is TrainerFailureCode.ADMISSION_REJECTED:
                status = QualificationStatus.ADMISSION_REJECTED
            else:
                raise TrainerError(
                    self._preflight_failure,
                    "scripted preflight failure",
                    trainer_identity=self.identity,
                    trial_id=request.trial_id,
                    attempt_id=request.attempt_id,
                )
            return QualificationReceipt(
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                status=status,
                checks=("canonical-request", "scripted-capability"),
                environment=environment,
                failure_code=self._preflight_failure,
                detail="scripted preflight failure",
            )
        if not isinstance(request.payload, ScriptedTrialPayload):
            return QualificationReceipt(
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                status=QualificationStatus.ADMISSION_REJECTED,
                checks=("canonical-request", "scripted-payload"),
                environment=environment,
                failure_code=TrainerFailureCode.ADMISSION_REJECTED,
                detail="scripted trainer requires ScriptedTrialPayload",
            )
        if len(request.payload.predictions) != len(request.ordered_row_ids):
            return QualificationReceipt(
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                status=QualificationStatus.ADMISSION_REJECTED,
                checks=("canonical-request", "prediction-row-count"),
                environment=environment,
                failure_code=TrainerFailureCode.ADMISSION_REJECTED,
                detail="scripted prediction count differs from ordered row identities",
            )
        return QualificationReceipt(
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.PREFLIGHT_PASSED,
            checks=("canonical-request", "scripted-payload", "prediction-row-count"),
            environment=environment,
        )

    def fit_predict(self, request: TrialRequest) -> TrialResult:
        validate_request_for_trainer(request, self.identity)
        preflight = self.preflight(request)
        if preflight.status is not QualificationStatus.PREFLIGHT_PASSED:
            raise TrainerError(
                preflight.failure_code or TrainerFailureCode.ADMISSION_REJECTED,
                preflight.detail or "scripted trial did not pass preflight",
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                attempt_id=request.attempt_id,
            )
        if self._fit_failure is not None:
            raise TrainerError(
                self._fit_failure,
                "scripted fit failure",
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                attempt_id=request.attempt_id,
            )
        payload = request.payload
        if not isinstance(payload, ScriptedTrialPayload):  # pragma: no cover - preflight narrows.
            raise TrainerContractError("scripted payload changed after preflight")

        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        prediction_sha256 = prediction_digest(payload.predictions)
        prediction_id = PredictionId.from_trial(
            ordered_row_ids=request.ordered_row_ids,
            prediction_sha256=prediction_sha256,
            trial_id=request.trial_id,
        )
        ended = time.perf_counter_ns()
        wall_seconds = (ended - started) / 1_000_000_000
        cpu_seconds = (time.process_time_ns() - cpu_started) / 1_000_000_000
        diagnostics = dict(payload.diagnostics)
        diagnostics.setdefault("scripted", True)
        diagnostics.setdefault(
            "script_sha256",
            hashlib.sha256(b"kuairand-scripted-trainer-v1").hexdigest(),
        )
        diagnostics.setdefault(
            "resource_measurements",
            {
                "cpu_seconds": "measured_process_clock",
                "peak_disk_bytes": "known_zero_no_workspace_io",
                "peak_process_count": "known_single_in_process_trainer",
                "peak_rss_bytes": "measured_process_high_water_mark",
                "threads": "declared_limit_not_observed",
                "wall_seconds": "measured_monotonic_clock",
            },
        )
        return TrialResult(
            trial_id=request.trial_id,
            attempt_id=request.attempt_id,
            prediction_id=prediction_id,
            trainer_identity=self.identity,
            ordered_row_ids=request.ordered_row_ids,
            predictions=np.asarray(payload.predictions, dtype=np.float64),
            prediction_sha256=prediction_sha256,
            data=DataReceipt(
                training_data_sha256=payload.training_data_sha256,
                prediction_data_sha256=payload.prediction_data_sha256,
                training_rows=payload.training_rows,
                prediction_rows=len(payload.predictions),
            ),
            features=FeatureReceipt(
                training_feature_sha256=payload.training_feature_sha256,
                prediction_feature_sha256=payload.prediction_feature_sha256,
                feature_schema_sha256=payload.feature_schema_sha256,
                feature_count=payload.feature_count,
            ),
            model=ModelReceipt(
                model_sha256=payload.model_sha256,
                config_sha256=payload.config_sha256,
                backend=self.identity.backend,
                selected_iteration=payload.selected_iteration,
            ),
            environment=EnvironmentReceipt.capture(self.identity),
            resources=ResourceReceipt(
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_rss_bytes=_observed_peak_rss_bytes(),
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
            diagnostics=diagnostics,
        )


__all__ = ["SCRIPTED_TRAINER_VERSION", "ScriptedTrainer", "ScriptedTrialPayload"]

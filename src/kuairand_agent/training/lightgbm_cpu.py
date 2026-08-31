"""Qualified deterministic CPU adapter over the existing LambdaRank primitive."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from kuairand_agent.candidates.grouping import UserGrouping
from kuairand_agent.candidates.tree_ranker import (
    PINNED_LIGHTGBM_VERSION,
    InnerValidationSet,
    LambdaRankBackend,
    LambdaRankConfig,
    TreeRankerDependencyError,
    TreeRankerError,
    fit_lambdarank,
    predict_lambdarank,
)
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.data.causal_features import FeatureMatrix
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

LIGHTGBM_CPU_TRAINER_VERSION = "v1"


def _digest_manifest(domain: bytes, value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + payload).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith("linux"):
        value *= 1024
    return max(value, 1)


@dataclass(frozen=True, slots=True)
class LightGBMCPUTrialPayload:
    """Trusted tree capabilities in canonical row order."""

    training_features: FeatureMatrix
    training_labels: Sequence[object] | npt.NDArray[np.generic] = field(repr=False)
    training_grouping: UserGrouping
    training_phase: DataPhase
    prediction_features: FeatureMatrix
    prediction_phase: DataPhase
    config: LambdaRankConfig
    inner_valid: InnerValidationSet | None = None
    backend: LambdaRankBackend | None = field(default=None, repr=False, compare=False)


class LightGBMCPUTrainer:
    """Preserve the CPU primitive's parameters, grouping, serialization, and replay checks."""

    def __init__(self, *, dependency_lock_sha256: str) -> None:
        self._identity = TrainerIdentity(
            trainer_id="lightgbm-lambdarank",
            trainer_version=LIGHTGBM_CPU_TRAINER_VERSION,
            backend="lightgbm-cpu",
            device="cpu",
            precision="float64",
            dependency_lock_sha256=dependency_lock_sha256,
        )

    @property
    def identity(self) -> TrainerIdentity:
        return self._identity

    def preflight(self, request: TrialRequest) -> QualificationReceipt:
        validate_request_for_trainer(request, self.identity)
        environment = EnvironmentReceipt.capture(self.identity)
        payload = request.payload
        if not isinstance(payload, LightGBMCPUTrialPayload):
            return self._receipt(
                request,
                environment,
                QualificationStatus.ADMISSION_REJECTED,
                TrainerFailureCode.ADMISSION_REJECTED,
                "LightGBM CPU requires LightGBMCPUTrialPayload",
                ("canonical-request", "cpu-payload"),
            )
        checks = ["canonical-request", "cpu-payload"]
        if payload.config.seed != request.seed:
            return self._admission(
                request,
                environment,
                "tree config seed differs from trial seed",
                checks,
            )
        if payload.config.num_threads != request.resource_limits.threads:
            return self._admission(
                request,
                environment,
                "tree config threads differ from admitted resource threads",
                checks,
            )
        expected_config = request.qualified_settings.get("config_sha256")
        if expected_config != payload.config.digest:
            return self._admission(
                request,
                environment,
                "qualified_settings.config_sha256 differs from tree config",
                checks,
            )
        checks.extend(("seed-and-thread-identity", "qualified-settings"))
        if payload.training_grouping.phase is not payload.training_phase:
            return self._admission(
                request,
                environment,
                "tree grouping phase differs from training phase",
                checks,
            )
        if payload.training_grouping.row_count != payload.training_features.row_count:
            return self._admission(
                request,
                environment,
                "tree grouping row count differs from training features",
                checks,
            )
        if payload.prediction_features.row_count != len(request.ordered_row_ids):
            return self._admission(
                request,
                environment,
                "tree prediction count differs from ordered row identities",
                checks,
            )
        if payload.prediction_features.feature_names != payload.training_features.feature_names:
            return self._admission(
                request,
                environment,
                "tree train/prediction feature schemas differ",
                checks,
            )
        checks.extend(("feature-and-grouping-capabilities", "prediction-row-count"))

        if payload.backend is None:
            try:
                installed = importlib.metadata.version("lightgbm")
            except importlib.metadata.PackageNotFoundError:
                return self._receipt(
                    request,
                    environment,
                    QualificationStatus.FAILED,
                    TrainerFailureCode.DEPENDENCY_ERROR,
                    "pinned LightGBM dependency is not installed",
                    tuple([*checks, "pinned-dependency"]),
                )
            if installed != PINNED_LIGHTGBM_VERSION:
                return self._receipt(
                    request,
                    environment,
                    QualificationStatus.FAILED,
                    TrainerFailureCode.DEPENDENCY_ERROR,
                    f"LightGBM {PINNED_LIGHTGBM_VERSION} is required, found {installed}",
                    tuple([*checks, "pinned-dependency"]),
                )
            checks.append("pinned-dependency")
        else:
            backend_identity = getattr(payload.backend, "identity", None)
            if type(backend_identity) is not str or not backend_identity:
                return self._admission(
                    request,
                    environment,
                    "injected LambdaRank backend has no stable identity",
                    checks,
                )
            checks.append("injected-backend-identity")

        return QualificationReceipt(
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.PREFLIGHT_PASSED,
            checks=tuple(checks),
            environment=environment,
        )

    def _admission(
        self,
        request: TrialRequest,
        environment: EnvironmentReceipt,
        detail: str,
        checks: list[str],
    ) -> QualificationReceipt:
        return self._receipt(
            request,
            environment,
            QualificationStatus.ADMISSION_REJECTED,
            TrainerFailureCode.ADMISSION_REJECTED,
            detail,
            tuple(checks),
        )

    def _receipt(
        self,
        request: TrialRequest,
        environment: EnvironmentReceipt,
        status: QualificationStatus,
        failure_code: TrainerFailureCode,
        detail: str,
        checks: tuple[str, ...],
    ) -> QualificationReceipt:
        return QualificationReceipt(
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            status=status,
            checks=checks,
            environment=environment,
            failure_code=failure_code,
            detail=detail,
        )

    def fit_predict(self, request: TrialRequest) -> TrialResult:
        validate_request_for_trainer(request, self.identity)
        preflight = self.preflight(request)
        if preflight.status is not QualificationStatus.PREFLIGHT_PASSED:
            raise TrainerError(
                preflight.failure_code or TrainerFailureCode.ADMISSION_REJECTED,
                preflight.detail or "LightGBM CPU trial did not pass preflight",
                trainer_identity=self.identity,
                trial_id=request.trial_id,
                attempt_id=request.attempt_id,
            )
        payload = request.payload
        assert isinstance(payload, LightGBMCPUTrialPayload)
        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        try:
            checkpoint = fit_lambdarank(
                payload.training_features,
                payload.training_labels,
                grouping=payload.training_grouping,
                phase=payload.training_phase,
                config=payload.config,
                inner_valid=payload.inner_valid,
                backend=payload.backend,
            )
            prediction = predict_lambdarank(
                checkpoint,
                payload.prediction_features,
                phase=payload.prediction_phase,
                backend=payload.backend,
            )
        except MemoryError as exc:
            raise self._error(
                request,
                TrainerFailureCode.OOM,
                "LightGBM CPU exhausted memory",
            ) from exc
        except TreeRankerDependencyError as exc:
            raise self._error(
                request,
                TrainerFailureCode.DEPENDENCY_ERROR,
                f"LightGBM CPU dependency failed: {exc}",
            ) from exc
        except TreeRankerError as exc:
            detail = str(exc)
            code = (
                TrainerFailureCode.NUMERICAL_ERROR
                if any(word in detail.casefold() for word in ("finite", "numeric", "prediction"))
                else TrainerFailureCode.ADMISSION_REJECTED
            )
            raise self._error(request, code, f"LightGBM CPU failed: {detail}") from exc
        except Exception as exc:  # pragma: no cover - backend-specific defensive boundary.
            raise self._error(
                request,
                TrainerFailureCode.INTERNAL_ERROR,
                f"unexpected LightGBM CPU failure: {type(exc).__name__}",
            ) from exc
        ended = time.perf_counter_ns()
        wall_seconds = (ended - started) / 1_000_000_000
        cpu_seconds = (time.process_time_ns() - cpu_started) / 1_000_000_000

        prediction_id = PredictionId.from_trial(
            ordered_row_ids=request.ordered_row_ids,
            prediction_sha256=prediction.prediction_digest,
            trial_id=request.trial_id,
        )
        training_data_sha256 = _digest_manifest(
            b"kuairand-tree-training-data-v1\0",
            {
                "grouping": checkpoint.training_grouping_digest,
                "targets": checkpoint.training_target_digest,
                "phase": checkpoint.training_phase.value,
            },
        )
        feature_schema_sha256 = _digest_manifest(
            b"kuairand-tree-feature-schema-v1\0",
            list(checkpoint.feature_names),
        )
        return TrialResult(
            trial_id=request.trial_id,
            attempt_id=request.attempt_id,
            prediction_id=prediction_id,
            trainer_identity=self.identity,
            ordered_row_ids=request.ordered_row_ids,
            predictions=prediction.scores,
            prediction_sha256=prediction.prediction_digest,
            data=DataReceipt(
                training_data_sha256=training_data_sha256,
                prediction_data_sha256=payload.prediction_features.digest,
                training_rows=payload.training_features.row_count,
                prediction_rows=payload.prediction_features.row_count,
            ),
            features=FeatureReceipt(
                training_feature_sha256=payload.training_features.digest,
                prediction_feature_sha256=payload.prediction_features.digest,
                feature_schema_sha256=feature_schema_sha256,
                feature_count=len(checkpoint.feature_names),
            ),
            model=ModelReceipt(
                model_sha256=checkpoint.model_sha256,
                config_sha256=checkpoint.config_digest,
                backend=self.identity.backend,
                selected_iteration=checkpoint.best_iteration,
            ),
            environment=EnvironmentReceipt.capture(self.identity),
            resources=ResourceReceipt(
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_rss_bytes=_peak_rss_bytes(),
                peak_disk_bytes=0,
                peak_process_count=1,
                threads=payload.config.num_threads,
                device=self.identity.device,
            ),
            timing=TimingReceipt(
                started_monotonic_ns=started,
                ended_monotonic_ns=ended,
                wall_seconds=wall_seconds,
            ),
            diagnostics={
                "checkpoint_sha256": checkpoint.digest,
                "backend_identity": checkpoint.backend_identity,
                "training_grouping_sha256": checkpoint.training_grouping_digest,
                "training_target_sha256": checkpoint.training_target_digest,
                "inner_validation_sha256": checkpoint.inner_validation_digest,
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
    "LIGHTGBM_CPU_TRAINER_VERSION",
    "LightGBMCPUTrainer",
    "LightGBMCPUTrialPayload",
]

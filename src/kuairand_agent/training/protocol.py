"""Typed trainer boundary and content-bound training receipts.

The trainer seam owns model execution, not scientific selection.  A request binds an already
canonical experiment to one exact backend/configuration and one monotonically numbered
infrastructure attempt.  A result binds finite float64 predictions in canonical row order to the
same identities and keeps observed machine resources outside model identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from kuairand_agent.domain.identity import (
    AttemptId,
    ExperimentId,
    PredictionId,
    TrialId,
)
from kuairand_agent.scoring.submission import prediction_digest

TRAINER_PROTOCOL_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_.-]{0,79}\Z")

type Float64Vector = npt.NDArray[np.float64]
type RowId = int | str
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class TrainerContractError(ValueError):
    """A trainer request, identity, or receipt violates the public trainer contract."""


class TrainerFailureCode(StrEnum):
    """Complete and closed trainer failure taxonomy from the laboratory contract."""

    UNSUPPORTED = "UNSUPPORTED"
    ADMISSION_REJECTED = "ADMISSION_REJECTED"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    CANCELLED = "CANCELLED"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class QualificationStatus(StrEnum):
    """A preflight or completed same-backend qualification outcome."""

    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    QUALIFIED = "QUALIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    ADMISSION_REJECTED = "ADMISSION_REJECTED"
    FAILED = "FAILED"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainerContractError("trainer evidence must be finite canonical JSON") from exc


def _manifest_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise TrainerContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise TrainerContractError(
            f"{name} must be a lowercase portable identifier of at most 80 characters"
        )
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value:
        raise TrainerContractError(f"{name} must be a non-empty single-line string")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerContractError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise TrainerContractError(f"{name} must be a finite non-negative number")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainerContractError(f"{name} must be a positive integer")
    return value


def _json_value(value: object, name: str) -> JsonValue:
    try:
        payload = _canonical_json(value)
        decoded = json.loads(payload)
    except TrainerContractError:
        raise
    if not isinstance(decoded, (dict, list, str, int, float, bool)) and decoded is not None:
        raise TrainerContractError(f"{name} must be canonical JSON")
    return cast(JsonValue, decoded)


def _frozen_mapping(value: Mapping[str, object], name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TrainerContractError(f"{name} must be a string-keyed mapping")
    normalized = _json_value(dict(value), name)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above.
        raise TrainerContractError(f"{name} must be an object")
    return MappingProxyType(dict(sorted(normalized.items())))


def _row_ids(value: Sequence[RowId]) -> tuple[RowId, ...]:
    if isinstance(value, (str, bytes)):
        raise TrainerContractError("ordered_row_ids must be a sequence of row identities")
    result: list[RowId] = []
    for index, item in enumerate(value):
        if (type(item) is int and item >= 0) or (type(item) is str and item and "\x00" not in item):
            result.append(item)
        else:
            raise TrainerContractError(
                f"ordered_row_ids[{index}] must be a non-negative integer or non-empty string"
            )
    if not result:
        raise TrainerContractError("ordered_row_ids cannot be empty")
    if len(set(result)) != len(result):
        raise TrainerContractError("ordered_row_ids must be unique")
    return tuple(result)


def _predictions(value: object, *, expected_rows: int) -> Float64Vector:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainerContractError("predictions must be a finite one-dimensional vector") from exc
    if raw.ndim != 1 or raw.size != expected_rows or raw.dtype.kind not in "iuf":
        raise TrainerContractError(
            f"predictions must be a numeric vector with exactly {expected_rows} rows"
        )
    try:
        contiguous = np.ascontiguousarray(raw, dtype=np.dtype("<f8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainerContractError("predictions must be representable as float64") from exc
    if not np.isfinite(contiguous).all():
        raise TrainerContractError("predictions must contain only finite values")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8"))
    frozen.setflags(write=False)
    return cast(Float64Vector, frozen)


@dataclass(frozen=True, slots=True)
class TrainerIdentity:
    """Stable implementation/backend identity used when deriving a :class:`TrialId`."""

    trainer_id: str
    trainer_version: str
    backend: str
    device: str
    precision: str
    dependency_lock_sha256: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_id", _identifier(self.trainer_id, "trainer_id"))
        object.__setattr__(
            self,
            "trainer_version",
            _identifier(self.trainer_version, "trainer_version"),
        )
        object.__setattr__(self, "backend", _identifier(self.backend, "backend"))
        object.__setattr__(self, "device", _identifier(self.device, "device"))
        object.__setattr__(self, "precision", _identifier(self.precision, "precision"))
        _digest(self.dependency_lock_sha256, "dependency_lock_sha256")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-trainer-identity-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": TRAINER_PROTOCOL_SCHEMA_VERSION,
            "trainer_id": self.trainer_id,
            "trainer_version": self.trainer_version,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Admission limits.  These are policy inputs, not observed resource evidence."""

    timeout_seconds: float
    memory_limit_bytes: int
    disk_limit_bytes: int
    threads: int

    def __post_init__(self) -> None:
        timeout = _nonnegative_float(self.timeout_seconds, "timeout_seconds")
        if timeout <= 0.0:
            raise TrainerContractError("timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", timeout)
        for name in ("memory_limit_bytes", "disk_limit_bytes", "threads"):
            _positive_int(getattr(self, name), name)

    def manifest(self) -> dict[str, object]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_bytes": self.memory_limit_bytes,
            "disk_limit_bytes": self.disk_limit_bytes,
            "threads": self.threads,
        }


@dataclass(frozen=True, slots=True)
class TrialRequest:
    """One canonical scientific trial and one infrastructure attempt."""

    experiment_id: ExperimentId
    trial_id: TrialId
    attempt_id: AttemptId
    trainer_identity: TrainerIdentity
    seed: int
    fold: str
    fidelity: JsonValue
    qualified_settings: Mapping[str, JsonValue]
    infrastructure_attempt: int
    ordered_row_ids: tuple[RowId, ...]
    resource_limits: ResourceLimits
    payload: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, ExperimentId):
            raise TrainerContractError("experiment_id must be ExperimentId")
        if not isinstance(self.trial_id, TrialId):
            raise TrainerContractError("trial_id must be TrialId")
        if not isinstance(self.attempt_id, AttemptId):
            raise TrainerContractError("attempt_id must be AttemptId")
        if not isinstance(self.trainer_identity, TrainerIdentity):
            raise TrainerContractError("trainer_identity must be TrainerIdentity")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise TrainerContractError("seed must be a uint32-compatible integer")
        _text(self.fold, "fold")
        fidelity = _json_value(self.fidelity, "fidelity")
        settings = _frozen_mapping(self.qualified_settings, "qualified_settings")
        attempt = _positive_int(self.infrastructure_attempt, "infrastructure_attempt")
        row_ids = _row_ids(self.ordered_row_ids)
        if not isinstance(self.resource_limits, ResourceLimits):
            raise TrainerContractError("resource_limits must be ResourceLimits")
        expected_trial = TrialId.derive(
            experiment_id=self.experiment_id,
            trainer_id=self.trainer_identity.trainer_id,
            trainer_version=self.trainer_identity.trainer_version,
            backend=self.trainer_identity.backend,
            precision=self.trainer_identity.precision,
            dependency_lock_sha256=self.trainer_identity.dependency_lock_sha256,
            seed=self.seed,
            fold=self.fold,
            fidelity=fidelity,
            qualified_settings=settings,
        )
        if self.trial_id != expected_trial:
            raise TrainerContractError("trial_id does not match the canonical trainer request")
        expected_attempt = AttemptId.derive(
            trial_id=self.trial_id,
            infrastructure_attempt=attempt,
        )
        if self.attempt_id != expected_attempt:
            raise TrainerContractError("attempt_id does not match trial_id/infrastructure_attempt")
        object.__setattr__(self, "fidelity", fidelity)
        object.__setattr__(self, "qualified_settings", settings)
        object.__setattr__(self, "ordered_row_ids", row_ids)


@dataclass(frozen=True, slots=True)
class DataReceipt:
    training_data_sha256: str
    prediction_data_sha256: str
    training_rows: int
    prediction_rows: int

    def __post_init__(self) -> None:
        _digest(self.training_data_sha256, "training_data_sha256")
        _digest(self.prediction_data_sha256, "prediction_data_sha256")
        _positive_int(self.training_rows, "training_rows")
        _positive_int(self.prediction_rows, "prediction_rows")

    def manifest(self) -> dict[str, object]:
        return {
            "training_data_sha256": self.training_data_sha256,
            "prediction_data_sha256": self.prediction_data_sha256,
            "training_rows": self.training_rows,
            "prediction_rows": self.prediction_rows,
        }


@dataclass(frozen=True, slots=True)
class FeatureReceipt:
    training_feature_sha256: str
    prediction_feature_sha256: str
    feature_schema_sha256: str
    feature_count: int

    def __post_init__(self) -> None:
        _digest(self.training_feature_sha256, "training_feature_sha256")
        _digest(self.prediction_feature_sha256, "prediction_feature_sha256")
        _digest(self.feature_schema_sha256, "feature_schema_sha256")
        _positive_int(self.feature_count, "feature_count")

    def manifest(self) -> dict[str, object]:
        return {
            "training_feature_sha256": self.training_feature_sha256,
            "prediction_feature_sha256": self.prediction_feature_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "feature_count": self.feature_count,
        }


@dataclass(frozen=True, slots=True)
class ModelReceipt:
    model_sha256: str
    config_sha256: str
    backend: str
    selected_iteration: int

    def __post_init__(self) -> None:
        _digest(self.model_sha256, "model_sha256")
        _digest(self.config_sha256, "config_sha256")
        _identifier(self.backend, "backend")
        _positive_int(self.selected_iteration, "selected_iteration")

    def manifest(self) -> dict[str, object]:
        return {
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
            "backend": self.backend,
            "selected_iteration": self.selected_iteration,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentReceipt:
    trainer_identity_sha256: str
    dependency_lock_sha256: str
    backend: str
    device: str
    precision: str
    python_version: str
    platform_system: str
    platform_machine: str

    def __post_init__(self) -> None:
        _digest(self.trainer_identity_sha256, "trainer_identity_sha256")
        _digest(self.dependency_lock_sha256, "dependency_lock_sha256")
        for name in (
            "backend",
            "device",
            "precision",
            "python_version",
            "platform_system",
            "platform_machine",
        ):
            _text(getattr(self, name), name)

    @classmethod
    def capture(cls, identity: TrainerIdentity) -> EnvironmentReceipt:
        if not isinstance(identity, TrainerIdentity):
            raise TrainerContractError("identity must be TrainerIdentity")
        return cls(
            trainer_identity_sha256=identity.digest,
            dependency_lock_sha256=identity.dependency_lock_sha256,
            backend=identity.backend,
            device=identity.device,
            precision=identity.precision,
            python_version=platform.python_version(),
            platform_system=platform.system() or sys.platform,
            platform_machine=platform.machine() or "unknown",
        )

    def manifest(self) -> dict[str, object]:
        return {
            "trainer_identity_sha256": self.trainer_identity_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
        }

    @property
    def digest(self) -> str:
        return _manifest_digest(b"kuairand-trainer-environment-v1\0", self.manifest())


@dataclass(frozen=True, slots=True)
class ResourceReceipt:
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    peak_disk_bytes: int
    peak_process_count: int
    threads: int
    device: str
    cpu_seconds_measured: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wall_seconds",
            _nonnegative_float(self.wall_seconds, "wall_seconds"),
        )
        object.__setattr__(self, "cpu_seconds", _nonnegative_float(self.cpu_seconds, "cpu_seconds"))
        for name in ("peak_rss_bytes", "peak_disk_bytes", "peak_process_count", "threads"):
            value = getattr(self, name)
            minimum = 1 if name in {"peak_process_count", "threads"} else 0
            if type(value) is not int or value < minimum:
                raise TrainerContractError(f"{name} is outside the supported non-negative range")
        _identifier(self.device, "device")
        if type(self.cpu_seconds_measured) is not bool:
            raise TrainerContractError("cpu_seconds_measured must be bool")

    def manifest(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_disk_bytes": self.peak_disk_bytes,
            "peak_process_count": self.peak_process_count,
            "threads": self.threads,
            "device": self.device,
            "cpu_seconds_measured": self.cpu_seconds_measured,
        }


@dataclass(frozen=True, slots=True)
class TimingReceipt:
    started_monotonic_ns: int
    ended_monotonic_ns: int
    wall_seconds: float

    def __post_init__(self) -> None:
        if type(self.started_monotonic_ns) is not int or self.started_monotonic_ns < 0:
            raise TrainerContractError("started_monotonic_ns must be a non-negative integer")
        if (
            type(self.ended_monotonic_ns) is not int
            or self.ended_monotonic_ns < self.started_monotonic_ns
        ):
            raise TrainerContractError("ended_monotonic_ns must not precede the start")
        observed = _nonnegative_float(self.wall_seconds, "wall_seconds")
        elapsed = (self.ended_monotonic_ns - self.started_monotonic_ns) / 1_000_000_000
        if not math.isclose(observed, elapsed, rel_tol=0.0, abs_tol=1e-6):
            raise TrainerContractError("timing wall_seconds differs from monotonic timestamps")
        object.__setattr__(self, "wall_seconds", observed)

    def manifest(self) -> dict[str, object]:
        return {
            "started_monotonic_ns": self.started_monotonic_ns,
            "ended_monotonic_ns": self.ended_monotonic_ns,
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Exact prediction vector plus all receipts required at the trainer boundary."""

    trial_id: TrialId
    attempt_id: AttemptId
    prediction_id: PredictionId
    trainer_identity: TrainerIdentity
    ordered_row_ids: tuple[RowId, ...]
    predictions: Float64Vector = field(repr=False)
    prediction_sha256: str
    data: DataReceipt
    features: FeatureReceipt
    model: ModelReceipt
    environment: EnvironmentReceipt
    resources: ResourceReceipt
    timing: TimingReceipt
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, TrialId):
            raise TrainerContractError("result trial_id must be TrialId")
        if not isinstance(self.attempt_id, AttemptId):
            raise TrainerContractError("result attempt_id must be AttemptId")
        if not isinstance(self.prediction_id, PredictionId):
            raise TrainerContractError("result prediction_id must be PredictionId")
        if not isinstance(self.trainer_identity, TrainerIdentity):
            raise TrainerContractError("result trainer_identity must be TrainerIdentity")
        row_ids = _row_ids(self.ordered_row_ids)
        predictions = _predictions(self.predictions, expected_rows=len(row_ids))
        observed_digest = prediction_digest(predictions)
        if observed_digest != _digest(self.prediction_sha256, "prediction_sha256"):
            raise TrainerContractError("prediction_sha256 differs from exact float64 predictions")
        expected_prediction_id = PredictionId.from_trial(
            ordered_row_ids=row_ids,
            prediction_sha256=observed_digest,
            trial_id=self.trial_id,
        )
        if self.prediction_id != expected_prediction_id:
            raise TrainerContractError("prediction_id differs from rows/predictions/trial identity")
        for name, expected_type in (
            ("data", DataReceipt),
            ("features", FeatureReceipt),
            ("model", ModelReceipt),
            ("environment", EnvironmentReceipt),
            ("resources", ResourceReceipt),
            ("timing", TimingReceipt),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TrainerContractError(f"result {name} receipt has the wrong type")
        if self.data.prediction_rows != len(row_ids):
            raise TrainerContractError("data receipt prediction_rows differs from predictions")
        if self.environment.trainer_identity_sha256 != self.trainer_identity.digest:
            raise TrainerContractError("environment receipt differs from trainer identity")
        if self.environment.backend != self.model.backend:
            raise TrainerContractError("environment and model backend identities differ")
        if self.resources.device != self.trainer_identity.device:
            raise TrainerContractError("resource and trainer device identities differ")
        object.__setattr__(self, "ordered_row_ids", row_ids)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics, "diagnostics"))

    @property
    def digest(self) -> str:
        return _manifest_digest(b"kuairand-trial-result-v1\0", self.manifest())

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": TRAINER_PROTOCOL_SCHEMA_VERSION,
            "trial_id": str(self.trial_id),
            "attempt_id": str(self.attempt_id),
            "prediction_id": str(self.prediction_id),
            "trainer_identity": self.trainer_identity.manifest(),
            "ordered_row_ids": list(self.ordered_row_ids),
            "prediction_sha256": self.prediction_sha256,
            "data": self.data.manifest(),
            "features": self.features.manifest(),
            "model": self.model.manifest(),
            "environment": self.environment.manifest(),
            "resources": self.resources.manifest(),
            "timing": self.timing.manifest(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class QualificationReceipt:
    """Preflight or deterministic same-backend replay evidence."""

    trainer_identity: TrainerIdentity
    trial_id: TrialId
    status: QualificationStatus
    checks: tuple[str, ...]
    environment: EnvironmentReceipt
    failure_code: TrainerFailureCode | None = None
    detail: str | None = None
    same_backend_replay_verified: bool = False
    first_prediction_sha256: str | None = None
    replay_prediction_sha256: str | None = None
    result_receipt_sha256: str | None = None
    replay_result_receipt_sha256: str | None = None
    resource_receipts: tuple[ResourceReceipt, ...] = ()
    p50_wall_seconds: float | None = None
    p95_wall_seconds: float | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.trainer_identity, TrainerIdentity):
            raise TrainerContractError("qualification trainer_identity must be TrainerIdentity")
        if not isinstance(self.trial_id, TrialId):
            raise TrainerContractError("qualification trial_id must be TrialId")
        if not isinstance(self.status, QualificationStatus):
            raise TrainerContractError("qualification status must be QualificationStatus")
        if not self.checks or any(
            type(check) is not str or not check or "\x00" in check for check in self.checks
        ):
            raise TrainerContractError("qualification checks must be non-empty strings")
        if not isinstance(self.environment, EnvironmentReceipt):
            raise TrainerContractError("qualification environment must be EnvironmentReceipt")
        if self.environment.trainer_identity_sha256 != self.trainer_identity.digest:
            raise TrainerContractError("qualification environment differs from trainer identity")
        if self.failure_code is not None and not isinstance(self.failure_code, TrainerFailureCode):
            raise TrainerContractError("qualification failure_code must be TrainerFailureCode")
        if self.detail is not None:
            _text(self.detail, "qualification detail")
        digest_names = (
            "first_prediction_sha256",
            "replay_prediction_sha256",
            "result_receipt_sha256",
            "replay_result_receipt_sha256",
        )
        for name in digest_names:
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if self.status in {QualificationStatus.UNSUPPORTED, QualificationStatus.ADMISSION_REJECTED}:
            expected = (
                TrainerFailureCode.UNSUPPORTED
                if self.status is QualificationStatus.UNSUPPORTED
                else TrainerFailureCode.ADMISSION_REJECTED
            )
            if self.failure_code is not expected or self.same_backend_replay_verified:
                raise TrainerContractError("terminal qualification status/failure evidence differs")
        elif self.failure_code is not None:
            if self.status is not QualificationStatus.FAILED:
                raise TrainerContractError("successful qualification cannot include a failure code")
        elif self.status is QualificationStatus.FAILED:
            raise TrainerContractError("FAILED qualification requires a typed failure code")
        if any(not isinstance(item, ResourceReceipt) for item in self.resource_receipts):
            raise TrainerContractError("resource_receipts must contain ResourceReceipt values")
        if self.p50_wall_seconds is not None:
            object.__setattr__(
                self,
                "p50_wall_seconds",
                _nonnegative_float(self.p50_wall_seconds, "p50_wall_seconds"),
            )
        if self.p95_wall_seconds is not None:
            object.__setattr__(
                self,
                "p95_wall_seconds",
                _nonnegative_float(self.p95_wall_seconds, "p95_wall_seconds"),
            )
        if self.status is QualificationStatus.QUALIFIED:
            if not self.same_backend_replay_verified or any(
                getattr(self, name) is None for name in digest_names
            ):
                raise TrainerContractError("qualified receipt requires complete replay evidence")
            if self.first_prediction_sha256 != self.replay_prediction_sha256:
                raise TrainerContractError("qualified replay predictions are not exact")
            if len(self.resource_receipts) < 2:
                raise TrainerContractError("qualified receipt requires both run resource receipts")
            ordered = sorted(item.wall_seconds for item in self.resource_receipts)
            expected_p50 = float(statistics.median(ordered))
            expected_p95 = ordered[max(1, math.ceil(0.95 * len(ordered))) - 1]
            if self.p50_wall_seconds != expected_p50 or self.p95_wall_seconds != expected_p95:
                raise TrainerContractError("qualification p50/p95 differ from resource receipts")
        elif self.same_backend_replay_verified:
            raise TrainerContractError("only QUALIFIED may claim same-backend replay")
        elif (
            self.resource_receipts
            or self.p50_wall_seconds is not None
            or self.p95_wall_seconds is not None
        ):
            raise TrainerContractError("non-qualified receipt cannot claim qualification resources")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-trainer-qualification-v1\0", self.manifest()),
        )

    @property
    def campaign_admissible(self) -> bool:
        return self.status is QualificationStatus.QUALIFIED and self.same_backend_replay_verified

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": TRAINER_PROTOCOL_SCHEMA_VERSION,
            "trainer_identity": self.trainer_identity.manifest(),
            "trial_id": str(self.trial_id),
            "status": self.status.value,
            "checks": list(self.checks),
            "environment": self.environment.manifest(),
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "detail": self.detail,
            "same_backend_replay_verified": self.same_backend_replay_verified,
            "first_prediction_sha256": self.first_prediction_sha256,
            "replay_prediction_sha256": self.replay_prediction_sha256,
            "result_receipt_sha256": self.result_receipt_sha256,
            "replay_result_receipt_sha256": self.replay_result_receipt_sha256,
            "resource_receipts": [item.manifest() for item in self.resource_receipts],
            "p50_wall_seconds": self.p50_wall_seconds,
            "p95_wall_seconds": self.p95_wall_seconds,
        }


class TrainerError(RuntimeError):
    """Typed trainer failure that always identifies the exact trial/attempt when admitted."""

    def __init__(
        self,
        code: TrainerFailureCode,
        detail: str,
        *,
        trainer_identity: TrainerIdentity,
        trial_id: TrialId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> None:
        if not isinstance(code, TrainerFailureCode):
            raise TrainerContractError("trainer error code must be TrainerFailureCode")
        _text(detail, "trainer error detail")
        if not isinstance(trainer_identity, TrainerIdentity):
            raise TrainerContractError("trainer error identity must be TrainerIdentity")
        if trial_id is not None and not isinstance(trial_id, TrialId):
            raise TrainerContractError("trainer error trial_id must be TrialId or None")
        if attempt_id is not None and not isinstance(attempt_id, AttemptId):
            raise TrainerContractError("trainer error attempt_id must be AttemptId or None")
        if attempt_id is not None and trial_id is None:
            raise TrainerContractError("attempt-scoped trainer error requires a trial_id")
        self.code = code
        self.detail = detail
        self.trainer_identity = trainer_identity
        self.trial_id = trial_id
        self.attempt_id = attempt_id
        super().__init__(f"{code.value}: {detail}")


@runtime_checkable
class QualifiedTrainer(Protocol):
    """Principal model/backend seam used by the scientific controller."""

    @property
    def identity(self) -> TrainerIdentity: ...

    def preflight(self, request: TrialRequest) -> QualificationReceipt: ...

    def fit_predict(self, request: TrialRequest) -> TrialResult: ...


def validate_request_for_trainer(request: TrialRequest, identity: TrainerIdentity) -> None:
    """Fail before model work when a request is assigned to a different trainer/backend."""

    if not isinstance(request, TrialRequest):
        raise TrainerContractError("request must be TrialRequest")
    if not isinstance(identity, TrainerIdentity):
        raise TrainerContractError("identity must be TrainerIdentity")
    if request.trainer_identity != identity:
        raise TrainerError(
            TrainerFailureCode.ADMISSION_REJECTED,
            "trial trainer identity differs from the selected adapter",
            trainer_identity=identity,
            trial_id=request.trial_id,
            attempt_id=request.attempt_id,
        )


__all__ = [
    "TRAINER_PROTOCOL_SCHEMA_VERSION",
    "DataReceipt",
    "EnvironmentReceipt",
    "FeatureReceipt",
    "Float64Vector",
    "ModelReceipt",
    "QualificationReceipt",
    "QualificationStatus",
    "QualifiedTrainer",
    "ResourceLimits",
    "ResourceReceipt",
    "RowId",
    "TimingReceipt",
    "TrainerContractError",
    "TrainerError",
    "TrainerFailureCode",
    "TrainerIdentity",
    "TrialRequest",
    "TrialResult",
    "validate_request_for_trainer",
]

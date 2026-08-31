"""Bounded, leakage-safe neural interaction primitives.

PyTorch is an optional research dependency.  Importing this module never imports
PyTorch; each executable neural operation resolves it lazily and fails with a
specific installation error when the ``research-neural`` dependency group is
absent.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Final, Protocol, cast, no_type_check, runtime_checkable

import numpy as np
import numpy.typing as npt

from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.submission import prediction_digest

NEURAL_SCHEMA_VERSION: Final = 1
MAX_CATEGORICAL_FIELDS: Final = 32
MAX_CATEGORY_CARDINALITY: Final = 10_000_000
MAX_EMBEDDING_DIMENSION: Final = 64
MAX_DENSE_FEATURES: Final = 512
MAX_HIDDEN_LAYERS: Final = 4
MAX_HIDDEN_WIDTH: Final = 2_048
MAX_CROSS_LAYERS: Final = 4
MAX_TRAINING_ROWS: Final = 5_000_000
MAX_PAIR_INDICES: Final = 1_000_000

type Int64Matrix = npt.NDArray[np.int64]
type Float32Matrix = npt.NDArray[np.float32]
type Int8Vector = npt.NDArray[np.int8]
type Int64Vector = npt.NDArray[np.int64]
type Float64Vector = npt.NDArray[np.float64]


@runtime_checkable
class _InnerAggregate(Protocol):
    """Inner-fold aggregate shape, independent of protected-evaluation result types."""

    gauc: float
    ndcg_at_5: float
    primary: float
    rows: int


type InnerValidScorer = Callable[[Float64Vector], _InnerAggregate]


class NeuralPrimitiveError(ValueError):
    """Raised when a neural primitive violates its scientific contract."""


class TorchUnavailableError(RuntimeError):
    """Raised when an executable neural operation lacks the optional backend."""


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise TorchUnavailableError(
            "PyTorch is optional; install the locked research-neural dependency group"
        ) from exc


class NeuralArchitecture(StrEnum):
    """The bounded WP7 interaction mechanisms and their required control."""

    CONTROL = "control"
    DEEPFM = "deepfm"
    DCNV2 = "dcnv2"


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise NeuralPrimitiveError(f"{name} must be an integer in [{minimum}, {maximum}]")
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise NeuralPrimitiveError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return normalized


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NeuralPrimitiveError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise NeuralPrimitiveError(f"{name} must be a finite real number")
    return normalized


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class NeuralConfigManifest:
    """Immutable, value-free identity of one compact neural architecture."""

    architecture: NeuralArchitecture
    categorical_cardinalities: tuple[int, ...]
    dense_feature_count: int
    embedding_dim: int
    hidden_dims: tuple[int, ...]
    cross_layers: int
    dropout: float
    schema_version: int = NEURAL_SCHEMA_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(_canonical_json(self.as_dict())).hexdigest()
        object.__setattr__(self, "digest", digest)

    def as_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe representation; row values are never included."""

        return {
            "schema_version": self.schema_version,
            "architecture": self.architecture.value,
            "categorical_cardinalities": list(self.categorical_cardinalities),
            "dense_feature_count": self.dense_feature_count,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": list(self.hidden_dims),
            "cross_layers": self.cross_layers,
            "dropout": self.dropout,
        }


@dataclass(frozen=True, slots=True)
class NeuralModelConfig:
    """Strictly bounded same-feature configuration for a WP7 neural primitive."""

    architecture: NeuralArchitecture
    categorical_cardinalities: tuple[int, ...]
    dense_feature_count: int
    embedding_dim: int = 8
    hidden_dims: tuple[int, ...] = (32, 16)
    cross_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        try:
            architecture = NeuralArchitecture(self.architecture)
        except (TypeError, ValueError) as exc:
            raise NeuralPrimitiveError("architecture must be control, deepfm, or dcnv2") from exc
        object.__setattr__(self, "architecture", architecture)

        try:
            raw_cardinalities = tuple(self.categorical_cardinalities)
        except TypeError as exc:
            raise NeuralPrimitiveError(
                "categorical_cardinalities must be a bounded sequence"
            ) from exc
        if not 2 <= len(raw_cardinalities) <= MAX_CATEGORICAL_FIELDS:
            raise NeuralPrimitiveError(
                f"categorical_cardinalities must contain 2 to {MAX_CATEGORICAL_FIELDS} fields"
            )
        cardinalities = tuple(
            _bounded_int(
                value,
                name=f"categorical_cardinalities[{index}]",
                minimum=2,
                maximum=MAX_CATEGORY_CARDINALITY,
            )
            for index, value in enumerate(raw_cardinalities)
        )
        object.__setattr__(self, "categorical_cardinalities", cardinalities)
        object.__setattr__(
            self,
            "dense_feature_count",
            _bounded_int(
                self.dense_feature_count,
                name="dense_feature_count",
                minimum=0,
                maximum=MAX_DENSE_FEATURES,
            ),
        )
        object.__setattr__(
            self,
            "embedding_dim",
            _bounded_int(
                self.embedding_dim,
                name="embedding_dim",
                minimum=1,
                maximum=MAX_EMBEDDING_DIMENSION,
            ),
        )
        try:
            raw_hidden = tuple(self.hidden_dims)
        except TypeError as exc:
            raise NeuralPrimitiveError("hidden_dims must be a bounded sequence") from exc
        if not 1 <= len(raw_hidden) <= MAX_HIDDEN_LAYERS:
            raise NeuralPrimitiveError(f"hidden_dims must contain 1 to {MAX_HIDDEN_LAYERS} layers")
        hidden = tuple(
            _bounded_int(
                value,
                name=f"hidden_dims[{index}]",
                minimum=1,
                maximum=MAX_HIDDEN_WIDTH,
            )
            for index, value in enumerate(raw_hidden)
        )
        object.__setattr__(self, "hidden_dims", hidden)
        object.__setattr__(
            self,
            "cross_layers",
            _bounded_int(
                self.cross_layers,
                name="cross_layers",
                minimum=1,
                maximum=MAX_CROSS_LAYERS,
            ),
        )
        dropout = _finite_real(self.dropout, name="dropout")
        if not 0.0 <= dropout < 1.0:
            raise NeuralPrimitiveError("dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", dropout)

    def manifest(self) -> NeuralConfigManifest:
        """Return the immutable manifest bound into checkpoints and result evidence."""

        return NeuralConfigManifest(
            architecture=self.architecture,
            categorical_cardinalities=self.categorical_cardinalities,
            dense_feature_count=self.dense_feature_count,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            cross_layers=self.cross_layers,
            dropout=self.dropout,
        )

    @property
    def digest(self) -> str:
        return self.manifest().digest


@dataclass(frozen=True, slots=True)
class HybridLossConfig:
    """Frozen calibration for binary pointwise plus same-user pairwise loss."""

    pointwise_weight: float = 0.5
    pairwise_weight: float = 0.5

    def __post_init__(self) -> None:
        pointwise = _finite_real(self.pointwise_weight, name="pointwise_weight")
        pairwise = _finite_real(self.pairwise_weight, name="pairwise_weight")
        if pointwise < 0.0 or pairwise < 0.0:
            raise NeuralPrimitiveError("hybrid loss weights must be non-negative")
        if not math.isclose(pointwise + pairwise, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise NeuralPrimitiveError("hybrid loss weights must sum to exactly one")
        object.__setattr__(self, "pointwise_weight", pointwise)
        object.__setattr__(self, "pairwise_weight", pairwise)

    def as_dict(self) -> dict[str, float]:
        return {
            "pointwise_weight": self.pointwise_weight,
            "pairwise_weight": self.pairwise_weight,
        }


def _digest_arrays(prefix: bytes, metadata: object, arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(prefix)
    digest.update(_canonical_json(metadata))
    for array in arrays:
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class NeuralFeatureBatch:
    """Immutable label-free categorical and dense features in canonical row order."""

    phase: DataPhase
    categorical: Int64Matrix = field(repr=False)
    dense: Float32Matrix = field(repr=False)
    digest: str

    def __init__(self, *, phase: DataPhase, categorical: object, dense: object) -> None:
        if not isinstance(phase, DataPhase):
            raise NeuralPrimitiveError("feature phase must be a DataPhase")
        try:
            raw_categorical = np.asarray(categorical)
            raw_dense = np.asarray(dense)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NeuralPrimitiveError("neural features must be numeric matrices") from exc
        if (
            raw_categorical.ndim != 2
            or raw_categorical.shape[0] == 0
            or raw_categorical.shape[0] > MAX_TRAINING_ROWS
            or raw_categorical.shape[1] == 0
            or raw_categorical.dtype.kind not in "iu"
        ):
            raise NeuralPrimitiveError(
                "categorical features need a bounded nonempty integer matrix"
            )
        if (
            raw_categorical.dtype.kind == "u"
            and raw_categorical.size
            and int(raw_categorical.max()) > np.iinfo(np.int64).max
        ):
            raise NeuralPrimitiveError("categorical feature indices must fit int64")
        categorical_array = np.array(raw_categorical, dtype=np.int64, order="C", copy=True)
        if np.any(categorical_array < 0):
            raise NeuralPrimitiveError("categorical feature indices must be non-negative")
        if (
            raw_dense.ndim != 2
            or raw_dense.shape[0] != raw_categorical.shape[0]
            or raw_dense.dtype.kind not in "iuf"
        ):
            raise NeuralPrimitiveError("dense features must be a row-aligned numeric matrix")
        dense_array = np.array(raw_dense, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(dense_array).all():
            raise NeuralPrimitiveError("dense features must contain only finite values")
        categorical_array.setflags(write=False)
        dense_array.setflags(write=False)
        digest = _digest_arrays(
            b"neural-features-v1\0",
            {"phase": phase.value},
            (categorical_array, dense_array),
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "categorical", categorical_array)
        object.__setattr__(self, "dense", dense_array)
        object.__setattr__(self, "digest", digest)

    @property
    def row_count(self) -> int:
        return int(self.categorical.shape[0])


@dataclass(frozen=True, slots=True, init=False)
class NeuralTrainingTargets:
    """Immutable binary targets constructible only for train-derived phases."""

    phase: DataPhase
    values: Int8Vector = field(repr=False)
    digest: str

    def __init__(self, *, phase: DataPhase, values: object) -> None:
        if not isinstance(phase, DataPhase) or phase not in {
            DataPhase.TRAIN,
            DataPhase.INNER_TRAIN,
        }:
            raise NeuralPrimitiveError("neural targets are allowed only for train or inner_train")
        try:
            raw = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NeuralPrimitiveError("neural targets must be a binary vector") from exc
        if raw.ndim != 1 or raw.size == 0 or raw.size > MAX_TRAINING_ROWS:
            raise NeuralPrimitiveError("neural targets must be one bounded nonempty vector")
        if raw.dtype.kind not in "biuf":
            raise NeuralPrimitiveError("neural targets must be a binary numeric vector")
        normalized = np.ascontiguousarray(raw, dtype=np.float64)
        if not np.isfinite(normalized).all() or not np.isin(normalized, (0.0, 1.0)).all():
            raise NeuralPrimitiveError("neural targets must contain only binary 0 and 1")
        target_array = np.array(normalized, dtype=np.int8, order="C", copy=True)
        target_array.setflags(write=False)
        digest = _digest_arrays(
            b"neural-targets-v1\0",
            {"phase": phase.value},
            (target_array,),
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "values", target_array)
        object.__setattr__(self, "digest", digest)


def _index_vector(values: object, *, name: str) -> Int64Vector:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NeuralPrimitiveError(f"{name} must be a bounded integer vector") from exc
    if raw.ndim != 1 or not 1 <= raw.size <= MAX_PAIR_INDICES or raw.dtype.kind not in "iu":
        raise NeuralPrimitiveError(f"{name} must be a bounded nonempty integer vector")
    if raw.dtype.kind == "u" and int(raw.max()) > np.iinfo(np.int64).max:
        raise NeuralPrimitiveError(f"{name} must fit int64")
    result = np.array(raw, dtype=np.int64, order="C", copy=True)
    if np.any(result < 0):
        raise NeuralPrimitiveError(f"{name} cannot contain negative indices")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class NeuralPairIndices:
    """Immutable sampler output; target semantics are verified only inside fit."""

    positive_indices: Int64Vector = field(repr=False)
    negative_indices: Int64Vector = field(repr=False)
    digest: str

    def __init__(self, positive_indices: object, negative_indices: object) -> None:
        positive = _index_vector(positive_indices, name="positive_indices")
        negative = _index_vector(negative_indices, name="negative_indices")
        if positive.shape != negative.shape:
            raise NeuralPrimitiveError("positive and negative pair indices must have equal lengths")
        digest = _digest_arrays(b"neural-pairs-v1\0", {}, (positive, negative))
        object.__setattr__(self, "positive_indices", positive)
        object.__setattr__(self, "negative_indices", negative)
        object.__setattr__(self, "digest", digest)

    @property
    def pair_count(self) -> int:
        return int(self.positive_indices.size)


@dataclass(frozen=True, slots=True)
class NeuralFitConfig:
    """Small deterministic optimizer configuration for a bounded evidence run."""

    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    loss: HybridLossConfig = HybridLossConfig()
    requested_device: str = "cpu"
    num_threads: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "seed", _bounded_int(self.seed, name="seed", minimum=0, maximum=2**32 - 1)
        )
        object.__setattr__(
            self, "epochs", _bounded_int(self.epochs, name="epochs", minimum=1, maximum=10)
        )
        object.__setattr__(
            self,
            "batch_size",
            _bounded_int(self.batch_size, name="batch_size", minimum=1, maximum=65_536),
        )
        learning_rate = _finite_real(self.learning_rate, name="learning_rate")
        if not 0.0 < learning_rate <= 1.0:
            raise NeuralPrimitiveError("learning_rate must be in (0, 1]")
        object.__setattr__(self, "learning_rate", learning_rate)
        if not isinstance(self.loss, HybridLossConfig):
            raise NeuralPrimitiveError("loss must be HybridLossConfig")
        if self.requested_device not in {"cpu", "mps"}:
            raise NeuralPrimitiveError("requested_device must be exactly 'cpu' or 'mps'")
        object.__setattr__(
            self,
            "num_threads",
            _bounded_int(self.num_threads, name="num_threads", minimum=1, maximum=16),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "loss": self.loss.as_dict(),
            "requested_device": self.requested_device,
            "num_threads": self.num_threads,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class NeuralCeilings:
    """Predeclared family limits; passing does not itself authorize promotion."""

    max_parameters: int
    max_p95_epoch_seconds: float
    min_examples_per_second: float
    max_checkpoint_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_parameters",
            _bounded_int(
                self.max_parameters,
                name="max_parameters",
                minimum=1,
                maximum=1_000_000_000,
            ),
        )
        p95 = _finite_real(self.max_p95_epoch_seconds, name="max_p95_epoch_seconds")
        throughput = _finite_real(self.min_examples_per_second, name="min_examples_per_second")
        if p95 <= 0.0 or throughput <= 0.0:
            raise NeuralPrimitiveError("runtime ceilings and throughput floors must be positive")
        object.__setattr__(self, "max_p95_epoch_seconds", p95)
        object.__setattr__(self, "min_examples_per_second", throughput)
        object.__setattr__(
            self,
            "max_checkpoint_bytes",
            _bounded_int(
                self.max_checkpoint_bytes,
                name="max_checkpoint_bytes",
                minimum=1,
                maximum=100_000_000_000,
            ),
        )


@dataclass(frozen=True, slots=True)
class NeuralPerformanceEvidence:
    """Immutable measured observations used by the family eligibility gate."""

    parameter_count: int
    epoch_seconds: tuple[float, ...]
    examples_per_second: tuple[float, ...]
    checkpoint_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_count",
            _bounded_int(
                self.parameter_count,
                name="parameter_count",
                minimum=1,
                maximum=1_000_000_000,
            ),
        )
        try:
            epoch_values = tuple(
                _finite_real(value, name=f"epoch_seconds[{index}]")
                for index, value in enumerate(self.epoch_seconds)
            )
            throughput_values = tuple(
                _finite_real(value, name=f"examples_per_second[{index}]")
                for index, value in enumerate(self.examples_per_second)
            )
        except TypeError as exc:
            raise NeuralPrimitiveError("performance observations must be finite sequences") from exc
        if not epoch_values or len(epoch_values) != len(throughput_values):
            raise NeuralPrimitiveError("epoch time and throughput need one equal nonempty shape")
        if any(value <= 0.0 for value in (*epoch_values, *throughput_values)):
            raise NeuralPrimitiveError("epoch time and throughput observations must be positive")
        object.__setattr__(self, "epoch_seconds", epoch_values)
        object.__setattr__(self, "examples_per_second", throughput_values)
        object.__setattr__(
            self,
            "checkpoint_bytes",
            _bounded_int(
                self.checkpoint_bytes,
                name="checkpoint_bytes",
                minimum=1,
                maximum=100_000_000_000,
            ),
        )

    @property
    def p95_epoch_seconds(self) -> float:
        """Conservative deterministic nearest-rank p95 (no interpolation drift)."""

        ordered = sorted(self.epoch_seconds)
        rank = max(1, math.ceil(0.95 * len(ordered)))
        return ordered[rank - 1]

    @property
    def minimum_examples_per_second(self) -> float:
        return min(self.examples_per_second)


@dataclass(frozen=True, slots=True)
class NeuralEligibility:
    """Result of applying one frozen ceiling set to measured observations."""

    passed: bool
    reasons: tuple[str, ...]
    evidence: NeuralPerformanceEvidence
    ceilings: NeuralCeilings


def assess_neural_eligibility(
    evidence: NeuralPerformanceEvidence,
    ceilings: NeuralCeilings,
) -> NeuralEligibility:
    """Apply all family ceilings without accessing labels or promotion metrics."""

    if not isinstance(evidence, NeuralPerformanceEvidence):
        raise NeuralPrimitiveError("evidence must be NeuralPerformanceEvidence")
    if not isinstance(ceilings, NeuralCeilings):
        raise NeuralPrimitiveError("ceilings must be NeuralCeilings")
    reasons: list[str] = []
    if evidence.parameter_count > ceilings.max_parameters:
        reasons.append("parameter_count_exceeds_ceiling")
    if evidence.p95_epoch_seconds > ceilings.max_p95_epoch_seconds:
        reasons.append("p95_epoch_seconds_exceeds_ceiling")
    if evidence.minimum_examples_per_second < ceilings.min_examples_per_second:
        reasons.append("throughput_below_floor")
    if evidence.checkpoint_bytes > ceilings.max_checkpoint_bytes:
        reasons.append("checkpoint_bytes_exceeds_ceiling")
    return NeuralEligibility(
        passed=not reasons,
        reasons=tuple(reasons),
        evidence=evidence,
        ceilings=ceilings,
    )


@dataclass(frozen=True, slots=True)
class NeuralResultManifest:
    """Immutable evidence from one bounded fit; never contains row-level outcomes."""

    model_config_digest: str
    fit_config_digest: str
    training_feature_digest: str
    training_target_digest: str
    pair_digest: str
    state_digest: str
    device: DeviceSelection
    eligibility: NeuralEligibility
    inner_valid_feature_digest: str | None
    inner_valid_gauc: float | None
    inner_valid_ndcg_at_5: float | None
    inner_valid_primary: float | None
    inner_valid_scorer_digest: str | None
    inner_valid_prediction_digest: str | None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "schema_version": NEURAL_SCHEMA_VERSION,
            "model_config_digest": self.model_config_digest,
            "fit_config_digest": self.fit_config_digest,
            "training_feature_digest": self.training_feature_digest,
            "training_target_digest": self.training_target_digest,
            "pair_digest": self.pair_digest,
            "state_digest": self.state_digest,
            "device": {
                "requested": self.device.requested_device,
                "actual": self.device.actual_device,
                "seed": self.device.seed,
                "torch_version": self.device.torch_version,
                "deterministic_algorithms": self.device.deterministic_algorithms,
                "num_threads": self.device.num_threads,
            },
            "performance": {
                "parameter_count": self.eligibility.evidence.parameter_count,
                "epoch_seconds": list(self.eligibility.evidence.epoch_seconds),
                "examples_per_second": list(self.eligibility.evidence.examples_per_second),
                "checkpoint_bytes": self.eligibility.evidence.checkpoint_bytes,
            },
            "eligibility": {
                "passed": self.eligibility.passed,
                "reasons": list(self.eligibility.reasons),
                "ceilings": {
                    "max_parameters": self.eligibility.ceilings.max_parameters,
                    "max_p95_epoch_seconds": (self.eligibility.ceilings.max_p95_epoch_seconds),
                    "min_examples_per_second": (self.eligibility.ceilings.min_examples_per_second),
                    "max_checkpoint_bytes": (self.eligibility.ceilings.max_checkpoint_bytes),
                },
            },
            "inner_valid": {
                "feature_digest": self.inner_valid_feature_digest,
                "gauc": self.inner_valid_gauc,
                "ndcg_at_5": self.inner_valid_ndcg_at_5,
                "primary": self.inner_valid_primary,
                "scorer_digest": self.inner_valid_scorer_digest,
                "prediction_digest": self.inner_valid_prediction_digest,
            },
        }
        object.__setattr__(self, "digest", hashlib.sha256(_canonical_json(payload)).hexdigest())


@dataclass(frozen=True, slots=True)
class NeuralCheckpoint:
    """Immutable serialized state plus semantic provenance for exact replay."""

    model_config: NeuralModelConfig
    fit_config: NeuralFitConfig
    state_bytes: bytes = field(repr=False)
    state_digest: str
    result: NeuralResultManifest
    digest: str


@dataclass(frozen=True, slots=True)
class NeuralPrediction:
    """One label-free canonical-order prediction and exact checkpoint identity."""

    scores: Float64Vector = field(repr=False)
    phase: DataPhase
    prediction_digest: str
    checkpoint_digest: str
    feature_digest: str
    device: DeviceSelection


@dataclass(frozen=True, slots=True)
class DeviceParityResult:
    """Explicit cross-device numerical comparison, never an identity claim."""

    tolerance: float
    max_absolute_difference: float
    within_tolerance: bool
    row_count: int


def assess_device_parity(
    cpu_predictions: object,
    mps_predictions: object,
    *,
    tolerance: float,
) -> DeviceParityResult:
    """Compare aligned finite vectors within one declared absolute tolerance."""

    normalized_tolerance = _finite_real(tolerance, name="tolerance")
    if normalized_tolerance < 0.0:
        raise NeuralPrimitiveError("tolerance must be non-negative")
    try:
        cpu = np.asarray(cpu_predictions, dtype=np.float64)
        mps = np.asarray(mps_predictions, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NeuralPrimitiveError("device predictions must be finite numeric vectors") from exc
    if cpu.ndim != 1 or mps.ndim != 1 or cpu.shape != mps.shape or cpu.size == 0:
        raise NeuralPrimitiveError("device predictions must have one equal nonempty shape")
    if not np.isfinite(cpu).all() or not np.isfinite(mps).all():
        raise NeuralPrimitiveError("device predictions must contain finite values")
    max_difference = float(np.max(np.abs(cpu - mps)))
    return DeviceParityResult(
        tolerance=normalized_tolerance,
        max_absolute_difference=max_difference,
        within_tolerance=max_difference <= normalized_tolerance,
        row_count=int(cpu.size),
    )


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """Exact deterministic runtime selection recorded for every neural result."""

    requested_device: str
    actual_device: str
    seed: int
    torch_version: str
    mps_available: bool
    deterministic_algorithms: bool
    num_threads: int


@dataclass(frozen=True, slots=True)
class BuiltNeuralModel:
    """A compact module plus immutable construction provenance and size facts."""

    module: Any = field(repr=False, compare=False)
    config: NeuralConfigManifest
    device: DeviceSelection
    parameter_count: int
    backbone_parameter_count: int


def configure_neural_runtime(
    *,
    seed: int,
    requested_device: str = "cpu",
    num_threads: int = 1,
) -> DeviceSelection:
    """Select only the mandatory CPU path or an explicitly requested available MPS path."""

    normalized_seed = _bounded_int(seed, name="seed", minimum=0, maximum=2**32 - 1)
    normalized_threads = _bounded_int(num_threads, name="num_threads", minimum=1, maximum=16)
    if requested_device not in {"cpu", "mps"}:
        raise NeuralPrimitiveError("requested_device must be exactly 'cpu' or 'mps'")
    torch = _torch()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if requested_device == "mps" and not mps_available:
        raise NeuralPrimitiveError("MPS was requested but is not available in this PyTorch runtime")

    torch.manual_seed(normalized_seed)
    if mps_available and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(normalized_seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(normalized_threads)
    return DeviceSelection(
        requested_device=requested_device,
        actual_device=requested_device,
        seed=normalized_seed,
        torch_version=str(torch.__version__),
        mps_available=mps_available,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        num_threads=normalized_threads,
    )


@no_type_check
def _make_model(config: NeuralModelConfig, *, device: str) -> Any:
    torch = _torch()
    nn = torch.nn

    class _FeatureBackbone(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                nn.Embedding(cardinality, config.embedding_dim)
                for cardinality in config.categorical_cardinalities
            )
            self.first_order_embeddings = nn.ModuleList(
                nn.Embedding(cardinality, 1) for cardinality in config.categorical_cardinalities
            )
            self.bias = nn.Parameter(torch.zeros(1))
            self.dense_first_order = (
                nn.Linear(config.dense_feature_count, 1, bias=False)
                if config.dense_feature_count > 0
                else None
            )
            input_width = len(config.categorical_cardinalities) * config.embedding_dim
            input_width += config.dense_feature_count
            layers: list[Any] = []
            previous = input_width
            for width in config.hidden_dims:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
                previous = width
            self.deep_stack = nn.Sequential(*layers)
            self.deep_head = nn.Linear(config.hidden_dims[-1], 1)
            self.input_width = input_width

        def parts(self, categorical: object, dense: object | None) -> tuple[Any, Any, Any, Any]:
            categorical_tensor, dense_tensor = _validated_model_inputs(
                categorical,
                dense,
                config=config,
                model_device=self.bias.device,
            )
            field_embeddings = torch.stack(
                [
                    embedding(categorical_tensor[:, index])
                    for index, embedding in enumerate(self.embeddings)
                ],
                dim=1,
            )
            first_order = self.bias.expand(categorical_tensor.shape[0])
            for index, embedding in enumerate(self.first_order_embeddings):
                first_order = first_order + embedding(categorical_tensor[:, index]).squeeze(1)
            flat_embeddings = field_embeddings.flatten(start_dim=1)
            if dense_tensor is not None:
                if self.dense_first_order is None:  # pragma: no cover - constructor invariant
                    raise RuntimeError("dense projection missing for configured dense inputs")
                first_order = first_order + self.dense_first_order(dense_tensor).squeeze(1)
                raw_features = torch.cat((flat_embeddings, dense_tensor), dim=1)
            else:
                raw_features = flat_embeddings
            deep_representation = self.deep_stack(raw_features)
            deep_logit = self.deep_head(deep_representation).squeeze(1)
            return first_order, field_embeddings, raw_features, deep_logit

    class _Control(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _FeatureBackbone()

        def forward(self, categorical: object, dense: object | None = None) -> Any:
            first, _embedded, _raw, deep = self.backbone.parts(categorical, dense)
            return first + deep

    class _DeepFM(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _FeatureBackbone()

        def forward(self, categorical: object, dense: object | None = None) -> Any:
            first, embedded, _raw, deep = self.backbone.parts(categorical, dense)
            return first + deepfm_interaction(embedded) + deep

    class _DCNv2(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _FeatureBackbone()
            self.cross_layers = nn.ModuleList(
                nn.Linear(self.backbone.input_width, self.backbone.input_width)
                for _ in range(config.cross_layers)
            )
            self.cross_head = nn.Linear(self.backbone.input_width, 1)

        def forward(self, categorical: object, dense: object | None = None) -> Any:
            first, _embedded, raw, deep = self.backbone.parts(categorical, dense)
            crossed = raw
            for layer in self.cross_layers:
                crossed = dcnv2_cross_step(raw, crossed, layer.weight, layer.bias)
            return first + deep + self.cross_head(crossed).squeeze(1)

    module_types = {
        NeuralArchitecture.CONTROL: _Control,
        NeuralArchitecture.DEEPFM: _DeepFM,
        NeuralArchitecture.DCNV2: _DCNv2,
    }
    return module_types[config.architecture]().to(torch.device(device))


def _validated_model_inputs(
    categorical: object,
    dense: object | None,
    *,
    config: NeuralModelConfig,
    model_device: object,
) -> tuple[Any, Any | None]:
    torch = _torch()
    if not isinstance(categorical, torch.Tensor):
        raise NeuralPrimitiveError("categorical inputs must be a PyTorch tensor")
    if categorical.ndim != 2 or categorical.shape[0] <= 0:
        raise NeuralPrimitiveError("categorical inputs must have nonempty [batch, fields] shape")
    if categorical.shape[1] != len(config.categorical_cardinalities):
        raise NeuralPrimitiveError("categorical input field count does not match the model config")
    if categorical.dtype != torch.int64:
        raise NeuralPrimitiveError("categorical inputs must use torch.int64 indices")
    if categorical.device != model_device:
        raise NeuralPrimitiveError("categorical inputs must be on the model device")
    for field_index, cardinality in enumerate(config.categorical_cardinalities):
        field_values = categorical[:, field_index]
        if bool(torch.any(field_values < 0).item()) or bool(
            torch.any(field_values >= cardinality).item()
        ):
            raise NeuralPrimitiveError(
                f"categorical field {field_index} contains an out-of-range index"
            )

    if config.dense_feature_count == 0:
        if dense is not None and (
            not isinstance(dense, torch.Tensor) or dense.shape != (categorical.shape[0], 0)
        ):
            raise NeuralPrimitiveError("dense inputs must be omitted or have [batch, 0] shape")
        return categorical, None
    if not isinstance(dense, torch.Tensor):
        raise NeuralPrimitiveError("dense inputs must be a PyTorch tensor")
    if dense.ndim != 2 or dense.shape != (categorical.shape[0], config.dense_feature_count):
        raise NeuralPrimitiveError("dense input shape does not match batch and model config")
    if not dense.is_floating_point() or not bool(torch.isfinite(dense).all().item()):
        raise NeuralPrimitiveError("dense inputs must contain finite floating-point values")
    if dense.device != model_device:
        raise NeuralPrimitiveError("dense inputs must be on the model device")
    return categorical, dense


def _trainable_parameter_count(module: Any) -> int:
    return sum(
        int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad
    )


def build_neural_model(
    config: NeuralModelConfig,
    *,
    seed: int,
    requested_device: str = "cpu",
    num_threads: int = 1,
) -> BuiltNeuralModel:
    """Build one seeded compact model without fitting or accessing any outcomes."""

    if not isinstance(config, NeuralModelConfig):
        raise NeuralPrimitiveError("config must be a NeuralModelConfig")
    selection = configure_neural_runtime(
        seed=seed,
        requested_device=requested_device,
        num_threads=num_threads,
    )
    module = _make_model(config, device=selection.actual_device)
    return BuiltNeuralModel(
        module=module,
        config=config.manifest(),
        device=selection,
        parameter_count=_trainable_parameter_count(module),
        backbone_parameter_count=_trainable_parameter_count(module.backbone),
    )


def deepfm_interaction(embeddings: object) -> Any:
    """Return the parameter-free DeepFM second-order interaction per row.

    ``embeddings`` must have shape ``[batch, categorical_fields, embedding_dim]``.
    The result is ``0.5 * sum((sum(v_i))^2 - sum(v_i^2))`` across latent
    dimensions, which is exactly the sum of all distinct field-pair dot products.
    """

    torch = _torch()
    if not isinstance(embeddings, torch.Tensor):
        raise NeuralPrimitiveError("embeddings must be a PyTorch tensor")
    if embeddings.ndim != 3 or embeddings.shape[0] <= 0:
        raise NeuralPrimitiveError(
            "embeddings must have shape [nonempty batch, categorical fields, embedding dim]"
        )
    if embeddings.shape[1] < 2 or embeddings.shape[2] <= 0:
        raise NeuralPrimitiveError(
            "DeepFM interaction requires at least two fields and a positive embedding dim"
        )
    if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all().item()):
        raise NeuralPrimitiveError("embeddings must contain finite floating-point values")
    summed = embeddings.sum(dim=1)
    interaction_by_dimension = summed.square() - embeddings.square().sum(dim=1)
    return 0.5 * interaction_by_dimension.sum(dim=1)


def dcnv2_cross_step(
    x0: object,
    current: object,
    weight: object,
    bias: object,
) -> Any:
    """Apply one full-matrix DCNv2 cross layer without hidden state.

    The public equation is ``x0 * linear(current, weight, bias) + current``.
    All tensors must be finite floating-point values on one device with a common
    dtype.  Keeping this primitive separate makes the cross-network math directly
    testable without inspecting a model's private modules.
    """

    torch = _torch()
    tensors = (x0, current, weight, bias)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise NeuralPrimitiveError("DCNv2 cross inputs must all be PyTorch tensors")
    typed_x0 = cast(Any, x0)
    typed_current = cast(Any, current)
    typed_weight = cast(Any, weight)
    typed_bias = cast(Any, bias)
    if typed_x0.ndim != 2 or typed_current.ndim != 2 or typed_x0.shape != typed_current.shape:
        raise NeuralPrimitiveError(
            "x0 and current must have the same nonempty [batch, width] shape"
        )
    if typed_x0.shape[0] <= 0 or typed_x0.shape[1] <= 0:
        raise NeuralPrimitiveError(
            "x0 and current must have the same nonempty [batch, width] shape"
        )
    width = typed_x0.shape[1]
    if typed_weight.ndim != 2 or typed_weight.shape != (width, width):
        raise NeuralPrimitiveError("weight must have shape [width, width]")
    if typed_bias.ndim != 1 or typed_bias.shape[0] != width:
        raise NeuralPrimitiveError("bias must have shape [width]")
    device = typed_x0.device
    dtype = typed_x0.dtype
    for value in tuple(cast(Any, tensor) for tensor in tensors):
        if not value.is_floating_point():
            raise NeuralPrimitiveError("DCNv2 cross tensors must be floating point")
        if value.device != device or value.dtype != dtype:
            raise NeuralPrimitiveError("DCNv2 cross tensors must share one dtype and device")
        if not bool(torch.isfinite(value).all().item()):
            raise NeuralPrimitiveError("DCNv2 cross tensors must contain only finite values")
    projected = torch.nn.functional.linear(typed_current, typed_weight, typed_bias)
    return typed_x0 * projected + typed_current


def hybrid_binary_pairwise_loss(
    pointwise_logits: object,
    targets: object,
    positive_logits: object | None,
    negative_logits: object | None,
    *,
    phase: DataPhase,
    mask: object | None = None,
    config: HybridLossConfig | None = None,
) -> Any:
    """Return a calibrated binary BCE and pairwise logistic loss.

    Phase authorization happens before target values are converted, indexed, or
    summarized.  A boolean mask may hide missing auxiliary targets; every active
    target and logit must remain finite and binary.  Pair logits come from a
    separately audited same-user sampler and are never formed by Cartesian
    enumeration here.
    """

    if not isinstance(phase, DataPhase) or phase not in {DataPhase.TRAIN, DataPhase.INNER_TRAIN}:
        raise NeuralPrimitiveError("neural losses are allowed only for train or inner_train")
    if config is None:
        config = HybridLossConfig()
    if not isinstance(config, HybridLossConfig):
        raise NeuralPrimitiveError("config must be a HybridLossConfig")
    torch = _torch()
    if not isinstance(pointwise_logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise NeuralPrimitiveError("pointwise_logits and targets must be PyTorch tensors")
    if (
        pointwise_logits.ndim != 1
        or targets.ndim != 1
        or pointwise_logits.shape != targets.shape
        or pointwise_logits.numel() == 0
    ):
        raise NeuralPrimitiveError(
            "pointwise_logits and targets must have one equal nonempty shape"
        )
    if not pointwise_logits.is_floating_point() or not targets.is_floating_point():
        raise NeuralPrimitiveError("pointwise_logits and targets must be floating point")
    if pointwise_logits.device != targets.device:
        raise NeuralPrimitiveError("pointwise_logits and targets must share one device")

    if mask is None:
        active_mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise NeuralPrimitiveError("mask must be a boolean PyTorch tensor")
        if mask.ndim != 1 or mask.shape != targets.shape or mask.device != targets.device:
            raise NeuralPrimitiveError("mask must match target shape and device")
        active_mask = mask
    if not bool(active_mask.any().item()):
        raise NeuralPrimitiveError("mask must retain at least one active target")
    active_logits = pointwise_logits[active_mask]
    active_targets = targets[active_mask]
    if not bool(torch.isfinite(active_logits).all().item()):
        raise NeuralPrimitiveError("active pointwise logits must be finite")
    if not bool(torch.isfinite(active_targets).all().item()) or not bool(
        torch.logical_or(active_targets == 0, active_targets == 1).all().item()
    ):
        raise NeuralPrimitiveError("active targets must contain finite binary values")
    pointwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        active_logits,
        active_targets.to(dtype=active_logits.dtype),
    )

    if config.pairwise_weight == 0.0:
        if positive_logits is not None or negative_logits is not None:
            raise NeuralPrimitiveError("pair logits must be omitted when pairwise_weight is zero")
        return config.pointwise_weight * pointwise_loss
    if not isinstance(positive_logits, torch.Tensor) or not isinstance(
        negative_logits, torch.Tensor
    ):
        raise NeuralPrimitiveError("positive_logits and negative_logits are required tensors")
    if (
        positive_logits.ndim != 1
        or negative_logits.ndim != 1
        or positive_logits.shape != negative_logits.shape
        or positive_logits.numel() == 0
    ):
        raise NeuralPrimitiveError(
            "positive_logits and negative_logits need one equal nonempty shape"
        )
    if (
        not positive_logits.is_floating_point()
        or not negative_logits.is_floating_point()
        or positive_logits.device != pointwise_logits.device
        or negative_logits.device != pointwise_logits.device
    ):
        raise NeuralPrimitiveError("all pair logits must be floating point on the pointwise device")
    if not bool(torch.isfinite(positive_logits).all().item()) or not bool(
        torch.isfinite(negative_logits).all().item()
    ):
        raise NeuralPrimitiveError("pair logits must contain only finite values")
    margins = positive_logits - negative_logits
    if not bool(torch.isfinite(margins).all().item()):
        raise NeuralPrimitiveError("pair margins must be finite")
    pairwise_loss = torch.nn.functional.softplus(-margins).mean()
    result = config.pointwise_weight * pointwise_loss + config.pairwise_weight * pairwise_loss
    if not bool(torch.isfinite(result).item()):
        raise NeuralPrimitiveError("hybrid loss must be finite")
    return result


def _validate_features_for_config(
    features: NeuralFeatureBatch,
    config: NeuralModelConfig,
) -> None:
    if features.categorical.shape[1] != len(config.categorical_cardinalities):
        raise NeuralPrimitiveError("feature categorical field count does not match model config")
    if features.dense.shape[1] != config.dense_feature_count:
        raise NeuralPrimitiveError("feature dense field count does not match model config")
    for field_index, cardinality in enumerate(config.categorical_cardinalities):
        if int(features.categorical[:, field_index].max()) >= cardinality:
            raise NeuralPrimitiveError(
                f"categorical field {field_index} contains an out-of-range index"
            )


@no_type_check
def _state_digest(state: object) -> str:
    digest = hashlib.sha256(b"neural-state-v1\0")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@no_type_check
def _predict_module(module: object, features: NeuralFeatureBatch, device: str) -> Float64Vector:
    torch = _torch()
    categorical = torch.as_tensor(
        np.array(features.categorical, copy=True), dtype=torch.int64, device=device
    )
    dense = torch.as_tensor(np.array(features.dense, copy=True), dtype=torch.float32, device=device)
    module.eval()
    with torch.no_grad():
        logits = module(categorical, dense)
    if device == "mps":
        torch.mps.synchronize()
    scores = np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float64)
    if scores.shape != (features.row_count,) or not np.isfinite(scores).all():
        raise NeuralPrimitiveError("neural inference must return one finite score per row")
    scores.setflags(write=False)
    return cast(Float64Vector, scores)


@no_type_check
def fit_neural(
    model_config: NeuralModelConfig,
    training_features: NeuralFeatureBatch,
    training_targets: NeuralTrainingTargets,
    pair_indices: NeuralPairIndices,
    *,
    fit_config: NeuralFitConfig,
    ceilings: NeuralCeilings,
    inner_valid_features: NeuralFeatureBatch | None = None,
    inner_valid_scorer: InnerValidScorer | None = None,
) -> NeuralCheckpoint:
    """Fit a bounded model using train-derived targets and an aggregate inner scorer only."""

    if not isinstance(model_config, NeuralModelConfig):
        raise NeuralPrimitiveError("model_config must be NeuralModelConfig")
    if not isinstance(training_features, NeuralFeatureBatch) or training_features.phase not in {
        DataPhase.TRAIN,
        DataPhase.INNER_TRAIN,
    }:
        raise NeuralPrimitiveError("fit features are allowed only for train or inner_train")
    if not isinstance(training_targets, NeuralTrainingTargets):
        raise NeuralPrimitiveError("training_targets must be NeuralTrainingTargets")
    if training_features.phase is not training_targets.phase:
        raise NeuralPrimitiveError("training feature and target phases must match")
    if training_features.row_count != training_targets.values.size:
        raise NeuralPrimitiveError("training feature and target row counts must match")
    if not isinstance(pair_indices, NeuralPairIndices):
        raise NeuralPrimitiveError("pair_indices must be NeuralPairIndices")
    if not isinstance(fit_config, NeuralFitConfig) or not isinstance(ceilings, NeuralCeilings):
        raise NeuralPrimitiveError("fit_config and ceilings must use frozen neural config types")
    _validate_features_for_config(training_features, model_config)

    row_count = training_features.row_count
    if (
        int(pair_indices.positive_indices.max()) >= row_count
        or int(pair_indices.negative_indices.max()) >= row_count
    ):
        raise NeuralPrimitiveError("pair indices must reference existing training rows")
    labels = training_targets.values
    if not np.all(labels[pair_indices.positive_indices] == 1):
        raise NeuralPrimitiveError("positive pair indices must reference positive targets")
    if not np.all(labels[pair_indices.negative_indices] == 0):
        raise NeuralPrimitiveError("negative pair indices must reference negative targets")

    if (inner_valid_features is None) != (inner_valid_scorer is None):
        raise NeuralPrimitiveError(
            "inner_valid features and aggregate scorer must be supplied together"
        )
    if inner_valid_features is not None:
        if inner_valid_features.phase is not DataPhase.INNER_VALID:
            raise NeuralPrimitiveError("fit may use an aggregate scorer only for inner_valid")
        _validate_features_for_config(inner_valid_features, model_config)

    built = build_neural_model(
        model_config,
        seed=fit_config.seed,
        requested_device=fit_config.requested_device,
        num_threads=fit_config.num_threads,
    )
    torch = _torch()
    device = built.device.actual_device
    categorical = torch.as_tensor(
        np.array(training_features.categorical, copy=True),
        dtype=torch.int64,
        device=device,
    )
    dense = torch.as_tensor(
        np.array(training_features.dense, copy=True),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(np.array(labels, copy=True), dtype=torch.float32, device=device)
    positive_indices = torch.as_tensor(
        np.array(pair_indices.positive_indices, copy=True), dtype=torch.int64, device=device
    )
    negative_indices = torch.as_tensor(
        np.array(pair_indices.negative_indices, copy=True), dtype=torch.int64, device=device
    )
    optimizer = torch.optim.Adam(built.module.parameters(), lr=fit_config.learning_rate)
    epoch_seconds: list[float] = []
    throughputs: list[float] = []
    pair_count = pair_indices.pair_count
    for epoch in range(fit_config.epochs):
        started = time.monotonic()
        permutation = np.random.default_rng(fit_config.seed + epoch).permutation(row_count)
        built.module.train()
        for batch_start in range(0, row_count, fit_config.batch_size):
            batch_numpy = permutation[batch_start : batch_start + fit_config.batch_size]
            batch = torch.as_tensor(batch_numpy, dtype=torch.int64, device=device)
            current_size = int(batch.numel())
            pair_positions_numpy = (
                np.arange(batch_start, batch_start + current_size, dtype=np.int64) % pair_count
            )
            pair_positions = torch.as_tensor(pair_positions_numpy, dtype=torch.int64, device=device)
            optimizer.zero_grad(set_to_none=True)
            pointwise_logits = built.module(categorical[batch], dense[batch])
            positive_logits = built.module(
                categorical[positive_indices[pair_positions]],
                dense[positive_indices[pair_positions]],
            )
            negative_logits = built.module(
                categorical[negative_indices[pair_positions]],
                dense[negative_indices[pair_positions]],
            )
            loss = hybrid_binary_pairwise_loss(
                pointwise_logits,
                targets[batch],
                positive_logits,
                negative_logits,
                phase=training_targets.phase,
                config=fit_config.loss,
            )
            loss.backward()
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elapsed = max(time.monotonic() - started, np.finfo(np.float64).tiny)
        epoch_seconds.append(elapsed)
        throughputs.append(row_count / elapsed)

    inner_score: _InnerAggregate | None = None
    if inner_valid_features is not None and inner_valid_scorer is not None:
        scores = _predict_module(built.module, inner_valid_features, device)
        inner_score = inner_valid_scorer(scores)
        if (
            not isinstance(inner_score, _InnerAggregate)
            or inner_score.rows != inner_valid_features.row_count
        ):
            raise NeuralPrimitiveError(
                "inner_valid scorer must return aligned aggregate ScoreResult"
            )
        metrics = (inner_score.gauc, inner_score.ndcg_at_5, inner_score.primary)
        if not all(math.isfinite(metric) for metric in metrics):
            raise NeuralPrimitiveError("inner_valid aggregate metrics must be finite")

    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in built.module.state_dict().items()
    }
    state_digest = _state_digest(cpu_state)
    buffer = io.BytesIO()
    torch.save(cpu_state, buffer)
    state_bytes = buffer.getvalue()
    evidence = NeuralPerformanceEvidence(
        parameter_count=built.parameter_count,
        epoch_seconds=tuple(epoch_seconds),
        examples_per_second=tuple(throughputs),
        checkpoint_bytes=len(state_bytes),
    )
    eligibility = assess_neural_eligibility(evidence, ceilings)
    result = NeuralResultManifest(
        model_config_digest=model_config.digest,
        fit_config_digest=fit_config.digest,
        training_feature_digest=training_features.digest,
        training_target_digest=training_targets.digest,
        pair_digest=pair_indices.digest,
        state_digest=state_digest,
        device=built.device,
        eligibility=eligibility,
        inner_valid_feature_digest=(
            inner_valid_features.digest if inner_valid_features is not None else None
        ),
        inner_valid_gauc=inner_score.gauc if inner_score is not None else None,
        inner_valid_ndcg_at_5=inner_score.ndcg_at_5 if inner_score is not None else None,
        inner_valid_primary=inner_score.primary if inner_score is not None else None,
        inner_valid_scorer_digest=(inner_score.scorer_digest if inner_score is not None else None),
        inner_valid_prediction_digest=(
            inner_score.prediction_digest if inner_score is not None else None
        ),
    )
    checkpoint_digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": NEURAL_SCHEMA_VERSION,
                "model_config_digest": model_config.digest,
                "fit_config_digest": fit_config.digest,
                "training_feature_digest": training_features.digest,
                "training_target_digest": training_targets.digest,
                "pair_digest": pair_indices.digest,
                "state_digest": state_digest,
            }
        )
    ).hexdigest()
    return NeuralCheckpoint(
        model_config=model_config,
        fit_config=fit_config,
        state_bytes=state_bytes,
        state_digest=state_digest,
        result=result,
        digest=checkpoint_digest,
    )


@no_type_check
def predict_neural(
    checkpoint: NeuralCheckpoint,
    features: NeuralFeatureBatch,
    *,
    requested_device: str = "cpu",
) -> NeuralPrediction:
    """Replay an immutable checkpoint on label-free inner/outer/final features."""

    if not isinstance(checkpoint, NeuralCheckpoint):
        raise NeuralPrimitiveError("checkpoint must be NeuralCheckpoint")
    if not isinstance(features, NeuralFeatureBatch) or features.phase not in {
        DataPhase.INNER_VALID,
        DataPhase.OUTER_VALID,
        DataPhase.FINAL,
    }:
        raise NeuralPrimitiveError(
            "prediction accepts only label-free validation or final features"
        )
    _validate_features_for_config(features, checkpoint.model_config)
    built = build_neural_model(
        checkpoint.model_config,
        seed=checkpoint.fit_config.seed,
        requested_device=requested_device,
        num_threads=checkpoint.fit_config.num_threads,
    )
    torch = _torch()
    try:
        state = torch.load(
            io.BytesIO(checkpoint.state_bytes),
            map_location=built.device.actual_device,
            weights_only=True,
        )
    except Exception as exc:
        raise NeuralPrimitiveError("checkpoint state bytes cannot be loaded safely") from exc
    if _state_digest(state) != checkpoint.state_digest:
        raise NeuralPrimitiveError("checkpoint state digest mismatch")
    built.module.load_state_dict(state, strict=True)
    scores = _predict_module(built.module, features, built.device.actual_device)
    return NeuralPrediction(
        scores=scores,
        phase=features.phase,
        prediction_digest=prediction_digest(scores),
        checkpoint_digest=checkpoint.digest,
        feature_digest=features.digest,
        device=built.device,
    )


__all__ = [
    "NEURAL_SCHEMA_VERSION",
    "BuiltNeuralModel",
    "DeviceParityResult",
    "DeviceSelection",
    "HybridLossConfig",
    "NeuralArchitecture",
    "NeuralCeilings",
    "NeuralCheckpoint",
    "NeuralConfigManifest",
    "NeuralEligibility",
    "NeuralFeatureBatch",
    "NeuralFitConfig",
    "NeuralModelConfig",
    "NeuralPairIndices",
    "NeuralPerformanceEvidence",
    "NeuralPrediction",
    "NeuralPrimitiveError",
    "NeuralResultManifest",
    "NeuralTrainingTargets",
    "TorchUnavailableError",
    "assess_device_parity",
    "assess_neural_eligibility",
    "build_neural_model",
    "configure_neural_runtime",
    "dcnv2_cross_step",
    "deepfm_interaction",
    "fit_neural",
    "hybrid_binary_pairwise_loss",
    "predict_neural",
]

"""Exact trusted adapter for the immutable organizer NumPy factorization machine."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import psutil  # type: ignore[import-untyped]

from kuairand_agent.baselines.artifacts import PredictionVector, StarterFMCheckpoint
from kuairand_agent.baselines.organizer import (
    OrganizerLoadError,
    OrganizerModules,
    load_verified_organizer,
)
from kuairand_agent.contract import OrganizerIntegrityError, verify_starter_kit
from kuairand_agent.data.canonical import CanonicalInputs

STARTER_FM_SCHEMA_VERSION: Final = 1
STARTER_FIELDS: Final = ("user_id", "video_id", "author_id", "tab", "dur_bucket")

type Int32Matrix = npt.NDArray[np.int32]
type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]


class StarterFMError(RuntimeError):
    """Raised when exact organizer parity or trusted input invariants fail."""


class EncodingProtocol(Protocol):
    """Structural boundary supplied by the trusted starter encoding module."""

    @property
    def digest(self) -> str: ...

    @property
    def training_inputs_digest(self) -> str: ...

    @property
    def field_names(self) -> tuple[str, ...]: ...

    @property
    def total_dim(self) -> int: ...

    def transform(self, inputs: CanonicalInputs) -> Int32Matrix: ...


class TrainTargetsProtocol(Protocol):
    """Primary-only view of a trusted official-train target capability."""

    @property
    def digest(self) -> str: ...

    @property
    def training_inputs_digest(self) -> str: ...

    @property
    def row_count(self) -> int: ...

    @property
    def primary(self) -> npt.NDArray[np.generic]: ...


class ValidationScorer(Protocol):
    """Protected aggregate scorer; its closure, not the adapter, owns labels."""

    @property
    def validation_inputs_digest(self) -> str: ...

    def __call__(self, scores: Float64Array, /) -> object: ...


class _OrganizerFMProtocol(Protocol):
    """Mutable state and methods of the hash-verified organizer ``FM`` class."""

    V: Float32Array
    W: Float32Array
    b: np.float32
    t: int

    def step(self, X: Int32Matrix, y: Float32Array) -> float: ...

    def predict(self, X: Int32Matrix, bs: int = 200_000) -> Float32Array: ...


class _MetricsObject(Protocol):
    @property
    def gauc(self) -> float: ...

    @property
    def ndcg_at_5(self) -> float: ...

    @property
    def primary(self) -> float: ...


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StarterFMError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class StarterFMConfig:
    """Frozen organizer hyperparameters; only the published seed is variable."""

    seed: int = 0
    k: int = field(init=False, default=16)
    learning_rate: float = field(init=False, default=0.001)
    l2: float = field(init=False, default=1e-6)
    batch_size: int = field(init=False, default=8192)
    max_epochs: int = field(init=False, default=40)
    patience: int = field(init=False, default=4)
    improvement_threshold: float = field(init=False, default=1e-5)
    predict_batch_size: int = field(init=False, default=200_000)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise StarterFMError("starter FM seed must be a uint32-compatible integer")
        object.__setattr__(self, "digest", _canonical_digest(self.manifest()))

    @property
    def lr(self) -> float:
        return self.learning_rate

    @property
    def bs(self) -> int:
        return self.batch_size

    @property
    def epochs(self) -> int:
        return self.max_epochs

    @property
    def threshold(self) -> float:
        return self.improvement_threshold

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": STARTER_FM_SCHEMA_VERSION,
            "algorithm": "immutable_organizer_numpy_fm",
            "seed": self.seed,
            "k": self.k,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "improvement_threshold": self.improvement_threshold,
            "predict_batch_size": self.predict_batch_size,
            "initializer_rng": "numpy.default_rng(seed)",
            "shuffle_rng": "independent_numpy.default_rng(seed)",
            "optimizer": "dense_adam",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "precision": "float32",
            "device": "cpu",
        }


def _metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise StarterFMError(f"protected validation metric {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise StarterFMError(f"protected validation metric {name} must be finite in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Only aggregate organizer metrics cross back from the protected scorer."""

    gauc: float
    ndcg_at_5: float
    primary: float

    def __post_init__(self) -> None:
        gauc = _metric(self.gauc, "GAUC")
        ndcg = _metric(self.ndcg_at_5, "nDCG@5")
        primary = _metric(self.primary, "primary")
        expected_primary = float((np.float32(gauc) + np.float32(ndcg)) / 2.0)
        if primary != expected_primary:
            raise StarterFMError("protected primary must equal mean(GAUC, nDCG@5)")
        object.__setattr__(self, "gauc", gauc)
        object.__setattr__(self, "ndcg_at_5", ndcg)
        object.__setattr__(self, "primary", primary)

    @property
    def ndcg5(self) -> float:
        return self.ndcg_at_5

    def manifest(self) -> dict[str, float]:
        return {"GAUC": self.gauc, "nDCG@5": self.ndcg_at_5, "primary": self.primary}


def _aggregate_metrics(value: object) -> AggregateMetrics:
    if isinstance(value, AggregateMetrics):
        return value
    if isinstance(value, Mapping):
        if set(value) != {"GAUC", "nDCG@5", "primary"}:
            raise StarterFMError(
                "protected validation scorer mapping must contain only organizer metrics"
            )
        try:
            return AggregateMetrics(
                gauc=cast(float, value["GAUC"]),
                ndcg_at_5=cast(float, value["nDCG@5"]),
                primary=cast(float, value["primary"]),
            )
        except KeyError as exc:
            raise StarterFMError("protected validation scorer omitted organizer metrics") from exc
    try:
        metrics = cast(_MetricsObject, value)
        return AggregateMetrics(
            gauc=metrics.gauc,
            ndcg_at_5=metrics.ndcg_at_5,
            primary=metrics.primary,
        )
    except AttributeError as exc:
        raise StarterFMError(
            "protected validation scorer must return aggregate organizer metrics"
        ) from exc


@dataclass(frozen=True, slots=True)
class EpochTrace:
    """Deterministic epoch evidence; wall-clock data is recorded separately."""

    epoch: int
    batch_count: int
    optimizer_steps: int
    mean_loss: float
    metrics: AggregateMetrics
    prediction_digest: str
    improved: bool
    bad_epochs: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise StarterFMError("trace epoch must be positive")
        if type(self.batch_count) is not int or self.batch_count <= 0:
            raise StarterFMError("trace batch_count must be positive")
        if type(self.optimizer_steps) is not int or self.optimizer_steps <= 0:
            raise StarterFMError("trace optimizer_steps must be positive")
        if not math.isfinite(self.mean_loss) or self.mean_loss < 0.0:
            raise StarterFMError("trace mean_loss must be finite and non-negative")
        _require_digest(self.prediction_digest, "trace prediction_digest")
        if type(self.improved) is not bool:
            raise StarterFMError("trace improved must be bool")
        if type(self.bad_epochs) is not int or self.bad_epochs < 0:
            raise StarterFMError("trace bad_epochs must be non-negative")

    def manifest(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "batch_count": self.batch_count,
            "optimizer_steps": self.optimizer_steps,
            "mean_loss": self.mean_loss,
            "metrics": self.metrics.manifest(),
            "prediction_digest": self.prediction_digest,
            "improved": self.improved,
            "bad_epochs": self.bad_epochs,
        }


@dataclass(frozen=True, slots=True)
class TrainingResources:
    """Observed local CPU resource use, excluded from deterministic model identity."""

    wall_seconds: float
    rss_before_bytes: int
    rss_after_bytes: int
    max_observed_rss_bytes: int
    train_rows: int
    validation_rows: int
    total_dim: int
    epochs_completed: int
    optimizer_steps: int
    device: str = "cpu"
    precision: str = "float32"

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0.0:
            raise StarterFMError("resource wall_seconds must be finite and non-negative")
        for name in (
            "rss_before_bytes",
            "rss_after_bytes",
            "max_observed_rss_bytes",
            "train_rows",
            "validation_rows",
            "total_dim",
            "epochs_completed",
            "optimizer_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise StarterFMError(f"resource {name} must be a positive integer")
        if self.max_observed_rss_bytes < max(self.rss_before_bytes, self.rss_after_bytes):
            raise StarterFMError("resource maximum RSS must cover before/after observations")
        if self.device != "cpu" or self.precision != "float32":
            raise StarterFMError("official FM reference resources must record cpu/float32")

    def manifest(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": self.rss_after_bytes,
            "max_observed_rss_bytes": self.max_observed_rss_bytes,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "total_dim": self.total_dim,
            "epochs_completed": self.epochs_completed,
            "optimizer_steps": self.optimizer_steps,
            "device": self.device,
            "precision": self.precision,
        }


@dataclass(frozen=True, slots=True)
class StarterFMRun:
    """Replayable best-restored baseline result and its complete trusted evidence."""

    checkpoint: StarterFMCheckpoint
    validation_predictions: PredictionVector
    validation_metrics: AggregateMetrics
    trace: tuple[EpochTrace, ...]
    resources: TrainingResources
    train_inputs_digest: str
    training_targets_digest: str
    validation_inputs_digest: str
    encoding_digest: str
    config_digest: str
    starter_manifest_digest: str
    logical_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.trace:
            raise StarterFMError("starter FM run requires at least one epoch trace")
        for name in (
            "train_inputs_digest",
            "training_targets_digest",
            "validation_inputs_digest",
            "encoding_digest",
            "config_digest",
            "starter_manifest_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.checkpoint.encoding_digest != self.encoding_digest:
            raise StarterFMError("run checkpoint encoding identity mismatch")
        if self.checkpoint.config_digest != self.config_digest:
            raise StarterFMError("run checkpoint config identity mismatch")
        if self.checkpoint.starter_manifest_digest != self.starter_manifest_digest:
            raise StarterFMError("run checkpoint starter identity mismatch")
        if self.checkpoint.epochs_completed != len(self.trace):
            raise StarterFMError("run checkpoint epoch count does not match trace")
        if tuple(epoch.epoch for epoch in self.trace) != tuple(range(1, len(self.trace) + 1)):
            raise StarterFMError("run trace epochs must be contiguous and one-based")
        if self.trace[-1].optimizer_steps != self.checkpoint.optimizer_steps:
            raise StarterFMError("run checkpoint optimizer steps do not match trace")
        prior_steps = 0
        for epoch in self.trace:
            if epoch.optimizer_steps - prior_steps != epoch.batch_count:
                raise StarterFMError("run trace optimizer steps do not match epoch batches")
            prior_steps = epoch.optimizer_steps
        if self.resources.validation_rows != self.validation_predictions.row_count:
            raise StarterFMError("run resource validation row count differs from predictions")
        if self.resources.total_dim != self.checkpoint.total_dim:
            raise StarterFMError("run resource dimension differs from checkpoint")
        if self.resources.epochs_completed != self.checkpoint.epochs_completed:
            raise StarterFMError("run resource epoch count differs from checkpoint")
        if self.resources.optimizer_steps != self.checkpoint.optimizer_steps:
            raise StarterFMError("run resource optimizer steps differ from checkpoint")
        best_trace = self.trace[self.checkpoint.best_epoch - 1]
        if best_trace.prediction_digest != self.validation_predictions.digest:
            raise StarterFMError("restored predictions do not match the best epoch")
        if best_trace.metrics != self.validation_metrics:
            raise StarterFMError("restored metrics do not match the best epoch")
        object.__setattr__(self, "logical_digest", _canonical_digest(self.logical_manifest()))

    @property
    def prediction_digest(self) -> str:
        return self.validation_predictions.digest

    @property
    def checkpoint_digest(self) -> str:
        return self.checkpoint.digest

    def logical_manifest(self) -> dict[str, object]:
        """Deterministic evidence identity excluding observed time and memory."""

        return {
            "schema_version": STARTER_FM_SCHEMA_VERSION,
            "train_inputs_digest": self.train_inputs_digest,
            "training_targets_digest": self.training_targets_digest,
            "validation_inputs_digest": self.validation_inputs_digest,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "starter_manifest_digest": self.starter_manifest_digest,
            "checkpoint": self.checkpoint.manifest(),
            "validation_predictions": self.validation_predictions.manifest(),
            "validation_metrics": self.validation_metrics.manifest(),
            "trace": [epoch.manifest() for epoch in self.trace],
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self.logical_manifest(),
            "logical_digest": self.logical_digest,
            "resources": self.resources.manifest(),
        }


def _require_model_state(
    model: _OrganizerFMProtocol,
    *,
    total_dim: int,
    factor_dim: int,
) -> tuple[Float32Array, Float32Array, np.float32]:
    V = getattr(model, "V", None)
    W = getattr(model, "W", None)
    b = np.asarray(getattr(model, "b", None))
    step = getattr(model, "t", None)
    if (
        not isinstance(V, np.ndarray)
        or V.dtype != np.dtype("float32")
        or V.shape != (total_dim, factor_dim)
        or not V.flags.c_contiguous
        or not np.isfinite(V).all()
    ):
        raise StarterFMError("verified organizer FM returned invalid factor state")
    if (
        not isinstance(W, np.ndarray)
        or W.dtype != np.dtype("float32")
        or W.shape != (total_dim,)
        or not W.flags.c_contiguous
        or not np.isfinite(W).all()
    ):
        raise StarterFMError("verified organizer FM returned invalid linear state")
    if b.shape != () or b.dtype != np.dtype("float32") or not np.isfinite(b):
        raise StarterFMError("verified organizer FM returned invalid bias state")
    if type(step) is not int or step < 0:
        raise StarterFMError("verified organizer FM returned invalid optimizer step")
    return cast(Float32Array, V), cast(Float32Array, W), np.float32(b)


def _model_predictions(
    model: _OrganizerFMProtocol,
    matrix: Int32Matrix,
    *,
    batch_size: int,
) -> Float32Array:
    raw = model.predict(matrix, bs=batch_size)
    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != np.dtype("float32")
        or raw.shape != (len(matrix),)
        or not raw.flags.c_contiguous
        or not np.isfinite(raw).all()
    ):
        raise StarterFMError("verified organizer FM returned invalid predictions")
    return raw


def _restore_model(
    model: _OrganizerFMProtocol,
    V: Float32Array,
    W: Float32Array,
    b: np.float32,
) -> None:
    model.V = np.array(V, dtype=np.float32, copy=True)
    model.W = np.array(W, dtype=np.float32, copy=True)
    model.b = np.float32(b)


def _encoded_matrix(
    encoding: EncodingProtocol, inputs: CanonicalInputs, *, name: str
) -> Int32Matrix:
    if not isinstance(inputs, CanonicalInputs) or len(inputs) == 0:
        raise StarterFMError(f"{name} must be non-empty CanonicalInputs")
    try:
        values = encoding.transform(inputs)
    except Exception as exc:
        raise StarterFMError(f"cannot encode {name}: {exc}") from exc
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("int32")
        or values.shape != (len(inputs), len(STARTER_FIELDS))
        or not values.flags.c_contiguous
        or values.flags.writeable
    ):
        raise StarterFMError(f"{name} encoding must be read-only C-contiguous int32 shape (N, 5)")
    total_dim = encoding.total_dim
    if type(total_dim) is not int or total_dim <= 0 or total_dim > np.iinfo(np.int32).max:
        raise StarterFMError("encoding total_dim must be a positive int32-compatible integer")
    if int(values.min()) < 0 or int(values.max()) >= total_dim:
        raise StarterFMError(f"{name} encoded IDs fall outside encoding total_dim")
    return values


def _training_labels(
    targets: TrainTargetsProtocol,
    *,
    expected_rows: int,
    expected_inputs_digest: str,
) -> tuple[Float32Array, str]:
    try:
        raw = np.asarray(targets.primary)
        row_count = targets.row_count
        digest = _require_digest(targets.digest, "training_targets_digest")
        inputs_digest = _require_digest(
            targets.training_inputs_digest,
            "target training_inputs_digest",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise StarterFMError("train_targets must be a trusted primary target capability") from exc
    if inputs_digest != expected_inputs_digest:
        raise StarterFMError("train targets are not aligned to the supplied training inputs")
    if (
        type(row_count) is not int
        or row_count != expected_rows
        or raw.ndim != 1
        or raw.shape[0] != expected_rows
        or raw.dtype.kind not in "biuf"
        or raw.dtype.kind == "b"
        or not np.isfinite(raw).all()
        or not np.logical_or(raw == 0, raw == 1).all()
    ):
        raise StarterFMError("training primary labels must be aligned binary numeric values")
    labels = np.ascontiguousarray(raw, dtype=np.float32)
    labels.setflags(write=False)
    return labels, digest


def _rss() -> int:
    try:
        current = int(psutil.Process().memory_info().rss)
    except (psutil.Error, OSError) as exc:
        raise StarterFMError("cannot sample current process RSS") from exc
    if current <= 0:
        raise StarterFMError("current process RSS must be positive")
    return current


class StarterFMAdapter:
    """Train, restore, and replay only the official organizer FM reference."""

    def __init__(
        self,
        *,
        starter_dir: str | Path,
        config: StarterFMConfig | None = None,
    ) -> None:
        self.config = StarterFMConfig() if config is None else config
        if not isinstance(self.config, StarterFMConfig):
            raise StarterFMError("config must be StarterFMConfig")
        try:
            organizer = load_verified_organizer(starter_dir)
        except (OrganizerIntegrityError, OrganizerLoadError, OSError, RuntimeError) as exc:
            raise StarterFMError("cannot load the verified organizer FM") from exc
        self._organizer: OrganizerModules = organizer
        self._starter_root = organizer.root
        self._starter_manifest_digest = organizer.manifest_sha256

    @property
    def starter_manifest_digest(self) -> str:
        return self._starter_manifest_digest

    def _require_starter_unchanged(self) -> None:
        try:
            current = verify_starter_kit(self._starter_root)
        except (OrganizerIntegrityError, OSError, RuntimeError) as exc:
            raise StarterFMError("cannot reverify the organizer starter") from exc
        if current.manifest_sha256 != self._starter_manifest_digest:
            raise StarterFMError("organizer starter changed after FM adapter construction")

    def _new_model(self, total_dim: int) -> _OrganizerFMProtocol:
        factory = self._organizer.baseline.FM
        try:
            raw = factory(
                total_dim,
                k=self.config.k,
                lr=self.config.learning_rate,
                l2=self.config.l2,
                seed=self.config.seed,
            )
        except Exception as exc:
            raise StarterFMError("verified organizer FM construction failed") from exc
        if raw.__class__ is not factory:
            raise StarterFMError("verified organizer FM construction returned another model class")
        model = cast(_OrganizerFMProtocol, raw)
        _require_model_state(model, total_dim=total_dim, factor_dim=self.config.k)
        return model

    def fit(
        self,
        *,
        encoding: EncodingProtocol,
        train_inputs: CanonicalInputs,
        train_targets: TrainTargetsProtocol,
        validation_inputs: CanonicalInputs,
        validation_scorer: ValidationScorer,
    ) -> StarterFMRun:
        """Fit with train labels and protected aggregate validation feedback only."""

        self._require_starter_unchanged()
        encoding_digest = _require_digest(getattr(encoding, "digest", None), "encoding_digest")
        training_source_digest = _require_digest(
            getattr(encoding, "training_inputs_digest", None),
            "encoding training_inputs_digest",
        )
        if getattr(encoding, "field_names", None) != STARTER_FIELDS:
            raise StarterFMError("encoding fields differ from the five organizer FM fields")
        if training_source_digest != train_inputs.digest:
            raise StarterFMError("encoding was not fitted from the supplied training inputs")
        train_matrix = _encoded_matrix(encoding, train_inputs, name="train_inputs")
        validation_matrix = _encoded_matrix(encoding, validation_inputs, name="validation_inputs")
        labels, targets_digest = _training_labels(
            train_targets,
            expected_rows=len(train_inputs),
            expected_inputs_digest=train_inputs.digest,
        )
        if not callable(validation_scorer):
            raise StarterFMError("validation_scorer must be a protected callable")
        scorer_inputs_digest = _require_digest(
            getattr(validation_scorer, "validation_inputs_digest", None),
            "validation scorer inputs digest",
        )
        if scorer_inputs_digest != validation_inputs.digest:
            raise StarterFMError(
                "validation scorer is not aligned to the supplied validation inputs"
            )

        rss_before = _rss()
        max_rss = rss_before
        started = time.perf_counter()
        model = self._new_model(encoding.total_dim)
        shuffle_rng = np.random.default_rng(self.config.seed)
        best_primary: int | np.float32 = -1
        best_state: tuple[Float32Array, Float32Array, np.float32] | None = None
        best_epoch = 0
        best_metrics: AggregateMetrics | None = None
        trace: list[EpochTrace] = []
        bad_epochs = 0

        for epoch in range(1, self.config.max_epochs + 1):
            permutation = shuffle_rng.permutation(len(labels))
            losses: list[float] = []
            for start in range(0, len(permutation), self.config.batch_size):
                try:
                    loss = float(
                        model.step(
                            train_matrix[permutation[start : start + self.config.batch_size]],
                            labels[permutation[start : start + self.config.batch_size]],
                        )
                    )
                except Exception as exc:
                    raise StarterFMError("verified organizer FM training step failed") from exc
                if not math.isfinite(loss) or loss < 0.0:
                    raise StarterFMError("verified organizer FM returned an invalid loss")
                losses.append(loss)
            state_v, state_w, state_b = _require_model_state(
                model,
                total_dim=encoding.total_dim,
                factor_dim=self.config.k,
            )
            predictions = PredictionVector(
                _model_predictions(
                    model,
                    validation_matrix,
                    batch_size=self.config.predict_batch_size,
                )
            )
            metrics = _aggregate_metrics(validation_scorer(predictions.scores))
            candidate_primary = np.float32(metrics.primary)
            improved = bool(candidate_primary > best_primary + self.config.improvement_threshold)
            if improved:
                best_primary = candidate_primary
                best_epoch = epoch
                best_metrics = metrics
                bad_epochs = 0
                best_state = (
                    np.array(state_v, dtype=np.float32, copy=True),
                    np.array(state_w, dtype=np.float32, copy=True),
                    np.float32(state_b),
                )
            else:
                bad_epochs += 1
            trace.append(
                EpochTrace(
                    epoch=epoch,
                    batch_count=len(losses),
                    optimizer_steps=model.t,
                    mean_loss=float(np.mean(losses)),
                    metrics=metrics,
                    prediction_digest=predictions.digest,
                    improved=improved,
                    bad_epochs=bad_epochs,
                )
            )
            max_rss = max(max_rss, _rss())
            if bad_epochs >= self.config.patience:
                break

        if best_state is None or best_metrics is None or best_epoch <= 0:
            raise StarterFMError("organizer FM did not produce a restorable best state")
        _restore_model(model, *best_state)
        _require_model_state(
            model,
            total_dim=encoding.total_dim,
            factor_dim=self.config.k,
        )
        final_predictions = PredictionVector(
            _model_predictions(
                model,
                validation_matrix,
                batch_size=self.config.predict_batch_size,
            )
        )
        final_metrics = _aggregate_metrics(validation_scorer(final_predictions.scores))
        if final_predictions.digest != trace[best_epoch - 1].prediction_digest:
            raise StarterFMError(
                "best-state restore did not reproduce exact validation predictions"
            )
        if final_metrics != best_metrics:
            raise StarterFMError("protected scorer changed for identical restored predictions")

        checkpoint = StarterFMCheckpoint(
            V=best_state[0],
            W=best_state[1],
            b=best_state[2],
            encoding_digest=encoding_digest,
            config_digest=self.config.digest,
            starter_manifest_digest=self._starter_manifest_digest,
            seed=self.config.seed,
            best_epoch=best_epoch,
            epochs_completed=len(trace),
            optimizer_steps=model.t,
        )
        rss_after = _rss()
        max_rss = max(max_rss, rss_after)
        resources = TrainingResources(
            wall_seconds=time.perf_counter() - started,
            rss_before_bytes=rss_before,
            rss_after_bytes=rss_after,
            max_observed_rss_bytes=max_rss,
            train_rows=len(train_inputs),
            validation_rows=len(validation_inputs),
            total_dim=encoding.total_dim,
            epochs_completed=len(trace),
            optimizer_steps=model.t,
        )
        result = StarterFMRun(
            checkpoint=checkpoint,
            validation_predictions=final_predictions,
            validation_metrics=final_metrics,
            trace=tuple(trace),
            resources=resources,
            train_inputs_digest=train_inputs.digest,
            training_targets_digest=targets_digest,
            validation_inputs_digest=validation_inputs.digest,
            encoding_digest=encoding_digest,
            config_digest=self.config.digest,
            starter_manifest_digest=self._starter_manifest_digest,
        )
        self._require_starter_unchanged()
        return result

    def predict(
        self,
        *,
        checkpoint: StarterFMCheckpoint,
        encoding: EncodingProtocol,
        inputs: CanonicalInputs,
        expected_prediction_digest: str | None = None,
    ) -> PredictionVector:
        """Run exact label-free checkpoint inference in canonical input order."""

        self._require_starter_unchanged()
        if not isinstance(checkpoint, StarterFMCheckpoint):
            raise StarterFMError("checkpoint must be StarterFMCheckpoint")
        if checkpoint.config_digest != self.config.digest:
            raise StarterFMError("checkpoint config does not match this adapter")
        if checkpoint.seed != self.config.seed:
            raise StarterFMError("checkpoint seed does not match this adapter")
        if checkpoint.starter_manifest_digest != self._starter_manifest_digest:
            raise StarterFMError("checkpoint starter source does not match this adapter")
        encoding_digest = _require_digest(getattr(encoding, "digest", None), "encoding_digest")
        if checkpoint.encoding_digest != encoding_digest:
            raise StarterFMError("checkpoint encoding does not match inference encoding")
        if checkpoint.total_dim != encoding.total_dim or checkpoint.factor_dim != self.config.k:
            raise StarterFMError("checkpoint dimensions do not match organizer FM configuration")
        matrix = _encoded_matrix(encoding, inputs, name="inference inputs")
        model = self._new_model(encoding.total_dim)
        _restore_model(model, checkpoint.V, checkpoint.W, checkpoint.b)
        predictions = PredictionVector(
            _model_predictions(
                model,
                matrix,
                batch_size=self.config.predict_batch_size,
            )
        )
        if expected_prediction_digest is not None and predictions.digest != _require_digest(
            expected_prediction_digest, "expected_prediction_digest"
        ):
            raise StarterFMError("replayed prediction digest mismatch")
        self._require_starter_unchanged()
        return predictions

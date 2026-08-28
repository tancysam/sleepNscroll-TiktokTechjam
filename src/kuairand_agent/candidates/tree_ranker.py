"""Deterministic CPU LambdaRank adapter over trusted numeric feature matrices.

Training is the only label-bearing operation.  The adapter builds a private user-contiguous
training view through :mod:`kuairand_agent.candidates.grouping`, while every prediction remains
in the caller's canonical row order.  LightGBM is an optional research dependency and is loaded
only when the default backend is actually requested.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.candidates.grouping import UserGrouping
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.data.causal_features import FeatureMatrix
from kuairand_agent.data.fields import USER_SNAPSHOT_HEADER, VIDEO_STATISTIC_HEADER
from kuairand_agent.scoring.submission import prediction_digest

type LabelInput = Sequence[object] | npt.NDArray[np.generic]
type Float64Matrix = npt.NDArray[np.float64]
type Float64Vector = npt.NDArray[np.float64]
type Int8Vector = npt.NDArray[np.int8]
type ParameterValue = str | int | float | bool | tuple[int, ...]

TREE_RANKER_SCHEMA_VERSION: Final = 1
PINNED_LIGHTGBM_VERSION: Final = "4.7.0"
DEFAULT_TREE_SEED: Final = 0
DEFAULT_TREE_THREADS: Final = 4
EVAL_AT: Final = (5,)
_TRAINING_PHASES: Final = frozenset({DataPhase.TRAIN, DataPhase.INNER_TRAIN})
_PREDICTION_PHASES: Final = frozenset(
    {DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL}
)
_BLOCKED_RAW_TREE_FEATURES: Final = (
    (frozenset(USER_SNAPSHOT_HEADER) - {"user_id"})
    | (frozenset(VIDEO_STATISTIC_HEADER) - {"video_id"})
    | {"visible_status", "is_rand"}
)
_SHA256_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")


class TreeRankerError(ValueError):
    """Raised when the ranking adapter contract is violated."""


class TreeRankerDependencyError(RuntimeError):
    """Raised when the optional, pinned LightGBM backend is unavailable."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_manifest(domain: bytes, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise TreeRankerError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _uint32(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise TreeRankerError(f"{name} must be a uint32-compatible integer")
    return value


def _finite_float(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if type(value) is not float or not math.isfinite(value) or not minimum <= value <= maximum:
        raise TreeRankerError(f"{name} must be a finite float in [{minimum}, {maximum}]")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
        raise TreeRankerError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class LambdaRankConfig:
    """Bounded tunables plus frozen deterministic CPU execution controls."""

    seed: int = DEFAULT_TREE_SEED
    num_threads: int = DEFAULT_TREE_THREADS
    num_boost_round: int = 300
    early_stopping_rounds: int = 30
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    lambda_l2: float = 1.0
    lambdarank_truncation_level: int = 8

    def __post_init__(self) -> None:
        _uint32(self.seed, name="seed")
        _positive_int(self.num_threads, name="num_threads", maximum=64)
        _positive_int(self.num_boost_round, name="num_boost_round", maximum=10_000)
        _positive_int(
            self.early_stopping_rounds,
            name="early_stopping_rounds",
            maximum=10_000,
        )
        _finite_float(self.learning_rate, name="learning_rate", minimum=1e-6, maximum=1.0)
        _positive_int(self.num_leaves, name="num_leaves", maximum=4096)
        _positive_int(self.min_data_in_leaf, name="min_data_in_leaf", maximum=10_000_000)
        _finite_float(self.lambda_l2, name="lambda_l2", minimum=0.0, maximum=1_000_000.0)
        _positive_int(
            self.lambdarank_truncation_level,
            name="lambdarank_truncation_level",
            maximum=1000,
        )

    def parameters(self) -> Mapping[str, ParameterValue]:
        """Return exact effective LightGBM parameters with no backend defaults for policy fields."""

        values: dict[str, ParameterValue] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "data_random_seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "extra_seed": self.seed,
            "label_gain": (0, 1),
            "lambdarank_norm": True,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "extra_trees": False,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.lambda_l2,
            "n_estimators": self.num_boost_round,
            "verbosity": -1,
        }
        return MappingProxyType(values)

    def manifest(self) -> dict[str, object]:
        """Return the complete tree-count, early-stop, and parameter identity."""

        return {
            "schema_version": TREE_RANKER_SCHEMA_VERSION,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_at": list(EVAL_AT),
            "parameters": dict(self.parameters()),
        }

    @property
    def digest(self) -> str:
        return _digest_manifest(b"kuairand-lambdarank-config-v1\0", self.manifest())


@dataclass(frozen=True, slots=True)
class BackendFitRequest:
    """Fully normalized grouped arrays supplied to a replaceable training backend."""

    params: Mapping[str, ParameterValue]
    train_features: Float64Matrix = field(repr=False)
    train_labels: Int8Vector = field(repr=False)
    train_group_sizes: tuple[int, ...]
    eval_at: tuple[int, ...]
    inner_valid_features: Float64Matrix | None = field(default=None, repr=False)
    inner_valid_labels: Int8Vector | None = field(default=None, repr=False)
    inner_valid_group_sizes: tuple[int, ...] | None = None
    early_stopping_rounds: int | None = None


@dataclass(frozen=True, slots=True)
class BackendFitResult:
    """Serialized backend model plus the exact selected tree count."""

    model_text: str
    best_iteration: int

    def __post_init__(self) -> None:
        if not self.model_text or "\x00" in self.model_text:
            raise TreeRankerError("backend model_text must be non-empty and contain no NUL")
        if type(self.best_iteration) is not int or self.best_iteration <= 0:
            raise TreeRankerError("backend best_iteration must be a positive integer")


class LambdaRankBackend(Protocol):
    """Narrow optional-backend seam used by deterministic contract tests."""

    @property
    def identity(self) -> str: ...

    def fit(self, request: BackendFitRequest) -> BackendFitResult: ...

    def predict(
        self,
        *,
        model_text: str,
        features: Float64Matrix,
        num_iteration: int,
    ) -> object: ...


class _LightGBMBooster(Protocol):
    best_iteration: int

    def model_to_string(self, *, num_iteration: int) -> str: ...

    def predict(self, data: object, *, num_iteration: int) -> object: ...


class _LightGBMModule(Protocol):
    __version__: str
    Booster: object
    Dataset: object
    early_stopping: object
    train: object


@dataclass(frozen=True, slots=True)
class _LightGBMBackend:
    """Lazy native-training adapter for the exact optional LightGBM version."""

    module: _LightGBMModule = field(repr=False)
    identity: str

    def fit(self, request: BackendFitRequest) -> BackendFitResult:
        dataset_constructor = cast(Callable[..., object], self.module.Dataset)
        train_set = dataset_constructor(
            request.train_features,
            label=request.train_labels,
            group=list(request.train_group_sizes),
            free_raw_data=False,
        )
        params: dict[str, object] = dict(request.params)
        configured_rounds = params.pop("n_estimators")
        assert type(configured_rounds) is int
        params["eval_at"] = list(request.eval_at)
        train_kwargs: dict[str, object] = {}
        if request.inner_valid_features is not None:
            assert request.inner_valid_labels is not None
            assert request.inner_valid_group_sizes is not None
            assert request.early_stopping_rounds is not None
            valid_set = dataset_constructor(
                request.inner_valid_features,
                label=request.inner_valid_labels,
                group=list(request.inner_valid_group_sizes),
                reference=train_set,
                free_raw_data=False,
            )
            callback_factory = cast(Callable[..., object], self.module.early_stopping)
            train_kwargs.update(
                {
                    "valid_sets": [valid_set],
                    "valid_names": ["inner_valid"],
                    "callbacks": [
                        callback_factory(
                            request.early_stopping_rounds,
                            first_metric_only=True,
                            verbose=False,
                        )
                    ],
                }
            )
        train_function = cast(Callable[..., _LightGBMBooster], self.module.train)
        booster = train_function(
            params,
            train_set,
            num_boost_round=configured_rounds,
            **train_kwargs,
        )
        observed_best = getattr(booster, "best_iteration", 0)
        best_iteration = (
            int(observed_best)
            if type(observed_best) is int and observed_best > 0
            else configured_rounds
        )
        model_text = booster.model_to_string(num_iteration=best_iteration)
        return BackendFitResult(model_text=model_text, best_iteration=best_iteration)

    def predict(
        self,
        *,
        model_text: str,
        features: Float64Matrix,
        num_iteration: int,
    ) -> object:
        constructor = cast(Callable[..., _LightGBMBooster], self.module.Booster)
        booster = constructor(model_str=model_text)
        return booster.predict(features, num_iteration=num_iteration)


@dataclass(frozen=True, slots=True, init=False)
class InnerValidationSet:
    """Train-derived holdout labels available only inside an ``inner_train`` fit call."""

    features: FeatureMatrix
    labels: Int8Vector = field(repr=False)
    grouping: UserGrouping
    target_digest: str
    digest: str

    def __init__(
        self,
        features: FeatureMatrix,
        labels: LabelInput,
        *,
        grouping: UserGrouping,
    ) -> None:
        if not isinstance(features, FeatureMatrix):
            raise TreeRankerError("inner validation features must be a trusted FeatureMatrix")
        if not isinstance(grouping, UserGrouping) or grouping.phase is not DataPhase.INNER_VALID:
            raise TreeRankerError("inner validation grouping must have phase inner_valid")
        if grouping.row_count != features.row_count:
            raise TreeRankerError("inner validation grouping and feature row counts differ")
        normalized = _labels(labels, expected=features.row_count, name="inner_valid.labels")
        target_digest = _target_digest(normalized, DataPhase.INNER_VALID)
        manifest = {
            "schema_version": TREE_RANKER_SCHEMA_VERSION,
            "phase": DataPhase.INNER_VALID.value,
            "feature_digest": features.digest,
            "grouping_digest": grouping.digest,
            "target_digest": target_digest,
        }
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", normalized)
        object.__setattr__(self, "grouping", grouping)
        object.__setattr__(self, "target_digest", target_digest)
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-lambdarank-inner-valid-v1\0", manifest),
        )


@dataclass(frozen=True, slots=True)
class TreeRankerCheckpoint:
    """Content-linked LambdaRank model with feature, order, target, config, and backend identity."""

    training_phase: DataPhase
    feature_names: tuple[str, ...]
    training_feature_digest: str
    training_grouping_digest: str
    training_target_digest: str
    inner_validation_digest: str | None
    config_digest: str
    backend_identity: str
    best_iteration: int
    model_text: str = field(repr=False)
    model_sha256: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.training_phase, DataPhase)
            or self.training_phase not in _TRAINING_PHASES
        ):
            raise TreeRankerError("checkpoint training_phase must be train or inner_train")
        if (
            not self.feature_names
            or any(
                type(name) is not str or not name or "\x00" in name for name in self.feature_names
            )
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise TreeRankerError("checkpoint feature_names must be non-empty, unique strings")
        _require_sha256(self.training_feature_digest, name="training_feature_digest")
        _require_sha256(self.training_grouping_digest, name="training_grouping_digest")
        _require_sha256(self.training_target_digest, name="training_target_digest")
        if self.inner_validation_digest is not None:
            _require_sha256(self.inner_validation_digest, name="inner_validation_digest")
        _require_sha256(self.config_digest, name="config_digest")
        if (
            type(self.backend_identity) is not str
            or not self.backend_identity
            or "\x00" in self.backend_identity
            or "\n" in self.backend_identity
        ):
            raise TreeRankerError(
                "checkpoint backend_identity must be a non-empty single-line string"
            )
        if type(self.best_iteration) is not int or self.best_iteration <= 0:
            raise TreeRankerError("checkpoint best_iteration must be a positive integer")
        if type(self.model_text) is not str or not self.model_text or "\x00" in self.model_text:
            raise TreeRankerError("checkpoint model_text must be non-empty and contain no NUL")
        model_sha256 = hashlib.sha256(self.model_text.encode("utf-8")).hexdigest()
        object.__setattr__(self, "model_sha256", model_sha256)
        manifest = self.manifest()
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-lambdarank-checkpoint-v1\0", manifest),
        )

    def manifest(self) -> dict[str, object]:
        """Return checkpoint identity without embedding serialized model text."""

        return {
            "schema_version": TREE_RANKER_SCHEMA_VERSION,
            "training_phase": self.training_phase.value,
            "feature_names": list(self.feature_names),
            "training_feature_digest": self.training_feature_digest,
            "training_grouping_digest": self.training_grouping_digest,
            "training_target_digest": self.training_target_digest,
            "inner_validation_digest": self.inner_validation_digest,
            "config_digest": self.config_digest,
            "backend_identity": self.backend_identity,
            "best_iteration": self.best_iteration,
            "model_sha256": getattr(
                self,
                "model_sha256",
                hashlib.sha256(self.model_text.encode("utf-8")).hexdigest(),
            ),
        }


@dataclass(frozen=True, slots=True)
class TreeRankerPrediction:
    """One finite canonical-order label-free prediction vector."""

    scores: Float64Vector = field(repr=False)
    phase: DataPhase
    checkpoint_digest: str
    feature_digest: str
    prediction_digest: str


def _require_training_phase(phase: DataPhase) -> None:
    # This check must run before label conversion or inspection.
    if not isinstance(phase, DataPhase):
        raise TreeRankerError("phase must be a DataPhase")
    if phase not in _TRAINING_PHASES:
        raise TreeRankerError("LambdaRank labels are allowed only for train or inner_train")


def _labels(value: LabelInput, *, expected: int, name: str) -> Int8Vector:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TreeRankerError(f"{name} must be a one-dimensional binary vector") from exc
    if raw.ndim != 1:
        raise TreeRankerError(f"{name} must be one-dimensional")
    if raw.size != expected:
        raise TreeRankerError(f"{name} must have length {expected}, got {raw.size}")
    if raw.dtype.kind not in "biuf":
        raise TreeRankerError(f"{name} must contain numeric binary values")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise TreeRankerError(f"{name} must contain only binary 0 and 1 values")
    contiguous = np.ascontiguousarray(numeric, dtype=np.int8)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.int8)
    frozen.setflags(write=False)
    return cast(Int8Vector, frozen)


def _target_digest(labels: Int8Vector, phase: DataPhase) -> str:
    digest = hashlib.sha256()
    digest.update(b"kuairand-lambdarank-target-v1\0")
    digest.update(phase.value.encode("ascii"))
    digest.update(b"\0")
    digest.update(len(labels).to_bytes(8, "little"))
    digest.update(labels.tobytes(order="C"))
    return digest.hexdigest()


def _backend_identity(backend: LambdaRankBackend) -> str:
    identity = getattr(backend, "identity", None)
    if type(identity) is not str or not identity or "\x00" in identity or "\n" in identity:
        raise TreeRankerError("backend identity must be a non-empty single-line string")
    return identity


def _validate_feature_policy(features: FeatureMatrix) -> None:
    for name in features.feature_names:
        if name in _BLOCKED_RAW_TREE_FEATURES or any(
            name.startswith(f"{blocked}__") for blocked in _BLOCKED_RAW_TREE_FEATURES
        ):
            raise TreeRankerError(f"blocked raw feature cannot enter tree ranker: {name}")


def _default_backend() -> LambdaRankBackend:
    try:
        raw_module = importlib.import_module("lightgbm")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TreeRankerDependencyError(
            "LightGBM is an optional dependency; install the research-tree dependency group"
        ) from exc
    except OSError as exc:
        raise TreeRankerDependencyError(
            "LightGBM's native runtime is unavailable; install the research-tree dependency "
            "group and the platform OpenMP runtime"
        ) from exc
    # The macOS PyTorch wheel and Homebrew LightGBM can load incompatible OpenMP runtimes into
    # one interpreter; constructing a LightGBM Dataset then aborts the process before Python can
    # raise.  Candidate executions are process-isolated by design, so fail closed and require a
    # fresh tree worker instead of permitting a native crash after a neural worker was imported.
    if "torch" in sys.modules:
        raise TreeRankerDependencyError(
            "LightGBM must run in a fresh process when PyTorch has already been imported"
        )
    module = cast(_LightGBMModule, raw_module)
    version = getattr(module, "__version__", None)
    if version != PINNED_LIGHTGBM_VERSION:
        raise TreeRankerDependencyError(
            f"LightGBM {PINNED_LIGHTGBM_VERSION} is required, found {version!r}"
        )
    for name in ("Booster", "Dataset", "early_stopping", "train"):
        if not callable(getattr(module, name, None)):
            raise TreeRankerDependencyError(f"LightGBM backend is missing callable {name}")
    return _LightGBMBackend(
        module=module,
        identity=f"lightgbm:{PINNED_LIGHTGBM_VERSION}",
    )


def fit_lambdarank(
    features: FeatureMatrix,
    labels: LabelInput,
    *,
    grouping: UserGrouping,
    phase: DataPhase,
    config: LambdaRankConfig | None = None,
    inner_valid: InnerValidationSet | None = None,
    backend: LambdaRankBackend | None = None,
) -> TreeRankerCheckpoint:
    """Fit on official-train labels after stable private grouping."""

    _require_training_phase(phase)
    if not isinstance(features, FeatureMatrix):
        raise TreeRankerError("features must be a trusted FeatureMatrix")
    _validate_feature_policy(features)
    if not isinstance(grouping, UserGrouping):
        raise TreeRankerError("grouping must be a UserGrouping")
    if grouping.phase is not phase:
        raise TreeRankerError("training grouping phase must match the label phase")
    if grouping.row_count != features.row_count:
        raise TreeRankerError("training grouping and feature matrix row counts differ")
    if inner_valid is not None:
        if phase is not DataPhase.INNER_TRAIN:
            raise TreeRankerError("early stopping is allowed only with inner_train")
        if not isinstance(inner_valid, InnerValidationSet):
            raise TreeRankerError("inner_valid must be an InnerValidationSet")
        if inner_valid.features.feature_names != features.feature_names:
            raise TreeRankerError("inner validation feature names/order differ from training")
    effective_config = LambdaRankConfig() if config is None else config
    if not isinstance(effective_config, LambdaRankConfig):
        raise TreeRankerError("config must be a LambdaRankConfig")
    effective_backend = _default_backend() if backend is None else backend
    backend_identity = _backend_identity(effective_backend)

    canonical_labels = _labels(labels, expected=features.row_count, name="labels")
    grouped_features = cast(Float64Matrix, grouping.to_grouped(features.values))
    grouped_labels = cast(Int8Vector, grouping.to_grouped(canonical_labels))
    inner_features: Float64Matrix | None = None
    inner_labels: Int8Vector | None = None
    inner_groups: tuple[int, ...] | None = None
    early_stopping_rounds: int | None = None
    if inner_valid is not None:
        inner_features = cast(
            Float64Matrix,
            inner_valid.grouping.to_grouped(inner_valid.features.values),
        )
        inner_labels = cast(Int8Vector, inner_valid.grouping.to_grouped(inner_valid.labels))
        inner_groups = inner_valid.grouping.group_sizes
        early_stopping_rounds = effective_config.early_stopping_rounds
    request = BackendFitRequest(
        params=effective_config.parameters(),
        train_features=grouped_features,
        train_labels=grouped_labels,
        train_group_sizes=grouping.group_sizes,
        eval_at=EVAL_AT,
        inner_valid_features=inner_features,
        inner_valid_labels=inner_labels,
        inner_valid_group_sizes=inner_groups,
        early_stopping_rounds=early_stopping_rounds,
    )
    result = effective_backend.fit(request)
    if not isinstance(result, BackendFitResult):
        raise TreeRankerError("backend fit must return BackendFitResult")
    if result.best_iteration > effective_config.num_boost_round:
        raise TreeRankerError("backend best_iteration exceeds configured num_boost_round")

    return TreeRankerCheckpoint(
        training_phase=phase,
        feature_names=features.feature_names,
        training_feature_digest=features.digest,
        training_grouping_digest=grouping.digest,
        training_target_digest=_target_digest(canonical_labels, phase),
        inner_validation_digest=None if inner_valid is None else inner_valid.digest,
        config_digest=effective_config.digest,
        backend_identity=backend_identity,
        best_iteration=result.best_iteration,
        model_text=result.model_text,
    )


def _prediction_scores(value: object, *, expected: int) -> Float64Vector:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TreeRankerError(
            "backend predictions must be a finite one-dimensional vector"
        ) from exc
    if raw.ndim != 1:
        raise TreeRankerError("backend predictions must be one-dimensional")
    if raw.size != expected:
        raise TreeRankerError(f"backend returned {raw.size} predictions, expected {expected}")
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TreeRankerError("backend predictions must have a real numeric dtype")
    try:
        contiguous = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TreeRankerError("backend predictions must be representable as float64") from exc
    if not np.isfinite(contiguous).all():
        raise TreeRankerError("backend predictions must contain only finite values")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    frozen.setflags(write=False)
    return cast(Float64Vector, frozen)


def predict_lambdarank(
    checkpoint: TreeRankerCheckpoint,
    features: FeatureMatrix,
    *,
    phase: DataPhase,
    backend: LambdaRankBackend | None = None,
) -> TreeRankerPrediction:
    """Predict one finite score per canonical row without accepting a label argument."""

    if not isinstance(phase, DataPhase):
        raise TreeRankerError("phase must be a DataPhase")
    if phase not in _PREDICTION_PHASES:
        raise TreeRankerError("prediction is allowed only for inner_valid, outer_valid, or final")
    if not isinstance(checkpoint, TreeRankerCheckpoint):
        raise TreeRankerError("checkpoint must be a TreeRankerCheckpoint")
    if not isinstance(features, FeatureMatrix):
        raise TreeRankerError("features must be a trusted FeatureMatrix")
    if features.feature_names != checkpoint.feature_names:
        raise TreeRankerError("prediction feature names/order differ from the checkpoint schema")
    effective_backend = _default_backend() if backend is None else backend
    if _backend_identity(effective_backend) != checkpoint.backend_identity:
        raise TreeRankerError("prediction backend identity differs from the checkpoint backend")

    raw_scores = effective_backend.predict(
        model_text=checkpoint.model_text,
        features=features.values,
        num_iteration=checkpoint.best_iteration,
    )
    scores = _prediction_scores(raw_scores, expected=features.row_count)
    return TreeRankerPrediction(
        scores=scores,
        phase=phase,
        checkpoint_digest=checkpoint.digest,
        feature_digest=features.digest,
        prediction_digest=prediction_digest(scores),
    )


__all__ = [
    "DEFAULT_TREE_SEED",
    "DEFAULT_TREE_THREADS",
    "EVAL_AT",
    "PINNED_LIGHTGBM_VERSION",
    "TREE_RANKER_SCHEMA_VERSION",
    "BackendFitRequest",
    "BackendFitResult",
    "InnerValidationSet",
    "LambdaRankBackend",
    "LambdaRankConfig",
    "TreeRankerCheckpoint",
    "TreeRankerDependencyError",
    "TreeRankerError",
    "TreeRankerPrediction",
    "fit_lambdarank",
    "predict_lambdarank",
]

"""Deterministic five-field FM optimized with the GAUC-aligned pairwise objective.

The candidate deliberately shares the starter FM's five encoded fields, 16-factor float32
state, seeded initializer, dense L2 regularization, and dense Adam optimizer.  Its scientific
intervention is limited to drawing logged same-user positive/negative impressions and replacing
the pointwise objective with pairwise logistic loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import PredictionVector
from kuairand_agent.candidates.pairwise import (
    MAX_SAMPLED_PAIRS,
    GAUCPairSampler,
    PairwisePrimitiveError,
    pairwise_logistic_loss_and_gradient,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.capabilities import DataPhase

PAIRWISE_FM_SCHEMA_VERSION: Final = 1
STARTER_FIELDS: Final = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
FACTOR_DIM: Final = 16

type Int32Matrix = npt.NDArray[np.int32]
type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]
ZERO_BIAS: Final = np.float32(0.0)


class PairwiseFMError(ValueError):
    """Raised when candidate state or data violates the pairwise-FM contract."""


class EncodingProtocol(Protocol):
    """Narrow structural view of the trusted five-field encoding."""

    @property
    def digest(self) -> str: ...

    @property
    def training_inputs_digest(self) -> str: ...

    @property
    def field_names(self) -> tuple[str, ...]: ...

    @property
    def total_dim(self) -> int: ...

    def transform(self, inputs: CanonicalInputs) -> Int32Matrix: ...


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
        raise PairwiseFMError("pairwise FM identity must be finite canonical JSON") from exc


def _manifest_digest(domain: bytes, value: object) -> str:
    digest = hashlib.sha256(domain)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _require_digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PairwiseFMError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _uint32(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise PairwiseFMError(f"{name} must be a uint32-compatible integer")
    return value


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise PairwiseFMError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _finite_nonnegative(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PairwiseFMError(
            f"{name} must be a finite {'positive' if positive else 'non-negative'} float"
        )
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        raise PairwiseFMError(
            f"{name} must be a finite {'positive' if positive else 'non-negative'} float"
        )
    return result


def _immutable_float32(value: npt.NDArray[np.generic]) -> Float32Array:
    """Copy through immutable bytes so callers cannot re-enable WRITEABLE."""

    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f4")).reshape(
        contiguous.shape
    )
    frozen.setflags(write=False)
    return cast(Float32Array, frozen)


def _encoded_pair_matrix(value: object, name: str) -> Int32Matrix:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("int32")
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] != len(STARTER_FIELDS)
        or not value.flags.c_contiguous
    ):
        raise PairwiseFMError(f"{name} must be C-contiguous int32 shape (N, 5)")
    return cast(Int32Matrix, value)


def _model_state(V: object, W: object) -> tuple[Float32Array, Float32Array]:
    if (
        not isinstance(V, np.ndarray)
        or V.dtype != np.dtype("float32")
        or V.ndim != 2
        or V.shape[0] <= 0
        or V.shape[1] != FACTOR_DIM
        or not V.flags.c_contiguous
        or not np.isfinite(V).all()
    ):
        raise PairwiseFMError("V must be finite C-contiguous float32 shape (D, 16)")
    if (
        not isinstance(W, np.ndarray)
        or W.dtype != np.dtype("float32")
        or W.shape != (V.shape[0],)
        or not W.flags.c_contiguous
        or not np.isfinite(W).all()
    ):
        raise PairwiseFMError("W must be finite C-contiguous float32 shape (D,)")
    return cast(Float32Array, V), cast(Float32Array, W)


def _bias(value: object) -> np.float32:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PairwiseFMError("b must be a finite float32 scalar") from exc
    if raw.shape != () or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise PairwiseFMError("b must be a finite float32 scalar")
    result = np.float32(raw)
    if not np.isfinite(result):
        raise PairwiseFMError("b must be a finite float32 scalar")
    return result


def _fm_scores(
    X: Int32Matrix,
    V: Float32Array,
    W: Float32Array,
    b: np.float32 = ZERO_BIAS,
) -> Float32Array:
    embeddings = V[X]
    summed = embeddings.sum(axis=1, dtype=np.float32)
    # Preserve the organizer's reduction order byte-for-byte.  Moving the subtraction inside
    # the factor-axis sum is algebraically equivalent but changes float32 rounding.
    interactions = 0.5 * ((summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2)))
    scores = b + W[X].sum(axis=1) + interactions
    return np.ascontiguousarray(scores, dtype=np.float32)


def pairwise_fm_scores(
    X: object,
    *,
    V: object,
    W: object,
    b: object = np.float32(0.0),
) -> Float32Array:
    """Score one encoded matrix with exact organizer float32 reduction semantics."""

    encoded = _encoded_pair_matrix(X, "X")
    factors, linear = _model_state(V, W)
    if int(encoded.min()) < 0 or int(encoded.max()) >= factors.shape[0]:
        raise PairwiseFMError("encoded IDs must be non-negative and below model dimension")
    return _immutable_float32(_fm_scores(encoded, factors, linear, _bias(b)))


def _state_digest(
    domain: bytes,
    *,
    V: Float32Array,
    W: Float32Array,
    b: np.float32,
    identity: Mapping[str, object],
) -> str:
    digest = hashlib.sha256(domain)
    digest.update(_canonical_json(dict(identity)))
    digest.update(V.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(W.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(np.asarray(b, dtype="<f4").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class PairwiseFMInitialState:
    """Exact immutable pointwise-control initialization bytes."""

    V: Float32Array = field(repr=False)
    W: Float32Array = field(repr=False)
    b: np.float32
    seed: int
    digest: str

    def __init__(self, *, V: object, W: object, b: object, seed: int) -> None:
        factors, linear = _model_state(
            np.ascontiguousarray(V) if isinstance(V, np.ndarray) else V,
            np.ascontiguousarray(W) if isinstance(W, np.ndarray) else W,
        )
        frozen_v = _immutable_float32(factors)
        frozen_w = _immutable_float32(linear)
        bias = _bias(b)
        normalized_seed = _uint32(seed, "initialization seed")
        identity = {
            "schema_version": PAIRWISE_FM_SCHEMA_VERSION,
            "total_dim": int(frozen_w.size),
            "factor_dim": FACTOR_DIM,
            "seed": normalized_seed,
        }
        object.__setattr__(self, "V", frozen_v)
        object.__setattr__(self, "W", frozen_w)
        object.__setattr__(self, "b", bias)
        object.__setattr__(self, "seed", normalized_seed)
        object.__setattr__(
            self,
            "digest",
            _state_digest(
                b"kuairand-pairwise-fm-initial-state-v1\0",
                V=frozen_v,
                W=frozen_w,
                b=bias,
                identity=identity,
            ),
        )


def initialize_pairwise_fm(*, total_dim: int, seed: int) -> PairwiseFMInitialState:
    """Build the exact organizer-compatible seeded FM initialization."""

    dimension = _positive_int(total_dim, "total_dim", np.iinfo(np.int32).max)
    normalized_seed = _uint32(seed, "initialization seed")
    rng = np.random.default_rng(normalized_seed)
    return PairwiseFMInitialState(
        V=rng.normal(0, 0.01, (dimension, FACTOR_DIM)).astype(np.float32),
        W=np.zeros(dimension, dtype=np.float32),
        b=np.float32(0.0),
        seed=normalized_seed,
    )


@dataclass(frozen=True, slots=True)
class PairwiseFMGradientResult:
    """One mean-loss dense gradient, suitable for exact optimizer goldens."""

    loss: float
    pair_count: int
    V_gradient: Float32Array = field(repr=False)
    W_gradient: Float32Array = field(repr=False)


def pairwise_fm_batch_loss_and_gradients(
    positive_X: object,
    negative_X: object,
    *,
    V: object,
    W: object,
    l2: float,
) -> PairwiseFMGradientResult:
    """Return exact dense gradients for one encoded positive/negative batch.

    The data term is the mean ``softplus(-(score_pos-score_neg))``.  Dense L2 matches the
    organizer pointwise control: ``l2 * (sum(V**2) + sum(W**2)) / 2``.
    """

    positive = _encoded_pair_matrix(positive_X, "positive_X")
    negative = _encoded_pair_matrix(negative_X, "negative_X")
    if positive.shape != negative.shape:
        raise PairwiseFMError("positive_X and negative_X must have identical shape")
    factors, linear = _model_state(V, W)
    if isinstance(l2, bool) or not isinstance(l2, (int, float)):
        raise PairwiseFMError("l2 must be a finite non-negative float")
    l2_value = float(l2)
    if not math.isfinite(l2_value) or l2_value < 0.0:
        raise PairwiseFMError("l2 must be a finite non-negative float")
    if int(positive.min()) < 0 or int(negative.min()) < 0:
        raise PairwiseFMError("encoded IDs must be non-negative and below model dimension")
    if int(positive.max()) >= factors.shape[0] or int(negative.max()) >= factors.shape[0]:
        raise PairwiseFMError("encoded IDs must be non-negative and below model dimension")

    positive_scores = _fm_scores(positive, factors, linear)
    negative_scores = _fm_scores(negative, factors, linear)
    loss_result = pairwise_logistic_loss_and_gradient(positive_scores, negative_scores)
    positive_gradient = np.asarray(loss_result.positive_gradient, dtype=np.float32)
    negative_gradient = np.asarray(loss_result.negative_gradient, dtype=np.float32)

    V_gradient = np.multiply(factors, np.float32(l2_value), dtype=np.float32)
    W_gradient = np.multiply(linear, np.float32(l2_value), dtype=np.float32)
    for matrix, score_gradient in (
        (positive, positive_gradient),
        (negative, negative_gradient),
    ):
        embeddings = factors[matrix]
        summed = embeddings.sum(axis=1, dtype=np.float32)
        row_gradients = score_gradient[:, None, None] * (summed[:, None, :] - embeddings)
        np.add.at(V_gradient, matrix, row_gradients)
        np.add.at(W_gradient, matrix, score_gradient[:, None])

    loss = loss_result.loss
    if (
        not math.isfinite(loss)
        or not np.isfinite(V_gradient).all()
        or not np.isfinite(W_gradient).all()
    ):
        raise PairwiseFMError("pairwise FM loss or gradient became non-finite")
    return PairwiseFMGradientResult(
        loss=loss,
        pair_count=int(positive.shape[0]),
        V_gradient=_immutable_float32(V_gradient),
        W_gradient=_immutable_float32(W_gradient),
    )


@dataclass(frozen=True, slots=True, init=False)
class EncodedFMInputs:
    """Immutable phase-typed five-field matrix at the candidate execution seam."""

    values: Int32Matrix = field(repr=False)
    phase: DataPhase
    inputs_digest: str
    encoding_digest: str
    total_dim: int
    row_count: int
    digest: str

    def __init__(
        self,
        *,
        values: object,
        phase: DataPhase,
        inputs_digest: str,
        encoding_digest: str,
        total_dim: int,
    ) -> None:
        if not isinstance(phase, DataPhase):
            raise PairwiseFMError("encoded input phase must be a DataPhase")
        matrix = _encoded_pair_matrix(values, "encoded inputs")
        dimension = _positive_int(total_dim, "encoded total_dim", np.iinfo(np.int32).max)
        if int(matrix.min()) < 0 or int(matrix.max()) >= dimension:
            raise PairwiseFMError("encoded IDs must be non-negative and below total_dim")
        input_identity = _require_digest(inputs_digest, "inputs_digest")
        encoding_identity = _require_digest(encoding_digest, "encoding_digest")
        frozen_raw = np.frombuffer(matrix.tobytes(order="C"), dtype=np.dtype("<i4")).reshape(
            matrix.shape
        )
        frozen_raw.setflags(write=False)
        frozen = cast(Int32Matrix, frozen_raw)
        manifest = {
            "schema_version": PAIRWISE_FM_SCHEMA_VERSION,
            "phase": phase.value,
            "inputs_digest": input_identity,
            "encoding_digest": encoding_identity,
            "total_dim": dimension,
            "row_count": int(frozen.shape[0]),
            "shape": list(frozen.shape),
            "dtype": "<i4",
        }
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "inputs_digest", input_identity)
        object.__setattr__(self, "encoding_digest", encoding_identity)
        object.__setattr__(self, "total_dim", dimension)
        object.__setattr__(self, "row_count", int(frozen.shape[0]))
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-pairwise-fm-encoded-inputs-v1\0", manifest),
        )

    @classmethod
    def from_encoding(
        cls,
        encoding: EncodingProtocol,
        inputs: CanonicalInputs,
        *,
        phase: DataPhase,
    ) -> Self:
        """Bridge the trusted starter encoding into the artifact-oriented matrix contract."""

        if not isinstance(inputs, CanonicalInputs) or len(inputs) == 0:
            raise PairwiseFMError("inputs must be non-empty CanonicalInputs")
        if getattr(encoding, "field_names", None) != STARTER_FIELDS:
            raise PairwiseFMError("encoding fields differ from the five starter FM fields")
        encoding_digest = _require_digest(getattr(encoding, "digest", None), "encoding_digest")
        try:
            values = encoding.transform(inputs)
            total_dim = encoding.total_dim
        except Exception as exc:
            raise PairwiseFMError(f"cannot encode inputs: {exc}") from exc
        return cls(
            values=values,
            phase=phase,
            inputs_digest=inputs.digest,
            encoding_digest=encoding_digest,
            total_dim=total_dim,
        )


@dataclass(frozen=True, slots=True, init=False)
class PairwiseFMTrainingData:
    """Only label-bearing pairwise input, restricted to fitting phases before label access."""

    inputs: EncodedFMInputs
    labels: npt.NDArray[np.int8] = field(repr=False)
    user_ids: tuple[str, ...] = field(repr=False)
    training_targets_digest: str
    target_inputs_digest: str

    def __init__(
        self,
        *,
        inputs: EncodedFMInputs,
        labels: object,
        user_ids: object,
        training_targets_digest: str,
        target_inputs_digest: str,
    ) -> None:
        # This check must precede every observation, conversion, hash, repr, or length request
        # involving labels or user IDs from a non-training phase.
        if not isinstance(inputs, EncodedFMInputs):
            raise PairwiseFMError("training inputs must be EncodedFMInputs")
        if inputs.phase not in {DataPhase.TRAIN, DataPhase.INNER_TRAIN}:
            raise PairwiseFMError("labels are allowed only for train or inner_train")
        target_identity = _require_digest(training_targets_digest, "training_targets_digest")
        bound_identity = _require_digest(target_inputs_digest, "target_inputs_digest")
        if bound_identity != inputs.inputs_digest:
            raise PairwiseFMError("training targets are not aligned to training inputs")
        try:
            raw = np.asarray(labels)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PairwiseFMError("training labels must be aligned binary numeric values") from exc
        if (
            raw.ndim != 1
            or raw.shape[0] != inputs.row_count
            or raw.dtype.kind not in "biuf"
            or raw.dtype.kind == "b"
        ):
            raise PairwiseFMError("training labels must be aligned binary numeric values")
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise PairwiseFMError("training labels must be aligned binary numeric values")
        if isinstance(user_ids, (str, bytes)) or not isinstance(user_ids, Iterable):
            raise PairwiseFMError("training user_ids must be an aligned identity vector")
        try:
            normalized_users: tuple[object, ...] = tuple(user_ids)
        except TypeError as exc:
            raise PairwiseFMError("training user_ids must be an aligned identity vector") from exc
        if len(normalized_users) != inputs.row_count or any(
            type(value) is not str or not value or "\x00" in value for value in normalized_users
        ):
            raise PairwiseFMError("training user_ids must be aligned non-empty strings")
        label_bytes = np.ascontiguousarray(numeric, dtype=np.int8).tobytes(order="C")
        frozen_labels = np.frombuffer(label_bytes, dtype=np.int8)
        frozen_labels.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "labels", cast(npt.NDArray[np.int8], frozen_labels))
        object.__setattr__(self, "user_ids", cast(tuple[str, ...], normalized_users))
        object.__setattr__(self, "training_targets_digest", target_identity)
        object.__setattr__(self, "target_inputs_digest", bound_identity)


@dataclass(frozen=True, slots=True)
class PairwiseFMConfig:
    """Bounded pure-pairwise tunables plus frozen pointwise-control numerics."""

    seed: int = 0
    learning_rate: float = 0.001
    l2: float = 1e-6
    pair_batch_size: int = 8192
    pairs_per_epoch: int = 8192
    max_epochs: int = 40
    device_metadata: str | None = None
    factor_dim: int = field(init=False, default=FACTOR_DIM)
    predict_batch_size: int = field(init=False, default=200_000)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _uint32(self.seed, "seed")
        _finite_nonnegative(self.learning_rate, "learning_rate", positive=True)
        _finite_nonnegative(self.l2, "l2")
        _positive_int(self.pair_batch_size, "pair_batch_size", MAX_SAMPLED_PAIRS)
        _positive_int(self.pairs_per_epoch, "pairs_per_epoch", MAX_SAMPLED_PAIRS)
        _positive_int(self.max_epochs, "max_epochs", 1_000)
        if self.device_metadata is not None and (
            type(self.device_metadata) is not str
            or not self.device_metadata
            or len(self.device_metadata) > 120
            or "\x00" in self.device_metadata
            or "\n" in self.device_metadata
            or "\r" in self.device_metadata
        ):
            raise PairwiseFMError("device_metadata must be short single-line text or None")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-pairwise-fm-config-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": PAIRWISE_FM_SCHEMA_VERSION,
            "candidate_family": "pairwise_fm",
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "pair_batch_size": self.pair_batch_size,
            "pairs_per_epoch": self.pairs_per_epoch,
            "max_epochs": self.max_epochs,
            "predict_batch_size": self.predict_batch_size,
            "execution_device": "cpu",
            "device_metadata": self.device_metadata,
            "epoch_selection": "trusted_inner_fold_controller_only",
            "pointwise_control": {
                "fields": list(STARTER_FIELDS),
                "factor_dim": FACTOR_DIM,
                "initializer": "numpy.default_rng(seed).normal(0,0.01).astype(float32)",
                "optimizer": "dense_adam(beta1=0.9,beta2=0.999,epsilon=1e-8)",
                "precision": "float32",
                "changed_mechanism": ("gauc_aligned_same_user_pairwise_logistic_objective"),
            },
        }


@dataclass(frozen=True, slots=True)
class PairwiseFMEpochTrace:
    """Deterministic training evidence with no official metric fields."""

    epoch: int
    sample_seed: int
    batch_sizes: tuple[int, ...]
    optimizer_steps: int
    sampled_pairs: int
    mean_pairwise_loss: float
    regularization_value: float
    sampling_digest: str
    state_digest: str

    def __post_init__(self) -> None:
        _positive_int(self.epoch, "trace epoch", 1_000)
        _uint32(self.sample_seed, "trace sample_seed")
        if not self.batch_sizes or any(
            type(value) is not int or value <= 0 for value in self.batch_sizes
        ):
            raise PairwiseFMError("trace batch_sizes must contain positive integers")
        _positive_int(self.optimizer_steps, "trace optimizer_steps", 2**63 - 1)
        _positive_int(self.sampled_pairs, "trace sampled_pairs", 2**63 - 1)
        _finite_nonnegative(self.mean_pairwise_loss, "trace mean_pairwise_loss")
        _finite_nonnegative(self.regularization_value, "trace regularization_value")
        _require_digest(self.sampling_digest, "trace sampling_digest")
        _require_digest(self.state_digest, "trace state_digest")

    def manifest(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "sample_seed": self.sample_seed,
            "batch_sizes": list(self.batch_sizes),
            "optimizer_steps": self.optimizer_steps,
            "sampled_pairs": self.sampled_pairs,
            "mean_pairwise_data_loss": self.mean_pairwise_loss,
            "regularization_value": self.regularization_value,
            "sampling_digest": self.sampling_digest,
            "state_digest": self.state_digest,
        }


@dataclass(frozen=True, slots=True, init=False)
class PairwiseFMCheckpoint:
    """Immutable inference checkpoint with complete replay identity."""

    V: Float32Array = field(repr=False)
    W: Float32Array = field(repr=False)
    b: np.float32
    training_phase: DataPhase
    encoding_digest: str
    config_digest: str
    source_digest: str
    train_inputs_digest: str
    training_targets_digest: str
    initialization_digest: str
    seed: int
    epochs_completed: int
    optimizer_steps: int
    sampled_pairs: int
    digest: str

    def __init__(
        self,
        *,
        V: object,
        W: object,
        b: object,
        training_phase: DataPhase,
        encoding_digest: str,
        config_digest: str,
        source_digest: str,
        train_inputs_digest: str,
        training_targets_digest: str,
        initialization_digest: str,
        seed: int,
        epochs_completed: int,
        optimizer_steps: int,
        sampled_pairs: int,
    ) -> None:
        factors, linear = _model_state(
            np.ascontiguousarray(V) if isinstance(V, np.ndarray) else V,
            np.ascontiguousarray(W) if isinstance(W, np.ndarray) else W,
        )
        frozen_v = _immutable_float32(factors)
        frozen_w = _immutable_float32(linear)
        bias = _bias(b)
        if bias.tobytes() != b"\x00\x00\x00\x00":
            raise PairwiseFMError("pure pairwise checkpoint bias must remain exact positive zero")
        if training_phase not in {DataPhase.TRAIN, DataPhase.INNER_TRAIN}:
            raise PairwiseFMError("checkpoint training_phase must be train or inner_train")
        identities = {
            "encoding_digest": _require_digest(encoding_digest, "encoding_digest"),
            "config_digest": _require_digest(config_digest, "config_digest"),
            "source_digest": _require_digest(source_digest, "source_digest"),
            "train_inputs_digest": _require_digest(train_inputs_digest, "train_inputs_digest"),
            "training_targets_digest": _require_digest(
                training_targets_digest, "training_targets_digest"
            ),
            "initialization_digest": _require_digest(
                initialization_digest, "initialization_digest"
            ),
        }
        normalized_seed = _uint32(seed, "checkpoint seed")
        completed = _positive_int(epochs_completed, "checkpoint epochs_completed", 1_000)
        steps = _positive_int(optimizer_steps, "checkpoint optimizer_steps", 2**63 - 1)
        pairs = _positive_int(sampled_pairs, "checkpoint sampled_pairs", 2**63 - 1)
        object.__setattr__(self, "V", frozen_v)
        object.__setattr__(self, "W", frozen_w)
        object.__setattr__(self, "b", bias)
        object.__setattr__(self, "training_phase", training_phase)
        for name, value in identities.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "seed", normalized_seed)
        object.__setattr__(self, "epochs_completed", completed)
        object.__setattr__(self, "optimizer_steps", steps)
        object.__setattr__(self, "sampled_pairs", pairs)
        object.__setattr__(
            self,
            "digest",
            _state_digest(
                b"kuairand-pairwise-fm-checkpoint-v1\0",
                V=frozen_v,
                W=frozen_w,
                b=bias,
                identity=self.identity_manifest(),
            ),
        )

    @property
    def total_dim(self) -> int:
        return int(self.W.size)

    @property
    def factor_dim(self) -> int:
        return int(self.V.shape[1])

    def identity_manifest(self) -> dict[str, object]:
        return {
            "schema_version": PAIRWISE_FM_SCHEMA_VERSION,
            "model": "gauc_aligned_pairwise_fm",
            "V_shape": list(self.V.shape),
            "W_shape": list(self.W.shape),
            "dtype": "<f4",
            "training_phase": self.training_phase.value,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "source_digest": self.source_digest,
            "train_inputs_digest": self.train_inputs_digest,
            "training_targets_digest": self.training_targets_digest,
            "initialization_digest": self.initialization_digest,
            "seed": self.seed,
            "epochs_completed": self.epochs_completed,
            "optimizer_steps": self.optimizer_steps,
            "sampled_pairs": self.sampled_pairs,
        }

    def manifest(self) -> dict[str, object]:
        return {**self.identity_manifest(), "checkpoint_digest": self.digest}


@dataclass(frozen=True, slots=True)
class PairwiseFMRun:
    """Replayable train-only candidate result, deliberately devoid of official metrics."""

    checkpoint: PairwiseFMCheckpoint
    trace: tuple[PairwiseFMEpochTrace, ...]
    eligible_user_count: int
    eligible_positive_count: int
    stored_row_index_count: int
    pair_space_size: int
    pairs_per_epoch: int
    train_inputs_digest: str
    training_targets_digest: str
    encoding_digest: str
    config_digest: str
    source_digest: str
    initialization_digest: str
    logical_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.trace:
            raise PairwiseFMError("pairwise FM run requires a non-empty trace")
        for name in (
            "train_inputs_digest",
            "training_targets_digest",
            "encoding_digest",
            "config_digest",
            "source_digest",
            "initialization_digest",
        ):
            _require_digest(getattr(self, name), name)
        if len(self.trace) != self.checkpoint.epochs_completed:
            raise PairwiseFMError("run trace and checkpoint epoch counts differ")
        if tuple(item.epoch for item in self.trace) != tuple(range(1, len(self.trace) + 1)):
            raise PairwiseFMError("run trace epochs must be contiguous and one-based")
        if self.trace[-1].optimizer_steps != self.checkpoint.optimizer_steps:
            raise PairwiseFMError("run optimizer-step evidence differs from checkpoint")
        if self.trace[-1].sampled_pairs != self.checkpoint.sampled_pairs:
            raise PairwiseFMError("run pair-count evidence differs from checkpoint")
        if self.checkpoint.encoding_digest != self.encoding_digest:
            raise PairwiseFMError("run encoding identity differs from checkpoint")
        if self.checkpoint.config_digest != self.config_digest:
            raise PairwiseFMError("run config identity differs from checkpoint")
        if self.checkpoint.source_digest != self.source_digest:
            raise PairwiseFMError("run source identity differs from checkpoint")
        if self.checkpoint.train_inputs_digest != self.train_inputs_digest:
            raise PairwiseFMError("run input identity differs from checkpoint")
        if self.checkpoint.training_targets_digest != self.training_targets_digest:
            raise PairwiseFMError("run target identity differs from checkpoint")
        if self.checkpoint.initialization_digest != self.initialization_digest:
            raise PairwiseFMError("run initialization identity differs from checkpoint")
        for name in (
            "eligible_user_count",
            "eligible_positive_count",
            "stored_row_index_count",
            "pair_space_size",
            "pairs_per_epoch",
        ):
            _positive_int(getattr(self, name), f"run {name}", 2**63 - 1)
        prior_steps = 0
        prior_pairs = 0
        for item in self.trace:
            if item.optimizer_steps - prior_steps != len(item.batch_sizes):
                raise PairwiseFMError("trace optimizer steps do not match batch sizes")
            if item.sampled_pairs - prior_pairs != sum(item.batch_sizes):
                raise PairwiseFMError("trace sampled pairs do not match batch sizes")
            if sum(item.batch_sizes) != self.pairs_per_epoch:
                raise PairwiseFMError("trace epoch pair count differs from configuration")
            prior_steps = item.optimizer_steps
            prior_pairs = item.sampled_pairs
        object.__setattr__(
            self,
            "logical_digest",
            _manifest_digest(b"kuairand-pairwise-fm-run-v1\0", self.logical_manifest()),
        )

    def logical_manifest(self) -> dict[str, object]:
        return {
            "schema_version": PAIRWISE_FM_SCHEMA_VERSION,
            "candidate_family": "pairwise_fm",
            "training_objective": "gauc_aligned_pairwise_logistic",
            "checkpoint": self.checkpoint.manifest(),
            "trace": [item.manifest() for item in self.trace],
            "diagnostics": {
                "eligible_user_count": self.eligible_user_count,
                "eligible_positive_count": self.eligible_positive_count,
                "stored_row_index_count": self.stored_row_index_count,
                "pair_space_size": self.pair_space_size,
                "pairs_per_epoch": self.pairs_per_epoch,
            },
            "train_inputs_digest": self.train_inputs_digest,
            "training_targets_digest": self.training_targets_digest,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "source_digest": self.source_digest,
            "initialization_digest": self.initialization_digest,
        }

    def candidate_result_manifest(self) -> dict[str, object]:
        return {
            **self.logical_manifest(),
            "checkpoint_digest": self.checkpoint.digest,
            "logical_digest": self.logical_digest,
        }


@dataclass(frozen=True, slots=True)
class PairwiseFMPrediction:
    """Strict label-free prediction result with request and checkpoint identity."""

    vector: PredictionVector
    phase: DataPhase
    checkpoint_digest: str
    inputs_digest: str
    encoding_digest: str
    config_digest: str
    source_digest: str
    logical_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.phase not in {DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL}:
            raise PairwiseFMError("prediction phase must be label-free inference")
        for name in (
            "checkpoint_digest",
            "inputs_digest",
            "encoding_digest",
            "config_digest",
            "source_digest",
        ):
            _require_digest(getattr(self, name), name)
        object.__setattr__(
            self,
            "logical_digest",
            _manifest_digest(b"kuairand-pairwise-fm-prediction-v1\0", self.manifest()),
        )

    @property
    def scores(self) -> Float64Array:
        return self.vector.scores

    @property
    def prediction_digest(self) -> str:
        return self.vector.digest

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": PAIRWISE_FM_SCHEMA_VERSION,
            "kind": "prediction",
            "phase": self.phase.value,
            "expected_count": self.vector.row_count,
            "dtype": "<f8",
            "prediction_digest": self.vector.digest,
            "checkpoint_digest": self.checkpoint_digest,
            "inputs_digest": self.inputs_digest,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "source_digest": self.source_digest,
        }


def _sampling_digest(
    *,
    sample_seed: int,
    positive_indices: npt.NDArray[np.int64],
    negative_indices: npt.NDArray[np.int64],
) -> str:
    digest = hashlib.sha256(b"kuairand-pairwise-fm-sampling-v1\0")
    digest.update(sample_seed.to_bytes(4, "little"))
    digest.update(positive_indices.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(negative_indices.astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _regularization_value(V: Float32Array, W: Float32Array, l2: float) -> float:
    value = np.float32(0.5 * l2) * (
        np.sum(V * V, dtype=np.float32) + np.sum(W * W, dtype=np.float32)
    )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PairwiseFMError("pairwise FM regularization became non-finite")
    return result


class PairwiseFMAdapter:
    """Reference pure-pairwise trainer and label-free predictor for generated seed source."""

    def __init__(
        self,
        *,
        source_digest: str,
        config: PairwiseFMConfig | None = None,
    ) -> None:
        self.source_digest = _require_digest(source_digest, "source_digest")
        self.config = PairwiseFMConfig() if config is None else config
        if not isinstance(self.config, PairwiseFMConfig):
            raise PairwiseFMError("config must be PairwiseFMConfig")

    def fit(self, training: PairwiseFMTrainingData, /) -> PairwiseFMRun:
        """Train for a fixed epoch count selected only by the trusted inner-fold controller."""

        if not isinstance(training, PairwiseFMTrainingData):
            raise PairwiseFMError("training must be PairwiseFMTrainingData")
        try:
            sampler = GAUCPairSampler(
                training.user_ids,
                training.labels,
                phase=training.inputs.phase,
            )
        except PairwisePrimitiveError as exc:
            raise PairwiseFMError(f"cannot build GAUC pair sampler: {exc}") from exc
        initial = initialize_pairwise_fm(
            total_dim=training.inputs.total_dim,
            seed=self.config.seed,
        )
        V = np.array(initial.V, dtype=np.float32, copy=True)
        W = np.array(initial.W, dtype=np.float32, copy=True)
        b = np.float32(initial.b)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        sampling_rng = np.random.default_rng(self.config.seed)
        optimizer_steps = 0
        sampled_pairs = 0
        trace: list[PairwiseFMEpochTrace] = []

        for epoch in range(1, self.config.max_epochs + 1):
            sample_seed = int(sampling_rng.integers(0, 2**32, dtype=np.uint64))
            sampled = sampler.sample(self.config.pairs_per_epoch, seed=sample_seed)
            batch_sizes: list[int] = []
            weighted_loss = 0.0
            for start in range(0, sampled.pair_count, self.config.pair_batch_size):
                stop = min(start + self.config.pair_batch_size, sampled.pair_count)
                positive_rows = sampled.positive_indices[start:stop]
                negative_rows = sampled.negative_indices[start:stop]
                gradients = pairwise_fm_batch_loss_and_gradients(
                    training.inputs.values[positive_rows],
                    training.inputs.values[negative_rows],
                    V=V,
                    W=W,
                    l2=self.config.l2,
                )
                batch_size = gradients.pair_count
                batch_sizes.append(batch_size)
                weighted_loss += gradients.loss * batch_size
                optimizer_steps += 1
                for parameter, gradient, first_moment, second_moment in (
                    (V, gradients.V_gradient, mV, vV),
                    (W, gradients.W_gradient, mW, vW),
                ):
                    first_moment *= 0.9
                    first_moment += 0.1 * gradient
                    second_moment *= 0.999
                    second_moment += 0.001 * (gradient * gradient)
                    parameter -= (
                        self.config.learning_rate
                        * (first_moment / (1.0 - 0.9**optimizer_steps))
                        / (np.sqrt(second_moment / (1.0 - 0.999**optimizer_steps)) + 1e-8)
                    )
                if not np.isfinite(V).all() or not np.isfinite(W).all():
                    raise PairwiseFMError("pairwise FM optimizer produced non-finite state")
                if b.tobytes() != b"\x00\x00\x00\x00":
                    raise PairwiseFMError("pure pairwise bias changed unexpectedly")
            sampled_pairs += sampled.pair_count
            identity = {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "sampled_pairs": sampled_pairs,
            }
            trace.append(
                PairwiseFMEpochTrace(
                    epoch=epoch,
                    sample_seed=sample_seed,
                    batch_sizes=tuple(batch_sizes),
                    optimizer_steps=optimizer_steps,
                    sampled_pairs=sampled_pairs,
                    mean_pairwise_loss=weighted_loss / sampled.pair_count,
                    regularization_value=_regularization_value(V, W, self.config.l2),
                    sampling_digest=_sampling_digest(
                        sample_seed=sample_seed,
                        positive_indices=sampled.positive_indices,
                        negative_indices=sampled.negative_indices,
                    ),
                    state_digest=_state_digest(
                        b"kuairand-pairwise-fm-epoch-state-v1\0",
                        V=V,
                        W=W,
                        b=b,
                        identity=identity,
                    ),
                )
            )

        checkpoint = PairwiseFMCheckpoint(
            V=V,
            W=W,
            b=b,
            training_phase=training.inputs.phase,
            encoding_digest=training.inputs.encoding_digest,
            config_digest=self.config.digest,
            source_digest=self.source_digest,
            train_inputs_digest=training.inputs.inputs_digest,
            training_targets_digest=training.training_targets_digest,
            initialization_digest=initial.digest,
            seed=self.config.seed,
            epochs_completed=self.config.max_epochs,
            optimizer_steps=optimizer_steps,
            sampled_pairs=sampled_pairs,
        )
        return PairwiseFMRun(
            checkpoint=checkpoint,
            trace=tuple(trace),
            eligible_user_count=sampler.eligible_user_count,
            eligible_positive_count=sampler.eligible_positive_count,
            stored_row_index_count=sampler.stored_row_index_count,
            pair_space_size=sampler.pair_space_size,
            pairs_per_epoch=self.config.pairs_per_epoch,
            train_inputs_digest=training.inputs.inputs_digest,
            training_targets_digest=training.training_targets_digest,
            encoding_digest=training.inputs.encoding_digest,
            config_digest=self.config.digest,
            source_digest=self.source_digest,
            initialization_digest=initial.digest,
        )

    def predict(
        self,
        checkpoint: PairwiseFMCheckpoint,
        inputs: EncodedFMInputs,
        /,
        *,
        expected_prediction_digest: str | None = None,
    ) -> PairwiseFMPrediction:
        """Predict from encoded label-free inputs without accepting outcomes or a scorer."""

        if not isinstance(inputs, EncodedFMInputs) or inputs.phase not in {
            DataPhase.INNER_VALID,
            DataPhase.OUTER_VALID,
            DataPhase.FINAL,
        }:
            raise PairwiseFMError("prediction inputs must have a label-free inference phase")
        if not isinstance(checkpoint, PairwiseFMCheckpoint):
            raise PairwiseFMError("checkpoint must be PairwiseFMCheckpoint")
        if checkpoint.config_digest != self.config.digest or checkpoint.seed != self.config.seed:
            raise PairwiseFMError("checkpoint config identity does not match adapter")
        if checkpoint.source_digest != self.source_digest:
            raise PairwiseFMError("checkpoint source identity does not match adapter")
        if checkpoint.encoding_digest != inputs.encoding_digest:
            raise PairwiseFMError("checkpoint encoding identity does not match inputs")
        if checkpoint.total_dim != inputs.total_dim:
            raise PairwiseFMError("checkpoint dimension does not match inputs")
        chunks = [
            _fm_scores(
                inputs.values[start : start + self.config.predict_batch_size],
                checkpoint.V,
                checkpoint.W,
                checkpoint.b,
            )
            for start in range(0, inputs.row_count, self.config.predict_batch_size)
        ]
        vector = PredictionVector(np.concatenate(chunks))
        if expected_prediction_digest is not None:
            expected = _require_digest(
                expected_prediction_digest,
                "expected prediction digest",
            )
            if vector.digest != expected:
                raise PairwiseFMError("expected prediction digest does not match replay")
        return PairwiseFMPrediction(
            vector=vector,
            phase=inputs.phase,
            checkpoint_digest=checkpoint.digest,
            inputs_digest=inputs.inputs_digest,
            encoding_digest=inputs.encoding_digest,
            config_digest=self.config.digest,
            source_digest=self.source_digest,
        )


# A shorter family name is useful to generated candidate seed code without changing the API.
PairwiseFM = PairwiseFMAdapter

"""Deterministic, corruption-checked artifacts for the trusted starter FM."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.scoring.submission import prediction_digest

ARTIFACT_SCHEMA_VERSION: Final = 1
_DIGEST_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_MEMBERS: Final = ("V.npy", "W.npy", "b.npy", "metadata.npy")
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_MAX_CHECKPOINT_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_METADATA_BYTES: Final = 64 * 1024
_OFFICIAL_FACTOR_DIM: Final = 16
_OFFICIAL_MAX_EPOCHS: Final = 40

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]


class BaselineArtifactError(RuntimeError):
    """Raised when a baseline artifact cannot be safely written or verified."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise BaselineArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
    dimensions: int,
    name: str,
) -> Float32Array | Float64Array:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselineArtifactError(f"{name} must contain real numeric values") from exc
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise BaselineArtifactError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselineArtifactError(f"{name} must be representable as {dtype}") from exc
    if array.ndim != dimensions:
        raise BaselineArtifactError(f"{name} must be {dimensions}-dimensional")
    if array.size == 0:
        raise BaselineArtifactError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise BaselineArtifactError(f"{name} must contain only finite values")
    canonical = np.ascontiguousarray(array, dtype=dtype)
    frozen = np.frombuffer(canonical.tobytes(order="C"), dtype=dtype).reshape(canonical.shape)
    frozen.setflags(write=False)
    return cast(Float32Array | Float64Array, frozen)


def _checkpoint_digest(checkpoint: StarterFMCheckpoint) -> str:
    digest = hashlib.sha256()
    digest.update(b"kuairand-starter-fm-checkpoint-v1\0")
    digest.update(_canonical_json(checkpoint.identity_manifest()))
    digest.update(checkpoint.V.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(checkpoint.W.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(np.asarray(checkpoint.b, dtype="<f4").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class StarterFMCheckpoint:
    """Immutable best-restored FM weights and their exact training identity."""

    V: Float32Array = field(repr=False)
    W: Float32Array = field(repr=False)
    b: np.float32
    encoding_digest: str
    config_digest: str
    starter_manifest_digest: str
    seed: int
    best_epoch: int
    epochs_completed: int
    optimizer_steps: int
    digest: str

    def __init__(
        self,
        *,
        V: object,
        W: object,
        b: object,
        encoding_digest: str,
        config_digest: str,
        starter_manifest_digest: str,
        seed: int,
        best_epoch: int,
        epochs_completed: int,
        optimizer_steps: int,
    ) -> None:
        factors = cast(
            Float32Array,
            _immutable_array(V, dtype=np.dtype("<f4"), dimensions=2, name="checkpoint V"),
        )
        linear = cast(
            Float32Array,
            _immutable_array(W, dtype=np.dtype("<f4"), dimensions=1, name="checkpoint W"),
        )
        if factors.shape[0] != linear.shape[0]:
            raise BaselineArtifactError("checkpoint V and W dimensions must agree")
        if factors.shape[1] != _OFFICIAL_FACTOR_DIM:
            raise BaselineArtifactError("checkpoint V must use the official factor dimension 16")
        try:
            raw_bias = np.asarray(b)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BaselineArtifactError("checkpoint b must be a finite float32 scalar") from exc
        if raw_bias.shape != () or raw_bias.dtype.kind not in "iuf" or raw_bias.dtype.kind == "b":
            raise BaselineArtifactError("checkpoint b must be a finite float32 scalar")
        try:
            bias_array = np.asarray(raw_bias, dtype=np.dtype("<f4"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise BaselineArtifactError("checkpoint b must be a finite float32 scalar") from exc
        if not math.isfinite(float(bias_array)):
            raise BaselineArtifactError("checkpoint b must be a finite float32 scalar")
        bias = np.float32(bias_array[()])

        encoding = _require_digest(encoding_digest, "encoding_digest")
        config = _require_digest(config_digest, "config_digest")
        starter_manifest = _require_digest(
            starter_manifest_digest,
            "starter_manifest_digest",
        )
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise BaselineArtifactError("checkpoint seed must be a uint32-compatible integer")
        for name, value in (
            ("best_epoch", best_epoch),
            ("epochs_completed", epochs_completed),
            ("optimizer_steps", optimizer_steps),
        ):
            if type(value) is not int or value <= 0:
                raise BaselineArtifactError(f"checkpoint {name} must be a positive integer")
        if best_epoch > epochs_completed:
            raise BaselineArtifactError("checkpoint best_epoch cannot exceed epochs_completed")
        if epochs_completed > _OFFICIAL_MAX_EPOCHS:
            raise BaselineArtifactError("checkpoint epochs_completed exceeds the official maximum")
        if optimizer_steps < epochs_completed or optimizer_steps % epochs_completed != 0:
            raise BaselineArtifactError(
                "checkpoint optimizer_steps must contain a fixed positive batch count per epoch"
            )

        object.__setattr__(self, "V", factors)
        object.__setattr__(self, "W", linear)
        object.__setattr__(self, "b", bias)
        object.__setattr__(self, "encoding_digest", encoding)
        object.__setattr__(self, "config_digest", config)
        object.__setattr__(self, "starter_manifest_digest", starter_manifest)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "best_epoch", best_epoch)
        object.__setattr__(self, "epochs_completed", epochs_completed)
        object.__setattr__(self, "optimizer_steps", optimizer_steps)
        object.__setattr__(self, "digest", _checkpoint_digest(self))

    @property
    def total_dim(self) -> int:
        return int(self.W.shape[0])

    @property
    def factor_dim(self) -> int:
        return int(self.V.shape[1])

    @property
    def checkpoint_digest(self) -> str:
        return self.digest

    def identity_manifest(self) -> dict[str, object]:
        """Return metadata covered by the weight digest, without numeric weights."""

        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model": "organizer_numpy_fm",
            "V_shape": list(self.V.shape),
            "W_shape": list(self.W.shape),
            "dtype": "<f4",
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "starter_manifest_digest": self.starter_manifest_digest,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "epochs_completed": self.epochs_completed,
            "optimizer_steps": self.optimizer_steps,
        }

    def manifest(self) -> dict[str, object]:
        return {**self.identity_manifest(), "checkpoint_digest": self.digest}


@dataclass(frozen=True, slots=True, init=False)
class PredictionVector:
    """Immutable finite float64 scores with the submission-layer digest."""

    scores: Float64Array = field(repr=False)
    digest: str

    def __init__(self, scores: Iterable[object] | npt.NDArray[np.generic]) -> None:
        try:
            raw = (
                np.asarray(scores) if isinstance(scores, np.ndarray) else np.asarray(tuple(scores))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise BaselineArtifactError(
                "predictions must be a one-dimensional numeric vector"
            ) from exc
        if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
            raise BaselineArtifactError("predictions must be a non-empty numeric vector")
        values = cast(
            Float64Array,
            _immutable_array(raw, dtype=np.dtype("<f8"), dimensions=1, name="predictions"),
        )
        object.__setattr__(self, "scores", values)
        object.__setattr__(self, "digest", prediction_digest(values))

    @property
    def row_count(self) -> int:
        return int(self.scores.shape[0])

    @property
    def prediction_digest(self) -> str:
        return self.digest

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "row_count": self.row_count,
            "dtype": "<f8",
            "prediction_digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    path: Path
    file_sha256: str
    checkpoint_digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_sha256", _require_digest(self.file_sha256, "file_sha256"))
        object.__setattr__(
            self,
            "checkpoint_digest",
            _require_digest(self.checkpoint_digest, "checkpoint_digest"),
        )
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise BaselineArtifactError("checkpoint artifact size_bytes must be positive")

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    path: Path
    file_sha256: str
    prediction_digest: str
    row_count: int
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_sha256", _require_digest(self.file_sha256, "file_sha256"))
        object.__setattr__(
            self,
            "prediction_digest",
            _require_digest(self.prediction_digest, "prediction_digest"),
        )
        if type(self.row_count) is not int or self.row_count <= 0:
            raise BaselineArtifactError("prediction artifact row_count must be positive")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise BaselineArtifactError("prediction artifact size_bytes must be positive")

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "prediction_digest": self.prediction_digest,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
        }


def file_sha256(path: Path | str) -> str:
    """Hash exact artifact bytes without trusting a stored manifest."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise BaselineArtifactError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _npy_bytes(array: npt.NDArray[np.generic]) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _checkpoint_bytes(checkpoint: StarterFMCheckpoint) -> bytes:
    metadata = _canonical_json(checkpoint.manifest())
    members = (
        ("V.npy", checkpoint.V.astype("<f4", copy=False)),
        ("W.npy", checkpoint.W.astype("<f4", copy=False)),
        ("b.npy", np.asarray(checkpoint.b, dtype="<f4")),
        ("metadata.npy", np.frombuffer(metadata, dtype=np.uint8)),
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in members:
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array))
    return output.getvalue()


def _atomic_no_replace(path: Path, payload: bytes) -> None:
    if not payload:
        raise BaselineArtifactError("refusing to write an empty artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BaselineArtifactError(
                f"artifact already exists and cannot be overwritten: {path}"
            ) from exc
        except OSError as exc:
            raise BaselineArtifactError(
                f"cannot atomically install artifact {path}: {exc}"
            ) from exc
        directory_descriptor = -1
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(directory_descriptor)
        except OSError as exc:
            try:
                temporary_status = temporary.stat(follow_symlinks=False)
                destination_status = path.stat(follow_symlinks=False)
                if os.path.samestat(temporary_status, destination_status):
                    path.unlink()
            except OSError:
                pass
            raise BaselineArtifactError(
                f"cannot make installed artifact directory durable: {path.parent}"
            ) from exc
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def save_checkpoint(path: Path | str, checkpoint: StarterFMCheckpoint) -> CheckpointArtifact:
    """Write one byte-deterministic NPZ checkpoint without overwriting a prior artifact."""

    if not isinstance(checkpoint, StarterFMCheckpoint):
        raise BaselineArtifactError("checkpoint must be StarterFMCheckpoint")
    destination = Path(path)
    if destination.suffix != ".npz":
        raise BaselineArtifactError("checkpoint path must end in .npz")
    payload = _checkpoint_bytes(checkpoint)
    _atomic_no_replace(destination, payload)
    return CheckpointArtifact(
        path=destination.resolve(),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_digest=checkpoint.digest,
        size_bytes=len(payload),
    )


def _checked_payload(path: Path | str, expected_file_sha256: str | None) -> bytes:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise BaselineArtifactError(
            f"artifact must be a readable regular non-symlink file: {source}"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            status = os.fstat(handle.fileno())
            if not stat.S_ISREG(status.st_mode):
                raise BaselineArtifactError(
                    f"artifact must be a regular non-symlink file: {source}"
                )
            if status.st_size <= 0 or status.st_size > _MAX_CHECKPOINT_BYTES:
                raise BaselineArtifactError("artifact size is outside the supported bound")
            payload = handle.read(_MAX_CHECKPOINT_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) == 0 or len(payload) > _MAX_CHECKPOINT_BYTES:
        raise BaselineArtifactError("artifact size is outside the supported bound")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and observed != _require_digest(
        expected_file_sha256, "expected_file_sha256"
    ):
        raise BaselineArtifactError("artifact file SHA-256 mismatch")
    return payload


def _metadata_from_array(array: npt.NDArray[np.generic]) -> dict[str, object]:
    if array.ndim != 1 or array.dtype != np.dtype("uint8") or array.size > _MAX_METADATA_BYTES:
        raise BaselineArtifactError("checkpoint metadata array is invalid")
    try:
        decoded = bytes(array).decode("ascii")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineArtifactError("checkpoint metadata is not canonical JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != bytes(array):
        raise BaselineArtifactError("checkpoint metadata is not a canonical JSON object")
    return cast(dict[str, object], value)


def load_checkpoint(
    path: Path | str,
    *,
    expected_file_sha256: str | None = None,
    expected_checkpoint_digest: str | None = None,
    expected_encoding_digest: str | None = None,
    expected_starter_manifest_digest: str | None = None,
    expected_config_digest: str | None = None,
    expected_seed: int | None = None,
) -> StarterFMCheckpoint:
    """Load an exact non-pickle checkpoint and verify every requested identity."""

    payload = _checked_payload(path, expected_file_sha256)
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != _CHECKPOINT_MEMBERS:
                raise BaselineArtifactError(
                    "checkpoint NPZ members are missing, duplicated, or reordered"
                )
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.file_size < 0
                or info.file_size > _MAX_CHECKPOINT_BYTES
                for info in infos
            ):
                raise BaselineArtifactError("checkpoint NPZ member encoding is invalid")
            if sum(info.file_size for info in infos) > _MAX_CHECKPOINT_BYTES:
                raise BaselineArtifactError("checkpoint NPZ payload exceeds the supported bound")

        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(f"{name}.npy" for name in archive.files) != _CHECKPOINT_MEMBERS:
                raise BaselineArtifactError("checkpoint arrays do not match the exact schema")
            V = np.asarray(archive["V"])
            W = np.asarray(archive["W"])
            b = np.asarray(archive["b"])
            metadata = _metadata_from_array(np.asarray(archive["metadata"]))
    except BaselineArtifactError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        raise BaselineArtifactError(f"cannot decode checkpoint artifact: {exc}") from exc

    if V.dtype != np.dtype("float32") or W.dtype != np.dtype("float32"):
        raise BaselineArtifactError("checkpoint weights must use exact float32 storage")
    if b.shape != () or b.dtype != np.dtype("float32"):
        raise BaselineArtifactError("checkpoint bias must be one float32 scalar")
    required = {
        "schema_version",
        "model",
        "V_shape",
        "W_shape",
        "dtype",
        "encoding_digest",
        "config_digest",
        "starter_manifest_digest",
        "seed",
        "best_epoch",
        "epochs_completed",
        "optimizer_steps",
        "checkpoint_digest",
    }
    if set(metadata) != required:
        raise BaselineArtifactError("checkpoint metadata fields do not match the exact schema")
    try:
        checkpoint = StarterFMCheckpoint(
            V=V,
            W=W,
            b=np.float32(b),
            encoding_digest=cast(str, metadata["encoding_digest"]),
            config_digest=cast(str, metadata["config_digest"]),
            starter_manifest_digest=cast(str, metadata["starter_manifest_digest"]),
            seed=cast(int, metadata["seed"]),
            best_epoch=cast(int, metadata["best_epoch"]),
            epochs_completed=cast(int, metadata["epochs_completed"]),
            optimizer_steps=cast(int, metadata["optimizer_steps"]),
        )
    except (TypeError, ValueError, BaselineArtifactError) as exc:
        raise BaselineArtifactError(f"checkpoint metadata or values are invalid: {exc}") from exc
    if checkpoint.manifest() != metadata:
        raise BaselineArtifactError(
            "checkpoint logical digest or manifest does not match its values"
        )
    if expected_checkpoint_digest is not None and checkpoint.digest != _require_digest(
        expected_checkpoint_digest, "expected_checkpoint_digest"
    ):
        raise BaselineArtifactError("checkpoint logical digest mismatch")
    if expected_encoding_digest is not None and checkpoint.encoding_digest != _require_digest(
        expected_encoding_digest, "expected_encoding_digest"
    ):
        raise BaselineArtifactError("checkpoint encoding digest mismatch")
    if (
        expected_starter_manifest_digest is not None
        and checkpoint.starter_manifest_digest
        != _require_digest(
            expected_starter_manifest_digest,
            "expected_starter_manifest_digest",
        )
    ):
        raise BaselineArtifactError("checkpoint starter manifest digest mismatch")
    if expected_config_digest is not None and checkpoint.config_digest != _require_digest(
        expected_config_digest,
        "expected_config_digest",
    ):
        raise BaselineArtifactError("checkpoint config digest mismatch")
    if expected_seed is not None:
        if type(expected_seed) is not int or not 0 <= expected_seed <= 2**32 - 1:
            raise BaselineArtifactError("expected_seed must be a uint32-compatible integer")
        if checkpoint.seed != expected_seed:
            raise BaselineArtifactError("checkpoint seed mismatch")
    return checkpoint


def save_predictions(path: Path | str, predictions: PredictionVector) -> PredictionArtifact:
    """Write deterministic float64 NPY predictions without overwrite."""

    if not isinstance(predictions, PredictionVector):
        raise BaselineArtifactError("predictions must be PredictionVector")
    destination = Path(path)
    if destination.suffix != ".npy":
        raise BaselineArtifactError("prediction path must end in .npy")
    payload = _npy_bytes(predictions.scores.astype("<f8", copy=False))
    _atomic_no_replace(destination, payload)
    return PredictionArtifact(
        path=destination.resolve(),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        prediction_digest=predictions.digest,
        row_count=predictions.row_count,
        size_bytes=len(payload),
    )


def load_predictions(
    path: Path | str,
    *,
    expected_file_sha256: str | None = None,
    expected_prediction_digest: str | None = None,
    expected_row_count: int | None = None,
) -> PredictionVector:
    """Load exact finite float64 NPY predictions with optional identity checks."""

    payload = _checked_payload(path, expected_file_sha256)
    try:
        raw = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise BaselineArtifactError(f"cannot decode prediction artifact: {exc}") from exc
    if not isinstance(raw, np.ndarray) or raw.dtype != np.dtype("float64") or raw.ndim != 1:
        raise BaselineArtifactError("prediction artifact must contain one float64 NPY vector")
    predictions = PredictionVector(raw)
    if expected_prediction_digest is not None and predictions.digest != _require_digest(
        expected_prediction_digest, "expected_prediction_digest"
    ):
        raise BaselineArtifactError("prediction logical digest mismatch")
    if expected_row_count is not None:
        if type(expected_row_count) is not int or expected_row_count <= 0:
            raise BaselineArtifactError("expected_row_count must be a positive integer")
        if predictions.row_count != expected_row_count:
            raise BaselineArtifactError("prediction row count mismatch")
    return predictions

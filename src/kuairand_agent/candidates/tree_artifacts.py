"""Canonical, corruption-checked artifacts for trusted LambdaRank checkpoints.

The container is deliberately small and non-executable: canonical JSON carries the complete
checkpoint identity, while a length-delimited UTF-8 tail carries LightGBM's native model text.
Loading never imports LightGBM and never follows a checkpoint symlink.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from kuairand_agent.candidates.tree_ranker import (
    TREE_RANKER_SCHEMA_VERSION,
    TreeRankerCheckpoint,
    TreeRankerError,
)
from kuairand_agent.data.capabilities import DataPhase

TREE_CHECKPOINT_ARTIFACT_SCHEMA_VERSION: Final = 1
_MAGIC: Final = b"KUAIRAND_TREE_CHECKPOINT_V1\n"
_LENGTH_BYTES: Final = 8
_MAX_METADATA_BYTES: Final = 1024 * 1024
_MAX_MODEL_BYTES: Final = 1024 * 1024 * 1024
_MAX_ARTIFACT_BYTES: Final = len(_MAGIC) + _LENGTH_BYTES + _MAX_METADATA_BYTES + _MAX_MODEL_BYTES
_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_TYPE: Final = "kuairand_lambdarank_checkpoint"
_CHECKPOINT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "training_phase",
        "feature_names",
        "training_feature_digest",
        "training_grouping_digest",
        "training_target_digest",
        "inner_validation_digest",
        "config_digest",
        "backend_identity",
        "best_iteration",
        "model_sha256",
    }
)
_METADATA_FIELDS: Final = frozenset(
    {
        "schema_version",
        "artifact_type",
        "checkpoint",
        "checkpoint_digest",
        "model_encoding",
        "model_size_bytes",
    }
)


class TreeCheckpointArtifactError(RuntimeError):
    """Raised when a tree checkpoint cannot be safely encoded, stored, or verified."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TreeCheckpointArtifactError("checkpoint metadata is not canonical JSON") from exc


def _require_digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise TreeCheckpointArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_checkpoint(checkpoint: object) -> TreeRankerCheckpoint:
    if not isinstance(checkpoint, TreeRankerCheckpoint):
        raise TreeCheckpointArtifactError("checkpoint must be TreeRankerCheckpoint")
    try:
        rebuilt = TreeRankerCheckpoint(
            training_phase=checkpoint.training_phase,
            feature_names=checkpoint.feature_names,
            training_feature_digest=checkpoint.training_feature_digest,
            training_grouping_digest=checkpoint.training_grouping_digest,
            training_target_digest=checkpoint.training_target_digest,
            inner_validation_digest=checkpoint.inner_validation_digest,
            config_digest=checkpoint.config_digest,
            backend_identity=checkpoint.backend_identity,
            best_iteration=checkpoint.best_iteration,
            model_text=checkpoint.model_text,
        )
    except (TreeRankerError, TypeError, ValueError, UnicodeError) as exc:
        raise TreeCheckpointArtifactError(f"checkpoint values are invalid: {exc}") from exc
    if (
        rebuilt.manifest() != checkpoint.manifest()
        or rebuilt.digest != checkpoint.digest
        or rebuilt.model_sha256 != checkpoint.model_sha256
    ):
        raise TreeCheckpointArtifactError("checkpoint identity does not match its current values")
    return rebuilt


@dataclass(frozen=True, slots=True)
class TreeCheckpointArtifact:
    """Identity of one exclusively published canonical tree checkpoint file."""

    path: Path
    file_sha256: str
    checkpoint_digest: str
    model_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_sha256", _require_digest(self.file_sha256, "file_sha256"))
        object.__setattr__(
            self,
            "checkpoint_digest",
            _require_digest(self.checkpoint_digest, "checkpoint_digest"),
        )
        object.__setattr__(self, "model_sha256", _require_digest(self.model_sha256, "model_sha256"))
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise TreeCheckpointArtifactError("checkpoint artifact size_bytes must be positive")

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "model_sha256": self.model_sha256,
            "size_bytes": self.size_bytes,
        }


def serialize_tree_checkpoint(checkpoint: TreeRankerCheckpoint) -> bytes:
    """Return the unique canonical byte representation of a validated checkpoint."""

    validated = _validated_checkpoint(checkpoint)
    try:
        model_bytes = validated.model_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TreeCheckpointArtifactError("checkpoint model_text is not valid UTF-8 text") from exc
    if not model_bytes or len(model_bytes) > _MAX_MODEL_BYTES:
        raise TreeCheckpointArtifactError(
            "checkpoint model_text size is outside the supported bound"
        )
    metadata = {
        "schema_version": TREE_CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "checkpoint": validated.manifest(),
        "checkpoint_digest": validated.digest,
        "model_encoding": "utf-8",
        "model_size_bytes": len(model_bytes),
    }
    metadata_bytes = _canonical_json(metadata)
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise TreeCheckpointArtifactError("checkpoint metadata exceeds the supported bound")
    return b"".join(
        (
            _MAGIC,
            len(metadata_bytes).to_bytes(_LENGTH_BYTES, "big"),
            metadata_bytes,
            model_bytes,
        )
    )


def _decode_metadata(payload: bytes) -> tuple[dict[str, object], bytes]:
    minimum_size = len(_MAGIC) + _LENGTH_BYTES + 1
    if not minimum_size <= len(payload) <= _MAX_ARTIFACT_BYTES:
        raise TreeCheckpointArtifactError("checkpoint artifact size is outside the supported bound")
    if not payload.startswith(_MAGIC):
        raise TreeCheckpointArtifactError("checkpoint artifact magic or schema is invalid")
    length_start = len(_MAGIC)
    length_end = length_start + _LENGTH_BYTES
    metadata_size = int.from_bytes(payload[length_start:length_end], "big")
    if not 1 <= metadata_size <= _MAX_METADATA_BYTES:
        raise TreeCheckpointArtifactError("checkpoint metadata size is outside the supported bound")
    metadata_end = length_end + metadata_size
    if metadata_end >= len(payload):
        raise TreeCheckpointArtifactError("checkpoint artifact framing is truncated")
    metadata_bytes = payload[length_end:metadata_end]
    model_bytes = payload[metadata_end:]
    try:
        decoded = metadata_bytes.decode("ascii", errors="strict")
        raw_metadata = json.loads(decoded)
    except (UnicodeDecodeError, ValueError) as exc:
        raise TreeCheckpointArtifactError("checkpoint metadata is not canonical JSON") from exc
    if (
        not isinstance(raw_metadata, dict)
        or _canonical_json(raw_metadata) != metadata_bytes
        or set(raw_metadata) != _METADATA_FIELDS
    ):
        raise TreeCheckpointArtifactError(
            "checkpoint metadata is not an exact canonical JSON object"
        )
    return cast(dict[str, object], raw_metadata), model_bytes


def _decode_checkpoint(metadata: dict[str, object], model_bytes: bytes) -> TreeRankerCheckpoint:
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise TreeCheckpointArtifactError("checkpoint artifact schema_version is unsupported")
    if metadata["artifact_type"] != _ARTIFACT_TYPE or metadata["model_encoding"] != "utf-8":
        raise TreeCheckpointArtifactError("checkpoint artifact type or model encoding is invalid")
    model_size = metadata["model_size_bytes"]
    if type(model_size) is not int or not 1 <= model_size <= _MAX_MODEL_BYTES:
        raise TreeCheckpointArtifactError("checkpoint model size is outside the supported bound")
    if len(model_bytes) != model_size:
        raise TreeCheckpointArtifactError(
            "checkpoint model size does not match the artifact framing"
        )
    try:
        model_text = model_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TreeCheckpointArtifactError("checkpoint model_text is not valid UTF-8 text") from exc

    raw_manifest = metadata["checkpoint"]
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != _CHECKPOINT_FIELDS:
        raise TreeCheckpointArtifactError(
            "checkpoint manifest fields do not match the exact schema"
        )
    manifest = cast(dict[str, object], raw_manifest)
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != TREE_RANKER_SCHEMA_VERSION
    ):
        raise TreeCheckpointArtifactError("tree ranker checkpoint schema_version is unsupported")
    stored_model_sha256 = _require_digest(manifest["model_sha256"], "model_sha256")
    if hashlib.sha256(model_bytes).hexdigest() != stored_model_sha256:
        raise TreeCheckpointArtifactError("checkpoint model SHA-256 mismatch")
    raw_feature_names = manifest["feature_names"]
    if not isinstance(raw_feature_names, list):
        raise TreeCheckpointArtifactError("checkpoint feature_names must be an ordered JSON array")
    raw_phase = manifest["training_phase"]
    if type(raw_phase) is not str:
        raise TreeCheckpointArtifactError("checkpoint training_phase must be a string")
    try:
        phase = DataPhase(raw_phase)
    except ValueError as exc:
        raise TreeCheckpointArtifactError("checkpoint training_phase is invalid") from exc
    try:
        checkpoint = TreeRankerCheckpoint(
            training_phase=phase,
            feature_names=tuple(raw_feature_names),
            training_feature_digest=cast(str, manifest["training_feature_digest"]),
            training_grouping_digest=cast(str, manifest["training_grouping_digest"]),
            training_target_digest=cast(str, manifest["training_target_digest"]),
            inner_validation_digest=cast(str | None, manifest["inner_validation_digest"]),
            config_digest=cast(str, manifest["config_digest"]),
            backend_identity=cast(str, manifest["backend_identity"]),
            best_iteration=cast(int, manifest["best_iteration"]),
            model_text=model_text,
        )
    except (TreeRankerError, TypeError, ValueError, UnicodeError) as exc:
        raise TreeCheckpointArtifactError(
            f"checkpoint manifest or model is invalid: {exc}"
        ) from exc
    if checkpoint.manifest() != manifest:
        raise TreeCheckpointArtifactError("checkpoint manifest does not match its decoded values")
    stored_checkpoint_digest = _require_digest(metadata["checkpoint_digest"], "checkpoint_digest")
    if checkpoint.digest != stored_checkpoint_digest:
        raise TreeCheckpointArtifactError("checkpoint logical digest mismatch")
    return checkpoint


def _expected_feature_names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TreeCheckpointArtifactError("expected_feature_names must be an ordered sequence")
    try:
        normalized = tuple(value)
    except TypeError as exc:
        raise TreeCheckpointArtifactError(
            "expected_feature_names must be an ordered sequence"
        ) from exc
    if (
        not normalized
        or any(type(name) is not str or not name or "\x00" in name for name in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise TreeCheckpointArtifactError(
            "expected_feature_names must contain unique non-empty strings"
        )
    return normalized


def _verify_expected_identity(
    checkpoint: TreeRankerCheckpoint,
    *,
    expected_checkpoint_digest: str | None,
    expected_model_sha256: str | None,
    expected_training_feature_digest: str | None,
    expected_training_grouping_digest: str | None,
    expected_training_target_digest: str | None,
    expected_inner_validation_digest: str | None,
    expected_config_digest: str | None,
    expected_training_phase: DataPhase | None,
    expected_feature_names: Sequence[str] | None,
    expected_backend_identity: str | None,
    expected_best_iteration: int | None,
) -> None:
    digest_checks = (
        ("checkpoint logical", checkpoint.digest, expected_checkpoint_digest),
        ("model", checkpoint.model_sha256, expected_model_sha256),
        ("training feature", checkpoint.training_feature_digest, expected_training_feature_digest),
        (
            "training grouping",
            checkpoint.training_grouping_digest,
            expected_training_grouping_digest,
        ),
        ("training target", checkpoint.training_target_digest, expected_training_target_digest),
        ("inner validation", checkpoint.inner_validation_digest, expected_inner_validation_digest),
        ("config", checkpoint.config_digest, expected_config_digest),
    )
    for label, observed, expected in digest_checks:
        expected_name = f"expected_{label.replace(' ', '_')}_digest"
        if expected is not None and observed != _require_digest(expected, expected_name):
            raise TreeCheckpointArtifactError(f"checkpoint {label} digest mismatch")

    if expected_training_phase is not None:
        if not isinstance(expected_training_phase, DataPhase):
            raise TreeCheckpointArtifactError("expected_training_phase must be a DataPhase")
        if checkpoint.training_phase is not expected_training_phase:
            raise TreeCheckpointArtifactError("checkpoint training phase mismatch")
    if expected_feature_names is not None and checkpoint.feature_names != _expected_feature_names(
        expected_feature_names
    ):
        raise TreeCheckpointArtifactError("checkpoint feature names/order mismatch")
    if expected_backend_identity is not None:
        if (
            type(expected_backend_identity) is not str
            or not expected_backend_identity
            or "\x00" in expected_backend_identity
            or "\n" in expected_backend_identity
        ):
            raise TreeCheckpointArtifactError(
                "expected_backend_identity must be a non-empty single-line string"
            )
        if checkpoint.backend_identity != expected_backend_identity:
            raise TreeCheckpointArtifactError("checkpoint backend identity mismatch")
    if expected_best_iteration is not None:
        if type(expected_best_iteration) is not int or expected_best_iteration <= 0:
            raise TreeCheckpointArtifactError("expected_best_iteration must be a positive integer")
        if checkpoint.best_iteration != expected_best_iteration:
            raise TreeCheckpointArtifactError("checkpoint best iteration mismatch")


def deserialize_tree_checkpoint(
    payload: bytes,
    *,
    expected_checkpoint_digest: str | None = None,
    expected_model_sha256: str | None = None,
    expected_training_feature_digest: str | None = None,
    expected_training_grouping_digest: str | None = None,
    expected_training_target_digest: str | None = None,
    expected_inner_validation_digest: str | None = None,
    expected_config_digest: str | None = None,
    expected_training_phase: DataPhase | None = None,
    expected_feature_names: Sequence[str] | None = None,
    expected_backend_identity: str | None = None,
    expected_best_iteration: int | None = None,
) -> TreeRankerCheckpoint:
    """Decode canonical bytes and verify every supplied logical identity."""

    if type(payload) is not bytes:
        raise TreeCheckpointArtifactError("checkpoint payload must be bytes")
    metadata, model_bytes = _decode_metadata(payload)
    checkpoint = _decode_checkpoint(metadata, model_bytes)
    _verify_expected_identity(
        checkpoint,
        expected_checkpoint_digest=expected_checkpoint_digest,
        expected_model_sha256=expected_model_sha256,
        expected_training_feature_digest=expected_training_feature_digest,
        expected_training_grouping_digest=expected_training_grouping_digest,
        expected_training_target_digest=expected_training_target_digest,
        expected_inner_validation_digest=expected_inner_validation_digest,
        expected_config_digest=expected_config_digest,
        expected_training_phase=expected_training_phase,
        expected_feature_names=expected_feature_names,
        expected_backend_identity=expected_backend_identity,
        expected_best_iteration=expected_best_iteration,
    )
    return checkpoint


def _atomic_no_replace(path: Path, payload: bytes) -> None:
    if not payload:
        raise TreeCheckpointArtifactError("refusing to write an empty checkpoint artifact")
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
            raise TreeCheckpointArtifactError(
                f"checkpoint artifact already exists and cannot be overwritten: {path}"
            ) from exc
        except OSError as exc:
            raise TreeCheckpointArtifactError(
                f"cannot atomically install checkpoint artifact {path}: {exc}"
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
            raise TreeCheckpointArtifactError(
                f"cannot make checkpoint artifact directory durable: {path.parent}"
            ) from exc
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def save_tree_checkpoint(
    path: Path | str,
    checkpoint: TreeRankerCheckpoint,
) -> TreeCheckpointArtifact:
    """Atomically publish one deterministic checkpoint without replacing any path entry."""

    destination = Path(path)
    if destination.suffix != ".tree":
        raise TreeCheckpointArtifactError("tree checkpoint path must end in .tree")
    payload = serialize_tree_checkpoint(checkpoint)
    _atomic_no_replace(destination, payload)
    return TreeCheckpointArtifact(
        path=destination.resolve(),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_digest=checkpoint.digest,
        model_sha256=checkpoint.model_sha256,
        size_bytes=len(payload),
    )


def _checked_payload(path: Path | str, expected_file_sha256: str | None) -> bytes:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise TreeCheckpointArtifactError(
            f"checkpoint artifact must be a readable regular non-symlink file: {source}"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            status = os.fstat(handle.fileno())
            if not stat.S_ISREG(status.st_mode):
                raise TreeCheckpointArtifactError(
                    f"checkpoint artifact must be a regular non-symlink file: {source}"
                )
            if status.st_size <= 0 or status.st_size > _MAX_ARTIFACT_BYTES:
                raise TreeCheckpointArtifactError(
                    "checkpoint artifact size is outside the supported bound"
                )
            payload = handle.read(_MAX_ARTIFACT_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise TreeCheckpointArtifactError("checkpoint artifact size is outside the supported bound")
    observed_file_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and observed_file_sha256 != _require_digest(
        expected_file_sha256, "expected_file_sha256"
    ):
        raise TreeCheckpointArtifactError("checkpoint artifact file SHA-256 mismatch")
    return payload


def load_tree_checkpoint(
    path: Path | str,
    *,
    expected_file_sha256: str | None = None,
    expected_checkpoint_digest: str | None = None,
    expected_model_sha256: str | None = None,
    expected_training_feature_digest: str | None = None,
    expected_training_grouping_digest: str | None = None,
    expected_training_target_digest: str | None = None,
    expected_inner_validation_digest: str | None = None,
    expected_config_digest: str | None = None,
    expected_training_phase: DataPhase | None = None,
    expected_feature_names: Sequence[str] | None = None,
    expected_backend_identity: str | None = None,
    expected_best_iteration: int | None = None,
) -> TreeRankerCheckpoint:
    """Read one inode snapshot, verify its file hash, then decode all logical identities."""

    payload = _checked_payload(path, expected_file_sha256)
    return deserialize_tree_checkpoint(
        payload,
        expected_checkpoint_digest=expected_checkpoint_digest,
        expected_model_sha256=expected_model_sha256,
        expected_training_feature_digest=expected_training_feature_digest,
        expected_training_grouping_digest=expected_training_grouping_digest,
        expected_training_target_digest=expected_training_target_digest,
        expected_inner_validation_digest=expected_inner_validation_digest,
        expected_config_digest=expected_config_digest,
        expected_training_phase=expected_training_phase,
        expected_feature_names=expected_feature_names,
        expected_backend_identity=expected_backend_identity,
        expected_best_iteration=expected_best_iteration,
    )


__all__ = [
    "TREE_CHECKPOINT_ARTIFACT_SCHEMA_VERSION",
    "TreeCheckpointArtifact",
    "TreeCheckpointArtifactError",
    "deserialize_tree_checkpoint",
    "load_tree_checkpoint",
    "save_tree_checkpoint",
    "serialize_tree_checkpoint",
]

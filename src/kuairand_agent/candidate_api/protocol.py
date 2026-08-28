"""Strict trusted parsing for generated-candidate train and prediction outputs.

Candidate processes are allowed to write artifacts, not to establish their meaning.  This module
therefore treats every result field as an untrusted declaration and binds it to controller-owned
expectations plus the bytes that actually exist under one exact output directory.  In particular,
candidate diagnostics may not declare organizer metrics; only the protected scorer can do that.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import numpy.typing as npt

PROTOCOL_SCHEMA_VERSION: Final = 1
TRAIN_RESULT_FILENAME: Final = "candidate_result.json"
PREDICTION_RESULT_FILENAME: Final = "prediction_result.json"
DEFAULT_SCORES_PATH: Final = "scores.npy"
SCORES_DTYPE: Final = "<f8"
MAX_RESULT_JSON_BYTES: Final = 256 * 1024
MAX_DIAGNOSTIC_DEPTH: Final = 8
MAX_DIAGNOSTIC_NODES: Final = 512
MAX_DIAGNOSTIC_STRING_BYTES: Final = 4096
MAX_CHECKPOINT_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_NPY_HEADER_BYTES: Final = 64 * 1024

type Float64Vector = npt.NDArray[np.float64]

_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_DIAGNOSTIC_KEY_RE: Final = re.compile(r"[A-Za-z][A-Za-z0-9_.@-]{0,127}\Z")
_OFFICIAL_TEXT_RE: Final = re.compile(
    r"(?:\bgauc\b|\bndcg(?:\s*@?\s*5)?\b|\bprimary(?:\s+metric|\s+score)?\b)",
    re.IGNORECASE,
)
_OFFICIAL_EXACT_KEYS: Final = frozenset(
    {
        "auc",
        "evaluation_metric",
        "evaluation_metrics",
        "evaluation_score",
        "metric",
        "metrics",
        "official_metric",
        "official_metrics",
        "official_score",
        "public_metric",
        "public_metrics",
        "public_score",
        "recall_50",
        "validation_metric",
        "validation_metrics",
        "validation_score",
    }
)


class CandidateProtocolError(ValueError):
    """An untrusted result or output artifact violates the candidate protocol."""


def _require_digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise CandidateProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_split_token(value: object, name: str = "split_token") -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 256
        or not value.isascii()
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise CandidateProtocolError(
            f"{name} must be a non-empty printable ASCII token without whitespace"
        )
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 < value <= (2**63 - 1):
        raise CandidateProtocolError(f"{name} must be a positive signed 64-bit integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= (2**63 - 1):
        raise CandidateProtocolError(f"{name} must be a non-negative signed 64-bit integer")
    return value


def _relative_path(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CandidateProtocolError(f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or part.startswith(".") for part in path.parts)
    ):
        raise CandidateProtocolError(f"{name} must be a canonical non-hidden relative POSIX path")
    return value


def _normalized_metric_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_official_metric_key(value: str) -> bool:
    normalized = _normalized_metric_key(value)
    components = set(normalized.split("_"))
    return (
        normalized in _OFFICIAL_EXACT_KEYS
        or "gauc" in components
        or "ndcg" in components
        or "primary" in components
    )


def _freeze_diagnostics(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {cast(str, key): _freeze_diagnostics(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_diagnostics(child) for child in value)
    return value


def _thaw_diagnostics(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_diagnostics(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_diagnostics(child) for child in value]
    return value


def _validate_diagnostics(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CandidateProtocolError("diagnostics must be a JSON object")
    nodes = 0

    def visit(child: object, *, path: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_DIAGNOSTIC_NODES:
            raise CandidateProtocolError("diagnostics exceed the node-count limit")
        if depth > MAX_DIAGNOSTIC_DEPTH:
            raise CandidateProtocolError("diagnostics exceed the nesting-depth limit")
        if child is None or isinstance(child, bool) or type(child) is int:
            return
        if type(child) is float:
            if not math.isfinite(child):
                raise CandidateProtocolError(f"diagnostics contain a non-finite number at {path}")
            return
        if isinstance(child, str):
            if len(child.encode("utf-8")) > MAX_DIAGNOSTIC_STRING_BYTES:
                raise CandidateProtocolError(f"diagnostic string is oversized at {path}")
            if _OFFICIAL_TEXT_RE.search(child) is not None:
                raise CandidateProtocolError(
                    f"diagnostics contain a candidate-declared official metric at {path}"
                )
            return
        if isinstance(child, dict):
            for key, nested in child.items():
                if type(key) is not str or _DIAGNOSTIC_KEY_RE.fullmatch(key) is None:
                    raise CandidateProtocolError(f"diagnostic key is invalid at {path}")
                if _is_official_metric_key(key):
                    raise CandidateProtocolError(
                        f"diagnostics contain a candidate-declared official metric at {path}.{key}"
                    )
                visit(nested, path=f"{path}.{key}", depth=depth + 1)
            return
        if isinstance(child, list):
            for index, nested in enumerate(child):
                visit(nested, path=f"{path}[{index}]", depth=depth + 1)
            return
        raise CandidateProtocolError(f"diagnostics contain a non-JSON value at {path}")

    visit(value, path="diagnostics", depth=0)
    frozen = _freeze_diagnostics(value)
    assert isinstance(frozen, Mapping)
    return cast(Mapping[str, object], frozen)


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateProtocolError(f"result JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CandidateProtocolError(f"result JSON contains non-finite constant {value}")


def _parse_json_object(payload: bytes | str) -> dict[str, object]:
    if not isinstance(payload, (bytes, str)):
        raise CandidateProtocolError("result JSON must be bytes or text")
    size = len(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    if size > MAX_RESULT_JSON_BYTES:
        raise CandidateProtocolError("result JSON exceeds the byte limit")
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError as exc:
        raise CandidateProtocolError("result JSON must be UTF-8") from exc
    try:
        parsed = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_constant,
            ),
        )
    except CandidateProtocolError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise CandidateProtocolError("result JSON is malformed") from exc
    if not isinstance(parsed, dict):
        raise CandidateProtocolError("result JSON must contain one object")
    return cast(dict[str, object], parsed)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CandidateProtocolError(
            f"{name} must contain exact keys; missing={missing!r}, extra={extra!r}"
        )


@dataclass(frozen=True, slots=True)
class ArtifactDeclaration:
    """One candidate-declared artifact bound later to actual output bytes."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _relative_path(self.path, "artifact.path")
        _require_digest(self.sha256, "artifact.sha256")
        _require_nonnegative_int(self.size_bytes, "artifact.size_bytes")

    def to_wire(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class TrainResultManifest:
    source_digest: str
    config_digest: str
    data_digest: str
    split_token: str
    checkpoint_digest: str
    artifacts: tuple[ArtifactDeclaration, ...]
    diagnostics: Mapping[str, object]
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    kind: str = "train"

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "data_digest": self.data_digest,
            "split_token": self.split_token,
            "checkpoint_digest": self.checkpoint_digest,
            "artifacts": [artifact.to_wire() for artifact in self.artifacts],
            "diagnostics": _thaw_diagnostics(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class PredictionResultManifest:
    source_digest: str
    config_digest: str
    data_digest: str
    split_token: str
    checkpoint_digest: str
    expected_count: int
    dtype: str
    scores_path: str
    scores_sha256: str
    diagnostics: Mapping[str, object]
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    kind: str = "prediction"

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "data_digest": self.data_digest,
            "split_token": self.split_token,
            "checkpoint_digest": self.checkpoint_digest,
            "expected_count": self.expected_count,
            "dtype": self.dtype,
            "scores_path": self.scores_path,
            "scores_sha256": self.scores_sha256,
            "diagnostics": _thaw_diagnostics(self.diagnostics),
        }


_TRAIN_KEYS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "source_digest",
        "config_digest",
        "data_digest",
        "split_token",
        "checkpoint_digest",
        "artifacts",
        "diagnostics",
    }
)
_ARTIFACT_KEYS: Final = frozenset({"path", "sha256", "size_bytes"})
_PREDICTION_KEYS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "source_digest",
        "config_digest",
        "data_digest",
        "split_token",
        "checkpoint_digest",
        "expected_count",
        "dtype",
        "scores_path",
        "scores_sha256",
        "diagnostics",
    }
)


def parse_train_result_json(payload: bytes | str) -> TrainResultManifest:
    """Parse one exact candidate training result without trusting any declared artifact."""

    value = _parse_json_object(payload)
    _exact_keys(value, _TRAIN_KEYS, "training result")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise CandidateProtocolError("training result schema_version must be integer 1")
    if value["kind"] != "train":
        raise CandidateProtocolError("training result kind must be 'train'")
    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise CandidateProtocolError("training result artifacts must be a non-empty list")
    artifacts: list[ArtifactDeclaration] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise CandidateProtocolError(f"artifact {index} must be an object")
        artifact = cast(dict[str, object], raw_artifact)
        _exact_keys(artifact, _ARTIFACT_KEYS, f"artifact {index}")
        artifacts.append(
            ArtifactDeclaration(
                path=_relative_path(artifact["path"], f"artifact {index}.path"),
                sha256=_require_digest(artifact["sha256"], f"artifact {index}.sha256"),
                size_bytes=_require_nonnegative_int(
                    artifact["size_bytes"], f"artifact {index}.size_bytes"
                ),
            )
        )
    paths = tuple(artifact.path for artifact in artifacts)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise CandidateProtocolError("training artifact paths must be unique and sorted")
    return TrainResultManifest(
        source_digest=_require_digest(value["source_digest"], "source_digest"),
        config_digest=_require_digest(value["config_digest"], "config_digest"),
        data_digest=_require_digest(value["data_digest"], "data_digest"),
        split_token=_require_split_token(value["split_token"]),
        checkpoint_digest=_require_digest(value["checkpoint_digest"], "checkpoint_digest"),
        artifacts=tuple(artifacts),
        diagnostics=_validate_diagnostics(value["diagnostics"]),
    )


def parse_prediction_result_json(payload: bytes | str) -> PredictionResultManifest:
    """Parse one exact candidate prediction result before inspecting ``scores.npy``."""

    value = _parse_json_object(payload)
    _exact_keys(value, _PREDICTION_KEYS, "prediction result")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise CandidateProtocolError("prediction result schema_version must be integer 1")
    if value["kind"] != "prediction":
        raise CandidateProtocolError("prediction result kind must be 'prediction'")
    if value["dtype"] != SCORES_DTYPE:
        raise CandidateProtocolError(f"prediction dtype must be exactly {SCORES_DTYPE!r}")
    return PredictionResultManifest(
        source_digest=_require_digest(value["source_digest"], "source_digest"),
        config_digest=_require_digest(value["config_digest"], "config_digest"),
        data_digest=_require_digest(value["data_digest"], "data_digest"),
        split_token=_require_split_token(value["split_token"]),
        checkpoint_digest=_require_digest(value["checkpoint_digest"], "checkpoint_digest"),
        expected_count=_require_positive_int(value["expected_count"], "expected_count"),
        dtype=value["dtype"],
        scores_path=_relative_path(value["scores_path"], "scores_path"),
        scores_sha256=_require_digest(value["scores_sha256"], "scores_sha256"),
        diagnostics=_validate_diagnostics(value["diagnostics"]),
    )


@dataclass(frozen=True, slots=True)
class TrainExpectation:
    """Controller-owned identities and exact artifact paths for one training execution."""

    source_digest: str
    config_digest: str
    data_digest: str
    split_token: str
    checkpoint_path: str
    artifact_paths: tuple[str, ...] = ()
    expected_checkpoint_digest: str | None = None
    max_artifact_bytes: int = MAX_CHECKPOINT_BYTES

    def __post_init__(self) -> None:
        _require_digest(self.source_digest, "source_digest")
        _require_digest(self.config_digest, "config_digest")
        _require_digest(self.data_digest, "data_digest")
        _require_split_token(self.split_token)
        checkpoint = _relative_path(self.checkpoint_path, "checkpoint_path")
        paths = self.artifact_paths or (checkpoint,)
        for index, path in enumerate(paths):
            _relative_path(path, f"artifact_paths[{index}]")
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise CandidateProtocolError("artifact_paths must be unique and sorted")
        if checkpoint not in paths:
            raise CandidateProtocolError("artifact_paths must contain checkpoint_path")
        object.__setattr__(self, "artifact_paths", paths)
        if self.expected_checkpoint_digest is not None:
            _require_digest(self.expected_checkpoint_digest, "expected_checkpoint_digest")
        _require_positive_int(self.max_artifact_bytes, "max_artifact_bytes")


@dataclass(frozen=True, slots=True)
class PredictionExpectation:
    """Controller-owned identity, shape, dtype, token, and path for one inference execution."""

    source_digest: str
    config_digest: str
    data_digest: str
    split_token: str
    checkpoint_digest: str
    expected_count: int
    dtype: str = SCORES_DTYPE
    scores_path: str = DEFAULT_SCORES_PATH

    def __post_init__(self) -> None:
        _require_digest(self.source_digest, "source_digest")
        _require_digest(self.config_digest, "config_digest")
        _require_digest(self.data_digest, "data_digest")
        _require_split_token(self.split_token)
        _require_digest(self.checkpoint_digest, "checkpoint_digest")
        _require_positive_int(self.expected_count, "expected_count")
        if self.dtype != SCORES_DTYPE:
            raise CandidateProtocolError(f"prediction dtype must be exactly {SCORES_DTYPE!r}")
        _relative_path(self.scores_path, "scores_path")


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ValidatedTrainResult:
    manifest: TrainResultManifest
    output_dir: Path
    checkpoint_path: Path
    checkpoint_digest: str
    checkpoint_size_bytes: int
    artifacts: tuple[ValidatedArtifact, ...]


@dataclass(frozen=True, slots=True)
class ValidatedPredictionResult:
    manifest: PredictionResultManifest
    output_dir: Path
    scores_path: Path
    scores_sha256: str
    scores: Float64Vector


def _validate_output_root(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise CandidateProtocolError("output_dir must be an absolute pathlib.Path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CandidateProtocolError("output_dir cannot be inspected") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CandidateProtocolError("output_dir must be a real directory, not a symlink")
    return path


def _inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CandidateProtocolError(f"output inventory contains a symlink: {relative}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise CandidateProtocolError(
                    f"output inventory contains a special directory entry: {relative}"
                )
            directories.add(relative)
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CandidateProtocolError(f"output inventory contains a symlink: {relative}")
            if not stat.S_ISREG(metadata.st_mode):
                raise CandidateProtocolError(
                    f"output inventory contains a special file: {relative}"
                )
            if metadata.st_nlink != 1:
                raise CandidateProtocolError(f"output inventory contains a hardlink: {relative}")
            files.add(relative)
    return files, directories


def _expected_directories(paths: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for value in paths:
        for parent in PurePosixPath(value).parents:
            if parent != PurePosixPath("."):
                result.add(parent.as_posix())
    return result


def _read_regular(path: Path, *, max_bytes: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateProtocolError(f"{name} cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CandidateProtocolError(f"{name} must be a single-link regular file")
        if metadata.st_size > max_bytes:
            raise CandidateProtocolError(f"{name} exceeds its byte limit")
        payload = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(payload))):
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise CandidateProtocolError(f"{name} exceeds its byte limit")
        after = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(payload) != after.st_size:
            raise CandidateProtocolError(f"{name} changed while it was inspected")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, *, max_bytes: int, name: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateProtocolError(f"{name} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CandidateProtocolError(f"{name} must be a single-link regular file")
        if before.st_size > max_bytes:
            raise CandidateProtocolError(f"{name} exceeds its byte limit")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise CandidateProtocolError(f"{name} exceeds its byte limit")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise CandidateProtocolError(f"{name} changed while it was inspected")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _assert_identity(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise CandidateProtocolError(f"candidate {name} does not match the trusted expectation")


def validate_train_outputs(
    output_dir: Path,
    expectation: TrainExpectation,
) -> ValidatedTrainResult:
    """Bind training JSON and each declared artifact to an exact trusted execution."""

    if not isinstance(expectation, TrainExpectation):
        raise CandidateProtocolError("expectation must be a TrainExpectation")
    root = _validate_output_root(output_dir)
    result_path = root / TRAIN_RESULT_FILENAME
    payload = _read_regular(
        result_path,
        max_bytes=MAX_RESULT_JSON_BYTES,
        name=TRAIN_RESULT_FILENAME,
    )
    manifest = parse_train_result_json(payload)
    _assert_identity(manifest.source_digest, expectation.source_digest, "source_digest")
    _assert_identity(manifest.config_digest, expectation.config_digest, "config_digest")
    _assert_identity(manifest.data_digest, expectation.data_digest, "data_digest")
    _assert_identity(manifest.split_token, expectation.split_token, "split_token")
    manifest_paths = tuple(artifact.path for artifact in manifest.artifacts)
    if manifest_paths != expectation.artifact_paths:
        raise CandidateProtocolError(
            "candidate artifact paths do not match the trusted expectation"
        )
    expected_files = {TRAIN_RESULT_FILENAME, *expectation.artifact_paths}
    files, directories = _inventory(root)
    if files != expected_files or directories != _expected_directories(expectation.artifact_paths):
        raise CandidateProtocolError("training output inventory is not exact")

    validated: list[ValidatedArtifact] = []
    for declaration in manifest.artifacts:
        path = root.joinpath(*PurePosixPath(declaration.path).parts)
        digest, size = _hash_regular(
            path,
            max_bytes=expectation.max_artifact_bytes,
            name=f"artifact {declaration.path}",
        )
        _assert_identity(digest, declaration.sha256, f"artifact {declaration.path} sha256")
        _assert_identity(size, declaration.size_bytes, f"artifact {declaration.path} size_bytes")
        validated.append(ValidatedArtifact(path, declaration.path, digest, size))

    checkpoint = next(
        artifact for artifact in validated if artifact.relative_path == expectation.checkpoint_path
    )
    _assert_identity(
        manifest.checkpoint_digest,
        checkpoint.sha256,
        "checkpoint_digest",
    )
    if expectation.expected_checkpoint_digest is not None:
        _assert_identity(
            checkpoint.sha256,
            expectation.expected_checkpoint_digest,
            "checkpoint_digest",
        )
    return ValidatedTrainResult(
        manifest=manifest,
        output_dir=root,
        checkpoint_path=checkpoint.path,
        checkpoint_digest=checkpoint.sha256,
        checkpoint_size_bytes=checkpoint.size_bytes,
        artifacts=tuple(validated),
    )


def validate_prediction_outputs(
    output_dir: Path,
    expectation: PredictionExpectation,
) -> ValidatedPredictionResult:
    """Validate exact prediction identities and return an immutable finite float64 vector."""

    if not isinstance(expectation, PredictionExpectation):
        raise CandidateProtocolError("expectation must be a PredictionExpectation")
    root = _validate_output_root(output_dir)
    result_path = root / PREDICTION_RESULT_FILENAME
    payload = _read_regular(
        result_path,
        max_bytes=MAX_RESULT_JSON_BYTES,
        name=PREDICTION_RESULT_FILENAME,
    )
    manifest = parse_prediction_result_json(payload)
    for name in (
        "source_digest",
        "config_digest",
        "data_digest",
        "split_token",
        "checkpoint_digest",
        "expected_count",
        "dtype",
        "scores_path",
    ):
        _assert_identity(getattr(manifest, name), getattr(expectation, name), name)
    expected_files = {PREDICTION_RESULT_FILENAME, expectation.scores_path}
    files, directories = _inventory(root)
    if files != expected_files or directories != _expected_directories((expectation.scores_path,)):
        raise CandidateProtocolError("prediction output inventory is not exact")

    scores_path = root.joinpath(*PurePosixPath(expectation.scores_path).parts)
    max_scores_bytes = expectation.expected_count * np.dtype(SCORES_DTYPE).itemsize
    scores_payload = _read_regular(
        scores_path,
        max_bytes=max_scores_bytes + MAX_NPY_HEADER_BYTES,
        name=expectation.scores_path,
    )
    scores_sha256 = hashlib.sha256(scores_payload).hexdigest()
    _assert_identity(scores_sha256, manifest.scores_sha256, "scores_sha256")
    buffer = io.BytesIO(scores_payload)
    try:
        raw = np.load(buffer, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise CandidateProtocolError("scores.npy is not a safe NumPy array") from exc
    if not isinstance(raw, np.ndarray):
        if hasattr(raw, "close"):
            raw.close()
        raise CandidateProtocolError("scores.npy must contain one NumPy array")
    if buffer.tell() != len(scores_payload):
        raise CandidateProtocolError("scores.npy contains trailing bytes")
    if raw.shape != (expectation.expected_count,):
        raise CandidateProtocolError(
            f"scores.npy shape must be exactly ({expectation.expected_count},)"
        )
    if raw.dtype.str != expectation.dtype:
        raise CandidateProtocolError(f"scores.npy dtype must be exactly {expectation.dtype!r}")
    if not raw.flags.c_contiguous:
        raise CandidateProtocolError("scores.npy must be C-contiguous")
    if not bool(np.isfinite(raw).all()):
        raise CandidateProtocolError("scores.npy values must all be finite")
    scores = np.frombuffer(raw.tobytes(order="C"), dtype=np.dtype(SCORES_DTYPE))
    scores.setflags(write=False)
    return ValidatedPredictionResult(
        manifest=manifest,
        output_dir=root,
        scores_path=scores_path,
        scores_sha256=scores_sha256,
        scores=cast(Float64Vector, scores),
    )

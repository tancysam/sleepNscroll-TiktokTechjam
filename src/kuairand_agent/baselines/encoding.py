"""Exact, replayable five-field encoding for the immutable organizer FM."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO, Final, Self, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.data.canonical import CanonicalInputs

ENCODING_SCHEMA_VERSION: Final = 1
STARTER_FIELD_NAMES: Final = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
MAX_ENCODING_ARCHIVE_BYTES: Final = 512 * 1024 * 1024
_EXPECTED_ARCHIVE_KEYS: Final = (
    "schema_version",
    "field_names",
    "edges",
    "field_dims",
    "offsets",
    "unknown_ids",
    "total_dim",
    "training_inputs_digest",
    "digest",
    *(f"vocab_{index}" for index in range(len(STARTER_FIELD_NAMES))),
)
_EXPECTED_ZIP_MEMBERS: Final = tuple(f"{key}.npy" for key in _EXPECTED_ARCHIVE_KEYS)

type Int32Matrix = npt.NDArray[np.int32]


class StarterEncodingError(ValueError):
    """Raised when encoding metadata or an encoding artifact is invalid."""


@dataclass(frozen=True, slots=True)
class EncodingArtifact:
    """Exact file and logical identities of one persisted encoding."""

    path: Path
    digest: str
    file_sha256: str


def _semantic_digest(
    *,
    edges: tuple[float, ...],
    vocabs: tuple[tuple[str, ...], ...],
    training_inputs_digest: str,
) -> str:
    payload = {
        "schema_version": ENCODING_SCHEMA_VERSION,
        "field_names": list(STARTER_FIELD_NAMES),
        "edges_hex": [value.hex() for value in edges],
        "vocabs": [list(vocab) for vocab in vocabs],
        "training_inputs_digest": training_inputs_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def _readonly_int32(values: Int32Matrix) -> Int32Matrix:
    result = np.ascontiguousarray(values, dtype=np.int32)
    result.setflags(write=False)
    return result


def _validated_sha256(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StarterEncodingError(f"{field_name} must be lowercase SHA-256")
    return value


def _sha256_open_file(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Durably persist a newly installed directory entry before reporting success."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_regular_metadata(metadata: os.stat_result, source: Path) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise StarterEncodingError(f"encoding archive must not be a symlink: {source}")
    if not stat.S_ISREG(metadata.st_mode):
        raise StarterEncodingError(f"encoding archive must be a regular file: {source}")
    if metadata.st_size == 0:
        raise StarterEncodingError(f"encoding archive must not be empty: {source}")
    if metadata.st_size > MAX_ENCODING_ARCHIVE_BYTES:
        raise StarterEncodingError(
            f"encoding archive exceeds {MAX_ENCODING_ARCHIVE_BYTES} bytes: {source}"
        )


@dataclass(frozen=True, slots=True)
class StarterEncoding:
    """Organizer-compatible categorical vocabularies and train-derived duration buckets.

    Vocabularies store values in first-seen training order.  Local unknown IDs occupy the slot
    immediately after each vocabulary; :meth:`transform` applies field offsets and returns the
    global IDs consumed by the organizer FM.
    """

    edges: tuple[float, ...]
    vocabs: tuple[tuple[str, ...], ...]
    training_inputs_digest: str
    digest: str = field(init=False)
    field_names: tuple[str, ...] = field(init=False, default=STARTER_FIELD_NAMES)

    def __post_init__(self) -> None:
        edges = tuple(float(value) for value in self.edges)
        if len(edges) != 9 or any(not np.isfinite(value) for value in edges):
            raise StarterEncodingError("duration encoding requires exactly nine finite edges")
        if any(left > right for left, right in pairwise(edges)):
            raise StarterEncodingError("duration edges must be non-decreasing")
        if len(self.vocabs) != len(STARTER_FIELD_NAMES):
            raise StarterEncodingError("encoding requires exactly five vocabularies")

        normalized_vocabs: list[tuple[str, ...]] = []
        for field_name, values in zip(STARTER_FIELD_NAMES, self.vocabs, strict=True):
            vocab = tuple(values)
            if not vocab:
                raise StarterEncodingError(f"{field_name} vocabulary must not be empty")
            if any(type(value) is not str or not value or "\x00" in value for value in vocab):
                raise StarterEncodingError(
                    f"{field_name} vocabulary values must be non-empty canonical text"
                )
            if len(vocab) != len(set(vocab)):
                raise StarterEncodingError(f"{field_name} vocabulary contains duplicates")
            normalized_vocabs.append(vocab)

        source_digest = self.training_inputs_digest
        if (
            type(source_digest) is not str
            or len(source_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_digest)
        ):
            raise StarterEncodingError("training_inputs_digest must be lowercase SHA-256")

        normalized = tuple(normalized_vocabs)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "vocabs", normalized)
        object.__setattr__(
            self,
            "digest",
            _semantic_digest(
                edges=edges,
                vocabs=normalized,
                training_inputs_digest=source_digest,
            ),
        )

    @classmethod
    def fit(cls, train_inputs: CanonicalInputs) -> Self:
        """Fit the exact starter encoding from training inputs in physical row order."""

        if not isinstance(train_inputs, CanonicalInputs):
            raise StarterEncodingError("train_inputs must be CanonicalInputs")
        if len(train_inputs) == 0:
            raise StarterEncodingError("cannot fit starter encoding on empty training inputs")
        durations = np.asarray(train_inputs.duration_ms, dtype=np.float64)
        edges_array = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])
        edges = tuple(float(value) for value in edges_array)

        buckets = tuple(
            str(int(np.searchsorted(edges_array, duration)))
            for duration in train_inputs.duration_ms
        )
        raw_fields = (
            train_inputs.user_id,
            train_inputs.video_id,
            train_inputs.author_id,
            train_inputs.tab,
            buckets,
        )
        vocabs: list[tuple[str, ...]] = []
        for values in raw_fields:
            # dict preserves first-seen order and exactly mirrors organizer data.encode().
            vocabs.append(tuple(dict.fromkeys(values)))
        return cls(
            edges=edges,
            vocabs=tuple(vocabs),
            training_inputs_digest=train_inputs.digest,
        )

    @property
    def field_dims(self) -> tuple[int, ...]:
        return tuple(len(vocab) + 1 for vocab in self.vocabs)

    @property
    def unknown_ids(self) -> tuple[int, ...]:
        return tuple(len(vocab) for vocab in self.vocabs)

    @property
    def offsets(self) -> tuple[int, ...]:
        dimensions = self.field_dims
        return tuple(int(value) for value in np.cumsum((0, *dimensions[:-1]), dtype=np.int64))

    @property
    def total_dim(self) -> int:
        return sum(self.field_dims)

    def manifest(self) -> dict[str, object]:
        """Return deterministic replay metadata without changing vocabulary order."""

        return {
            "schema_version": ENCODING_SCHEMA_VERSION,
            "field_names": list(self.field_names),
            "edges_hex": [value.hex() for value in self.edges],
            "vocabs": [list(vocab) for vocab in self.vocabs],
            "field_dims": list(self.field_dims),
            "offsets": list(self.offsets),
            "unknown_ids": list(self.unknown_ids),
            "total_dim": self.total_dim,
            "training_inputs_digest": self.training_inputs_digest,
            "digest": self.digest,
        }

    def transform(self, inputs: CanonicalInputs) -> Int32Matrix:
        """Encode train, validation, or label-free final inputs as read-only ``(N, 5)`` int32."""

        if not isinstance(inputs, CanonicalInputs):
            raise StarterEncodingError("inputs must be CanonicalInputs")
        edges = np.asarray(self.edges, dtype=np.float64)
        buckets = tuple(
            str(int(np.searchsorted(edges, duration))) for duration in inputs.duration_ms
        )
        raw_fields = (
            inputs.user_id,
            inputs.video_id,
            inputs.author_id,
            inputs.tab,
            buckets,
        )
        matrix = np.empty((len(inputs), len(self.field_names)), dtype=np.int32)
        for column, (values, vocabulary, unknown_id, offset) in enumerate(
            zip(raw_fields, self.vocabs, self.unknown_ids, self.offsets, strict=True)
        ):
            mapping = {value: index for index, value in enumerate(vocabulary)}
            matrix[:, column] = tuple(mapping.get(value, unknown_id) + offset for value in values)
        return _readonly_int32(matrix)

    def save(self, path: str | Path) -> EncodingArtifact:
        """Validate then atomically install a new object-free archive without overwriting."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "schema_version": np.asarray([ENCODING_SCHEMA_VERSION], dtype=np.int64),
            "field_names": np.asarray(self.field_names, dtype=np.str_),
            "edges": np.asarray(self.edges, dtype=np.float64),
            "field_dims": np.asarray(self.field_dims, dtype=np.int64),
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            "unknown_ids": np.asarray(self.unknown_ids, dtype=np.int64),
            "total_dim": np.asarray([self.total_dim], dtype=np.int64),
            "training_inputs_digest": np.asarray([self.training_inputs_digest], dtype=np.str_),
            "digest": np.asarray([self.digest], dtype=np.str_),
        }
        arrays.update(
            {
                f"vocab_{index}": np.asarray(vocabulary, dtype=np.str_)
                for index, vocabulary in enumerate(self.vocabs)
            }
        )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez(handle, **cast(Any, arrays))
                handle.flush()
                os.fsync(handle.fileno())
            with temporary_path.open("rb") as handle:
                file_sha256 = _sha256_open_file(handle)
            restored = type(self).load(
                temporary_path,
                expected_file_sha256=file_sha256,
            )
            if restored.digest != self.digest or restored.manifest() != self.manifest():
                raise StarterEncodingError("saved encoding did not replay exactly")
            try:
                # The temporary file is on the destination filesystem. link() installs that
                # already-validated inode atomically and fails if any destination entry exists.
                os.link(temporary_path, destination, follow_symlinks=False)
                _fsync_directory(destination.parent)
            except FileExistsError as exc:
                raise StarterEncodingError(
                    f"refusing to overwrite existing encoding archive: {destination}"
                ) from exc
            except OSError as exc:
                raise StarterEncodingError(
                    f"cannot atomically install encoding archive: {destination}"
                ) from exc
            return EncodingArtifact(
                path=destination.resolve(),
                digest=self.digest,
                file_sha256=file_sha256,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_file_sha256: str | None = None,
    ) -> Self:
        """Load one bounded regular archive and verify its retained file identity."""

        source = Path(path)
        expected_sha256 = (
            None
            if expected_file_sha256 is None
            else _validated_sha256(expected_file_sha256, "expected_file_sha256")
        )
        try:
            path_metadata = source.lstat()
            _validate_regular_metadata(path_metadata, source)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(source, flags)
            with os.fdopen(descriptor, "rb") as handle:
                opened_metadata = os.fstat(handle.fileno())
                _validate_regular_metadata(opened_metadata, source)
                if (
                    opened_metadata.st_dev != path_metadata.st_dev
                    or opened_metadata.st_ino != path_metadata.st_ino
                ):
                    raise StarterEncodingError("encoding archive changed while opening")

                actual_sha256 = _sha256_open_file(handle)
                if expected_sha256 is not None and actual_sha256 != expected_sha256:
                    raise StarterEncodingError(
                        "encoding archive SHA-256 mismatch: "
                        f"expected {expected_sha256}, got {actual_sha256}"
                    )

                with zipfile.ZipFile(handle, mode="r") as zipped:
                    members = zipped.infolist()
                    member_names = tuple(member.filename for member in members)
                    if member_names != _EXPECTED_ZIP_MEMBERS:
                        raise StarterEncodingError(
                            "encoding archive members are duplicated, reordered, or unexpected"
                        )
                    if any(
                        member.is_dir() or member.compress_type != zipfile.ZIP_STORED
                        for member in members
                    ):
                        raise StarterEncodingError(
                            "encoding archive members must be regular uncompressed NPY entries"
                        )
                    if sum(member.file_size for member in members) > MAX_ENCODING_ARCHIVE_BYTES:
                        raise StarterEncodingError("encoding archive expands beyond its size limit")

                handle.seek(0)
                with np.load(handle, allow_pickle=False) as archive:
                    actual_keys = tuple(archive.files)
                    if actual_keys != _EXPECTED_ARCHIVE_KEYS:
                        raise StarterEncodingError(
                            "encoding NPZ members differ from the frozen ordered schema"
                        )
                    schema = archive["schema_version"]
                    if schema.shape != (1,) or int(schema[0]) != ENCODING_SCHEMA_VERSION:
                        raise StarterEncodingError("unsupported encoding schema version")
                    field_names = tuple(str(value) for value in archive["field_names"].tolist())
                    if field_names != STARTER_FIELD_NAMES:
                        raise StarterEncodingError(
                            "encoding field names differ from organizer fields"
                        )
                    edges = tuple(float(value) for value in archive["edges"].tolist())
                    vocabs = tuple(
                        tuple(str(value) for value in archive[f"vocab_{index}"].tolist())
                        for index in range(len(STARTER_FIELD_NAMES))
                    )
                    training_digest_values = archive["training_inputs_digest"]
                    digest_values = archive["digest"]
                    if training_digest_values.shape != (1,) or digest_values.shape != (1,):
                        raise StarterEncodingError("encoding digest fields have invalid shapes")
                    restored = cls(
                        edges=edges,
                        vocabs=vocabs,
                        training_inputs_digest=str(training_digest_values[0]),
                    )
                    expected_derived = {
                        "field_dims": restored.field_dims,
                        "offsets": restored.offsets,
                        "unknown_ids": restored.unknown_ids,
                    }
                    for key, expected in expected_derived.items():
                        actual = tuple(int(value) for value in archive[key].tolist())
                        if actual != expected:
                            raise StarterEncodingError(f"encoding {key} metadata is corrupt")
                    total_dim = archive["total_dim"]
                    if total_dim.shape != (1,) or int(total_dim[0]) != restored.total_dim:
                        raise StarterEncodingError("encoding total_dim metadata is corrupt")
                    if str(digest_values[0]) != restored.digest:
                        raise StarterEncodingError("encoding logical digest mismatch")
                    return restored
        except StarterEncodingError:
            raise
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise StarterEncodingError(f"cannot load encoding archive: {source}") from exc


__all__ = [
    "ENCODING_SCHEMA_VERSION",
    "MAX_ENCODING_ARCHIVE_BYTES",
    "STARTER_FIELD_NAMES",
    "EncodingArtifact",
    "StarterEncoding",
    "StarterEncodingError",
]

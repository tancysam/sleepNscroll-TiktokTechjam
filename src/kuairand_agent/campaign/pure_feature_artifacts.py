"""Deterministic replay artifacts for the frozen KuaiRand-Pure feature table.

The trusted campaign stores one canonical, non-pickle NPZ beside one canonical JSON identity
manifest.  The JSON digest is the externally anchored artifact identity; it binds the exact NPZ
bytes and the complete logical :class:`~kuairand_agent.campaign.pure_features.PureFeaturePair`
manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final, cast

import numpy as np

from kuairand_agent.campaign.pure_features import PureFeaturePair
from kuairand_agent.campaign.strict_past_exposure import (
    STRICT_PAST_EXPOSURE_FEATURE_NAMES,
    StrictPastExposureError,
    StrictPastExposurePair,
)
from kuairand_agent.data.causal_features import CausalFeatureError, FeatureMatrix

PURE_FEATURE_ARTIFACT_SCHEMA_VERSION: Final = 1
PURE_FEATURE_ARTIFACT_TYPE: Final = "kuairand-pure-feature-pair"
_NPZ_MEMBERS: Final = ("prefix.npy", "query.npy")
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_MAX_MANIFEST_BYTES: Final = 1024 * 1024
_MAX_NPZ_BYTES: Final = 1024 * 1024 * 1024
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class PureFeatureArtifactError(RuntimeError):
    """Raised when a feature artifact cannot be saved, decoded, or verified."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise PureFeatureArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PureFeatureArtifactError(f"{name} must be a positive integer")
    return value


def _artifact_paths(path: Path | str) -> tuple[Path, Path]:
    npz_path = Path(path)
    if npz_path.suffix != ".npz":
        raise PureFeatureArtifactError("pure feature artifact path must end in .npz")
    return npz_path, npz_path.with_suffix(".manifest.json")


@dataclass(frozen=True, slots=True)
class PureFeatureArtifact:
    """Exact filesystem and logical identities for one persisted feature pair."""

    npz_path: Path
    manifest_path: Path
    npz_sha256: str
    manifest_sha256: str
    pair_digest: str
    prefix_row_count: int
    query_row_count: int
    feature_count: int
    npz_size_bytes: int
    manifest_size_bytes: int

    def __post_init__(self) -> None:
        for name in ("npz_sha256", "manifest_sha256", "pair_digest"):
            object.__setattr__(self, name, _require_digest(getattr(self, name), name))
        for name in (
            "prefix_row_count",
            "query_row_count",
            "feature_count",
            "npz_size_bytes",
            "manifest_size_bytes",
        ):
            _require_positive_int(getattr(self, name), name)

    def manifest(self) -> dict[str, object]:
        """Return the path-independent artifact identity used by campaign evidence."""

        return {
            "schema_version": PURE_FEATURE_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": PURE_FEATURE_ARTIFACT_TYPE,
            "npz_sha256": self.npz_sha256,
            "manifest_sha256": self.manifest_sha256,
            "pair_digest": self.pair_digest,
            "prefix_row_count": self.prefix_row_count,
            "query_row_count": self.query_row_count,
            "feature_count": self.feature_count,
            "npz_size_bytes": self.npz_size_bytes,
            "manifest_size_bytes": self.manifest_size_bytes,
        }


def _write_npz(handle: BinaryIO, pair: PureFeaturePair) -> None:
    try:
        handle.seek(0)
        handle.truncate(0)
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name, values in (
                (_NPZ_MEMBERS[0], pair.prefix.values),
                (_NPZ_MEMBERS[1], pair.query.values),
            ):
                info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                with archive.open(info, mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member,
                        values.astype("<f8", copy=False),
                        allow_pickle=False,
                    )
        handle.flush()
        os.fsync(handle.fileno())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise PureFeatureArtifactError(f"cannot encode pure feature NPZ: {exc}") from exc


def _hash_handle(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        handle.seek(0)
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        handle.seek(0)
    except OSError as exc:
        raise PureFeatureArtifactError(f"cannot hash pure feature artifact: {exc}") from exc
    if size <= 0:
        raise PureFeatureArtifactError("pure feature artifact cannot be empty")
    return digest.hexdigest(), size


def _write_payload(handle: BinaryIO, payload: bytes) -> None:
    if not payload:
        raise PureFeatureArtifactError("pure feature manifest cannot be empty")
    try:
        handle.seek(0)
        handle.truncate(0)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
    except OSError as exc:
        raise PureFeatureArtifactError(f"cannot write pure feature manifest: {exc}") from exc


type _FileIdentity = tuple[int, int]


def _file_identity(status: os.stat_result) -> _FileIdentity:
    return status.st_dev, status.st_ino


def _unlink_if_identity(path: Path, identity: _FileIdentity) -> None:
    try:
        if _file_identity(path.stat(follow_symlinks=False)) == identity:
            path.unlink()
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _recoverable_npz_matches(path: Path, *, sha256: str, size_bytes: int) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            status = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or bool(stat.S_IMODE(status.st_mode) & 0o222)
                or status.st_size != size_bytes
            ):
                return False
            observed_digest, observed_size = _hash_handle(cast(BinaryIO, handle))
            final_status = os.fstat(handle.fileno())
            return (
                observed_digest == sha256
                and observed_size == size_bytes
                and _file_identity(final_status) == _file_identity(status)
                and final_status.st_size == status.st_size
                and final_status.st_ctime_ns == status.st_ctime_ns
                and final_status.st_mtime_ns == status.st_mtime_ns
            )
    except OSError:
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _install_pair(
    *,
    staged_npz: Path,
    npz_path: Path,
    staged_manifest: Path,
    manifest_path: Path,
    staged_npz_identity: _FileIdentity,
    staged_manifest_identity: _FileIdentity,
    npz_sha256: str,
    npz_size_bytes: int,
) -> None:
    installed_npz = False
    installed_manifest = False
    recovered_npz = False
    try:
        if os.path.lexists(manifest_path):
            raise PureFeatureArtifactError(
                "pure feature artifact already exists and cannot be overwritten"
            )
        if os.path.lexists(npz_path):
            if not _recoverable_npz_matches(
                npz_path,
                sha256=npz_sha256,
                size_bytes=npz_size_bytes,
            ):
                raise PureFeatureArtifactError(
                    "pure feature artifact already exists and cannot be overwritten"
                )
            recovered_npz = True
            _unlink_if_identity(staged_npz, staged_npz_identity)
        else:
            os.link(staged_npz, npz_path)
            installed_npz = True
            published_npz_identity = _file_identity(npz_path.stat(follow_symlinks=False))
            if published_npz_identity != staged_npz_identity:
                _unlink_if_identity(npz_path, published_npz_identity)
                installed_npz = False
                raise PureFeatureArtifactError("published pure feature NPZ identity changed")
            _unlink_if_identity(staged_npz, staged_npz_identity)
        _fsync_directory(npz_path.parent)

        try:
            os.link(staged_manifest, manifest_path)
        except FileExistsError as exc:
            raise PureFeatureArtifactError(
                "pure feature artifact already exists and cannot be overwritten"
            ) from exc
        installed_manifest = True
        published_manifest_identity = _file_identity(manifest_path.stat(follow_symlinks=False))
        if published_manifest_identity != staged_manifest_identity:
            _unlink_if_identity(manifest_path, published_manifest_identity)
            installed_manifest = False
            raise PureFeatureArtifactError("published pure feature manifest identity changed")
        _unlink_if_identity(staged_manifest, staged_manifest_identity)
        if recovered_npz and not _recoverable_npz_matches(
            npz_path,
            sha256=npz_sha256,
            size_bytes=npz_size_bytes,
        ):
            raise PureFeatureArtifactError("recovered pure feature NPZ identity changed")
        _fsync_directory(npz_path.parent)
    except (OSError, PureFeatureArtifactError) as exc:
        if installed_manifest:
            _unlink_if_identity(manifest_path, staged_manifest_identity)
        if installed_npz:
            _unlink_if_identity(npz_path, staged_npz_identity)
        with suppress(OSError):
            _fsync_directory(npz_path.parent)
        if isinstance(exc, PureFeatureArtifactError):
            raise
        raise PureFeatureArtifactError(
            f"cannot atomically install pure feature artifact: {exc}"
        ) from exc


def save_pure_feature_pair(path: Path | str, pair: PureFeaturePair) -> PureFeatureArtifact:
    """Persist canonical NPZ and JSON identities without replacing existing bytes."""

    if not isinstance(pair, PureFeaturePair):
        raise PureFeatureArtifactError("pair must be PureFeaturePair")
    npz_path, manifest_path = _artifact_paths(path)
    try:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PureFeatureArtifactError(f"cannot create feature artifact directory: {exc}") from exc

    staged_npz: Path | None = None
    staged_manifest: Path | None = None
    staged_npz_identity: _FileIdentity | None = None
    staged_manifest_identity: _FileIdentity | None = None
    npz_handle: BinaryIO | None = None
    manifest_handle: BinaryIO | None = None
    try:
        npz_descriptor, npz_name = tempfile.mkstemp(
            prefix=f".{npz_path.name}.", suffix=".tmp", dir=npz_path.parent
        )
        staged_npz = Path(npz_name)
        npz_handle = cast(BinaryIO, os.fdopen(npz_descriptor, "w+b"))
        staged_npz_identity = _file_identity(os.fstat(npz_handle.fileno()))
        _write_npz(npz_handle, pair)
        npz_sha256, npz_size = _hash_handle(npz_handle)
        if npz_size > _MAX_NPZ_BYTES:
            raise PureFeatureArtifactError("pure feature NPZ size is outside the supported bound")
        os.fchmod(npz_handle.fileno(), 0o444)
        os.fsync(npz_handle.fileno())

        storage_manifest = {
            "schema_version": PURE_FEATURE_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": PURE_FEATURE_ARTIFACT_TYPE,
            "npz": {
                "format": "canonical-npz",
                "members": list(_NPZ_MEMBERS),
                "dtype": "<f8",
                "sha256": npz_sha256,
                "size_bytes": npz_size,
            },
            "pair_digest": pair.digest,
            "pair": pair.manifest(),
        }
        manifest_payload = _canonical_json(storage_manifest)
        if len(manifest_payload) > _MAX_MANIFEST_BYTES:
            raise PureFeatureArtifactError(
                "pure feature manifest size is outside the supported bound"
            )
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        manifest_descriptor, manifest_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=npz_path.parent
        )
        staged_manifest = Path(manifest_name)
        manifest_handle = cast(BinaryIO, os.fdopen(manifest_descriptor, "w+b"))
        staged_manifest_identity = _file_identity(os.fstat(manifest_handle.fileno()))
        _write_payload(manifest_handle, manifest_payload)
        os.fchmod(manifest_handle.fileno(), 0o444)
        os.fsync(manifest_handle.fileno())

        _install_pair(
            staged_npz=staged_npz,
            npz_path=npz_path,
            staged_manifest=staged_manifest,
            manifest_path=manifest_path,
            staged_npz_identity=staged_npz_identity,
            staged_manifest_identity=staged_manifest_identity,
            npz_sha256=npz_sha256,
            npz_size_bytes=npz_size,
        )
        return PureFeatureArtifact(
            npz_path=npz_path.resolve(),
            manifest_path=manifest_path.resolve(),
            npz_sha256=npz_sha256,
            manifest_sha256=manifest_sha256,
            pair_digest=pair.digest,
            prefix_row_count=pair.prefix.row_count,
            query_row_count=pair.query.row_count,
            feature_count=pair.prefix.feature_count,
            npz_size_bytes=npz_size,
            manifest_size_bytes=len(manifest_payload),
        )
    finally:
        for handle in (npz_handle, manifest_handle):
            if handle is not None:
                handle.close()
        for staged, identity in (
            (staged_npz, staged_npz_identity),
            (staged_manifest, staged_manifest_identity),
        ):
            if staged is not None and identity is not None:
                _unlink_if_identity(staged, identity)


@contextmanager
def _checked_file(
    path: Path,
    *,
    maximum: int,
    kind: str,
) -> Iterator[tuple[BinaryIO, str, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    handle: BinaryIO | None = None
    try:
        descriptor = os.open(path, flags)
        handle = cast(BinaryIO, os.fdopen(descriptor, "rb"))
        descriptor = -1
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise PureFeatureArtifactError(
                f"pure feature {kind} must be a regular non-symlink, non-hardlinked file"
            )
        if status.st_size <= 0 or status.st_size > maximum:
            raise PureFeatureArtifactError(
                f"pure feature {kind} size is outside the supported bound"
            )
        digest = hashlib.sha256()
        observed_size = 0
        while chunk := handle.read(1024 * 1024):
            observed_size += len(chunk)
            digest.update(chunk)
        verified_status = os.fstat(handle.fileno())
        if (
            observed_size != status.st_size
            or _file_identity(verified_status) != _file_identity(status)
            or verified_status.st_size != status.st_size
            or verified_status.st_ctime_ns != status.st_ctime_ns
            or verified_status.st_mtime_ns != status.st_mtime_ns
            or stat.S_IMODE(verified_status.st_mode) != stat.S_IMODE(status.st_mode)
        ):
            raise PureFeatureArtifactError(
                f"pure feature {kind} changed while its identity was verified"
            )
        initial_digest = digest.hexdigest()
        handle.seek(0)
        yield handle, initial_digest, observed_size
        final_digest, final_size = _hash_handle(handle)
        final_status = os.fstat(handle.fileno())
        if (
            final_digest != initial_digest
            or final_size != observed_size
            or _file_identity(final_status) != _file_identity(status)
            or final_status.st_size != status.st_size
            or final_status.st_ctime_ns != status.st_ctime_ns
            or final_status.st_mtime_ns != status.st_mtime_ns
            or stat.S_IMODE(final_status.st_mode) != stat.S_IMODE(status.st_mode)
        ):
            raise PureFeatureArtifactError(f"pure feature {kind} changed while it was decoded")
    except OSError as exc:
        raise PureFeatureArtifactError(
            f"pure feature {kind} must be a readable regular non-symlink file"
        ) from exc
    finally:
        if handle is not None:
            handle.close()
        if descriptor >= 0:
            os.close(descriptor)


def _checked_manifest(path: Path) -> tuple[bytes, str]:
    with _checked_file(path, maximum=_MAX_MANIFEST_BYTES, kind="manifest") as checked:
        handle, digest, size = checked
        payload = handle.read(_MAX_MANIFEST_BYTES + 1)
    if len(payload) != size:
        raise PureFeatureArtifactError("pure feature manifest changed while it was read")
    return payload, digest


def _parse_manifest(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("ascii")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PureFeatureArtifactError("pure feature manifest is not canonical JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise PureFeatureArtifactError("pure feature manifest is not a canonical JSON object")
    return cast(dict[str, object], value)


def _require_mapping(value: object, *, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PureFeatureArtifactError(f"pure feature {name} fields do not match the exact schema")
    if any(not isinstance(key, str) for key in value):
        raise PureFeatureArtifactError(f"pure feature {name} keys must be strings")
    return cast(Mapping[str, object], value)


def _matrix_manifest(
    value: object,
    *,
    name: str,
) -> tuple[Mapping[str, object], int, int, tuple[str, ...]]:
    manifest = _require_mapping(
        value,
        fields={"row_count", "feature_count", "feature_names", "logical_digest"},
        name=f"{name} matrix manifest",
    )
    row_count = _require_positive_int(manifest.get("row_count"), f"{name}.row_count")
    feature_count = _require_positive_int(
        manifest.get("feature_count"),
        f"{name}.feature_count",
    )
    names_value = manifest.get("feature_names")
    if not isinstance(names_value, list) or any(not isinstance(name, str) for name in names_value):
        raise PureFeatureArtifactError("pure feature names must be a JSON string array")
    names = tuple(cast(list[str], names_value))
    if len(names) != feature_count or len(set(names)) != len(names):
        raise PureFeatureArtifactError("pure feature names do not match the declared schema")
    _require_digest(manifest.get("logical_digest"), f"{name}.logical_digest")
    return manifest, row_count, feature_count, names


def _validate_npy_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_shape: tuple[int, int],
) -> None:
    try:
        with archive.open(info, mode="r") as handle:
            if np.lib.format.read_magic(handle) != (1, 0):
                raise PureFeatureArtifactError(
                    "pure feature NPY member must use canonical version 1.0"
                )
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            header_bytes = handle.tell()
    except PureFeatureArtifactError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise PureFeatureArtifactError(f"cannot decode pure feature NPY header: {exc}") from exc
    if shape != expected_shape:
        raise PureFeatureArtifactError("pure feature NPY shape does not match its manifest")
    if fortran_order or dtype.str != "<f8":
        raise PureFeatureArtifactError(
            "pure feature NPY member must be C-order little-endian float64"
        )
    payload_bytes = expected_shape[0] * expected_shape[1] * np.dtype("<f8").itemsize
    if payload_bytes > _MAX_NPZ_BYTES or header_bytes + payload_bytes != info.file_size:
        raise PureFeatureArtifactError(
            "pure feature NPY member size does not match its declared shape"
        )
    expected_header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        expected_header,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype("<f8")),
            "fortran_order": False,
            "shape": expected_shape,
        },
    )
    with archive.open(info, mode="r") as handle:
        observed_header = handle.read(header_bytes)
    if observed_header != expected_header.getvalue():
        raise PureFeatureArtifactError("pure feature NPY header is not canonical")


def _read_matrix(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    manifest: Mapping[str, object],
    row_count: int,
    feature_count: int,
    feature_names: tuple[str, ...],
) -> FeatureMatrix:
    _validate_npy_member(
        archive,
        info,
        expected_shape=(row_count, feature_count),
    )
    try:
        with archive.open(info, mode="r") as handle:
            values = np.lib.format.read_array(handle, allow_pickle=False)
        if (
            values.dtype.str != "<f8"
            or values.shape != (row_count, feature_count)
            or not values.flags.c_contiguous
        ):
            raise PureFeatureArtifactError(
                "pure feature matrix storage does not match its exact schema"
            )
        matrix = FeatureMatrix(values, feature_names)
    except PureFeatureArtifactError:
        raise
    except (OSError, ValueError, EOFError, MemoryError) as exc:
        raise PureFeatureArtifactError(f"cannot decode pure feature matrix: {exc}") from exc
    if _canonical_json(matrix.manifest()) != _canonical_json(manifest):
        raise PureFeatureArtifactError(
            "pure feature matrix logical manifest does not match its values"
        )
    return matrix


def _decode_pair(npz_handle: BinaryIO, stored_pair: Mapping[str, object]) -> PureFeaturePair:
    if type(stored_pair.get("schema_version")) is not int:
        raise PureFeatureArtifactError("pure feature pair schema_version must be an integer")
    prefix_manifest, prefix_rows, prefix_features, prefix_names = _matrix_manifest(
        stored_pair.get("prefix"),
        name="prefix",
    )
    query_manifest, query_rows, query_features, query_names = _matrix_manifest(
        stored_pair.get("query"),
        name="query",
    )
    if query_features != prefix_features or query_names != prefix_names:
        raise PureFeatureArtifactError("pure feature prefix and query schemas differ")
    try:
        npz_handle.seek(0)
        with zipfile.ZipFile(npz_handle, mode="r") as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != _NPZ_MEMBERS:
                raise PureFeatureArtifactError(
                    "pure feature NPZ members are missing, duplicated, or reordered"
                )
            if archive.comment:
                raise PureFeatureArtifactError("pure feature NPZ archive comment is not canonical")
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.flag_bits != 0
                or info.file_size <= 0
                or info.file_size > _MAX_NPZ_BYTES
                or info.date_time != _ZIP_TIMESTAMP
                or info.create_system != 3
                or info.create_version != 45
                or info.extract_version != 45
                or info.external_attr != 0o600 << 16
                or bool(info.comment)
                or bool(info.extra)
                for info in infos
            ):
                raise PureFeatureArtifactError("pure feature NPZ member encoding is invalid")
            if sum(info.file_size for info in infos) > _MAX_NPZ_BYTES:
                raise PureFeatureArtifactError(
                    "pure feature NPZ payload exceeds the supported bound"
                )
            prefix = _read_matrix(
                archive,
                infos[0],
                manifest=prefix_manifest,
                row_count=prefix_rows,
                feature_count=prefix_features,
                feature_names=prefix_names,
            )
            query = _read_matrix(
                archive,
                infos[1],
                manifest=query_manifest,
                row_count=query_rows,
                feature_count=query_features,
                feature_names=query_names,
            )
    except PureFeatureArtifactError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        raise PureFeatureArtifactError(f"cannot decode pure feature NPZ: {exc}") from exc

    raw_categorical_codes = stored_pair.get("categorical_codes")
    categorical_encoding_digest = (
        cast(str | None, raw_categorical_codes.get("encoding_digest"))
        if isinstance(raw_categorical_codes, Mapping)
        else None
    )
    raw_auxiliary_history = stored_pair.get("auxiliary_history")
    auxiliary_history_cache_key = (
        cast(str | None, raw_auxiliary_history.get("causal_cache_key"))
        if isinstance(raw_auxiliary_history, Mapping)
        else None
    )
    input_exposure: StrictPastExposurePair | None = None
    raw_input_exposure = stored_pair.get("input_exposure")
    if stored_pair.get("schema_version") == 8:
        if not isinstance(raw_input_exposure, Mapping):
            raise PureFeatureArtifactError(
                "schema-v8 input-exposure manifest must be an object"
            )
        width = len(STRICT_PAST_EXPOSURE_FEATURE_NAMES)
        if prefix.feature_names[-width:] != STRICT_PAST_EXPOSURE_FEATURE_NAMES:
            raise PureFeatureArtifactError(
                "schema-v8 feature matrix lacks the fixed input-exposure suffix"
            )
        try:
            input_exposure = StrictPastExposurePair(
                prefix=FeatureMatrix(
                    prefix.values[:, -width:], STRICT_PAST_EXPOSURE_FEATURE_NAMES
                ),
                query=FeatureMatrix(
                    query.values[:, -width:], STRICT_PAST_EXPOSURE_FEATURE_NAMES
                ),
                prefix_input_digest=cast(
                    str, raw_input_exposure.get("prefix_input_digest")
                ),
                query_input_digest=cast(
                    str, raw_input_exposure.get("query_input_digest")
                ),
                builder_source_digest=cast(
                    str, raw_input_exposure.get("builder_source_digest")
                ),
            )
            if input_exposure.digest != _require_digest(
                raw_input_exposure.get("build_digest"), "input_exposure.build_digest"
            ):
                raise PureFeatureArtifactError(
                    "input-exposure build digest does not match its values"
                )
        except StrictPastExposureError as exc:
            raise PureFeatureArtifactError(
                f"input-exposure values or identity are invalid: {exc}"
            ) from exc
    try:
        pair = PureFeaturePair(
            prefix=prefix,
            query=query,
            dataset_digest=cast(str, stored_pair.get("dataset_digest")),
            split_role=cast(str, stored_pair.get("split_role")),
            causal_cache_key=cast(str, stored_pair.get("causal_cache_key")),
            categorical_encoding_digest=categorical_encoding_digest,
            auxiliary_history_cache_key=auxiliary_history_cache_key,
            input_exposure=input_exposure,
            feature_schema_version=cast(int, stored_pair.get("schema_version")),
            feature_policy=cast(str, stored_pair.get("policy")),
        )
    except (TypeError, ValueError, CausalFeatureError, StrictPastExposureError) as exc:
        raise PureFeatureArtifactError(
            f"pure feature values or identity are invalid: {exc}"
        ) from exc
    if _canonical_json(pair.manifest()) != _canonical_json(stored_pair):
        raise PureFeatureArtifactError("pure feature logical manifest does not match its values")
    return pair


def _load_verified(
    path: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_npz_sha256: str | None = None,
    expected_pair_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    expected_split_role: str | None = None,
    expected_prefix_row_count: int | None = None,
    expected_query_row_count: int | None = None,
    expected_feature_count: int | None = None,
) -> tuple[PureFeaturePair, PureFeatureArtifact]:
    npz_path, manifest_path = _artifact_paths(path)
    expected_manifest = _require_digest(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    manifest_payload, manifest_sha256 = _checked_manifest(manifest_path)
    if manifest_sha256 != expected_manifest:
        raise PureFeatureArtifactError("pure feature manifest SHA-256 mismatch")
    manifest = _parse_manifest(manifest_payload)
    top = _require_mapping(
        manifest,
        fields={"schema_version", "artifact_type", "npz", "pair_digest", "pair"},
        name="artifact manifest",
    )
    if (
        type(top.get("schema_version")) is not int
        or top.get("schema_version") != PURE_FEATURE_ARTIFACT_SCHEMA_VERSION
        or top.get("artifact_type") != PURE_FEATURE_ARTIFACT_TYPE
    ):
        raise PureFeatureArtifactError("pure feature artifact schema or type is unsupported")
    npz_manifest = _require_mapping(
        top.get("npz"),
        fields={"format", "members", "dtype", "sha256", "size_bytes"},
        name="NPZ manifest",
    )
    if (
        npz_manifest.get("format") != "canonical-npz"
        or npz_manifest.get("members") != list(_NPZ_MEMBERS)
        or npz_manifest.get("dtype") != "<f8"
    ):
        raise PureFeatureArtifactError("pure feature NPZ manifest is invalid")
    declared_npz_digest = _require_digest(npz_manifest.get("sha256"), "npz.sha256")
    declared_npz_size = _require_positive_int(npz_manifest.get("size_bytes"), "npz.size_bytes")
    raw_pair = top.get("pair")
    if not isinstance(raw_pair, Mapping):
        raise PureFeatureArtifactError("pure feature logical pair manifest must be an object")
    pair_schema_version = raw_pair.get("schema_version")
    pair_fields = {
        "schema_version",
        "policy",
        "dataset_digest",
        "split_role",
        "aggregate_specs",
        "static_features",
        "causal_cache_key",
        "prefix",
        "query",
    }
    if pair_schema_version in {3, 4}:
        pair_fields.update(("recency", "categorical_codes"))
    elif pair_schema_version == 5:
        pair_fields.update(("recency", "categorical_codes", "auxiliary_history"))
    elif pair_schema_version in {6, 7, 8}:
        pair_fields.update(
            (
                "recency",
                "categorical_codes",
                "auxiliary_history",
                "watch_progress_history",
            )
        )
        if pair_schema_version == 7:
            pair_fields.add("video_type_code")
        elif pair_schema_version == 8:
            pair_fields.update(("video_type_code", "input_exposure"))
    elif pair_schema_version == 2:
        pair_fields.add("recency")
    elif pair_schema_version != 1:
        raise PureFeatureArtifactError("pure feature logical pair schema is unsupported")
    stored_pair = _require_mapping(
        raw_pair,
        fields=pair_fields,
        name="logical pair manifest",
    )
    with _checked_file(npz_path, maximum=_MAX_NPZ_BYTES, kind="NPZ") as checked_npz:
        npz_handle, npz_sha256, npz_size = checked_npz
        if npz_sha256 != declared_npz_digest:
            raise PureFeatureArtifactError("pure feature NPZ SHA-256 mismatch")
        if expected_npz_sha256 is not None and npz_sha256 != _require_digest(
            expected_npz_sha256,
            "expected_npz_sha256",
        ):
            raise PureFeatureArtifactError("pure feature expected NPZ SHA-256 mismatch")
        if npz_size != declared_npz_size:
            raise PureFeatureArtifactError("pure feature NPZ size mismatch")
        pair = _decode_pair(npz_handle, stored_pair)
    declared_pair_digest = _require_digest(top.get("pair_digest"), "pair_digest")
    if pair.digest != declared_pair_digest:
        raise PureFeatureArtifactError("pure feature pair logical digest mismatch")
    if expected_pair_digest is not None and pair.digest != _require_digest(
        expected_pair_digest,
        "expected_pair_digest",
    ):
        raise PureFeatureArtifactError("pure feature expected pair digest mismatch")
    if expected_dataset_digest is not None and pair.dataset_digest != _require_digest(
        expected_dataset_digest,
        "expected_dataset_digest",
    ):
        raise PureFeatureArtifactError("pure feature dataset digest mismatch")
    if expected_split_role is not None:
        if (
            not isinstance(expected_split_role, str)
            or not expected_split_role
            or "\n" in expected_split_role
            or "\r" in expected_split_role
        ):
            raise PureFeatureArtifactError("expected_split_role must be non-empty single-line text")
        if pair.split_role != expected_split_role:
            raise PureFeatureArtifactError("pure feature split role mismatch")
    for expected, observed, name in (
        (expected_prefix_row_count, pair.prefix.row_count, "prefix row count"),
        (expected_query_row_count, pair.query.row_count, "query row count"),
        (expected_feature_count, pair.prefix.feature_count, "feature count"),
    ):
        if expected is not None:
            normalized = _require_positive_int(expected, f"expected_{name.replace(' ', '_')}")
            if observed != normalized:
                raise PureFeatureArtifactError(f"pure feature {name} mismatch")
    artifact = PureFeatureArtifact(
        npz_path=npz_path.resolve(),
        manifest_path=manifest_path.resolve(),
        npz_sha256=npz_sha256,
        manifest_sha256=manifest_sha256,
        pair_digest=pair.digest,
        prefix_row_count=pair.prefix.row_count,
        query_row_count=pair.query.row_count,
        feature_count=pair.prefix.feature_count,
        npz_size_bytes=npz_size,
        manifest_size_bytes=len(manifest_payload),
    )
    return pair, artifact


def load_pure_feature_pair(
    path: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_npz_sha256: str | None = None,
    expected_pair_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    expected_split_role: str | None = None,
    expected_prefix_row_count: int | None = None,
    expected_query_row_count: int | None = None,
    expected_feature_count: int | None = None,
) -> PureFeaturePair:
    """Load exact matrices only after verifying the externally anchored JSON identity."""

    pair, _ = _load_verified(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_npz_sha256=expected_npz_sha256,
        expected_pair_digest=expected_pair_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_split_role=expected_split_role,
        expected_prefix_row_count=expected_prefix_row_count,
        expected_query_row_count=expected_query_row_count,
        expected_feature_count=expected_feature_count,
    )
    return pair


def verify_pure_feature_artifact(
    path: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_npz_sha256: str | None = None,
    expected_pair_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    expected_split_role: str | None = None,
    expected_prefix_row_count: int | None = None,
    expected_query_row_count: int | None = None,
    expected_feature_count: int | None = None,
) -> PureFeatureArtifact:
    """Decode and validate an artifact, returning its exact path-independent identities."""

    _, artifact = _load_verified(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_npz_sha256=expected_npz_sha256,
        expected_pair_digest=expected_pair_digest,
        expected_dataset_digest=expected_dataset_digest,
        expected_split_role=expected_split_role,
        expected_prefix_row_count=expected_prefix_row_count,
        expected_query_row_count=expected_query_row_count,
        expected_feature_count=expected_feature_count,
    )
    return artifact


__all__ = [
    "PURE_FEATURE_ARTIFACT_SCHEMA_VERSION",
    "PURE_FEATURE_ARTIFACT_TYPE",
    "PureFeatureArtifact",
    "PureFeatureArtifactError",
    "load_pure_feature_pair",
    "save_pure_feature_pair",
    "verify_pure_feature_artifact",
]

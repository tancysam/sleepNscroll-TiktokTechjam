"""Atomic qualification coordinator for the immutable KuaiRand-Pure baselines.

The public interface is intentionally one deep operation: :func:`run_qualification`.  It owns
the required sequence, fail-fast policy, six-launch accounting, replay checks, high-precision
submission round trips, immutable fallback selection, evidence indexing, and atomic no-overwrite
commit.  Model and rung computation sit behind one structural backend seam because both a local
organizer adapter and a deterministic fixture adapter are required.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.contract import BENCHMARK_CONTRACT
from kuairand_agent.scoring.submission import (
    AlignmentRow,
    ValidatedSubmission,
    compare_within_user_order,
    prediction_digest,
    write_submission,
)

QUALIFICATION_SCHEMA_VERSION: Final = 1
FM_SEEDS: Final = (0, 1, 2, 3, 4)
RANDOM_SEEDS: Final = (0, 1, 2, 3, 4)
EXPECTED_TRAINING_LAUNCHES: Final = 6
_FOUR_PLACES: Final = Decimal("0.0001")
_METRIC_TOLERANCE: Final = 1e-15
_FLOAT32_METRIC_TOLERANCE: Final = float(np.finfo(np.float32).eps)
_INTEGER_LABEL_PROTOCOL: Final = "integer_labels_float64"
_ENCODED_LABEL_PROTOCOL: Final = "encoded_labels_float32"
_RETRAIN_ABSOLUTE_TOLERANCE: Final = 0.0
_RENAME_NOREPLACE: Final = 1
_AT_FDCWD: Final = -100

type Float64Vector = npt.NDArray[np.float64]
type JsonMapping = Mapping[str, object]


class QualificationError(RuntimeError):
    """Raised when any mandatory qualification gate fails."""


def _lower_sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationError(f"{location} must be a lowercase SHA-256")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise QualificationError(f"{location} must be non-empty text without NUL")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise QualificationError("qualification evidence is not canonical JSON") from exc
    return (rendered + "\n").encode("ascii")


def _json_digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value).rstrip(b"\n")).hexdigest()


def _scores(values: Iterable[object], expected_count: int, location: str) -> Float64Vector:
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
        if array.ndim != 1 or array.dtype.kind not in "iuf":
            raise QualificationError(f"{location} must be a one-dimensional real vector")
        normalized = np.ascontiguousarray(array, dtype=np.float64)
    else:
        converted: list[float] = []
        for index, raw in enumerate(values):
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.number)):
                raise QualificationError(f"{location}[{index}] must be a real number")
            converted.append(float(raw))
        normalized = np.ascontiguousarray(converted, dtype=np.float64)
    if len(normalized) != expected_count:
        raise QualificationError(
            f"{location} row count mismatch: expected {expected_count}, got {len(normalized)}"
        )
    if not np.isfinite(normalized).all():
        raise QualificationError(f"{location} must contain only finite values")
    normalized.setflags(write=False)
    return normalized


def _labels(values: Iterable[object], expected_count: int) -> npt.NDArray[np.int8]:
    array = np.asarray(tuple(values))
    if array.ndim != 1 or len(array) != expected_count:
        raise QualificationError("validation labels do not match validation alignment")
    if array.dtype.kind not in "biuf" or not np.logical_or(array == 0, array == 1).all():
        raise QualificationError("validation labels must be binary")
    result = np.ascontiguousarray(array, dtype=np.int8)
    result.setflags(write=False)
    return result


def _rounded(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_FOUR_PLACES, rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class QualificationMetrics:
    """Exact protected validation metric triplet."""

    gauc: float
    ndcg_at_5: float
    primary: float
    label_protocol: str = _INTEGER_LABEL_PROTOCOL

    def __post_init__(self) -> None:
        values = (self.gauc, self.ndcg_at_5, self.primary)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise QualificationError("qualification metrics must be finite built-in floats")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise QualificationError("qualification metrics must be in [0, 1]")
        if self.label_protocol not in {_INTEGER_LABEL_PROTOCOL, _ENCODED_LABEL_PROTOCOL}:
            raise QualificationError("qualification metrics use an unknown label protocol")
        expected = (self.gauc + self.ndcg_at_5) / 2.0
        tolerance = (
            _METRIC_TOLERANCE
            if self.label_protocol == _INTEGER_LABEL_PROTOCOL
            else _FLOAT32_METRIC_TOLERANCE
        )
        if not math.isclose(self.primary, expected, rel_tol=0.0, abs_tol=tolerance):
            raise QualificationError(
                "primary differs from the mean beyond its declared label-arithmetic tolerance"
            )

    def manifest(self) -> dict[str, float]:
        return {"GAUC": self.gauc, "nDCG@5": self.ndcg_at_5, "primary": self.primary}


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Per-operation resource evidence supplied by the trusted local backend."""

    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0
            for value in (self.wall_seconds, self.cpu_seconds)
        ):
            raise QualificationError("resource times must be finite non-negative floats")
        if type(self.peak_rss_bytes) is not int or self.peak_rss_bytes < 0:
            raise QualificationError("peak_rss_bytes must be a non-negative integer")
        _text(self.device, "resource device")

    def manifest(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class QualificationRequest:
    """Paths for one fresh, fixed-policy official qualification run."""

    data_dir: Path
    starter_dir: Path
    run_dir: Path

    def __post_init__(self) -> None:
        for name in ("data_dir", "starter_dir", "run_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise QualificationError(f"{name} must be a pathlib.Path")
            if "\x00" in os.fspath(value):
                raise QualificationError(f"{name} must not contain NUL")


@dataclass(frozen=True, slots=True)
class QualificationSnapshot:
    """One independently rebuilt trusted input identity plus opaque backend payload."""

    starter_manifest_digest: str
    audit_digest: str
    audit_manifest: JsonMapping
    canonical_digest: str
    canonical_manifest: JsonMapping
    evaluator_golden_digest: str
    evaluator_golden_passed: bool
    validation_alignment: Sequence[AlignmentRow]
    validation_labels: Iterable[object] = field(repr=False)
    final_alignment: Sequence[AlignmentRow] = field(repr=False)
    payload: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        for name in (
            "starter_manifest_digest",
            "audit_digest",
            "canonical_digest",
            "evaluator_golden_digest",
        ):
            object.__setattr__(self, name, _lower_sha256(getattr(self, name), name))
        if self.evaluator_golden_passed is not True:
            raise QualificationError("organizer evaluator golden gate did not pass")
        validation = tuple(self.validation_alignment)
        final = tuple(self.final_alignment)
        if not validation or not final:
            raise QualificationError("qualification alignments must be non-empty")
        if tuple(row.row_id for row in validation) != tuple(range(len(validation))):
            raise QualificationError("validation alignment is not canonical")
        if tuple(row.row_id for row in final) != tuple(range(len(final))):
            raise QualificationError("final alignment is not canonical")
        normalized_labels = _labels(self.validation_labels, len(validation))
        # Prove manifests are serializable before any model launch.
        _json_bytes(dict(self.audit_manifest))
        _json_bytes(dict(self.canonical_manifest))
        object.__setattr__(self, "validation_alignment", validation)
        object.__setattr__(self, "validation_labels", normalized_labels)
        object.__setattr__(self, "final_alignment", final)

    @property
    def validation_count(self) -> int:
        return len(self.validation_alignment)

    @property
    def final_count(self) -> int:
        return len(self.final_alignment)

    @property
    def validation_label_digest(self) -> str:
        labels = cast(npt.NDArray[np.int8], self.validation_labels)
        digest = hashlib.sha256(b"kuairand-qualification-valid-labels-v1\0")
        digest.update(labels.tobytes(order="C"))
        return digest.hexdigest()

    def identity_manifest(self) -> dict[str, object]:
        return {
            "benchmark_digest": BENCHMARK_CONTRACT.digest,
            "starter_manifest_digest": self.starter_manifest_digest,
            "audit_digest": self.audit_digest,
            "audit_manifest": dict(self.audit_manifest),
            "canonical_digest": self.canonical_digest,
            "canonical_manifest": dict(self.canonical_manifest),
            "evaluator_golden_digest": self.evaluator_golden_digest,
            "evaluator_golden_passed": self.evaluator_golden_passed,
            "validation_alignment_count": self.validation_count,
            "validation_label_digest": self.validation_label_digest,
            "final_alignment_count": self.final_count,
            "final_target_capability": None,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self.identity_manifest())


@dataclass(frozen=True, slots=True)
class RungEvaluationEvidence:
    """One deterministic protected baseline evaluation."""

    seed: int | None
    metrics: QualificationMetrics
    users: int
    rows: int
    scorer_digest: str
    prediction_digest: str
    split_digest: str
    runtime_seconds: float

    def __post_init__(self) -> None:
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise QualificationError("rung seed must be a non-negative integer or None")
        if type(self.users) is not int or self.users <= 0:
            raise QualificationError("rung users must be positive")
        if type(self.rows) is not int or self.rows <= 0:
            raise QualificationError("rung rows must be positive")
        for name in ("scorer_digest", "prediction_digest", "split_digest"):
            _lower_sha256(getattr(self, name), f"rung {name}")
        if (
            type(self.runtime_seconds) is not float
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0
        ):
            raise QualificationError("rung runtime must be finite and non-negative")

    def manifest(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "metrics": self.metrics.manifest(),
            "users": self.users,
            "rows": self.rows,
            "scorer_digest": self.scorer_digest,
            "prediction_digest": self.prediction_digest,
            "split_digest": self.split_digest,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True, slots=True)
class RungSummaryEvidence:
    """Aggregate random or popularity reference qualification."""

    name: str
    evaluations: tuple[RungEvaluationEvidence, ...]
    reference_metrics: QualificationMetrics
    reference_passed: bool
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.name, "rung name")
        if not self.evaluations:
            raise QualificationError("rung summary must contain evaluations")
        if self.reference_passed is not True:
            raise QualificationError(f"{self.name} reference rung did not pass")
        object.__setattr__(self, "digest", _json_digest(self._base_manifest()))

    @property
    def mean_metrics(self) -> QualificationMetrics:
        count = len(self.evaluations)
        return QualificationMetrics(
            gauc=sum(item.metrics.gauc for item in self.evaluations) / count,
            ndcg_at_5=sum(item.metrics.ndcg_at_5 for item in self.evaluations) / count,
            primary=sum(item.metrics.primary for item in self.evaluations) / count,
        )

    def _base_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "evaluations": [item.manifest() for item in self.evaluations],
            "mean_metrics": self.mean_metrics.manifest(),
            "rounded_mean": {
                key: str(_rounded(value)) for key, value in self.mean_metrics.manifest().items()
            },
            "reference_metrics": self.reference_metrics.manifest(),
            "reference_passed": self.reference_passed,
        }

    def manifest(self) -> dict[str, object]:
        return {**self._base_manifest(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class FMTrainingEvidence:
    """One FM launch and its already-persisted replay artifacts."""

    seed: int
    validation_scores: Iterable[object] = field(repr=False)
    validation_metrics: QualificationMetrics
    checkpoint_path: Path
    checkpoint_digest: str
    encoding_digest: str
    config_digest: str
    starter_manifest_digest: str
    artifact_paths: tuple[Path, ...]
    artifact_sha256: Mapping[str, str]
    training_trace: JsonMapping
    resources: ResourceUsage
    organizer_parity_passed: bool
    prediction_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FM_SEEDS:
            raise QualificationError("FM evidence seed must be one of 0..4")
        for name in (
            "checkpoint_digest",
            "encoding_digest",
            "config_digest",
            "starter_manifest_digest",
        ):
            _lower_sha256(getattr(self, name), f"FM {name}")
        if not isinstance(self.checkpoint_path, Path):
            raise QualificationError("FM checkpoint_path must be a pathlib.Path")
        paths = tuple(self.artifact_paths)
        if not paths or any(not isinstance(path, Path) for path in paths):
            raise QualificationError("FM artifact_paths must contain pathlib.Path values")
        names = tuple(path.name for path in paths)
        if len(names) != len(set(names)):
            raise QualificationError("FM artifact_paths must have unique filenames")
        hashes = dict(self.artifact_sha256)
        if set(hashes) != set(names):
            raise QualificationError("FM artifact SHA-256 evidence does not match artifact_paths")
        for name, digest in hashes.items():
            _lower_sha256(digest, f"FM artifact {name} SHA-256")
        if self.organizer_parity_passed is not True:
            raise QualificationError(f"FM seed {self.seed} did not match untouched organizer logic")
        _json_bytes(dict(self.training_trace))
        object.__setattr__(self, "artifact_paths", paths)
        object.__setattr__(self, "artifact_sha256", MappingProxyType(hashes))

    def normalize_scores(self, expected_count: int) -> Float64Vector:
        values = _scores(
            self.validation_scores,
            expected_count,
            f"FM seed {self.seed} validation predictions",
        )
        object.__setattr__(self, "validation_scores", values)
        object.__setattr__(self, "prediction_digest", prediction_digest(values))
        return values

    def manifest(self, artifacts: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not hasattr(self, "prediction_digest"):
            raise QualificationError("FM predictions were not normalized before manifesting")
        return {
            "seed": self.seed,
            "validation_metrics": self.validation_metrics.manifest(),
            "validation_prediction_digest": self.prediction_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "starter_manifest_digest": self.starter_manifest_digest,
            "artifact_file_sha256": dict(self.artifact_sha256),
            "organizer_parity_passed": self.organizer_parity_passed,
            "training_trace": dict(self.training_trace),
            "resources": self.resources.manifest(),
            "artifacts": [dict(item) for item in artifacts],
        }


@dataclass(frozen=True, slots=True)
class FMReplayEvidence:
    seed: int
    validation_scores: Iterable[object] = field(repr=False)
    validation_metrics: QualificationMetrics
    checkpoint_digest: str
    resources: ResourceUsage


@dataclass(frozen=True, slots=True)
class FinalPredictionEvidence:
    scores: Iterable[object] = field(repr=False)
    checkpoint_digest: str
    resources: ResourceUsage


class QualificationBackend(Protocol):
    """Structural seam implemented by the local organizer stack and fixture adapter."""

    def snapshot(self, request: QualificationRequest) -> QualificationSnapshot: ...

    def random_rungs(self, snapshot: QualificationSnapshot) -> RungSummaryEvidence: ...

    def popularity_rung(self, snapshot: QualificationSnapshot) -> RungSummaryEvidence: ...

    def train_fm(
        self,
        snapshot: QualificationSnapshot,
        seed: int,
        artifact_dir: Path,
    ) -> FMTrainingEvidence: ...

    def replay_fm(
        self,
        snapshot: QualificationSnapshot,
        training: FMTrainingEvidence,
    ) -> FMReplayEvidence: ...

    def score_validation(
        self,
        snapshot: QualificationSnapshot,
        scores: Float64Vector,
    ) -> QualificationMetrics: ...

    def predict_final(
        self,
        snapshot: QualificationSnapshot,
        training: FMTrainingEvidence,
    ) -> FinalPredictionEvidence: ...


@dataclass(frozen=True, slots=True)
class QualificationResult:
    run_dir: Path
    manifest_digest: str
    fallback_seed: int
    launch_count: int
    validation_metrics: QualificationMetrics
    validation_submission: ValidatedSubmission
    final_submission: ValidatedSubmission


def _published_manifest(name: str) -> dict[str, float]:
    for rung in BENCHMARK_CONTRACT.reference_rungs:
        if rung.name == name:
            return {
                "GAUC": float(rung.validation.gauc),
                "nDCG@5": float(rung.validation.ndcg_at_5),
                "primary": float(rung.validation.primary),
            }
    raise QualificationError(f"benchmark contract has no reference rung {name!r}")


def _published_metrics(name: str) -> QualificationMetrics:
    """Return an exact-formula metric object for a published rounded triplet.

    The FM reference components average to 0.60155 before the separately published primary is
    rounded to 0.6016.  Runtime metrics always retain their unrounded exact formula; published
    parity compares every component after the declared four-place rounding.
    """

    published = _published_manifest(name)
    return QualificationMetrics(
        published["GAUC"],
        published["nDCG@5"],
        (published["GAUC"] + published["nDCG@5"]) / 2.0,
    )


def _require_rung(
    summary: RungSummaryEvidence,
    *,
    name: str,
    seeds: tuple[int, ...] | None,
) -> None:
    if summary.name != name:
        raise QualificationError(f"expected {name} rung, got {summary.name}")
    observed_seeds = tuple(item.seed for item in summary.evaluations)
    expected_seeds: tuple[int | None, ...] = seeds if seeds is not None else (None,)
    if observed_seeds != expected_seeds:
        raise QualificationError(
            f"{name} rung seed order mismatch: expected {expected_seeds}, got {observed_seeds}"
        )
    published_name = "item_popularity" if name == "item_popularity" else "random"
    published_manifest = _published_manifest(published_name)
    if summary.reference_metrics.manifest() != published_manifest:
        raise QualificationError(f"{name} rung uses the wrong published reference")
    mean = summary.mean_metrics
    for metric_name, expected in published_manifest.items():
        if _rounded(mean.manifest()[metric_name]) != _rounded(expected):
            raise QualificationError(f"{name} did not reproduce published {metric_name}")


def _mean_fm_metrics(runs: Sequence[FMTrainingEvidence]) -> QualificationMetrics:
    count = len(runs)
    protocols = {run.validation_metrics.label_protocol for run in runs}
    if len(protocols) != 1:
        raise QualificationError("FM seed metrics mix incompatible label protocols")
    return QualificationMetrics(
        gauc=sum(run.validation_metrics.gauc for run in runs) / count,
        ndcg_at_5=sum(run.validation_metrics.ndcg_at_5 for run in runs) / count,
        primary=sum(run.validation_metrics.primary for run in runs) / count,
        label_protocol=protocols.pop(),
    )


def _require_fm_reference(runs: Sequence[FMTrainingEvidence]) -> QualificationMetrics:
    if tuple(run.seed for run in runs) != FM_SEEDS:
        raise QualificationError("FM qualification must run seeds 0 through 4 exactly once")
    mean = _mean_fm_metrics(runs)
    published = _published_manifest("fm_official")
    for name, expected in published.items():
        if _rounded(mean.manifest()[name]) != _rounded(expected):
            raise QualificationError(
                f"FM five-seed mean did not reproduce published {name}: "
                f"observed={mean.manifest()[name]}, reference={expected}"
            )
    return mean


def _require_metrics_equal(
    expected: QualificationMetrics,
    actual: QualificationMetrics,
    location: str,
) -> None:
    for name, expected_value in expected.manifest().items():
        actual_value = actual.manifest()[name]
        if not math.isclose(expected_value, actual_value, rel_tol=0.0, abs_tol=_METRIC_TOLERANCE):
            raise QualificationError(f"{location} changed protected metric {name}")


def _artifact_record(path: Path, root: Path, evidence_root: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    expected_root = root.resolve(strict=True)
    if not resolved.is_relative_to(expected_root):
        raise QualificationError(f"artifact resolved outside its assigned directory: {path}")
    mode = resolved.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise QualificationError(f"artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": resolved.relative_to(evidence_root.resolve(strict=True)).as_posix(),
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _fm_artifacts(
    training: FMTrainingEvidence,
    artifact_dir: Path,
    evidence_root: Path,
) -> list[dict[str, object]]:
    resolved_paths = tuple(path.resolve(strict=True) for path in training.artifact_paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise QualificationError("FM artifact_paths contain duplicates")
    checkpoint = training.checkpoint_path.resolve(strict=True)
    if checkpoint not in resolved_paths:
        raise QualificationError("FM checkpoint is missing from artifact_paths")
    records: list[dict[str, object]] = []
    for path in training.artifact_paths:
        record = _artifact_record(path, artifact_dir, evidence_root)
        if record["sha256"] != training.artifact_sha256[path.name]:
            raise QualificationError(f"FM artifact changed after creation: {path.name}")
        records.append(record)
    return records


def _exclusive_directory(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise QualificationError(f"qualification artifact path already exists: {path}") from exc
    return path


def _write_exclusive(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise QualificationError(f"qualification evidence already exists: {path}") from exc
    return {
        "path": path.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
    return _write_exclusive(path, _json_bytes(value))


def _tree_manifest(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root).as_posix() in exclude:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise QualificationError(f"artifact tree contains a non-regular file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    return records


def _tree_digest(root: Path) -> str:
    return _json_digest(_tree_manifest(root))


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o400)
        elif path.is_dir():
            path.chmod(0o500)
    root.chmod(0o500)


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in (root, *root.rglob("*")):
        try:
            if path.is_dir():
                path.chmod(0o700)
            elif path.is_file():
                path.chmod(0o600)
        except OSError:
            pass


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically install a directory without replacing any destination object."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int
    if sys.platform == "darwin":
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:  # pragma: no cover - supported macOS contract.
            raise QualificationError("renamex_np is unavailable; cannot commit without overwrite")
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
        result = int(renamex(source_bytes, destination_bytes, 0x00000004))
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - modern glibc contract.
            raise QualificationError("renameat2 is unavailable; cannot commit without overwrite")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                _AT_FDCWD,
                source_bytes,
                _AT_FDCWD,
                destination_bytes,
                _RENAME_NOREPLACE,
            )
        )
    else:  # pragma: no cover - qualification reference platforms are macOS/Linux.
        raise QualificationError("platform lacks a supported atomic no-overwrite rename")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise QualificationError(f"qualification run directory already exists: {destination}")
    raise QualificationError(f"cannot atomically commit qualification: {os.strerror(error)}")


def _aggregate_resources(resources: Sequence[ResourceUsage]) -> dict[str, object]:
    by_device = CounterLike()
    for usage in resources:
        by_device.add(usage)
    return by_device.manifest()


@dataclass(slots=True)
class CounterLike:
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_rss_bytes: int = 0
    operations: int = 0
    devices: set[str] = field(default_factory=set)

    def add(self, usage: ResourceUsage) -> None:
        self.wall_seconds += usage.wall_seconds
        self.cpu_seconds += usage.cpu_seconds
        self.peak_rss_bytes = max(self.peak_rss_bytes, usage.peak_rss_bytes)
        self.operations += 1
        self.devices.add(usage.device)

    def manifest(self) -> dict[str, object]:
        return {
            "operation_count": self.operations,
            "summed_wall_seconds": self.wall_seconds,
            "summed_cpu_seconds": self.cpu_seconds,
            "maximum_peak_rss_bytes": self.peak_rss_bytes,
            "devices": sorted(self.devices),
        }


def _local_backend() -> QualificationBackend:
    # Import lazily to keep the coordinator's structural fixture seam free of heavyweight data and
    # organizer imports, and to avoid a module cycle while the local backend implements this API.
    from kuairand_agent.baselines.qualification_local import LocalQualificationBackend

    return LocalQualificationBackend()


def run_qualification(
    request: QualificationRequest,
    *,
    backend: QualificationBackend | None = None,
) -> QualificationResult:
    """Run every mandatory baseline gate and atomically publish one fallback bundle.

    ``run_dir`` must not exist.  Any mismatch or injected failure removes private staging and
    leaves no visible run.  FM seeds 0..4 plus one clean seed-0 retrain are the only charged model
    launches; random/popularity rungs and checkpoint replays are evidence checks, not training
    launches.
    """

    if not isinstance(request, QualificationRequest):
        raise QualificationError("request must be a QualificationRequest")
    selected_backend = _local_backend() if backend is None else backend
    run_dir = request.run_dir.expanduser().absolute()
    if os.path.lexists(run_dir):
        raise QualificationError(f"qualification run directory already exists: {run_dir}")
    run_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.qualification-", dir=run_dir.parent))
    committed = False
    try:
        verification_dir = _exclusive_directory(staging / "verification")
        first = selected_backend.snapshot(request)
        second = selected_backend.snapshot(request)
        if first.identity_manifest() != second.identity_manifest() or first.digest != second.digest:
            raise QualificationError("audit/canonical double build changed logical identity")
        _write_json(verification_dir / "snapshot-first.json", first.identity_manifest())
        _write_json(verification_dir / "snapshot-second.json", second.identity_manifest())
        # A full canonical snapshot is intentionally substantial.  The second build proves logical
        # identity and is no longer needed after its evidence bytes have been committed to staging.
        del second

        rungs_dir = _exclusive_directory(staging / "rungs")
        random_summary = selected_backend.random_rungs(first)
        _require_rung(random_summary, name="random", seeds=RANDOM_SEEDS)
        popularity_summary = selected_backend.popularity_rung(first)
        _require_rung(popularity_summary, name="item_popularity", seeds=None)
        _write_json(rungs_dir / "random.json", random_summary.manifest())
        _write_json(rungs_dir / "item-popularity.json", popularity_summary.manifest())

        fm_root = _exclusive_directory(staging / "fm")
        training_runs: list[FMTrainingEvidence] = []
        training_manifests: list[dict[str, object]] = []
        launch_records: list[dict[str, object]] = []
        resources: list[ResourceUsage] = []
        for launch_number, seed in enumerate(FM_SEEDS, start=1):
            artifact_dir = _exclusive_directory(fm_root / f"seed-{seed}")
            training = selected_backend.train_fm(first, seed, artifact_dir)
            if training.seed != seed:
                raise QualificationError("FM backend returned evidence for the wrong seed")
            scores = training.normalize_scores(first.validation_count)
            rescored = selected_backend.score_validation(first, scores)
            _require_metrics_equal(training.validation_metrics, rescored, f"FM seed {seed}")
            artifacts = _fm_artifacts(training, artifact_dir, staging)
            manifest = training.manifest(artifacts)
            _write_json(artifact_dir / "run.json", manifest)
            training_runs.append(training)
            training_manifests.append(manifest)
            resources.append(training.resources)
            launch_records.append(
                {
                    "launch_number": launch_number,
                    "kind": "official_fm_training",
                    "seed": seed,
                    "charged": True,
                }
            )

        fm_mean = _require_fm_reference(training_runs)

        replay_dir = _exclusive_directory(staging / "replays")
        replay_manifests: list[dict[str, object]] = []
        for training in training_runs:
            replay = selected_backend.replay_fm(first, training)
            if replay.seed != training.seed:
                raise QualificationError("checkpoint replay returned the wrong seed")
            if replay.checkpoint_digest != training.checkpoint_digest:
                raise QualificationError(f"FM seed {training.seed} replay used another checkpoint")
            replay_scores = _scores(
                replay.validation_scores,
                first.validation_count,
                f"FM seed {training.seed} replay predictions",
            )
            expected_scores = cast(Float64Vector, training.validation_scores)
            if replay_scores.tobytes(order="C") != expected_scores.tobytes(order="C"):
                raise QualificationError(
                    f"FM seed {training.seed} checkpoint replay changed predictions"
                )
            rescored = selected_backend.score_validation(first, replay_scores)
            _require_metrics_equal(training.validation_metrics, replay.validation_metrics, "replay")
            _require_metrics_equal(training.validation_metrics, rescored, "replay rescore")
            replay_manifest = {
                "seed": replay.seed,
                "checkpoint_digest": replay.checkpoint_digest,
                "prediction_digest": prediction_digest(replay_scores),
                "prediction_identity": True,
                "metrics": replay.validation_metrics.manifest(),
                "resources": replay.resources.manifest(),
                "charged_launch": False,
            }
            _write_json(replay_dir / f"seed-{training.seed}.json", replay_manifest)
            replay_manifests.append(replay_manifest)
            resources.append(replay.resources)

        clean_dir = _exclusive_directory(staging / "clean-retrain-seed-0")
        clean = selected_backend.train_fm(first, 0, clean_dir)
        clean_scores = clean.normalize_scores(first.validation_count)
        clean_rescored = selected_backend.score_validation(first, clean_scores)
        _require_metrics_equal(clean.validation_metrics, clean_rescored, "clean seed-0 rescore")
        seed_zero = training_runs[0]
        seed_zero_scores = cast(Float64Vector, seed_zero.validation_scores)
        if not np.allclose(
            clean_scores,
            seed_zero_scores,
            rtol=0.0,
            atol=_RETRAIN_ABSOLUTE_TOLERANCE,
            equal_nan=False,
        ):
            raise QualificationError("clean seed-0 retrain changed validation predictions")
        if not compare_within_user_order(
            first.validation_alignment, seed_zero_scores, clean_scores
        ):
            raise QualificationError("clean seed-0 retrain changed within-user ordering")
        _require_metrics_equal(
            seed_zero.validation_metrics, clean.validation_metrics, "clean retrain"
        )
        if clean.checkpoint_digest != seed_zero.checkpoint_digest:
            raise QualificationError("clean seed-0 retrain changed checkpoint identity")
        if (
            clean.encoding_digest != seed_zero.encoding_digest
            or clean.config_digest != seed_zero.config_digest
            or clean.starter_manifest_digest != seed_zero.starter_manifest_digest
        ):
            raise QualificationError(
                "clean seed-0 retrain changed encoding, config, or organizer identity"
            )
        clean_artifacts = _fm_artifacts(clean, clean_dir, staging)
        clean_manifest = {
            **clean.manifest(clean_artifacts),
            "source_seed": 0,
            "prediction_identity": True,
            "within_user_order_identity": True,
            "absolute_tolerance": _RETRAIN_ABSOLUTE_TOLERANCE,
        }
        _write_json(clean_dir / "run.json", clean_manifest)
        resources.append(clean.resources)
        launch_records.append(
            {
                "launch_number": 6,
                "kind": "clean_source_retrain",
                "seed": 0,
                "charged": True,
            }
        )
        if len(launch_records) != EXPECTED_TRAINING_LAUNCHES:
            raise QualificationError("qualification launch accounting is not exactly six")

        best = max(
            training_runs,
            key=lambda run: (
                run.validation_metrics.primary,
                run.validation_metrics.gauc,
                run.validation_metrics.ndcg_at_5,
                -run.seed,
            ),
        )
        best_scores = cast(Float64Vector, best.validation_scores)

        validation_dir = _exclusive_directory(staging / "validation")

        def metric_evaluator(values: Float64Vector) -> Mapping[str, object]:
            return selected_backend.score_validation(first, values).manifest()

        validation_submission = write_submission(
            validation_dir / "submission.csv",
            first.validation_alignment,
            best_scores,
            protected_metric_evaluator=metric_evaluator,
            metric_tolerance=0.0,
        )

        final_prediction = selected_backend.predict_final(first, best)
        if final_prediction.checkpoint_digest != best.checkpoint_digest:
            raise QualificationError("final inference did not use the selected fallback checkpoint")
        final_scores = _scores(final_prediction.scores, first.final_count, "final predictions")
        final_dir = _exclusive_directory(staging / "final")
        final_submission = write_submission(
            final_dir / "submission.csv", first.final_alignment, final_scores
        )
        resources.append(final_prediction.resources)

        fallback_dir = _exclusive_directory(staging / "fallback")
        fallback_model = fallback_dir / "model"
        shutil.copytree(fm_root / f"seed-{best.seed}", fallback_model, copy_function=shutil.copy2)
        source_tree_digest = _tree_digest(fm_root / f"seed-{best.seed}")
        fallback_tree_digest = _tree_digest(fallback_model)
        if source_tree_digest != fallback_tree_digest:
            raise QualificationError("fallback copy differs from the selected FM artifact tree")
        fallback_manifest = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "kind": "immutable_official_fm_fallback",
            "seed": best.seed,
            "validation_metrics": best.validation_metrics.manifest(),
            "checkpoint_digest": best.checkpoint_digest,
            "encoding_digest": best.encoding_digest,
            "config_digest": best.config_digest,
            "validation_prediction_digest": best.prediction_digest,
            "validation_submission": {
                "path": "validation/submission.csv",
                "sha256": validation_submission.submission_digest,
                "prediction_digest": validation_submission.prediction_digest,
                "round_trip_identity": validation_submission.round_trip_identity,
                "protected_metrics_preserved": validation_submission.protected_metrics_preserved,
            },
            "final_submission": {
                "path": "final/submission.csv",
                "sha256": final_submission.submission_digest,
                "prediction_digest": final_submission.prediction_digest,
                "round_trip_identity": final_submission.round_trip_identity,
                "final_outcomes_accessed": False,
            },
            "source_model_tree_digest": source_tree_digest,
            "fallback_model_tree_digest": fallback_tree_digest,
            "replay_verified": True,
            "clean_seed_zero_retrain_verified": True,
        }
        fallback_manifest["digest"] = _json_digest(fallback_manifest)
        _write_json(fallback_dir / "manifest.json", fallback_manifest)

        root_manifest: dict[str, object] = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "status": "baseline_reproduced",
            "benchmark_digest": BENCHMARK_CONTRACT.digest,
            "qualification_input_digest": first.digest,
            "double_build_identity": True,
            "rungs": {
                "random": random_summary.manifest(),
                "item_popularity": popularity_summary.manifest(),
            },
            "fm": {
                "seeds": list(FM_SEEDS),
                "runs": training_manifests,
                "five_seed_mean": fm_mean.manifest(),
                "published_reference": _published_manifest("fm_official"),
                "reference_passed": True,
                "checkpoint_replays": replay_manifests,
                "clean_seed_zero": clean_manifest,
            },
            "launch_accounting": {
                "charged_launches": len(launch_records),
                "expected_launches": EXPECTED_TRAINING_LAUNCHES,
                "records": launch_records,
                "random_rungs_charged": False,
                "popularity_rung_charged": False,
                "checkpoint_replays_charged": False,
            },
            "fallback": fallback_manifest,
            "resource_usage": _aggregate_resources(resources),
            "final_period": {
                "input_rows": first.final_count,
                "target_capability": None,
                "outcomes_accessed": False,
                "outcomes_scored": False,
            },
        }
        # Index exact evidence bytes before writing the self-referential root manifest.
        root_manifest["artifacts"] = _tree_manifest(staging)
        root_manifest["digest"] = _json_digest(root_manifest)
        root_manifest_path = staging / "manifest.json"
        root_manifest_record = _write_json(root_manifest_path, root_manifest)
        manifest_digest = cast(str, root_manifest["digest"])
        if (
            hashlib.sha256(root_manifest_path.read_bytes()).hexdigest()
            != root_manifest_record["sha256"]
        ):
            raise QualificationError("root manifest changed after writing")

        _make_read_only(fallback_dir)
        _rename_exclusive(staging, run_dir)
        committed = True
        return QualificationResult(
            run_dir=run_dir,
            manifest_digest=manifest_digest,
            fallback_seed=best.seed,
            launch_count=len(launch_records),
            validation_metrics=best.validation_metrics,
            validation_submission=ValidatedSubmission(
                path=run_dir / "validation" / "submission.csv",
                scores=validation_submission.scores,
                row_count=validation_submission.row_count,
                prediction_digest=validation_submission.prediction_digest,
                submission_digest=validation_submission.submission_digest,
                round_trip_identity=validation_submission.round_trip_identity,
                within_user_order_preserved=validation_submission.within_user_order_preserved,
                top5_preserved=validation_submission.top5_preserved,
                protected_metrics_preserved=validation_submission.protected_metrics_preserved,
            ),
            final_submission=ValidatedSubmission(
                path=run_dir / "final" / "submission.csv",
                scores=final_submission.scores,
                row_count=final_submission.row_count,
                prediction_digest=final_submission.prediction_digest,
                submission_digest=final_submission.submission_digest,
                round_trip_identity=final_submission.round_trip_identity,
                within_user_order_preserved=final_submission.within_user_order_preserved,
                top5_preserved=final_submission.top5_preserved,
                protected_metrics_preserved=final_submission.protected_metrics_preserved,
            ),
        )
    finally:
        if not committed and staging.exists():
            _make_writable(staging)
            shutil.rmtree(staging, ignore_errors=False)


__all__ = [
    "EXPECTED_TRAINING_LAUNCHES",
    "FM_SEEDS",
    "QUALIFICATION_SCHEMA_VERSION",
    "RANDOM_SEEDS",
    "FMReplayEvidence",
    "FMTrainingEvidence",
    "FinalPredictionEvidence",
    "QualificationBackend",
    "QualificationError",
    "QualificationMetrics",
    "QualificationRequest",
    "QualificationResult",
    "QualificationSnapshot",
    "ResourceUsage",
    "RungEvaluationEvidence",
    "RungSummaryEvidence",
    "run_qualification",
]

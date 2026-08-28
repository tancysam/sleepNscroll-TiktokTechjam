"""Clean, identity-bound replay over leakage-safe candidate capabilities.

The replay backend receives only a read-only frozen candidate workspace and the already-built
``OUTER_VALID`` or ``FINAL`` input capability.  It never receives raw archive paths, validation
labels, final outcomes, or the protected metric evaluator.  All scientific identity checks and
all scoring/serialization checks remain in this trusted orchestration layer.
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
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from kuairand_agent.campaign.provenance import ENVIRONMENT_IDENTITY_PACKAGES
from kuairand_agent.contract import sha256_file
from kuairand_agent.data.capabilities import CandidateInputs, DataPhase
from kuairand_agent.execution.artifacts import (
    ArtifactError,
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
)
from kuairand_agent.scoring.submission import (
    AlignmentRow,
    compare_protected_metrics,
    compare_within_user_top5,
    prediction_digest,
    validate_alignment,
    write_submission,
)

REPLAY_SCHEMA_VERSION: Final = 1
_DIGEST_LENGTH: Final = 64
_MAX_PREDICTION_BYTES: Final = 2 * 1024 * 1024 * 1024
_AT_FDCWD: Final = -100
_RENAME_NOREPLACE: Final = 1
_ENVIRONMENT_IDENTITY_DOMAINS: Final = {
    1: b"kuairand-environment-v1\0",
    2: b"kuairand-environment-v2\0",
}
_ENVIRONMENT_V2_KEYS: Final = frozenset(
    {"schema_version", "python", "platform", "packages", "uv_lock_sha256"}
)
_ENVIRONMENT_V2_PYTHON_KEYS: Final = frozenset({"implementation", "version"})
_ENVIRONMENT_V2_PLATFORM_KEYS: Final = frozenset({"system", "release", "machine"})

type Float64Vector = npt.NDArray[np.float64]
type MetricEvaluator = Callable[[Float64Vector], Mapping[str, object]]
type ReplayArtifact = ArtifactRef | DirectoryArtifactRef


class ReplayError(RuntimeError):
    """Raised when a clean replay cannot prove every required invariant."""


class ReplayCancelledError(ReplayError):
    """Cooperative cancellation stopped replay before immutable publication."""


def _check_cancellation(
    cancel_event: threading.Event | None,
    *,
    stage: str,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ReplayCancelledError(f"clean replay cancelled before {stage}")


class ReplayEquality(StrEnum):
    """Declared prediction-equivalence policy for the selected model family."""

    EXACT = "exact_same_host_bytes"
    TOLERANT_TOP5 = "numeric_tolerance_top5_and_metric_parity"


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplayError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ReplayError(f"{location} must be one non-empty line of text")
    return value


def _evidence_metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ReplayError(f"{location} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise ReplayError(f"{location} must be finite in [0, 1]")
    return rendered


def _canonical_json(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReplayError("replay evidence must be finite canonical JSON") from exc
    return rendered.encode("ascii")


def manifest_digest(value: object) -> str:
    """Return an ordinary canonical-JSON SHA-256 (not a provenance-domain digest)."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact_environment_object(
    value: object,
    *,
    expected_keys: frozenset[str],
    location: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ReplayError(f"{location} must be an object")
    rendered = dict(value)
    if any(type(key) is not str for key in rendered) or set(rendered) != expected_keys:
        raise ReplayError(f"{location} must contain exactly {sorted(expected_keys)!r}")
    return rendered


def _validate_portable_environment_v2(body: dict[str, object]) -> None:
    if set(body) != _ENVIRONMENT_V2_KEYS:
        raise ReplayError(
            f"environment identity v2 must contain exactly {sorted(_ENVIRONMENT_V2_KEYS)!r}"
        )
    python = _exact_environment_object(
        body["python"],
        expected_keys=_ENVIRONMENT_V2_PYTHON_KEYS,
        location="environment.python",
    )
    _text(python["implementation"], "environment.python.implementation")
    _text(python["version"], "environment.python.version")
    runtime_platform = _exact_environment_object(
        body["platform"],
        expected_keys=_ENVIRONMENT_V2_PLATFORM_KEYS,
        location="environment.platform",
    )
    for name in sorted(_ENVIRONMENT_V2_PLATFORM_KEYS):
        _text(runtime_platform[name], f"environment.platform.{name}")
    packages = _exact_environment_object(
        body["packages"],
        expected_keys=frozenset(ENVIRONMENT_IDENTITY_PACKAGES),
        location="environment.packages",
    )
    for name in ENVIRONMENT_IDENTITY_PACKAGES:
        version = packages[name]
        if version is not None:
            _text(version, f"environment.packages.{name}")
    _digest(body["uv_lock_sha256"], "environment.uv_lock_sha256")


def environment_identity_digest(value: Mapping[str, object]) -> str:
    """Reprove a campaign environment identity in its provenance hash domain.

    Campaign provenance deliberately domain-separates the environment body from ordinary JSON
    manifests. Replay must select the domain declared by the schema: v1 keeps legacy campaign
    evidence internally verifiable, while v2 is the path-independent clean-environment identity.
    Hashing the rendered ``environment.json`` bytes would create a second, incompatible digest
    namespace.
    """

    if not isinstance(value, Mapping) or not value:
        raise ReplayError("environment identity must be a non-empty mapping")
    body = dict(value)
    declared = _digest(body.pop("digest", None), "environment.digest")
    if not body or any(type(key) is not str or not key for key in body):
        raise ReplayError("environment identity body must use non-empty string keys")
    schema_version = body.get("schema_version")
    if type(schema_version) is not int or schema_version not in _ENVIRONMENT_IDENTITY_DOMAINS:
        raise ReplayError("environment identity schema_version must be supported")
    if schema_version == 2:
        _validate_portable_environment_v2(body)
    observed = hashlib.sha256(
        _ENVIRONMENT_IDENTITY_DOMAINS[schema_version] + _canonical_json(body)
    ).hexdigest()
    if observed != declared:
        raise ReplayError("environment provenance digest does not match its manifest body")
    return observed


@dataclass(frozen=True, slots=True)
class FrozenReplayIdentity:
    """Complete selected-candidate identity that replay must independently reprove."""

    source_sha256: str
    config_sha256: str
    features_sha256: str
    checkpoint_sha256: str
    validation_prediction_artifact_sha256: str
    validation_prediction_digest: str
    data_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "config_sha256",
            "features_sha256",
            "checkpoint_sha256",
            "validation_prediction_artifact_sha256",
            "validation_prediction_digest",
            "data_sha256",
            "environment_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def manifest(self) -> dict[str, str]:
        return {
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "features_sha256": self.features_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "validation_prediction_artifact_sha256": (self.validation_prediction_artifact_sha256),
            "validation_prediction_digest": self.validation_prediction_digest,
            "data_sha256": self.data_sha256,
            "environment_sha256": self.environment_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReplayArtifacts:
    """Content-addressed objects restored into the clean replay workspace."""

    source: DirectoryArtifactRef
    config: ReplayArtifact
    features: ReplayArtifact
    checkpoint: ReplayArtifact
    validation_predictions: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, DirectoryArtifactRef):
            raise ReplayError("source must be a directory artifact")
        if self.source.kind is not ArtifactKind.SOURCE:
            raise ReplayError("source directory must use ArtifactKind.SOURCE")
        for name in ("config", "features", "checkpoint"):
            if not isinstance(getattr(self, name), (ArtifactRef, DirectoryArtifactRef)):
                raise ReplayError(f"{name} must be a content-addressed artifact")
        checkpoint_kind = self.checkpoint.kind
        if checkpoint_kind not in {ArtifactKind.CHECKPOINT, ArtifactKind.INPUT}:
            raise ReplayError("checkpoint artifact has an invalid kind")
        if not isinstance(self.validation_predictions, ArtifactRef):
            raise ReplayError("validation_predictions must be a file artifact")
        if self.validation_predictions.kind is not ArtifactKind.PREDICTION:
            raise ReplayError("validation_predictions must use ArtifactKind.PREDICTION")

    def identity_manifest(self) -> dict[str, str]:
        return {
            "source_sha256": self.source.sha256,
            "config_sha256": self.config.sha256,
            "features_sha256": self.features.sha256,
            "checkpoint_sha256": self.checkpoint.sha256,
            "validation_prediction_artifact_sha256": self.validation_predictions.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReplayCapabilities:
    """Only label-free input capabilities and trusted alignment reach final replay."""

    data_sha256: str
    validation_inputs: CandidateInputs
    final_inputs: CandidateInputs
    validation_alignment: Sequence[AlignmentRow]
    final_alignment: Sequence[AlignmentRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_sha256", _digest(self.data_sha256, "data_sha256"))
        if not isinstance(self.validation_inputs, CandidateInputs):
            raise ReplayError("validation_inputs must be a CandidateInputs capability")
        if self.validation_inputs.phase is not DataPhase.OUTER_VALID:
            raise ReplayError("validation replay requires an OUTER_VALID input capability")
        if not isinstance(self.final_inputs, CandidateInputs):
            raise ReplayError("final_inputs must be a CandidateInputs capability")
        if self.final_inputs.phase is not DataPhase.FINAL:
            raise ReplayError("final inference requires a FINAL input capability")
        validation_alignment = tuple(self.validation_alignment)
        final_alignment = tuple(self.final_alignment)
        validate_alignment(validation_alignment)
        validate_alignment(final_alignment)
        _match_alignment(self.validation_inputs, validation_alignment, "validation")
        _match_alignment(self.final_inputs, final_alignment, "final")
        object.__setattr__(self, "validation_alignment", validation_alignment)
        object.__setattr__(self, "final_alignment", final_alignment)


def _match_alignment(
    inputs: CandidateInputs, alignment: Sequence[AlignmentRow], location: str
) -> None:
    if inputs.row_count != len(alignment):
        raise ReplayError(f"{location} capability row count differs from trusted alignment")
    try:
        users = inputs.column("user_id")
        videos = inputs.column("video_id")
    except Exception as exc:
        raise ReplayError(f"{location} capability omits trusted user/video identifiers") from exc
    if tuple(str(value) for value in users) != tuple(row.user_id for row in alignment):
        raise ReplayError(f"{location} capability user order differs from trusted alignment")
    if tuple(str(value) for value in videos) != tuple(row.video_id for row in alignment):
        raise ReplayError(f"{location} capability video order differs from trusted alignment")


@dataclass(frozen=True, slots=True)
class CleanReplayRequest:
    """One no-overwrite clean replay request for a frozen candidate."""

    candidate_id: str
    output_dir: Path
    identity: FrozenReplayIdentity
    artifacts: ReplayArtifacts
    environment: Mapping[str, object]
    equality: ReplayEquality = ReplayEquality.EXACT
    absolute_tolerance: float = 0.0
    metric_tolerance: float = 0.0
    training_replay: str = "checkpoint_replay"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not isinstance(self.output_dir, Path):
            raise ReplayError("output_dir must be a pathlib.Path")
        if not isinstance(self.identity, FrozenReplayIdentity):
            raise ReplayError("identity must be FrozenReplayIdentity")
        if not isinstance(self.artifacts, ReplayArtifacts):
            raise ReplayError("artifacts must be ReplayArtifacts")
        if not isinstance(self.environment, Mapping) or not self.environment:
            raise ReplayError("environment must be a non-empty mapping")
        environment = dict(self.environment)
        _canonical_json(environment)
        object.__setattr__(self, "environment", MappingProxyType(environment))
        if not isinstance(self.equality, ReplayEquality):
            raise ReplayError("equality must be ReplayEquality")
        for name in ("absolute_tolerance", "metric_tolerance"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ReplayError(f"{name} must be a finite non-negative float")
        if self.equality is ReplayEquality.EXACT and self.absolute_tolerance != 0.0:
            raise ReplayError("exact replay cannot declare a numeric tolerance")
        if self.equality is ReplayEquality.TOLERANT_TOP5 and self.absolute_tolerance <= 0.0:
            raise ReplayError("tolerant replay requires a positive absolute_tolerance")
        object.__setattr__(self, "training_replay", _text(self.training_replay, "training_replay"))


@dataclass(frozen=True, slots=True)
class FrozenCandidateWorkspace:
    """Read-only scientific files exposed to a candidate-specific replay adapter."""

    root: Path
    source_dir: Path
    config_path: Path
    features_path: Path
    checkpoint_path: Path
    identity: FrozenReplayIdentity


class ReplayBackend(Protocol):
    """Candidate-family adapter; implementations never own scoring or data loading."""

    def replay_validation(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Iterable[object]: ...

    def predict_final(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Iterable[object]: ...


@dataclass(frozen=True, slots=True)
class ValidationReplayEvidence:
    row_count: int
    reference_prediction_digest: str
    replay_prediction_digest: str
    replay_prediction_file_sha256: str
    exact_prediction_bytes: bool
    maximum_absolute_difference: float
    top5_order_identical: bool
    protected_metrics_identical: bool
    metrics: Mapping[str, float]
    public_submission_sha256: str
    public_submission_prediction_digest: str
    csv_round_trip_identity: bool
    csv_within_user_order_preserved: bool
    csv_top5_preserved: bool
    csv_protected_metrics_preserved: bool

    def __post_init__(self) -> None:
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ReplayError("validation replay row_count must be positive")
        for name in (
            "reference_prediction_digest",
            "replay_prediction_digest",
            "replay_prediction_file_sha256",
            "public_submission_sha256",
            "public_submission_prediction_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (
            type(self.maximum_absolute_difference) is not float
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference < 0
        ):
            raise ReplayError("maximum_absolute_difference must be finite and non-negative")
        for name in (
            "exact_prediction_bytes",
            "top5_order_identical",
            "protected_metrics_identical",
            "csv_round_trip_identity",
            "csv_within_user_order_preserved",
            "csv_top5_preserved",
            "csv_protected_metrics_preserved",
        ):
            if type(getattr(self, name)) is not bool:
                raise ReplayError(f"{name} must be boolean")
        if self.exact_prediction_bytes and self.maximum_absolute_difference != 0.0:
            raise ReplayError("exact replay cannot have a non-zero numeric difference")
        if not self.top5_order_identical or not self.protected_metrics_identical:
            raise ReplayError("successful validation replay requires top-five and metric parity")
        if not all(
            (
                self.csv_round_trip_identity,
                self.csv_within_user_order_preserved,
                self.csv_top5_preserved,
                self.csv_protected_metrics_preserved,
            )
        ):
            raise ReplayError(
                "successful public CSV must preserve bytes, order, top-five, and metrics"
            )
        if self.public_submission_prediction_digest != self.replay_prediction_digest:
            raise ReplayError("public CSV prediction digest differs from replayed predictions")
        if not isinstance(self.metrics, Mapping) or set(self.metrics) != {
            "GAUC",
            "nDCG@5",
            "primary",
        }:
            raise ReplayError("validation metrics must contain exactly GAUC, nDCG@5, and primary")
        normalized = {
            name: _evidence_metric(self.metrics[name], f"metrics.{name}")
            for name in ("GAUC", "nDCG@5", "primary")
        }
        if not math.isclose(
            normalized["primary"],
            (normalized["GAUC"] + normalized["nDCG@5"]) / 2.0,
            rel_tol=0.0,
            abs_tol=2e-7,
        ):
            raise ReplayError("validation evidence primary is not the organizer metric mean")
        object.__setattr__(self, "metrics", MappingProxyType(normalized))

    def manifest(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "reference_prediction_digest": self.reference_prediction_digest,
            "replay_prediction_digest": self.replay_prediction_digest,
            "replay_prediction_file_sha256": self.replay_prediction_file_sha256,
            "exact_prediction_bytes": self.exact_prediction_bytes,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "top5_order_identical": self.top5_order_identical,
            "protected_metrics_identical": self.protected_metrics_identical,
            "metrics": dict(self.metrics),
            "public_submission_sha256": self.public_submission_sha256,
            "public_submission_prediction_digest": self.public_submission_prediction_digest,
            "csv_serialization": {
                "float64_round_trip_identity": self.csv_round_trip_identity,
                "within_user_order_preserved": self.csv_within_user_order_preserved,
                "top5_preserved": self.csv_top5_preserved,
                "protected_metrics_preserved": self.csv_protected_metrics_preserved,
            },
        }


@dataclass(frozen=True, slots=True)
class FinalReplayEvidence:
    row_count: int
    prediction_digest: str
    prediction_file_sha256: str
    submission_sha256: str
    submission_prediction_digest: str
    finite_scores: bool
    csv_round_trip_identity: bool
    final_outcomes_accessed: bool = False

    def __post_init__(self) -> None:
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ReplayError("final replay row_count must be positive")
        for name in (
            "prediction_digest",
            "prediction_file_sha256",
            "submission_sha256",
            "submission_prediction_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("finite_scores", "csv_round_trip_identity", "final_outcomes_accessed"):
            if type(getattr(self, name)) is not bool:
                raise ReplayError(f"{name} must be boolean")
        if not self.finite_scores or not self.csv_round_trip_identity:
            raise ReplayError("successful final replay requires finite round-tripped scores")
        if self.final_outcomes_accessed:
            raise ReplayError("successful final replay cannot report final-outcome access")
        if self.submission_prediction_digest != self.prediction_digest:
            raise ReplayError("final CSV prediction digest differs from final predictions")

    def manifest(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "prediction_digest": self.prediction_digest,
            "prediction_file_sha256": self.prediction_file_sha256,
            "submission_sha256": self.submission_sha256,
            "submission_prediction_digest": self.submission_prediction_digest,
            "finite_scores": self.finite_scores,
            "csv_round_trip_identity": self.csv_round_trip_identity,
            "outcome_access": "none",
            "final_outcomes_accessed": self.final_outcomes_accessed,
            "final_outcomes_scored": False,
        }


@dataclass(frozen=True, slots=True)
class CleanReplayEvidence:
    candidate_id: str
    identity: FrozenReplayIdentity
    equality: ReplayEquality
    absolute_tolerance: float
    training_replay: str
    validation: ValidationReplayEvidence
    final: FinalReplayEvidence
    validation_capability_digest: str
    final_capability_digest: str
    clean_workspace_removed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not isinstance(self.identity, FrozenReplayIdentity):
            raise ReplayError("replay evidence identity must be FrozenReplayIdentity")
        if not isinstance(self.equality, ReplayEquality):
            raise ReplayError("replay evidence equality must be ReplayEquality")
        if (
            type(self.absolute_tolerance) is not float
            or not math.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0
        ):
            raise ReplayError("replay evidence tolerance must be finite and non-negative")
        object.__setattr__(self, "training_replay", _text(self.training_replay, "training_replay"))
        if not isinstance(self.validation, ValidationReplayEvidence):
            raise ReplayError("validation must be ValidationReplayEvidence")
        if not isinstance(self.final, FinalReplayEvidence):
            raise ReplayError("final must be FinalReplayEvidence")
        for name in ("validation_capability_digest", "final_capability_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.clean_workspace_removed) is not bool or not self.clean_workspace_removed:
            raise ReplayError("successful replay must remove its clean workspace")
        if self.equality is ReplayEquality.EXACT:
            if self.absolute_tolerance != 0.0 or not self.validation.exact_prediction_bytes:
                raise ReplayError("exact replay evidence must prove exact prediction bytes")
        elif (
            self.absolute_tolerance <= 0.0
            or self.validation.maximum_absolute_difference > self.absolute_tolerance
        ):
            raise ReplayError("tolerant replay evidence exceeds its declared tolerance")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "identity": self.identity.manifest(),
            "equality": {
                "policy": self.equality.value,
                "absolute_tolerance": self.absolute_tolerance,
            },
            "training_replay": self.training_replay,
            "validation": self.validation.manifest(),
            "final": self.final.manifest(),
            "capabilities": {
                "validation_digest": self.validation_capability_digest,
                "final_digest": self.final_capability_digest,
                "validation_phase": DataPhase.OUTER_VALID.value,
                "final_phase": DataPhase.FINAL.value,
                "labels_exposed_to_backend": False,
                "raw_data_path_exposed_to_backend": False,
            },
            "workspace": {
                "fresh_materialization": True,
                "artifact_identities_reverified_after_inference": True,
                "clean_workspace_removed": self.clean_workspace_removed,
            },
        }


@dataclass(frozen=True, slots=True)
class CleanReplayResult:
    root: Path
    evidence: CleanReplayEvidence
    evidence_path: Path
    final_submission: Path
    public_validation_submission: Path
    source_dir: Path
    config_dir: Path
    model_dir: Path
    preprocessing_dir: Path
    validation_evidence_dir: Path
    replay_dir: Path
    environment_path: Path


def _artifact_identity(ref: ReplayArtifact) -> str:
    return ref.sha256


def _verify_artifacts(request: CleanReplayRequest, store: ArtifactStore) -> None:
    observed = request.artifacts.identity_manifest()
    expected = request.identity.manifest()
    for name, digest in observed.items():
        if expected[name] != digest:
            raise ReplayError(f"frozen {name} differs from its selected identity")
    try:
        store.verify_directory(request.artifacts.source)
        for artifact in (
            request.artifacts.config,
            request.artifacts.features,
            request.artifacts.checkpoint,
        ):
            if isinstance(artifact, DirectoryArtifactRef):
                store.verify_directory(artifact)
            else:
                store.verify(artifact)
        store.verify(request.artifacts.validation_predictions)
    except ArtifactError as exc:
        raise ReplayError("selected replay artifact failed content verification") from exc


def _copy_file(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256_file(destination) != expected_sha256:
        raise ReplayError("artifact bytes changed while materializing clean replay")
    os.chmod(destination, 0o444, follow_symlinks=False)


def _materialize_artifact(
    store: ArtifactStore, artifact: ReplayArtifact, destination: Path
) -> Path:
    if isinstance(artifact, ArtifactRef):
        destination.mkdir(mode=0o700)
        target = destination / "artifact"
        _copy_file(store.verify(artifact), target, artifact.sha256)
        os.chmod(destination, 0o555, follow_symlinks=False)
        return target
    store.verify_directory(artifact)
    destination.mkdir(mode=0o700)
    for entry in artifact.entries:
        target = destination.joinpath(*Path(entry.path).parts)
        _copy_file(store.verify(entry.artifact), target, entry.artifact.sha256)
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()), reverse=True
    ):
        os.chmod(directory, 0o555, follow_symlinks=False)
    os.chmod(destination, 0o555, follow_symlinks=False)
    return destination


def _verify_materialized(artifact: ReplayArtifact, path: Path) -> None:
    if isinstance(artifact, ArtifactRef):
        if not path.is_file() or path.is_symlink() or sha256_file(path) != artifact.sha256:
            raise ReplayError("materialized file artifact changed during replay")
        return
    expected = {entry.path: entry.artifact.sha256 for entry in artifact.entries}
    observed: dict[str, str] = {}
    for candidate in sorted(path.rglob("*")):
        metadata = candidate.lstat()
        relative = candidate.relative_to(path).as_posix()
        if stat.S_ISLNK(metadata.st_mode) or (
            not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode)
        ):
            raise ReplayError("materialized directory contains an unsafe member")
        if stat.S_ISREG(metadata.st_mode):
            observed[relative] = sha256_file(candidate)
    if observed != expected:
        raise ReplayError("materialized directory artifact changed during replay")


def _scores(values: Iterable[object], expected: int, location: str) -> Float64Vector:
    try:
        array = np.asarray(values if isinstance(values, np.ndarray) else tuple(values))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError(f"{location} must be a one-dimensional real vector") from exc
    if array.ndim != 1 or array.dtype.kind not in "iuf" or len(array) != expected:
        raise ReplayError(f"{location} must contain exactly {expected} numeric scores")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ReplayError(f"{location} contains NaN or infinity")
    result.setflags(write=False)
    return result


def _load_reference(path: Path, expected_count: int) -> Float64Vector:
    try:
        if path.stat().st_size > _MAX_PREDICTION_BYTES:
            raise ReplayError("reference prediction artifact exceeds the replay size limit")
        with path.open("rb") as handle:
            loaded = np.load(handle, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReplayError("reference prediction artifact is not a safe NPY array") from exc
    if not isinstance(loaded, np.ndarray):
        raise ReplayError("reference prediction artifact must contain one NPY array")
    return _scores(loaded, expected_count, "reference validation predictions")


def _write_npy(path: Path, scores: Float64Vector) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _metrics(evaluator: MetricEvaluator, scores: Float64Vector) -> Mapping[str, float]:
    try:
        raw = evaluator(scores)
    except Exception as exc:
        raise ReplayError("protected validation scorer failed during replay") from exc
    required = ("GAUC", "nDCG@5", "primary")
    if not isinstance(raw, Mapping) or any(name not in raw for name in required):
        raise ReplayError("protected scorer omitted a required aggregate metric")
    values: dict[str, float] = {}
    for name in required:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ReplayError(f"protected metric {name} is not numeric")
        rendered = float(value)
        if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
            raise ReplayError(f"protected metric {name} must be finite in [0, 1]")
        values[name] = rendered
    if not math.isclose(
        values["primary"],
        (values["GAUC"] + values["nDCG@5"]) / 2.0,
        rel_tol=0.0,
        abs_tol=2e-7,
    ):
        raise ReplayError("protected primary is not the organizer metric mean")
    return MappingProxyType(values)


def _copy_snapshot(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copyfile)
    else:  # pragma: no cover - materialized paths are directories except returned leaf paths.
        destination.mkdir()
        shutil.copyfile(source, destination / source.name)


def _remove_private_tree(path: Path) -> None:
    """Make only the known temporary tree traversable, then remove it without following links."""

    if not path.exists():
        return
    os.chmod(path, 0o700, follow_symlinks=False)
    for candidate in path.rglob("*"):
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            os.chmod(candidate, 0o700, follow_symlinks=False)
    shutil.rmtree(path)


def _publish(staging: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:  # pragma: no cover - supported macOS contract.
            raise ReplayError("renamex_np is unavailable for exclusive replay publication")
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
        result = int(renamex(source_bytes, destination_bytes, 0x00000004))
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - modern glibc contract.
            raise ReplayError("renameat2 is unavailable for exclusive replay publication")
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
    else:  # pragma: no cover - supported reference platforms are macOS and Linux.
        raise ReplayError("platform lacks atomic no-overwrite replay publication")
    if result == 0:
        _fsync_directory(destination.parent)
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ReplayError(f"refusing to overwrite existing replay output: {destination}")
    raise ReplayError(f"could not publish replay output: {os.strerror(error)}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_clean_replay(
    request: CleanReplayRequest,
    *,
    artifact_store: ArtifactStore,
    capabilities: ReplayCapabilities,
    backend: ReplayBackend,
    protected_metric_evaluator: MetricEvaluator,
    cancel_event: threading.Event | None = None,
) -> CleanReplayResult:
    """Restore, replay, score, infer, serialize, and publish immutable replay evidence."""

    if not isinstance(request, CleanReplayRequest):
        raise ReplayError("request must be CleanReplayRequest")
    if not isinstance(artifact_store, ArtifactStore):
        raise ReplayError("artifact_store must be ArtifactStore")
    if not isinstance(capabilities, ReplayCapabilities):
        raise ReplayError("capabilities must be ReplayCapabilities")
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise ReplayError("cancel_event must be threading.Event or None")
    _check_cancellation(cancel_event, stage="replay admission")
    if capabilities.data_sha256 != request.identity.data_sha256:
        raise ReplayError("replay data identity differs from the selected candidate")
    environment_digest = environment_identity_digest(request.environment)
    if environment_digest != request.identity.environment_sha256:
        raise ReplayError("replay environment identity differs from the selected candidate")
    _verify_artifacts(request, artifact_store)

    destination = request.output_dir
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise ReplayError(f"refusing to overwrite existing replay output: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    clean_root = Path(tempfile.mkdtemp(prefix=".clean-replay-workspace-", dir=destination.parent))
    published = False
    try:
        source_dir = clean_root / "source"
        config_dir = clean_root / "config"
        features_dir = clean_root / "preprocessing"
        model_dir = clean_root / "model"
        _materialize_artifact(artifact_store, request.artifacts.source, source_dir)
        config_path = _materialize_artifact(artifact_store, request.artifacts.config, config_dir)
        features_path = _materialize_artifact(
            artifact_store, request.artifacts.features, features_dir
        )
        checkpoint_path = _materialize_artifact(
            artifact_store, request.artifacts.checkpoint, model_dir
        )
        workspace = FrozenCandidateWorkspace(
            root=clean_root,
            source_dir=source_dir,
            config_path=config_path,
            features_path=features_path,
            checkpoint_path=checkpoint_path,
            identity=request.identity,
        )
        reference_path = artifact_store.verify(request.artifacts.validation_predictions)
        reference = _load_reference(reference_path, len(capabilities.validation_alignment))
        if prediction_digest(reference) != request.identity.validation_prediction_digest:
            raise ReplayError("stored reference prediction semantics differ from selected identity")

        try:
            _check_cancellation(cancel_event, stage="validation inference")
            replayed = _scores(
                backend.replay_validation(
                    workspace=workspace, inputs=capabilities.validation_inputs
                ),
                len(capabilities.validation_alignment),
                "replayed validation predictions",
            )
            _check_cancellation(cancel_event, stage="final inference")
            final_scores = _scores(
                backend.predict_final(workspace=workspace, inputs=capabilities.final_inputs),
                len(capabilities.final_alignment),
                "final predictions",
            )
            _check_cancellation(cancel_event, stage="replay evidence construction")
        except ReplayCancelledError:
            raise
        except ReplayError:
            _check_cancellation(cancel_event, stage="replay backend failure handling")
            raise
        except Exception as exc:
            _check_cancellation(cancel_event, stage="replay backend failure handling")
            raise ReplayError("selected candidate replay backend failed") from exc

        _verify_materialized(request.artifacts.source, source_dir)
        _verify_materialized(request.artifacts.config, config_path)
        _verify_materialized(request.artifacts.features, features_path)
        _verify_materialized(request.artifacts.checkpoint, checkpoint_path)

        exact = reference.tobytes(order="C") == replayed.tobytes(order="C")
        maximum_difference = float(np.max(np.abs(reference - replayed), initial=0.0))
        top5 = compare_within_user_top5(capabilities.validation_alignment, reference, replayed)
        metrics_identical = compare_protected_metrics(
            reference,
            replayed,
            protected_metric_evaluator,
            absolute_tolerance=request.metric_tolerance,
        )
        if request.equality is ReplayEquality.EXACT and not exact:
            raise ReplayError("same-host validation prediction bytes differ from the incumbent")
        if request.equality is ReplayEquality.TOLERANT_TOP5 and (
            maximum_difference > request.absolute_tolerance or not top5 or not metrics_identical
        ):
            raise ReplayError("tolerant replay failed numeric, top-five, or metric parity")

        _copy_snapshot(source_dir, staging / "source")
        _copy_snapshot(config_dir, staging / "config")
        _copy_snapshot(model_dir, staging / "model")
        _copy_snapshot(features_dir, staging / "preprocessing")
        _remove_private_tree(clean_root)
        if clean_root.exists():  # pragma: no cover - shutil failure normally raises.
            raise ReplayError("clean replay workspace could not be removed")

        validation_evidence_dir = staging / "validation-evidence"
        replay_dir = staging / "replay"
        validation_evidence_dir.mkdir()
        replay_dir.mkdir()
        reference_copy = validation_evidence_dir / "reference-validation-predictions.npy"
        _copy_file(
            reference_path,
            reference_copy,
            request.artifacts.validation_predictions.sha256,
        )
        validation_npy = replay_dir / "validation-predictions.npy"
        validation_file_digest = _write_npy(validation_npy, replayed)
        final_npy = replay_dir / "final-predictions.npy"
        final_file_digest = _write_npy(final_npy, final_scores)
        metrics = _metrics(protected_metric_evaluator, replayed)
        public_submission = write_submission(
            validation_evidence_dir / "public-validation.csv",
            capabilities.validation_alignment,
            replayed,
            protected_metric_evaluator=protected_metric_evaluator,
            metric_tolerance=request.metric_tolerance,
        )
        final_submission = write_submission(
            staging / "submission.csv", capabilities.final_alignment, final_scores
        )
        environment_path = staging / "environment.json"
        environment_payload = _canonical_json(dict(request.environment)) + b"\n"
        with environment_path.open("xb") as handle:
            handle.write(environment_payload)
            handle.flush()
            os.fsync(handle.fileno())

        validation = ValidationReplayEvidence(
            row_count=len(replayed),
            reference_prediction_digest=prediction_digest(reference),
            replay_prediction_digest=prediction_digest(replayed),
            replay_prediction_file_sha256=validation_file_digest,
            exact_prediction_bytes=exact,
            maximum_absolute_difference=maximum_difference,
            top5_order_identical=top5,
            protected_metrics_identical=metrics_identical,
            metrics=metrics,
            public_submission_sha256=public_submission.submission_digest,
            public_submission_prediction_digest=public_submission.prediction_digest,
            csv_round_trip_identity=public_submission.round_trip_identity is True,
            csv_within_user_order_preserved=(public_submission.within_user_order_preserved is True),
            csv_top5_preserved=public_submission.top5_preserved is True,
            csv_protected_metrics_preserved=(public_submission.protected_metrics_preserved is True),
        )
        final = FinalReplayEvidence(
            row_count=len(final_scores),
            prediction_digest=prediction_digest(final_scores),
            prediction_file_sha256=final_file_digest,
            submission_sha256=final_submission.submission_digest,
            submission_prediction_digest=final_submission.prediction_digest,
            finite_scores=True,
            csv_round_trip_identity=final_submission.round_trip_identity is True,
        )
        evidence = CleanReplayEvidence(
            candidate_id=request.candidate_id,
            identity=request.identity,
            equality=request.equality,
            absolute_tolerance=request.absolute_tolerance,
            training_replay=request.training_replay,
            validation=validation,
            final=final,
            validation_capability_digest=capabilities.validation_inputs.digest,
            final_capability_digest=capabilities.final_inputs.digest,
        )
        evidence_path = replay_dir / "evidence.json"
        with evidence_path.open("xb") as handle:
            handle.write(_canonical_json(evidence.manifest()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _check_cancellation(cancel_event, stage="immutable replay publication")
        _publish(staging, destination)
        published = True
        root = destination.resolve()
        return CleanReplayResult(
            root=root,
            evidence=evidence,
            evidence_path=root / "replay" / "evidence.json",
            final_submission=root / "submission.csv",
            public_validation_submission=root / "validation-evidence" / "public-validation.csv",
            source_dir=root / "source",
            config_dir=root / "config",
            model_dir=root / "model",
            preprocessing_dir=root / "preprocessing",
            validation_evidence_dir=root / "validation-evidence",
            replay_dir=root / "replay",
            environment_path=root / "environment.json",
        )
    except ReplayError:
        raise
    except (OSError, ValueError) as exc:
        raise ReplayError("clean replay orchestration failed closed") from exc
    finally:
        if clean_root.exists():
            with suppress(OSError):
                _remove_private_tree(clean_root)
        if not published and staging.exists():
            with suppress(OSError):
                _remove_private_tree(staging)


__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "CleanReplayEvidence",
    "CleanReplayRequest",
    "CleanReplayResult",
    "FinalReplayEvidence",
    "FrozenCandidateWorkspace",
    "FrozenReplayIdentity",
    "ReplayArtifacts",
    "ReplayBackend",
    "ReplayCancelledError",
    "ReplayCapabilities",
    "ReplayEquality",
    "ReplayError",
    "ValidationReplayEvidence",
    "environment_identity_digest",
    "manifest_digest",
    "run_clean_replay",
]

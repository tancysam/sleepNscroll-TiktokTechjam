"""Trusted composition helpers for the provider-free full KuaiRand-Pure campaign.

The full production driver deliberately keeps raw training targets inside this module's trusted
data plane.  Generated code receives only numeric capabilities built from the returned feature
matrices.  Public-validation targets remain bound inside protected scorer closures, and final
period targets do not exist in the canonical contract.

This module also owns the one permitted fusion-selection operation.  Every point in the frozen
rank-fusion grid is scored on Fold B, the exact protected score receipt is checked against the
fused prediction bytes, and the winning point is frozen before Fold A or public validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from numbers import Integral
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from kuairand_agent.campaign.controller import CampaignEngine

from kuairand_agent.campaign.pure_features import (
    PureFeaturePair,
    build_pure_feature_pair,
    concat_canonical_inputs,
    split_feature_matrix,
    subset_canonical_inputs,
    subset_values,
)
from kuairand_agent.campaign.selector import OrganizerMetrics
from kuairand_agent.candidates.fusion import (
    FUSION_WEIGHT_GRID,
    FusionResult,
    fuse_ranked_predictions,
)
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import (
    OUTCOME_FIELDS,
    CanonicalDataset,
    CanonicalFinalSplit,
    CanonicalInputs,
    ProtectedTargets,
    TrainingTargets,
)
from kuairand_agent.data.capabilities import (
    CandidateInputs,
    DataPhase,
    build_candidate_inputs,
)
from kuairand_agent.data.causal_features import FeatureMatrix
from kuairand_agent.data.fields import (
    STANDARD_LATE_MEMBER,
    VIDEO_BASIC_MEMBER,
    FieldKey,
)
from kuairand_agent.data.folds import TemporalFold, TemporalFoldSet, build_temporal_folds
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
    DirectoryEntryRef,
)
from kuairand_agent.scoring.protected import ScoreResult

FULL_CAMPAIGN_SCHEMA_VERSION: Final = 1
_DIGEST_DOMAIN: Final = b"kuairand-full-campaign-v1\0"
_MAX_PROGRESS_CHECKPOINT_BYTES: Final = 1024 * 1024
_CHECKPOINT_NAME_RE: Final = re.compile(r"checkpoint-(\d{8})\.json\Z")

type Float64Vector = npt.NDArray[np.float64]
type Int64Vector = npt.NDArray[np.int64]
type ProtectedAggregateScorer = Callable[[Float64Vector], ScoreResult]


class FullCampaignError(RuntimeError):
    """A production-campaign composition gate failed closed."""


class FullCampaignCancelled(FullCampaignError):
    """Signal-driven stop before another launch; the durable campaign remains resumable."""


def build_finalization_candidate_inputs(
    phase: DataPhase,
    inputs: CanonicalInputs,
) -> CandidateInputs:
    """Project canonical validation/final inputs through the one final-replay capability seam."""

    if phase not in {DataPhase.OUTER_VALID, DataPhase.FINAL}:
        raise FullCampaignError("finalization candidate inputs require outer-valid or final phase")
    if not isinstance(inputs, CanonicalInputs):
        raise FullCampaignError("finalization candidate inputs require CanonicalInputs")
    member = STANDARD_LATE_MEMBER
    return build_candidate_inputs(
        phase,
        {
            FieldKey(member, "user_id"): inputs.user_id,
            FieldKey(member, "video_id"): inputs.video_id,
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): inputs.author_id,
            FieldKey(member, "tab"): inputs.tab,
            FieldKey(member, "duration_ms"): inputs.duration_ms,
        },
    )


class FullCampaignStage(StrEnum):
    """Strict durable stage order for the provider-free production campaign."""

    DATA_PREPARED = "data_prepared"
    QUALIFICATION_VERIFIED = "qualification_verified"
    FOLD_CONTROLS_READY = "fold_controls_ready"
    FEATURES_READY = "features_ready"
    LINEAGE_READY = "lineage_ready"
    SCIENCE_COMPLETE = "science_complete"
    REFLECTED = "reflected"
    FINALIZATION_REQUIRED = "finalization_required"


_FULL_CAMPAIGN_STAGE_ORDER: Final = tuple(FullCampaignStage)


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
        raise FullCampaignError("full-campaign evidence must be finite canonical JSON") from exc


def _manifest_digest(domain: bytes, value: object) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN + domain)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FullCampaignError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FullCampaignError(f"campaign checkpoint is not safely readable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_PROGRESS_CHECKPOINT_BYTES
        ):
            raise FullCampaignError("campaign checkpoint must be one bounded regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) != before.st_size or not stable:
            raise FullCampaignError("campaign checkpoint changed while it was read")
        return payload
    finally:
        os.close(descriptor)


def _plain_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise FullCampaignError(f"{name} must be a non-empty canonical JSON object")
    try:
        decoded = json.loads(_canonical_json(dict(value)))
    except json.JSONDecodeError as exc:  # pragma: no cover - encoder output is always valid JSON.
        raise FullCampaignError(f"{name} could not be normalized") from exc
    if not isinstance(decoded, dict) or any(type(key) is not str for key in decoded):
        raise FullCampaignError(f"{name} must be a canonical JSON object")
    return MappingProxyType(cast(dict[str, object], decoded))


@dataclass(frozen=True, slots=True)
class FullCampaignCheckpoint:
    """One immutable, hash-chained production orchestration cursor."""

    sequence: int
    request_digest: str
    stage: FullCampaignStage
    previous_digest: str | None
    evidence: Mapping[str, object]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise FullCampaignError("campaign checkpoint sequence must be positive")
        _digest(self.request_digest, "request_digest")
        if not isinstance(self.stage, FullCampaignStage):
            raise FullCampaignError("campaign checkpoint stage is unsupported")
        if self.sequence == 1:
            if self.previous_digest is not None:
                raise FullCampaignError("first campaign checkpoint cannot name a predecessor")
        else:
            _digest(self.previous_digest, "previous_digest")
        object.__setattr__(self, "evidence", _plain_mapping(self.evidence, "checkpoint evidence"))
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"progress-checkpoint\0", self.body()),
        )

    def body(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "sequence": self.sequence,
            "request_digest": self.request_digest,
            "stage": self.stage.value,
            "previous_digest": self.previous_digest,
            "evidence": dict(self.evidence),
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @classmethod
    def from_bytes(cls, payload: bytes) -> FullCampaignCheckpoint:
        if not payload or len(payload) > _MAX_PROGRESS_CHECKPOINT_BYTES:
            raise FullCampaignError("campaign checkpoint size is outside the supported bound")
        try:
            raw = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise FullCampaignError("campaign checkpoint is not canonical ASCII JSON") from exc
        fields = {
            "schema_version",
            "sequence",
            "request_digest",
            "stage",
            "previous_digest",
            "evidence",
            "digest",
        }
        if not isinstance(raw, dict) or set(raw) != fields or _canonical_json(raw) != payload:
            raise FullCampaignError("campaign checkpoint does not use the exact canonical schema")
        if raw["schema_version"] != FULL_CAMPAIGN_SCHEMA_VERSION:
            raise FullCampaignError("campaign checkpoint schema version is unsupported")
        try:
            stage = FullCampaignStage(raw["stage"])
        except (TypeError, ValueError) as exc:
            raise FullCampaignError("campaign checkpoint stage is unsupported") from exc
        evidence = raw["evidence"]
        if not isinstance(evidence, dict):
            raise FullCampaignError("campaign checkpoint evidence must be an object")
        checkpoint = cls(
            sequence=raw["sequence"],
            request_digest=raw["request_digest"],
            stage=stage,
            previous_digest=raw["previous_digest"],
            evidence=cast(dict[str, object], evidence),
        )
        if checkpoint.digest != raw["digest"]:
            raise FullCampaignError("campaign checkpoint digest mismatch")
        return checkpoint


class FullCampaignProgressLedger:
    """Strict append-only campaign cursor safe to open after a process restart."""

    def __init__(self, root: Path, *, create: bool = True) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise FullCampaignError("progress ledger root must be an absolute Path")
        self.root = root
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            metadata = self.root.lstat()
        except OSError as exc:
            raise FullCampaignError("progress ledger root does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FullCampaignError("progress ledger root must be a real directory")

    def _checkpoint_paths(self) -> tuple[Path, ...]:
        indexed: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _CHECKPOINT_NAME_RE.fullmatch(path.name)
            if match is None:
                if path.name.startswith(".staging-"):
                    metadata = path.lstat()
                    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                        raise FullCampaignError("progress staging entry is unsafe")
                    continue
                raise FullCampaignError(f"unexpected progress ledger entry: {path.name}")
            indexed.append((int(match.group(1)), path))
        indexed.sort()
        if tuple(index for index, _ in indexed) != tuple(range(1, len(indexed) + 1)):
            raise FullCampaignError("campaign checkpoint sequence is not contiguous")
        return tuple(path for _, path in indexed)

    def checkpoints(self) -> tuple[FullCampaignCheckpoint, ...]:
        result: list[FullCampaignCheckpoint] = []
        previous: FullCampaignCheckpoint | None = None
        for expected_sequence, path in enumerate(self._checkpoint_paths(), start=1):
            checkpoint = FullCampaignCheckpoint.from_bytes(_bounded_regular_bytes(path))
            if checkpoint.sequence != expected_sequence:
                raise FullCampaignError("campaign checkpoint file and sequence differ")
            if previous is not None:
                if checkpoint.request_digest != previous.request_digest:
                    raise FullCampaignError("campaign checkpoint request identity changed")
                if checkpoint.previous_digest != previous.digest:
                    raise FullCampaignError("campaign checkpoint chain is broken")
                next_index = _FULL_CAMPAIGN_STAGE_ORDER.index(previous.stage) + 1
                if next_index >= len(_FULL_CAMPAIGN_STAGE_ORDER):
                    raise FullCampaignError("campaign checkpoint follows the terminal stage")
                expected_stage = _FULL_CAMPAIGN_STAGE_ORDER[next_index]
                if checkpoint.stage is not expected_stage:
                    raise FullCampaignError("campaign checkpoint stages are not contiguous")
            elif checkpoint.stage is not FullCampaignStage.DATA_PREPARED:
                raise FullCampaignError(
                    "campaign checkpoint chain must start with data preparation"
                )
            previous = checkpoint
            result.append(checkpoint)
        return tuple(result)

    def append(
        self,
        *,
        request_digest: str,
        stage: FullCampaignStage,
        evidence: Mapping[str, object],
    ) -> FullCampaignCheckpoint:
        request = _digest(request_digest, "request_digest")
        if not isinstance(stage, FullCampaignStage):
            raise FullCampaignError("campaign stage is unsupported")
        normalized = _plain_mapping(evidence, "checkpoint evidence")
        existing = self.checkpoints()
        retained = next((item for item in existing if item.stage is stage), None)
        if retained is not None:
            if retained.request_digest != request or retained.evidence != normalized:
                raise FullCampaignError("campaign stage retry contradicts durable evidence")
            return retained
        if existing:
            latest_index = _FULL_CAMPAIGN_STAGE_ORDER.index(existing[-1].stage)
            expected_index = latest_index + 1
        else:
            expected_index = 0
        if expected_index >= len(_FULL_CAMPAIGN_STAGE_ORDER) or (
            stage is not _FULL_CAMPAIGN_STAGE_ORDER[expected_index]
        ):
            raise FullCampaignError("append must name the next campaign stage")
        checkpoint = FullCampaignCheckpoint(
            sequence=len(existing) + 1,
            request_digest=request,
            stage=stage,
            previous_digest=None if not existing else existing[-1].digest,
            evidence=normalized,
        )
        payload = _canonical_json(checkpoint.manifest())
        descriptor, staging_name = tempfile.mkstemp(prefix=".staging-", dir=self.root)
        staging = Path(staging_name)
        destination = self.root / f"checkpoint-{checkpoint.sequence:08d}.json"
        committed = False
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            try:
                os.link(staging, destination)
            except FileExistsError:
                retry = self.checkpoints()
                if not retry or retry[-1] != checkpoint:
                    raise FullCampaignError(
                        "campaign checkpoint append raced with a conflict"
                    ) from None
                return retry[-1]
            _fsync_directory(self.root)
            committed = True
        finally:
            os.close(descriptor)
            staging.unlink(missing_ok=True)
            if not committed and destination.exists():
                # A linked destination is complete and immutable even if the directory fsync
                # failed. Leave it for exact retry verification rather than deleting evidence.
                pass
        return checkpoint


def _artifact(reference: object, kind: ArtifactKind, name: str) -> ArtifactRef:
    if not isinstance(reference, ArtifactRef) or reference.kind is not kind:
        raise FullCampaignError(f"{name} must be an {kind.value} artifact reference")
    return reference


def _artifact_from_manifest(value: object, name: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise FullCampaignError(f"{name} artifact manifest must be an object")
    try:
        return ArtifactRef.from_manifest(value)
    except (TypeError, ValueError) as exc:
        raise FullCampaignError(f"{name} artifact manifest is invalid") from exc


def _directory_manifest(reference: DirectoryArtifactRef) -> dict[str, object]:
    return {
        "directory": reference.manifest(),
        "manifest_artifact": reference.manifest_artifact.manifest(),
    }


def _directory_from_manifest(value: object, name: str) -> DirectoryArtifactRef:
    if not isinstance(value, Mapping) or set(value) != {"directory", "manifest_artifact"}:
        raise FullCampaignError(f"{name} directory artifact manifest is invalid")
    directory = value["directory"]
    if not isinstance(directory, Mapping) or set(directory) != {
        "schema_version",
        "kind",
        "total_size_bytes",
        "entries",
    }:
        raise FullCampaignError(f"{name} directory manifest has invalid fields")
    entries_raw = directory["entries"]
    if not isinstance(entries_raw, list):
        raise FullCampaignError(f"{name} directory entries must be an array")
    entries: list[DirectoryEntryRef] = []
    for index, item in enumerate(entries_raw):
        if not isinstance(item, Mapping) or set(item) != {"path", "artifact"}:
            raise FullCampaignError(f"{name} directory entry {index} is invalid")
        path = item["path"]
        if type(path) is not str:
            raise FullCampaignError(f"{name} directory entry path must be text")
        entries.append(
            DirectoryEntryRef(
                path=path,
                artifact=_artifact_from_manifest(
                    item["artifact"],
                    f"{name} directory entry {index}",
                ),
            )
        )
    kind_raw = directory["kind"]
    try:
        kind = ArtifactKind(kind_raw)
    except (TypeError, ValueError) as exc:
        raise FullCampaignError(f"{name} directory kind is unsupported") from exc
    try:
        return DirectoryArtifactRef(
            schema_version=directory["schema_version"],
            kind=kind,
            manifest_artifact=_artifact_from_manifest(
                value["manifest_artifact"],
                f"{name} directory manifest",
            ),
            entries=tuple(entries),
            total_size_bytes=directory["total_size_bytes"],
        )
    except (TypeError, ValueError) as exc:
        raise FullCampaignError(f"{name} directory artifact is invalid") from exc


@dataclass(frozen=True, slots=True)
class QualifiedFMMemberPlan:
    """Exact qualified official-FM member paired with a generated full-train seed."""

    seed: int
    checkpoint_sha256: str
    checkpoint_digest: str
    encoding_sha256: str
    encoding_digest: str
    config_digest: str
    starter_manifest_digest: str
    validation_prediction_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise FullCampaignError("qualified FM member seed must be an unsigned 32-bit integer")
        for name in (
            "checkpoint_sha256",
            "checkpoint_digest",
            "encoding_sha256",
            "encoding_digest",
            "config_digest",
            "starter_manifest_digest",
            "validation_prediction_digest",
        ):
            _digest(getattr(self, name), name)
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"qualified-fm-member\0", self.body()),
        )

    def body(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "seed": self.seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "encoding_sha256": self.encoding_sha256,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "starter_manifest_digest": self.starter_manifest_digest,
            "validation_prediction_digest": self.validation_prediction_digest,
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @classmethod
    def from_manifest(cls, value: object) -> QualifiedFMMemberPlan:
        fields = {
            "schema_version",
            "seed",
            "checkpoint_sha256",
            "checkpoint_digest",
            "encoding_sha256",
            "encoding_digest",
            "config_digest",
            "starter_manifest_digest",
            "validation_prediction_digest",
            "digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise FullCampaignError("qualified FM member manifest has invalid fields")
        if value["schema_version"] != FULL_CAMPAIGN_SCHEMA_VERSION:
            raise FullCampaignError("qualified FM member schema is unsupported")
        member = cls(
            seed=value["seed"],
            checkpoint_sha256=value["checkpoint_sha256"],
            checkpoint_digest=value["checkpoint_digest"],
            encoding_sha256=value["encoding_sha256"],
            encoding_digest=value["encoding_digest"],
            config_digest=value["config_digest"],
            starter_manifest_digest=value["starter_manifest_digest"],
            validation_prediction_digest=value["validation_prediction_digest"],
        )
        if member.digest != value["digest"]:
            raise FullCampaignError("qualified FM member digest mismatch")
        return member


@dataclass(frozen=True, slots=True)
class MatchedSeedSelectionEvidence:
    """One matched public-validation seed retained for audit and final reporting."""

    seed: int
    scientific_request_digest: str
    scientific_record_digest: str
    checkpoint: ArtifactRef
    candidate_validation_prediction: ArtifactRef
    fm_validation_prediction: ArtifactRef
    candidate_metrics: OrganizerMetrics
    fm_metrics: OrganizerMetrics
    candidate_wall_seconds: float
    candidate_peak_rss_bytes: int
    candidate_disk_bytes: int
    fm_wall_seconds: float
    fm_peak_rss_bytes: int
    fm_disk_bytes: int
    fm_member: QualifiedFMMemberPlan
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise FullCampaignError("matched seed must be an unsigned 32-bit integer")
        for name in ("scientific_request_digest", "scientific_record_digest"):
            _digest(getattr(self, name), name)
        _artifact(self.checkpoint, ArtifactKind.CHECKPOINT, "matched checkpoint")
        _artifact(
            self.candidate_validation_prediction,
            ArtifactKind.PREDICTION,
            "matched candidate validation prediction",
        )
        _artifact(
            self.fm_validation_prediction,
            ArtifactKind.PREDICTION,
            "matched FM validation prediction",
        )
        if not isinstance(self.candidate_metrics, OrganizerMetrics) or not isinstance(
            self.fm_metrics, OrganizerMetrics
        ):
            raise FullCampaignError("matched seed metrics must use organizer aggregates")
        for role in ("candidate", "fm"):
            wall = getattr(self, f"{role}_wall_seconds")
            if type(wall) not in (int, float) or not math.isfinite(float(wall)) or wall < 0.0:
                raise FullCampaignError(
                    f"matched seed {role} wall time must be finite and non-negative"
                )
            for suffix in ("peak_rss_bytes", "disk_bytes"):
                resource = getattr(self, f"{role}_{suffix}")
                if type(resource) is not int or not 0 <= resource <= 2**63 - 1:
                    raise FullCampaignError(
                        f"matched seed {role} {suffix} is outside its supported bound"
                    )
        if not isinstance(self.fm_member, QualifiedFMMemberPlan):
            raise FullCampaignError("matched seed requires a qualified FM member")
        if self.fm_member.seed != self.seed:
            raise FullCampaignError("matched generated and FM seeds differ")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"matched-seed-selection\0", self.body()),
        )

    @staticmethod
    def _metrics(metrics: OrganizerMetrics) -> dict[str, float]:
        return {
            "GAUC": metrics.gauc,
            "nDCG@5": metrics.ndcg_at_5,
            "primary": metrics.primary,
        }

    def body(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "seed": self.seed,
            "scientific_request_digest": self.scientific_request_digest,
            "scientific_record_digest": self.scientific_record_digest,
            "checkpoint": self.checkpoint.manifest(),
            "candidate_validation_prediction": self.candidate_validation_prediction.manifest(),
            "fm_validation_prediction": self.fm_validation_prediction.manifest(),
            "candidate_metrics": self._metrics(self.candidate_metrics),
            "fm_metrics": self._metrics(self.fm_metrics),
            "candidate_resources": {
                "wall_seconds": float(self.candidate_wall_seconds),
                "peak_rss_bytes": self.candidate_peak_rss_bytes,
                "disk_bytes": self.candidate_disk_bytes,
            },
            "fm_resources": {
                "wall_seconds": float(self.fm_wall_seconds),
                "peak_rss_bytes": self.fm_peak_rss_bytes,
                "disk_bytes": self.fm_disk_bytes,
            },
            "fm_member": self.fm_member.manifest(),
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @classmethod
    def from_manifest(cls, value: object) -> MatchedSeedSelectionEvidence:
        fields = {
            "schema_version",
            "seed",
            "scientific_request_digest",
            "scientific_record_digest",
            "checkpoint",
            "candidate_validation_prediction",
            "fm_validation_prediction",
            "candidate_metrics",
            "fm_metrics",
            "candidate_resources",
            "fm_resources",
            "fm_member",
            "digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise FullCampaignError("matched seed manifest has invalid fields")
        if value["schema_version"] != FULL_CAMPAIGN_SCHEMA_VERSION:
            raise FullCampaignError("matched seed schema is unsupported")
        candidate_metrics = value["candidate_metrics"]
        fm_metrics = value["fm_metrics"]
        resources = value["candidate_resources"]
        fm_resources = value["fm_resources"]
        metric_fields = {"GAUC", "nDCG@5", "primary"}
        if (
            not isinstance(candidate_metrics, Mapping)
            or set(candidate_metrics) != metric_fields
            or not isinstance(fm_metrics, Mapping)
            or set(fm_metrics) != metric_fields
            or not isinstance(resources, Mapping)
            or set(resources) != {"wall_seconds", "peak_rss_bytes", "disk_bytes"}
            or not isinstance(fm_resources, Mapping)
            or set(fm_resources) != {"wall_seconds", "peak_rss_bytes", "disk_bytes"}
        ):
            raise FullCampaignError("matched seed metrics or resources are invalid")
        candidate = OrganizerMetrics(candidate_metrics["GAUC"], candidate_metrics["nDCG@5"])
        fm = OrganizerMetrics(fm_metrics["GAUC"], fm_metrics["nDCG@5"])
        if candidate_metrics["primary"] != candidate.primary or fm_metrics["primary"] != fm.primary:
            raise FullCampaignError("matched seed primary is not mean(GAUC, nDCG@5)")
        matched = cls(
            seed=value["seed"],
            scientific_request_digest=value["scientific_request_digest"],
            scientific_record_digest=value["scientific_record_digest"],
            checkpoint=_artifact_from_manifest(value["checkpoint"], "matched checkpoint"),
            candidate_validation_prediction=_artifact_from_manifest(
                value["candidate_validation_prediction"],
                "matched candidate validation prediction",
            ),
            fm_validation_prediction=_artifact_from_manifest(
                value["fm_validation_prediction"],
                "matched FM validation prediction",
            ),
            candidate_metrics=candidate,
            fm_metrics=fm,
            candidate_wall_seconds=resources["wall_seconds"],
            candidate_peak_rss_bytes=resources["peak_rss_bytes"],
            candidate_disk_bytes=resources["disk_bytes"],
            fm_wall_seconds=fm_resources["wall_seconds"],
            fm_peak_rss_bytes=fm_resources["peak_rss_bytes"],
            fm_disk_bytes=fm_resources["disk_bytes"],
            fm_member=QualifiedFMMemberPlan.from_manifest(value["fm_member"]),
        )
        if matched.digest != value["digest"]:
            raise FullCampaignError("matched seed digest mismatch")
        return matched


@dataclass(frozen=True, slots=True)
class InnerFoldSelectionEvidence:
    """Exact trusted candidate, parent, and fallback aggregates for one train-only fold."""

    fold_id: str
    candidate: OrganizerMetrics
    parent: OrganizerMetrics
    reference: OrganizerMetrics
    candidate_wall_seconds: float
    candidate_peak_rss_bytes: int
    candidate_disk_bytes: int
    parent_wall_seconds: float
    parent_peak_rss_bytes: int
    parent_disk_bytes: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.fold_id not in {"A", "B"}:
            raise FullCampaignError("inner fold selection evidence must name A or B")
        if any(
            not isinstance(item, OrganizerMetrics)
            for item in (self.candidate, self.parent, self.reference)
        ):
            raise FullCampaignError("inner fold selection evidence requires organizer metrics")
        for role in ("candidate", "parent"):
            wall = getattr(self, f"{role}_wall_seconds")
            if type(wall) not in (int, float) or not math.isfinite(float(wall)) or wall < 0.0:
                raise FullCampaignError(
                    f"inner fold {role} wall time must be finite and non-negative"
                )
            for suffix in ("peak_rss_bytes", "disk_bytes"):
                resource = getattr(self, f"{role}_{suffix}")
                if type(resource) is not int or not 0 <= resource <= 2**63 - 1:
                    raise FullCampaignError(
                        f"inner fold {role} {suffix} is outside its supported bound"
                    )
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"inner-fold-selection\0", self.body()),
        )

    @staticmethod
    def _metrics(metrics: OrganizerMetrics) -> dict[str, float]:
        return {
            "GAUC": metrics.gauc,
            "nDCG@5": metrics.ndcg_at_5,
            "primary": metrics.primary,
        }

    def body(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "fold_id": self.fold_id,
            "candidate": self._metrics(self.candidate),
            "parent": self._metrics(self.parent),
            "reference": self._metrics(self.reference),
            "candidate_resources": {
                "wall_seconds": float(self.candidate_wall_seconds),
                "peak_rss_bytes": self.candidate_peak_rss_bytes,
                "disk_bytes": self.candidate_disk_bytes,
            },
            "parent_resources": {
                "wall_seconds": float(self.parent_wall_seconds),
                "peak_rss_bytes": self.parent_peak_rss_bytes,
                "disk_bytes": self.parent_disk_bytes,
            },
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @classmethod
    def from_manifest(cls, value: object) -> InnerFoldSelectionEvidence:
        fields = {
            "schema_version",
            "fold_id",
            "candidate",
            "parent",
            "reference",
            "candidate_resources",
            "parent_resources",
            "digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise FullCampaignError("inner fold selection manifest has invalid fields")
        if value["schema_version"] != FULL_CAMPAIGN_SCHEMA_VERSION:
            raise FullCampaignError("inner fold selection schema is unsupported")

        def metrics(name: str) -> OrganizerMetrics:
            raw = value[name]
            if not isinstance(raw, Mapping) or set(raw) != {"GAUC", "nDCG@5", "primary"}:
                raise FullCampaignError(f"inner fold {name} metrics are invalid")
            result = OrganizerMetrics(raw["GAUC"], raw["nDCG@5"])
            if raw["primary"] != result.primary:
                raise FullCampaignError(f"inner fold {name} primary is not mean(GAUC, nDCG@5)")
            return result

        candidate_resources = value["candidate_resources"]
        parent_resources = value["parent_resources"]
        resource_fields = {"wall_seconds", "peak_rss_bytes", "disk_bytes"}
        if (
            not isinstance(candidate_resources, Mapping)
            or set(candidate_resources) != resource_fields
            or not isinstance(parent_resources, Mapping)
            or set(parent_resources) != resource_fields
        ):
            raise FullCampaignError("inner fold resource evidence is invalid")
        fold = cls(
            fold_id=value["fold_id"],
            candidate=metrics("candidate"),
            parent=metrics("parent"),
            reference=metrics("reference"),
            candidate_wall_seconds=candidate_resources["wall_seconds"],
            candidate_peak_rss_bytes=candidate_resources["peak_rss_bytes"],
            candidate_disk_bytes=candidate_resources["disk_bytes"],
            parent_wall_seconds=parent_resources["wall_seconds"],
            parent_peak_rss_bytes=parent_resources["peak_rss_bytes"],
            parent_disk_bytes=parent_resources["disk_bytes"],
        )
        if fold.digest != value["digest"]:
            raise FullCampaignError("inner fold selection digest mismatch")
        return fold


@dataclass(frozen=True, slots=True)
class FinalizationSelectionPlan:
    """Closed generated-candidate evidence sufficient for later provider-free finalization."""

    experiment_id: str
    candidate_id: str
    candidate_fingerprint: str
    source_digest: str
    parent_source_digest: str
    executable_change_digest: str
    config_digest: str
    training_policy_digest: str
    evidence_receipt_digest: str
    source_snapshot: DirectoryArtifactRef
    training_features: ArtifactRef
    training_targets: ArtifactRef = field(repr=False)
    training_user_groups: ArtifactRef
    validation_features: ArtifactRef
    final_features: ArtifactRef
    feature_bundle_digest: str
    feature_count: int
    dataset_digest: str
    validation_inputs_digest: str
    final_inputs_digest: str
    frozen_fusion_weights: tuple[float, float]
    representative_seed: int
    selected_outer_request_digest: str
    scientific_record_digest: str
    tree_checkpoint: ArtifactRef
    validation_prediction: ArtifactRef
    fm_member: QualifiedFMMemberPlan
    inner_folds: tuple[InnerFoldSelectionEvidence, ...]
    matched_seeds: tuple[MatchedSeedSelectionEvidence, ...]
    timeout_seconds: int
    memory_limit_bytes: int
    threads: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.experiment_id) is not str
            or not self.experiment_id
            or "\x00" in self.experiment_id
            or "\n" in self.experiment_id
            or "\r" in self.experiment_id
        ):
            raise FullCampaignError("selected experiment_id must be non-empty single-line text")
        if (
            type(self.candidate_id) is not str
            or not self.candidate_id
            or "\x00" in self.candidate_id
        ):
            raise FullCampaignError("selected candidate_id must be non-empty text")
        for name in (
            "candidate_fingerprint",
            "source_digest",
            "parent_source_digest",
            "executable_change_digest",
            "config_digest",
            "training_policy_digest",
            "evidence_receipt_digest",
            "feature_bundle_digest",
            "dataset_digest",
            "validation_inputs_digest",
            "final_inputs_digest",
            "selected_outer_request_digest",
            "scientific_record_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            not isinstance(self.source_snapshot, DirectoryArtifactRef)
            or self.source_snapshot.kind is not ArtifactKind.SOURCE
        ):
            raise FullCampaignError("selected source_snapshot must be a source directory artifact")
        for name in (
            "training_features",
            "training_targets",
            "training_user_groups",
            "validation_features",
            "final_features",
        ):
            _artifact(getattr(self, name), ArtifactKind.INPUT, name)
        _artifact(self.tree_checkpoint, ArtifactKind.CHECKPOINT, "tree_checkpoint")
        _artifact(
            self.validation_prediction,
            ArtifactKind.PREDICTION,
            "validation_prediction",
        )
        if type(self.feature_count) is not int or self.feature_count <= 0:
            raise FullCampaignError("selected feature_count must be positive")
        if self.frozen_fusion_weights not in FUSION_WEIGHT_GRID:
            raise FullCampaignError("selected fusion weights are outside the frozen grid")
        if type(self.representative_seed) is not int or not (
            0 <= self.representative_seed <= 2**32 - 1
        ):
            raise FullCampaignError("representative seed must be unsigned 32-bit")
        if not isinstance(self.fm_member, QualifiedFMMemberPlan):
            raise FullCampaignError("selected candidate requires a qualified FM member")
        if self.fm_member.seed != self.representative_seed:
            raise FullCampaignError("selected tree and FM member seeds differ")
        if any(not isinstance(item, InnerFoldSelectionEvidence) for item in self.inner_folds):
            raise FullCampaignError("selected inner-fold evidence is invalid")
        if tuple(item.fold_id for item in self.inner_folds) != ("A", "B"):
            raise FullCampaignError("selected inner-fold evidence must cover A then B")
        if any(not isinstance(item, MatchedSeedSelectionEvidence) for item in self.matched_seeds):
            raise FullCampaignError("selected matched-seed evidence is invalid")
        if tuple(item.seed for item in self.matched_seeds) != (0, 1, 2):
            raise FullCampaignError("selected matched-seed evidence must cover seeds 0, 1, 2")
        representative = next(
            item for item in self.matched_seeds if item.seed == self.representative_seed
        )
        if (
            representative.scientific_request_digest != self.selected_outer_request_digest
            or representative.scientific_record_digest != self.scientific_record_digest
            or representative.checkpoint != self.tree_checkpoint
            or representative.candidate_validation_prediction != self.validation_prediction
            or representative.fm_member != self.fm_member
        ):
            raise FullCampaignError("representative selection differs from matched-seed evidence")
        for name, maximum in (
            ("timeout_seconds", 86_400),
            ("memory_limit_bytes", 2**63 - 1),
            ("threads", 64),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise FullCampaignError(f"selected {name} is outside its supported bound")
        entries = {entry.path: entry.artifact for entry in self.source_snapshot.entries}
        config = entries.get("config.json")
        if config is None or config.sha256 != self.config_digest:
            raise FullCampaignError("selected source snapshot config identity differs")
        if "candidate.py" not in entries:
            raise FullCampaignError("selected source snapshot lacks candidate.py")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"finalization-selection\0", self.body()),
        )

    def body(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "source_digest": self.source_digest,
            "parent_source_digest": self.parent_source_digest,
            "executable_change_digest": self.executable_change_digest,
            "config_digest": self.config_digest,
            "training_policy_digest": self.training_policy_digest,
            "evidence_receipt_digest": self.evidence_receipt_digest,
            "source_snapshot": _directory_manifest(self.source_snapshot),
            "training_features": self.training_features.manifest(),
            "training_targets": self.training_targets.manifest(),
            "training_user_groups": self.training_user_groups.manifest(),
            "validation_features": self.validation_features.manifest(),
            "final_features": self.final_features.manifest(),
            "feature_bundle_digest": self.feature_bundle_digest,
            "feature_count": self.feature_count,
            "dataset_digest": self.dataset_digest,
            "validation_inputs_digest": self.validation_inputs_digest,
            "final_inputs_digest": self.final_inputs_digest,
            "frozen_fusion_weights": list(self.frozen_fusion_weights),
            "representative_seed": self.representative_seed,
            "selected_outer_request_digest": self.selected_outer_request_digest,
            "scientific_record_digest": self.scientific_record_digest,
            "tree_checkpoint": self.tree_checkpoint.manifest(),
            "validation_prediction": self.validation_prediction.manifest(),
            "fm_member": self.fm_member.manifest(),
            "inner_folds": [item.manifest() for item in self.inner_folds],
            "matched_seeds": [item.manifest() for item in self.matched_seeds],
            "limits": {
                "timeout_seconds": self.timeout_seconds,
                "memory_limit_bytes": self.memory_limit_bytes,
                "threads": self.threads,
            },
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @classmethod
    def from_manifest(cls, value: object) -> FinalizationSelectionPlan:
        fields = {
            "schema_version",
            "experiment_id",
            "candidate_id",
            "candidate_fingerprint",
            "source_digest",
            "parent_source_digest",
            "executable_change_digest",
            "config_digest",
            "training_policy_digest",
            "evidence_receipt_digest",
            "source_snapshot",
            "training_features",
            "training_targets",
            "training_user_groups",
            "validation_features",
            "final_features",
            "feature_bundle_digest",
            "feature_count",
            "dataset_digest",
            "validation_inputs_digest",
            "final_inputs_digest",
            "frozen_fusion_weights",
            "representative_seed",
            "selected_outer_request_digest",
            "scientific_record_digest",
            "tree_checkpoint",
            "validation_prediction",
            "fm_member",
            "inner_folds",
            "matched_seeds",
            "limits",
            "digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise FullCampaignError("finalization selection manifest has invalid fields")
        if value["schema_version"] != FULL_CAMPAIGN_SCHEMA_VERSION:
            raise FullCampaignError("finalization selection schema is unsupported")
        weights = value["frozen_fusion_weights"]
        limits = value["limits"]
        inner_folds = value["inner_folds"]
        matched_seeds = value["matched_seeds"]
        if (
            not isinstance(weights, list)
            or len(weights) != 2
            or not all(type(item) is float for item in weights)
            or not isinstance(limits, Mapping)
            or set(limits) != {"timeout_seconds", "memory_limit_bytes", "threads"}
            or not isinstance(inner_folds, list)
            or not isinstance(matched_seeds, list)
        ):
            raise FullCampaignError("finalization selection policy or limits are invalid")
        selection = cls(
            experiment_id=value["experiment_id"],
            candidate_id=value["candidate_id"],
            candidate_fingerprint=value["candidate_fingerprint"],
            source_digest=value["source_digest"],
            parent_source_digest=value["parent_source_digest"],
            executable_change_digest=value["executable_change_digest"],
            config_digest=value["config_digest"],
            training_policy_digest=value["training_policy_digest"],
            evidence_receipt_digest=value["evidence_receipt_digest"],
            source_snapshot=_directory_from_manifest(value["source_snapshot"], "source_snapshot"),
            training_features=_artifact_from_manifest(
                value["training_features"], "training_features"
            ),
            training_targets=_artifact_from_manifest(value["training_targets"], "training_targets"),
            training_user_groups=_artifact_from_manifest(
                value["training_user_groups"], "training_user_groups"
            ),
            validation_features=_artifact_from_manifest(
                value["validation_features"], "validation_features"
            ),
            final_features=_artifact_from_manifest(value["final_features"], "final_features"),
            feature_bundle_digest=value["feature_bundle_digest"],
            feature_count=value["feature_count"],
            dataset_digest=value["dataset_digest"],
            validation_inputs_digest=value["validation_inputs_digest"],
            final_inputs_digest=value["final_inputs_digest"],
            frozen_fusion_weights=(weights[0], weights[1]),
            representative_seed=value["representative_seed"],
            selected_outer_request_digest=value["selected_outer_request_digest"],
            scientific_record_digest=value["scientific_record_digest"],
            tree_checkpoint=_artifact_from_manifest(value["tree_checkpoint"], "tree_checkpoint"),
            validation_prediction=_artifact_from_manifest(
                value["validation_prediction"], "validation_prediction"
            ),
            fm_member=QualifiedFMMemberPlan.from_manifest(value["fm_member"]),
            inner_folds=tuple(
                InnerFoldSelectionEvidence.from_manifest(item) for item in inner_folds
            ),
            matched_seeds=tuple(
                MatchedSeedSelectionEvidence.from_manifest(item) for item in matched_seeds
            ),
            timeout_seconds=limits["timeout_seconds"],
            memory_limit_bytes=limits["memory_limit_bytes"],
            threads=limits["threads"],
        )
        if selection.digest != value["digest"]:
            raise FullCampaignError("finalization selection digest mismatch")
        return selection


@dataclass(frozen=True, slots=True)
class FullCampaignOutcome:
    """Persisted research closure consumed by a later fallback-aware finalizer."""

    run_dir: Path
    campaign_id: str
    request_digest: str
    progress_predecessor_digest: str
    fallback_candidate_id: str
    fallback_receipt_digest: str
    qualification_manifest_digest: str
    dataset_digest: str
    scorer_digest: str
    validation_row_count: int
    final_row_count: int
    scientific_result_digest: str | None
    reflection_request_digest: str | None
    reflection_response_digest: str | None
    reflection_transcript: ArtifactRef | None
    selection: FinalizationSelectionPlan | None
    launches_used: int
    outer_queries_used: int
    manual_interventions: int = 0
    status: str = "FINALIZATION_REQUIRED"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_dir, Path) or not self.run_dir.is_absolute():
            raise FullCampaignError("outcome run_dir must be an absolute Path")
        for name in ("campaign_id", "fallback_candidate_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or "\x00" in value:
                raise FullCampaignError(f"outcome {name} must be non-empty text")
        for name in (
            "request_digest",
            "progress_predecessor_digest",
            "fallback_receipt_digest",
            "qualification_manifest_digest",
            "dataset_digest",
            "scorer_digest",
        ):
            _digest(getattr(self, name), name)
        for name in ("validation_row_count", "final_row_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise FullCampaignError(f"outcome {name} must be positive")
        optional_digests = (
            self.scientific_result_digest,
            self.reflection_request_digest,
            self.reflection_response_digest,
        )
        for index, value in enumerate(optional_digests):
            if value is not None:
                _digest(value, f"optional outcome digest {index}")
        reflection_parts = (
            self.reflection_request_digest,
            self.reflection_response_digest,
            self.reflection_transcript,
        )
        if any(item is None for item in reflection_parts) and not all(
            item is None for item in reflection_parts
        ):
            raise FullCampaignError("outcome reflection evidence must be complete or absent")
        if self.reflection_transcript is not None:
            _artifact(self.reflection_transcript, ArtifactKind.LOG, "reflection_transcript")
        if self.selection is not None and not isinstance(self.selection, FinalizationSelectionPlan):
            raise FullCampaignError("outcome selection must be a finalization selection plan")
        if self.selection is not None and (
            self.scientific_result_digest is None or self.reflection_transcript is None
        ):
            raise FullCampaignError(
                "generated selection requires scientific and reflection evidence"
            )
        for name, maximum in (("launches_used", 50), ("outer_queries_used", 6)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise FullCampaignError(f"outcome {name} is outside the frozen campaign bound")
        if self.manual_interventions != 0:
            raise FullCampaignError("provider-free campaign manual interventions must be zero")
        if self.status != "FINALIZATION_REQUIRED":
            raise FullCampaignError("research outcome status must require finalization")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"outcome\0", self.body()),
        )

    @property
    def finalization_required(self) -> bool:
        return self.status == "FINALIZATION_REQUIRED"

    @property
    def fallback_preserved(self) -> bool:
        return self.finalization_required

    @property
    def has_generated_selection(self) -> bool:
        return self.selection is not None

    def body(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "campaign_id": self.campaign_id,
            "request_digest": self.request_digest,
            "progress_predecessor_digest": self.progress_predecessor_digest,
            "fallback_candidate_id": self.fallback_candidate_id,
            "fallback_receipt_digest": self.fallback_receipt_digest,
            "qualification_manifest_digest": self.qualification_manifest_digest,
            "dataset_digest": self.dataset_digest,
            "scorer_digest": self.scorer_digest,
            "validation_row_count": self.validation_row_count,
            "final_row_count": self.final_row_count,
            "scientific_result_digest": self.scientific_result_digest,
            "reflection_request_digest": self.reflection_request_digest,
            "reflection_response_digest": self.reflection_response_digest,
            "reflection_transcript": (
                None
                if self.reflection_transcript is None
                else self.reflection_transcript.manifest()
            ),
            "selection": None if self.selection is None else self.selection.manifest(),
            "launches_used": self.launches_used,
            "outer_queries_used": self.outer_queries_used,
            "manual_interventions": self.manual_interventions,
            "status": self.status,
            "fallback_preserved": True,
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest())

    @classmethod
    def from_bytes(cls, payload: bytes) -> FullCampaignOutcome:
        if not payload or len(payload) > _MAX_PROGRESS_CHECKPOINT_BYTES:
            raise FullCampaignError("full campaign outcome size is outside the supported bound")
        try:
            raw = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise FullCampaignError("full campaign outcome is not canonical ASCII JSON") from exc
        expected = {
            "schema_version",
            "run_dir",
            "campaign_id",
            "request_digest",
            "progress_predecessor_digest",
            "fallback_candidate_id",
            "fallback_receipt_digest",
            "qualification_manifest_digest",
            "dataset_digest",
            "scorer_digest",
            "validation_row_count",
            "final_row_count",
            "scientific_result_digest",
            "reflection_request_digest",
            "reflection_response_digest",
            "reflection_transcript",
            "selection",
            "launches_used",
            "outer_queries_used",
            "manual_interventions",
            "status",
            "fallback_preserved",
            "digest",
        }
        if not isinstance(raw, dict) or set(raw) != expected or _canonical_json(raw) != payload:
            raise FullCampaignError("full campaign outcome does not use the exact schema")
        if (
            raw["schema_version"] != FULL_CAMPAIGN_SCHEMA_VERSION
            or raw["fallback_preserved"] is not True
        ):
            raise FullCampaignError("full campaign outcome schema or fallback gate is invalid")
        transcript = raw["reflection_transcript"]
        selection_raw = raw["selection"]
        outcome = cls(
            run_dir=Path(raw["run_dir"]),
            campaign_id=raw["campaign_id"],
            request_digest=raw["request_digest"],
            progress_predecessor_digest=raw["progress_predecessor_digest"],
            fallback_candidate_id=raw["fallback_candidate_id"],
            fallback_receipt_digest=raw["fallback_receipt_digest"],
            qualification_manifest_digest=raw["qualification_manifest_digest"],
            dataset_digest=raw["dataset_digest"],
            scorer_digest=raw["scorer_digest"],
            validation_row_count=raw["validation_row_count"],
            final_row_count=raw["final_row_count"],
            scientific_result_digest=raw["scientific_result_digest"],
            reflection_request_digest=raw["reflection_request_digest"],
            reflection_response_digest=raw["reflection_response_digest"],
            reflection_transcript=(
                None
                if transcript is None
                else _artifact_from_manifest(transcript, "reflection_transcript")
            ),
            selection=(
                None
                if selection_raw is None
                else FinalizationSelectionPlan.from_manifest(selection_raw)
            ),
            launches_used=raw["launches_used"],
            outer_queries_used=raw["outer_queries_used"],
            manual_interventions=raw["manual_interventions"],
            status=raw["status"],
        )
        if outcome.digest != raw["digest"]:
            raise FullCampaignError("full campaign outcome digest mismatch")
        return outcome


class FullCampaignOutcomeRepository:
    """Content-addressed outcome store anchored by the terminal progress checkpoint."""

    def __init__(
        self,
        *,
        run_dir: Path,
        artifact_store: ArtifactStore,
        progress: FullCampaignProgressLedger,
    ) -> None:
        if not isinstance(run_dir, Path) or not run_dir.is_absolute():
            raise FullCampaignError("outcome repository run_dir must be an absolute Path")
        if not isinstance(artifact_store, ArtifactStore):
            raise FullCampaignError("outcome repository requires an ArtifactStore")
        if not isinstance(progress, FullCampaignProgressLedger):
            raise FullCampaignError("outcome repository requires a progress ledger")
        self.run_dir = run_dir
        self.artifact_store = artifact_store
        self.progress = progress

    def _verify_artifacts(self, outcome: FullCampaignOutcome) -> None:
        if outcome.reflection_transcript is not None:
            self.artifact_store.verify(outcome.reflection_transcript)
        selection = outcome.selection
        if selection is None:
            return
        self.artifact_store.verify_directory(selection.source_snapshot)
        for reference in (
            selection.training_features,
            selection.training_targets,
            selection.training_user_groups,
            selection.validation_features,
            selection.final_features,
            selection.tree_checkpoint,
            selection.validation_prediction,
        ):
            self.artifact_store.verify(reference)
        for matched in selection.matched_seeds:
            for reference in (
                matched.checkpoint,
                matched.candidate_validation_prediction,
                matched.fm_validation_prediction,
            ):
                self.artifact_store.verify(reference)

    def load(self, *, request_digest: str) -> FullCampaignOutcome:
        request = _digest(request_digest, "request_digest")
        checkpoints = self.progress.checkpoints()
        if not checkpoints or checkpoints[-1].stage is not FullCampaignStage.FINALIZATION_REQUIRED:
            raise FullCampaignError("full campaign outcome is not durably finalization-ready")
        terminal = checkpoints[-1]
        if terminal.request_digest != request:
            raise FullCampaignError("outcome request identity differs from progress")
        if set(terminal.evidence) != {
            "outcome_artifact",
            "outcome_digest",
            "selection_digest",
        }:
            raise FullCampaignError("terminal outcome checkpoint has invalid evidence fields")
        reference = _artifact_from_manifest(
            terminal.evidence["outcome_artifact"],
            "outcome",
        )
        _artifact(reference, ArtifactKind.MANIFEST, "outcome")
        payload = self.artifact_store.read_bytes(
            reference,
            max_bytes=_MAX_PROGRESS_CHECKPOINT_BYTES,
        )
        outcome = FullCampaignOutcome.from_bytes(payload)
        if outcome.run_dir != self.run_dir or outcome.request_digest != request:
            raise FullCampaignError("outcome run or request identity differs")
        if outcome.progress_predecessor_digest != terminal.previous_digest:
            raise FullCampaignError("outcome predecessor differs from terminal checkpoint")
        if terminal.evidence["outcome_digest"] != outcome.digest:
            raise FullCampaignError("terminal checkpoint and outcome digest differ")
        expected_selection = None if outcome.selection is None else outcome.selection.digest
        if terminal.evidence["selection_digest"] != expected_selection:
            raise FullCampaignError("terminal checkpoint and selection digest differ")
        self._verify_artifacts(outcome)
        return outcome

    def commit(self, outcome: FullCampaignOutcome) -> FullCampaignOutcome:
        if not isinstance(outcome, FullCampaignOutcome):
            raise FullCampaignError("outcome repository can commit only FullCampaignOutcome")
        if outcome.run_dir != self.run_dir:
            raise FullCampaignError("outcome run differs from its repository")
        checkpoints = self.progress.checkpoints()
        if checkpoints and checkpoints[-1].stage is FullCampaignStage.FINALIZATION_REQUIRED:
            retained = self.load(request_digest=outcome.request_digest)
            if retained != outcome:
                raise FullCampaignError("outcome retry contradicts durable terminal evidence")
            return retained
        if not checkpoints or checkpoints[-1].stage is not FullCampaignStage.REFLECTED:
            raise FullCampaignError("outcome can be committed only after durable reflection")
        if checkpoints[-1].digest != outcome.progress_predecessor_digest:
            raise FullCampaignError("outcome does not bind the latest reflected checkpoint")
        if checkpoints[-1].request_digest != outcome.request_digest:
            raise FullCampaignError("outcome request differs from reflected progress")
        self._verify_artifacts(outcome)
        reference = self.artifact_store.put_bytes(
            outcome.canonical_bytes,
            kind=ArtifactKind.MANIFEST,
            max_bytes=_MAX_PROGRESS_CHECKPOINT_BYTES,
        )
        self.progress.append(
            request_digest=outcome.request_digest,
            stage=FullCampaignStage.FINALIZATION_REQUIRED,
            evidence={
                "outcome_artifact": reference.manifest(),
                "outcome_digest": outcome.digest,
                "selection_digest": (
                    None if outcome.selection is None else outcome.selection.digest
                ),
            },
        )
        return self.load(request_digest=outcome.request_digest)


def load_full_campaign_outcome(
    run_dir: Path,
    *,
    engine: CampaignEngine | None = None,
) -> FullCampaignOutcome:
    """Strictly rehydrate a retained research outcome in a later local process.

    This is intentionally read-only: it verifies controller/store identity, the complete progress
    chain, every selected content-addressed artifact, and the official qualification member
    without appending a current-clock observation or opening the final-period outcome domain.
    """

    from kuairand_agent.campaign.controller import CampaignEngine
    from kuairand_agent.campaign.qualification_evidence import (
        QualificationExpectations,
        load_official_fm_qualification,
    )

    if not isinstance(run_dir, Path):
        raise FullCampaignError("run_dir must be a Path")
    try:
        resolved = run_dir.resolve(strict=True)
    except OSError as exc:
        raise FullCampaignError("campaign run directory is unavailable") from exc
    selected_engine = CampaignEngine() if engine is None else engine
    if not isinstance(selected_engine, CampaignEngine):
        raise FullCampaignError("engine must be a CampaignEngine")
    request = selected_engine.load_request(resolved)
    outcome = FullCampaignOutcomeRepository(
        run_dir=resolved,
        artifact_store=ArtifactStore(resolved / "artifacts"),
        progress=FullCampaignProgressLedger(
            resolved / "production" / "progress",
            create=False,
        ),
    ).load(request_digest=request.digest)
    status = selected_engine.status(resolved)
    if (
        outcome.campaign_id != request.campaign_id
        or outcome.qualification_manifest_digest != request.qualification_manifest_digest
        or outcome.dataset_digest != request.dataset_manifest_digest
    ):
        raise FullCampaignError("retained outcome differs from the immutable campaign request")
    if (
        not status.finalization_required
        or status.incumbent_id != outcome.fallback_candidate_id
        or not status.incumbent_is_fallback
        or not status.incumbent_replay_verified
        or status.launches_used != outcome.launches_used
        or status.outer_queries_used != outcome.outer_queries_used
    ):
        raise FullCampaignError("retained outcome differs from authoritative campaign status")
    qualification = load_official_fm_qualification(
        request.qualification_run_dir,
        expectations=QualificationExpectations(
            canonical_digest=outcome.dataset_digest,
            starter_manifest_digest=request.starter_manifest_digest,
            scorer_digest=outcome.scorer_digest,
            validation_row_count=outcome.validation_row_count,
            final_row_count=outcome.final_row_count,
        ),
    )
    if (
        qualification.manifest_digest != outcome.qualification_manifest_digest
        or qualification.benchmark_digest != request.benchmark_digest
    ):
        raise FullCampaignError("retained qualification differs from campaign outcome")
    if outcome.selection is not None:
        for matched in outcome.selection.matched_seeds:
            selected = qualification.outer_seed(matched.seed)
            member = matched.fm_member
            observed = (
                member.checkpoint_sha256,
                member.checkpoint_digest,
                member.encoding_sha256,
                member.encoding_digest,
                member.config_digest,
                member.starter_manifest_digest,
                member.validation_prediction_digest,
            )
            expected = (
                selected.checkpoint_file_sha256,
                selected.checkpoint_digest,
                selected.encoding_file_sha256,
                selected.encoding_digest,
                selected.config_digest,
                qualification.starter_manifest_digest,
                selected.validation_prediction_digest,
            )
            if observed != expected:
                raise FullCampaignError(
                    "selected FM member differs from strict qualification evidence"
                )
    checkpoints = FullCampaignProgressLedger(
        resolved / "production" / "progress",
        create=False,
    ).checkpoints()
    fold_checkpoint = next(
        (item for item in checkpoints if item.stage is FullCampaignStage.FOLD_CONTROLS_READY),
        None,
    )
    if (
        fold_checkpoint is None
        or fold_checkpoint.evidence.get("fallback_receipt_digest")
        != outcome.fallback_receipt_digest
    ):
        raise FullCampaignError("fallback receipt is absent from durable fold-control evidence")
    return outcome


@dataclass(frozen=True, slots=True)
class FoldDataPlane:
    """One train-only rolling-origin fold with targets kept in the trusted controller."""

    fold: TemporalFold
    prefix_inputs: CanonicalInputs
    prefix_labels: tuple[int, ...] = field(repr=False)
    prefix_click_labels: tuple[int, ...] = field(repr=False)
    prefix_watch_progress: tuple[float, ...] = field(repr=False)
    query_inputs: CanonicalInputs
    query_labels: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fold, TemporalFold):
            raise FullCampaignError("fold data plane requires a TemporalFold")
        if len(self.prefix_inputs) != len(self.prefix_labels):
            raise FullCampaignError("fold prefix inputs and targets differ in row count")
        if len(self.query_inputs) != len(self.query_labels):
            raise FullCampaignError("fold query inputs and targets differ in row count")
        if any(type(value) is not int or value not in (0, 1) for value in self.prefix_labels):
            raise FullCampaignError("fold prefix targets must remain binary integers")
        if len(self.prefix_inputs) != len(self.prefix_click_labels) or any(
            type(value) is not int or value not in (0, 1) for value in self.prefix_click_labels
        ):
            raise FullCampaignError("fold prefix click targets must remain aligned binary integers")
        if len(self.prefix_inputs) != len(self.prefix_watch_progress) or any(
            not math.isfinite(value) or not 0.0 <= value <= 2.0
            for value in self.prefix_watch_progress
        ):
            raise FullCampaignError(
                "fold prefix watch progress must remain aligned and finite in [0, 2]"
            )
        if any(type(value) is not int or value not in (0, 1) for value in self.query_labels):
            raise FullCampaignError("fold query targets must remain binary integers")

    @property
    def name(self) -> str:
        return self.fold.name


@dataclass(frozen=True, slots=True)
class CampaignDataPlane:
    """Prepared development inputs; final-period outcomes cannot be represented here."""

    dataset_digest: str
    folds: TemporalFoldSet
    fold_a: FoldDataPlane
    fold_b: FoldDataPlane
    outer_train_inputs: CanonicalInputs
    outer_train_labels: tuple[int, ...] = field(repr=False)
    outer_train_click_labels: tuple[int, ...] = field(repr=False)
    outer_train_watch_progress: tuple[float, ...] = field(repr=False)
    outer_validation_inputs: CanonicalInputs
    final_inputs: CanonicalInputs
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.dataset_digest, "dataset_digest")
        if not isinstance(self.folds, TemporalFoldSet):
            raise FullCampaignError("campaign data plane requires frozen temporal folds")
        if (self.fold_a.fold, self.fold_b.fold) != self.folds.folds:
            raise FullCampaignError("campaign fold data differs from the frozen fold set")
        if len(self.outer_train_inputs) != len(self.outer_train_labels):
            raise FullCampaignError("outer train inputs and targets differ in row count")
        if any(type(value) is not int or value not in (0, 1) for value in self.outer_train_labels):
            raise FullCampaignError("outer train targets must remain binary integers")
        if len(self.outer_train_inputs) != len(self.outer_train_click_labels) or any(
            type(value) is not int or value not in (0, 1) for value in self.outer_train_click_labels
        ):
            raise FullCampaignError("outer train click targets must remain aligned binary integers")
        if len(self.outer_train_inputs) != len(self.outer_train_watch_progress) or any(
            not math.isfinite(value) or not 0.0 <= value <= 2.0
            for value in self.outer_train_watch_progress
        ):
            raise FullCampaignError(
                "outer train watch progress must remain aligned and finite in [0, 2]"
            )
        manifest = self.manifest()
        object.__setattr__(self, "digest", _manifest_digest(b"data-plane\0", manifest))

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "dataset_digest": self.dataset_digest,
            "fold_set_digest": self.folds.digest,
            "folds": {
                "A": {
                    "fold_digest": self.fold_a.fold.digest,
                    "prefix_inputs_digest": self.fold_a.prefix_inputs.digest,
                    "prefix_rows": len(self.fold_a.prefix_inputs),
                    "query_inputs_digest": self.fold_a.query_inputs.digest,
                    "query_rows": len(self.fold_a.query_inputs),
                },
                "B": {
                    "fold_digest": self.fold_b.fold.digest,
                    "prefix_inputs_digest": self.fold_b.prefix_inputs.digest,
                    "prefix_rows": len(self.fold_b.prefix_inputs),
                    "query_inputs_digest": self.fold_b.query_inputs.digest,
                    "query_rows": len(self.fold_b.query_inputs),
                },
            },
            "outer_train_inputs_digest": self.outer_train_inputs.digest,
            "outer_train_rows": len(self.outer_train_inputs),
            "outer_validation_inputs_digest": self.outer_validation_inputs.digest,
            "outer_validation_rows": len(self.outer_validation_inputs),
            "final_inputs_digest": self.final_inputs.digest,
            "final_rows": len(self.final_inputs),
            "final_target_capability": None,
        }


def _fold_data(
    train_inputs: CanonicalInputs,
    labels: tuple[int, ...],
    click_labels: tuple[int, ...],
    watch_progress: tuple[float, ...],
    fold: TemporalFold,
) -> FoldDataPlane:
    return FoldDataPlane(
        fold=fold,
        prefix_inputs=subset_canonical_inputs(train_inputs, fold.train_positions),
        prefix_labels=subset_values(labels, fold.train_positions),
        prefix_click_labels=subset_values(click_labels, fold.train_positions),
        prefix_watch_progress=tuple(watch_progress[index] for index in fold.train_positions),
        query_inputs=subset_canonical_inputs(train_inputs, fold.valid_positions),
        query_labels=subset_values(labels, fold.valid_positions),
    )


def prepare_campaign_data_plane(
    dataset: CanonicalDataset,
    *,
    expected_dataset_digest: str,
) -> CampaignDataPlane:
    """Validate the final label boundary before deriving folds or touching train targets."""

    expected = _digest(expected_dataset_digest, "expected_dataset_digest")
    if not isinstance(dataset, CanonicalDataset):
        raise FullCampaignError("campaign data must be a CanonicalDataset")
    if dataset.digest != expected:
        raise FullCampaignError("canonical dataset identity differs from campaign identity")

    # This gate intentionally precedes access to any development target vector.  Canonical final
    # construction normally proves all of it already; repeat the facts at the orchestration seam
    # so a future alternate loader cannot silently broaden authority.
    final = dataset.final
    if final.name is not SplitName.TEST or not isinstance(final, CanonicalFinalSplit):
        raise FullCampaignError("final split must be label-free")
    if final.outcome_trace.parsed_fields or final.outcome_trace.parsed_cell_count != 0:
        raise FullCampaignError("final-period outcomes were parsed during development")
    if final.outcome_trace.skipped_fields != OUTCOME_FIELDS:
        raise FullCampaignError("final-period outcome skip evidence differs from the contract")

    train = dataset.train
    valid = dataset.valid
    if not isinstance(train.targets, TrainingTargets):
        raise FullCampaignError("official train split lacks trusted training targets")
    if not isinstance(valid.targets, ProtectedTargets):
        raise FullCampaignError("public validation lacks protected scorer targets")
    labels = train.targets.long_view
    click_labels = tuple(int(value) for value in train.targets.column("is_click"))
    play_time_ms = tuple(float(value) for value in train.targets.column("play_time_ms"))
    watch_progress = tuple(
        min(max(play_time / max(min(duration, 18_000.0), 1.0), 0.0), 2.0)
        for play_time, duration in zip(
            play_time_ms,
            train.inputs.duration_ms,
            strict=True,
        )
    )
    folds = build_temporal_folds(train)
    return CampaignDataPlane(
        dataset_digest=dataset.digest,
        folds=folds,
        fold_a=_fold_data(train.inputs, labels, click_labels, watch_progress, folds.fold_a),
        fold_b=_fold_data(train.inputs, labels, click_labels, watch_progress, folds.fold_b),
        outer_train_inputs=train.inputs,
        outer_train_labels=labels,
        outer_train_click_labels=click_labels,
        outer_train_watch_progress=watch_progress,
        outer_validation_inputs=valid.inputs,
        final_inputs=final.inputs,
    )


@dataclass(frozen=True, slots=True)
class ProductionFeatureBundle:
    """A/B and outer+final matrices with frozen outcomes and input-only query warm-up."""

    data_plane_digest: str
    fold_a: PureFeaturePair
    fold_b: PureFeaturePair
    outer_and_final: PureFeaturePair
    outer_validation: FeatureMatrix
    final: FeatureMatrix
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.data_plane_digest, "data_plane_digest")
        if self.outer_and_final.query.row_count != (
            self.outer_validation.row_count + self.final.row_count
        ):
            raise FullCampaignError("combined outer/final features lost their split boundary")
        if not (
            self.outer_and_final.query.feature_names
            == self.outer_validation.feature_names
            == self.final.feature_names
        ):
            raise FullCampaignError("outer/final feature schemas differ")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"feature-bundle\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "data_plane_digest": self.data_plane_digest,
            "fold_a_pair_digest": self.fold_a.digest,
            "fold_b_pair_digest": self.fold_b.digest,
            "outer_and_final_pair_digest": self.outer_and_final.digest,
            "outer_validation": self.outer_validation.manifest(),
            "final": self.final.manifest(),
            "final_target_capability": None,
        }


def build_production_feature_bundle(
    data: CampaignDataPlane,
    *,
    builder_source_digest: str,
    cache_dir: Path | str | None = None,
) -> ProductionFeatureBundle:
    """Build one outer query stream: frozen outcomes plus strict-earlier input exposures."""

    if not isinstance(data, CampaignDataPlane):
        raise FullCampaignError("data must be a prepared CampaignDataPlane")
    builder = _digest(builder_source_digest, "builder_source_digest")
    fold_a = build_pure_feature_pair(
        prefix_inputs=data.fold_a.prefix_inputs,
        prefix_labels=data.fold_a.prefix_labels,
        prefix_click_labels=data.fold_a.prefix_click_labels,
        prefix_watch_progress=data.fold_a.prefix_watch_progress,
        query_inputs=data.fold_a.query_inputs,
        dataset_digest=data.dataset_digest,
        split_role="inner_fold_A",
        builder_source_digest=builder,
        cache_dir=cache_dir,
    )
    fold_b = build_pure_feature_pair(
        prefix_inputs=data.fold_b.prefix_inputs,
        prefix_labels=data.fold_b.prefix_labels,
        prefix_click_labels=data.fold_b.prefix_click_labels,
        prefix_watch_progress=data.fold_b.prefix_watch_progress,
        query_inputs=data.fold_b.query_inputs,
        dataset_digest=data.dataset_digest,
        split_role="inner_fold_B",
        builder_source_digest=builder,
        cache_dir=cache_dir,
    )
    combined_query = concat_canonical_inputs((data.outer_validation_inputs, data.final_inputs))
    outer_and_final = build_pure_feature_pair(
        prefix_inputs=data.outer_train_inputs,
        prefix_labels=data.outer_train_labels,
        prefix_click_labels=data.outer_train_click_labels,
        prefix_watch_progress=data.outer_train_watch_progress,
        query_inputs=combined_query,
        dataset_digest=data.dataset_digest,
        split_role="outer_validation_and_final",
        builder_source_digest=builder,
        cache_dir=cache_dir,
    )
    outer_validation, final = split_feature_matrix(
        outer_and_final.query,
        (len(data.outer_validation_inputs), len(data.final_inputs)),
    )
    return ProductionFeatureBundle(
        data_plane_digest=data.digest,
        fold_a=fold_a,
        fold_b=fold_b,
        outer_and_final=outer_and_final,
        outer_validation=outer_validation,
        final=final,
    )


def encode_numeric_user_groups(user_ids: Sequence[object]) -> Int64Vector:
    """Encode equality-only user groups in stable first-seen order for generated training."""

    if isinstance(user_ids, (str, bytes)):
        raise FullCampaignError("user_ids must be a non-empty identity sequence")
    values = tuple(user_ids)
    if not values:
        raise FullCampaignError("user_ids must be a non-empty identity sequence")
    mapping: dict[int | str, int] = {}
    encoded = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if type(value) is bool:
            raise FullCampaignError("user identity cannot be boolean")
        if isinstance(value, Integral):
            identity: int | str = int(value)
        elif type(value) is str and value and "\x00" not in value:
            identity = value
        else:
            raise FullCampaignError("user identity must be an integer or non-empty string")
        code = mapping.setdefault(identity, len(mapping))
        encoded[index] = code
    encoded.setflags(write=False)
    return cast(Int64Vector, encoded)


@dataclass(frozen=True, slots=True)
class FusionGridEvidence:
    """One Fold-B grid point bound to exact fused predictions and organizer aggregates."""

    weights: tuple[float, float]
    fusion_digest: str
    prediction_digest: str
    scorer_digest: str
    metrics: OrganizerMetrics

    def __post_init__(self) -> None:
        if self.weights not in FUSION_WEIGHT_GRID:
            raise FullCampaignError("fusion evidence weights are outside the frozen grid")
        for name in ("fusion_digest", "prediction_digest", "scorer_digest"):
            _digest(getattr(self, name), name)
        if not isinstance(self.metrics, OrganizerMetrics):
            raise FullCampaignError("fusion evidence requires organizer aggregate metrics")

    def manifest(self) -> dict[str, object]:
        return {
            "weights": list(self.weights),
            "fusion_digest": self.fusion_digest,
            "prediction_digest": self.prediction_digest,
            "scorer_digest": self.scorer_digest,
            "metrics": {
                "GAUC": self.metrics.gauc,
                "nDCG@5": self.metrics.ndcg_at_5,
                "primary": self.metrics.primary,
            },
        }


@dataclass(frozen=True, slots=True)
class FrozenFoldBFusion:
    """Complete predeclared Fold-B search and the deterministically selected fixed point."""

    tree_prediction_digest: str
    fm_prediction_digest: str
    scorer_digest: str
    grid: tuple[FusionGridEvidence, ...]
    selected: FusionGridEvidence
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("tree_prediction_digest", "fm_prediction_digest", "scorer_digest"):
            _digest(getattr(self, name), name)
        if tuple(item.weights for item in self.grid) != FUSION_WEIGHT_GRID:
            raise FullCampaignError("Fold-B fusion evidence must cover the exact frozen grid")
        if self.selected not in self.grid:
            raise FullCampaignError("selected Fold-B fusion evidence is not in the grid")
        if any(item.scorer_digest != self.scorer_digest for item in self.grid):
            raise FullCampaignError("Fold-B fusion grid contains mixed scorer identities")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"fold-b-fusion\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": FULL_CAMPAIGN_SCHEMA_VERSION,
            "selection_split": "inner_fold_B",
            "selection_rule": "maximum primary; ties use frozen grid order",
            "weight_grid": [list(item) for item in FUSION_WEIGHT_GRID],
            "tree_prediction_digest": self.tree_prediction_digest,
            "fm_prediction_digest": self.fm_prediction_digest,
            "scorer_digest": self.scorer_digest,
            "grid": [item.manifest() for item in self.grid],
            "selected_weights": list(self.selected.weights),
            "selected_fusion_digest": self.selected.fusion_digest,
            "public_weight_selection": False,
        }


def _grid_evidence(
    fusion: FusionResult,
    result: ScoreResult,
    *,
    scorer_digest: str,
) -> FusionGridEvidence:
    if not isinstance(result, ScoreResult):
        raise FullCampaignError("protected fusion scorer returned an unsupported receipt")
    if result.scorer_digest != scorer_digest:
        raise FullCampaignError("protected fusion score used a different scorer identity")
    if result.prediction_digest != fusion.prediction_digest:
        raise FullCampaignError("protected fusion score prediction identity mismatch")
    if result.rows != fusion.scores.size:
        raise FullCampaignError("protected fusion score row count mismatch")
    metrics = OrganizerMetrics(result.gauc, result.ndcg_at_5)
    if not math.isclose(result.primary, metrics.primary, rel_tol=0.0, abs_tol=1e-15):
        raise FullCampaignError("protected fusion primary is not mean(GAUC, nDCG@5)")
    return FusionGridEvidence(
        # This legacy v1 grid remains exactly two-member even though FusionResult now supports
        # additive n-member fusion for the v2 replay path.
        weights=(fusion.weights[0], fusion.weights[1]),
        fusion_digest=fusion.fusion_digest,
        prediction_digest=fusion.prediction_digest,
        scorer_digest=scorer_digest,
        metrics=metrics,
    )


def select_fold_b_fusion(
    *,
    user_ids: Sequence[object],
    video_ids: Sequence[object],
    tree_scores: Sequence[object] | npt.NDArray[np.generic],
    fm_scores: Sequence[object] | npt.NDArray[np.generic],
    scorer_digest: str,
    score: ProtectedAggregateScorer,
) -> FrozenFoldBFusion:
    """Score the frozen rank grid once on Fold B and return a fixed downstream policy."""

    scorer = _digest(scorer_digest, "scorer_digest")
    if not callable(score):
        raise FullCampaignError("Fold-B protected score callback must be callable")
    evidence: list[FusionGridEvidence] = []
    tree_digest: str | None = None
    fm_digest: str | None = None
    for weights in FUSION_WEIGHT_GRID:
        fused = fuse_ranked_predictions(
            user_ids,
            video_ids,
            tree_scores,
            fm_scores,
            weights=weights,
            phase=DataPhase.INNER_VALID,
        )
        tree_digest, fm_digest = fused.member_prediction_digests
        evidence.append(_grid_evidence(fused, score(fused.scores), scorer_digest=scorer))
    assert tree_digest is not None and fm_digest is not None
    selected = evidence[0]
    selected_primary = selected.metrics.primary_decimal
    for item in evidence[1:]:
        primary: Decimal = item.metrics.primary_decimal
        if primary > selected_primary:
            selected = item
            selected_primary = primary
    return FrozenFoldBFusion(
        tree_prediction_digest=tree_digest,
        fm_prediction_digest=fm_digest,
        scorer_digest=scorer,
        grid=tuple(evidence),
        selected=selected,
    )


def run_provider_free_campaign(
    run_dir: Path,
    *,
    project_root: Path,
    engine: CampaignEngine | None = None,
    outer_ledger_path: Path | None = None,
    cancel_event: Event | None = None,
) -> FullCampaignOutcome:
    """Run or resume the bounded local research campaign through finalization handoff."""

    from kuairand_agent.campaign.full_campaign_runtime import (
        run_provider_free_campaign as _run,
    )

    return _run(
        run_dir,
        project_root=project_root,
        engine=engine,
        outer_ledger_path=outer_ledger_path,
        cancel_event=cancel_event,
    )


__all__ = [
    "FULL_CAMPAIGN_SCHEMA_VERSION",
    "CampaignDataPlane",
    "FinalizationSelectionPlan",
    "FoldDataPlane",
    "FrozenFoldBFusion",
    "FullCampaignCancelled",
    "FullCampaignCheckpoint",
    "FullCampaignError",
    "FullCampaignOutcome",
    "FullCampaignOutcomeRepository",
    "FullCampaignProgressLedger",
    "FullCampaignStage",
    "FusionGridEvidence",
    "InnerFoldSelectionEvidence",
    "MatchedSeedSelectionEvidence",
    "ProductionFeatureBundle",
    "QualifiedFMMemberPlan",
    "build_finalization_candidate_inputs",
    "build_production_feature_bundle",
    "encode_numeric_user_groups",
    "load_full_campaign_outcome",
    "prepare_campaign_data_plane",
    "run_provider_free_campaign",
    "select_fold_b_fusion",
]

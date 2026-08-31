"""Durable trusted adapter from scientific requests to generated candidate execution.

The generated process receives numeric training capabilities for ``train`` and exactly one
feature capability for each prediction.  Protected labels, scorer closures, organizer metrics,
trusted FM control predictions, and rank-fusion policy remain in this controller module.  One
successful scientific request is trained once, predicted twice in fresh workspaces, checked for
byte-exact replay, scored through an injected protected closure, and durably cached.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.campaign.scientific import (
    ResourceEvidence,
    RunIdentityEvidence,
    ScientificRunEvidence,
    ScientificRunRequest,
    ScientificTier,
)
from kuairand_agent.campaign.selector import GateEvidence, OrganizerMetrics
from kuairand_agent.candidates.fusion import (
    FUSION_WEIGHT_GRID,
    FusionResult,
    fuse_ranked_predictions,
    normalize_within_user_percentiles,
)
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactRef, ArtifactStore
from kuairand_agent.execution.candidate_executor import (
    CandidateAction,
    CandidateExecutionArtifacts,
    CandidateExecutionJournal,
    GeneratedCandidateExecutor,
    GeneratedCandidateIdentity,
    GeneratedPredictionRun,
    GeneratedPredictRequest,
    GeneratedTrainRequest,
    GeneratedTrainRun,
)
from kuairand_agent.execution.policy import SplitRole
from kuairand_agent.scoring.protected import ScoreResult
from kuairand_agent.scoring.submission import prediction_digest

GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION: Final = 1
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_TIER_ROLE: Final = {
    ScientificTier.FOLD_B_SCREEN: ("B", SplitRole.INNER_TRAIN, SplitRole.INNER_VALID),
    ScientificTier.FOLD_A_CONFIRMATION: ("A", SplitRole.INNER_TRAIN, SplitRole.INNER_VALID),
    ScientificTier.OUTER_MATCHED_SEED: (None, SplitRole.TRAIN, SplitRole.OUTER_VALID),
}
_TIER_PHASE: Final = {
    ScientificTier.FOLD_B_SCREEN: DataPhase.INNER_VALID,
    ScientificTier.FOLD_A_CONFIRMATION: DataPhase.INNER_VALID,
    ScientificTier.OUTER_MATCHED_SEED: DataPhase.OUTER_VALID,
}

type ScoreVector = Sequence[float] | npt.NDArray[np.generic]
type ProtectedScoreCallback = Callable[[npt.NDArray[np.float64]], ScoreResult]


class GeneratedScientificRunnerError(ValueError):
    """A scientific request is not exactly bound to prepared trusted capabilities."""


class GeneratedScientificRunnerCancelledError(GeneratedScientificRunnerError):
    """The trusted controller requested cancellation before another generated action."""


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise GeneratedScientificRunnerError(f"{name} must be a lowercase SHA-256 digest")
    return value


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
        raise GeneratedScientificRunnerError(
            "generated scientific evidence must be finite canonical JSON"
        ) from exc


def _manifest_digest(domain: bytes, value: object) -> str:
    result = hashlib.sha256(domain)
    result.update(_canonical_json(value))
    return result.hexdigest()


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != keys
        or any(type(key) is not str for key in value)
    ):
        raise GeneratedScientificRunnerError(f"{name} has an invalid schema")
    return cast(Mapping[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise GeneratedScientificRunnerError(f"{name} must be a JSON array")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str:
        raise GeneratedScientificRunnerError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise GeneratedScientificRunnerError(f"{name} must be boolean")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise GeneratedScientificRunnerError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeneratedScientificRunnerError(f"{name} must be finite numeric JSON")
    result = float(value)
    if not math.isfinite(result):
        raise GeneratedScientificRunnerError(f"{name} must be finite numeric JSON")
    return result


def _optional_digest(value: object, name: str) -> str | None:
    return None if value is None else _digest(value, name)


def _weight_from_json(value: object, name: str) -> tuple[float, float]:
    raw = _list(value, name)
    if len(raw) != 2:
        raise GeneratedScientificRunnerError(f"{name} must have exactly two values")
    weights = (_float(raw[0], f"{name}[0]"), _float(raw[1], f"{name}[1]"))
    return _fusion_weight(weights, name)


def _input(reference: object, name: str) -> ArtifactRef:
    if not isinstance(reference, ArtifactRef) or reference.kind is not ArtifactKind.INPUT:
        raise GeneratedScientificRunnerError(f"{name} must be an input artifact reference")
    return reference


def _fusion_weight(value: object, name: str) -> tuple[float, float]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not float for item in value)
        or value not in FUSION_WEIGHT_GRID
    ):
        raise GeneratedScientificRunnerError(
            f"{name} must be one exact float pair in {FUSION_WEIGHT_GRID}"
        )
    return value


def _vector(values: ScoreVector, *, expected: int, name: str) -> npt.NDArray[np.float64]:
    if type(expected) is not int or expected <= 0:
        raise GeneratedScientificRunnerError(f"{name} expected row count must be positive")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneratedScientificRunnerError(f"{name} must be a finite numeric vector") from exc
    if array.ndim != 1 or array.size != expected or array.dtype.kind not in "iuf":
        raise GeneratedScientificRunnerError(
            f"{name} must be a one-dimensional numeric vector with {expected} rows"
        )
    try:
        result = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneratedScientificRunnerError(f"{name} must be representable as float64") from exc
    if not bool(np.isfinite(result).all()):
        raise GeneratedScientificRunnerError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TrustedFMRankFusion:
    """One frozen, label-free FM-control fusion policy selected on train-derived evidence."""

    phase: DataPhase
    weights: tuple[float, float]
    user_ids: Sequence[object] = field(repr=False)
    video_ids: Sequence[object] = field(repr=False)
    control_scores: ScoreVector = field(repr=False)
    row_count: int = field(init=False)
    control_prediction_digest: str = field(init=False)
    alignment_digest: str = field(init=False)
    normalized_control_prediction_digest: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.phase not in {DataPhase.INNER_VALID, DataPhase.OUTER_VALID}:
            raise GeneratedScientificRunnerError(
                "scientific fusion phase must be inner_valid or outer_valid; final is forbidden"
            )
        _fusion_weight(self.weights, "fusion weights")
        if isinstance(self.user_ids, (str, bytes)) or isinstance(self.video_ids, (str, bytes)):
            raise GeneratedScientificRunnerError(
                "fusion user_ids and video_ids must be one-dimensional identity sequences"
            )
        try:
            users = tuple(self.user_ids)
            videos = tuple(self.video_ids)
        except TypeError as exc:
            raise GeneratedScientificRunnerError(
                "fusion user_ids and video_ids must be one-dimensional sequences"
            ) from exc
        if not users or len(users) != len(videos):
            raise GeneratedScientificRunnerError(
                "fusion user_ids and video_ids must be non-empty with equal lengths"
            )
        control = _vector(
            self.control_scores,
            expected=len(users),
            name="trusted FM control predictions",
        )
        normalized = normalize_within_user_percentiles(
            users,
            videos,
            control,
            phase=self.phase,
        )
        object.__setattr__(self, "user_ids", users)
        object.__setattr__(self, "video_ids", videos)
        object.__setattr__(self, "control_scores", control)
        object.__setattr__(self, "row_count", len(users))
        object.__setattr__(self, "control_prediction_digest", prediction_digest(control))
        object.__setattr__(self, "alignment_digest", normalized.alignment_digest)
        object.__setattr__(
            self,
            "normalized_control_prediction_digest",
            normalized.prediction_digest,
        )
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-trusted-fm-rank-fusion-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "phase": self.phase.value,
            "weights": list(self.weights),
            "row_count": self.row_count,
            "control_prediction_digest": self.control_prediction_digest,
            "normalized_control_prediction_digest": self.normalized_control_prediction_digest,
            "alignment_digest": self.alignment_digest,
            "weight_selection": "frozen_before_this_scientific_tier",
            "labels_exposed": False,
        }

    def fuse(self, generated_scores: ScoreVector) -> FusionResult:
        """Apply the already-frozen policy without accepting labels or choosing a weight."""

        return fuse_ranked_predictions(
            self.user_ids,
            self.video_ids,
            generated_scores,
            self.control_scores,
            weights=self.weights,
            phase=self.phase,
        )


@dataclass(frozen=True, slots=True)
class FoldBFusionSelector:
    """Controller-owned fixed-grid selector allowed only on the train-derived Fold B screen."""

    user_ids: Sequence[object] = field(repr=False)
    video_ids: Sequence[object] = field(repr=False)
    control_scores: ScoreVector = field(repr=False)
    row_count: int = field(init=False)
    control_prediction_digest: str = field(init=False)
    alignment_digest: str = field(init=False)
    normalized_control_prediction_digest: str = field(init=False)
    digest: str = field(init=False)
    phase: DataPhase = field(init=False, default=DataPhase.INNER_VALID)
    weight_grid: tuple[tuple[float, float], ...] = field(init=False, default=FUSION_WEIGHT_GRID)

    def __post_init__(self) -> None:
        if isinstance(self.user_ids, (str, bytes)) or isinstance(self.video_ids, (str, bytes)):
            raise GeneratedScientificRunnerError(
                "Fold B fusion user_ids and video_ids must be identity sequences"
            )
        try:
            users = tuple(self.user_ids)
            videos = tuple(self.video_ids)
        except TypeError as exc:
            raise GeneratedScientificRunnerError(
                "Fold B fusion user_ids and video_ids must be one-dimensional sequences"
            ) from exc
        if not users or len(users) != len(videos):
            raise GeneratedScientificRunnerError(
                "Fold B fusion user_ids and video_ids must be non-empty with equal lengths"
            )
        control = _vector(
            self.control_scores,
            expected=len(users),
            name="trusted Fold B FM control predictions",
        )
        normalized = normalize_within_user_percentiles(
            users,
            videos,
            control,
            phase=DataPhase.INNER_VALID,
        )
        object.__setattr__(self, "user_ids", users)
        object.__setattr__(self, "video_ids", videos)
        object.__setattr__(self, "control_scores", control)
        object.__setattr__(self, "row_count", len(users))
        object.__setattr__(self, "control_prediction_digest", prediction_digest(control))
        object.__setattr__(self, "alignment_digest", normalized.alignment_digest)
        object.__setattr__(
            self,
            "normalized_control_prediction_digest",
            normalized.prediction_digest,
        )
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-fold-b-fusion-selector-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "phase": self.phase.value,
            "weight_grid": [list(weights) for weights in self.weight_grid],
            "row_count": self.row_count,
            "control_prediction_digest": self.control_prediction_digest,
            "normalized_control_prediction_digest": self.normalized_control_prediction_digest,
            "alignment_digest": self.alignment_digest,
            "selection": "maximum_primary_then_grid_order",
            "selection_tier": ScientificTier.FOLD_B_SCREEN.value,
            "labels_exposed": False,
        }

    def fuse(self, generated_scores: ScoreVector, weights: tuple[float, float]) -> FusionResult:
        """Evaluate one predeclared point; callers cannot supply an adaptive/non-grid weight."""

        _fusion_weight(weights, "Fold B fusion weight")
        return fuse_ranked_predictions(
            self.user_ids,
            self.video_ids,
            generated_scores,
            self.control_scores,
            weights=weights,
            phase=self.phase,
        )


@dataclass(frozen=True, slots=True)
class ScientificTierCapabilities:
    """Prebuilt generated-plane inputs for exactly one non-final scientific tier."""

    tier: ScientificTier
    fold_id: str | None
    scientific_data_digest: str
    training_data_digest: str
    prediction_data_digest: str
    training_split_token: str
    prediction_split_token: str
    training_features: ArtifactRef
    training_targets: ArtifactRef = field(repr=False)
    training_user_groups: ArtifactRef
    prediction_features: ArtifactRef
    prediction_row_count: int
    fusion: TrustedFMRankFusion | FoldBFusionSelector | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tier, ScientificTier) or self.tier not in _TIER_ROLE:
            raise GeneratedScientificRunnerError("capability tier must be a scientific tier")
        expected_fold, _, _ = _TIER_ROLE[self.tier]
        if self.fold_id != expected_fold:
            raise GeneratedScientificRunnerError("capability fold_id does not match its tier")
        for name in (
            "scientific_data_digest",
            "training_data_digest",
            "prediction_data_digest",
        ):
            _digest(getattr(self, name), name)
        for name in ("training_split_token", "prediction_split_token"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value
                or len(value.encode("utf-8")) > 256
                or not value.isascii()
                or not value.isprintable()
                or any(character.isspace() for character in value)
            ):
                raise GeneratedScientificRunnerError(
                    f"{name} must be a printable ASCII token without whitespace"
                )
        for name in (
            "training_features",
            "training_targets",
            "training_user_groups",
            "prediction_features",
        ):
            _input(getattr(self, name), name)
        if (
            len(
                {
                    self.training_features.sha256,
                    self.training_targets.sha256,
                    self.training_user_groups.sha256,
                }
            )
            != 3
        ):
            raise GeneratedScientificRunnerError(
                "training feature, target, and user-group capabilities must be distinct"
            )
        if type(self.prediction_row_count) is not int or self.prediction_row_count <= 0:
            raise GeneratedScientificRunnerError("prediction_row_count must be positive")
        if self.fusion is not None:
            if not isinstance(self.fusion, (TrustedFMRankFusion, FoldBFusionSelector)):
                raise GeneratedScientificRunnerError(
                    "fusion must be a trusted fixed policy, Fold B selector, or None"
                )
            if isinstance(self.fusion, FoldBFusionSelector) and (
                self.tier is not ScientificTier.FOLD_B_SCREEN
            ):
                raise GeneratedScientificRunnerError(
                    "fusion weight selection is allowed only on the train-derived Fold B screen"
                )
            if self.fusion.phase is not _TIER_PHASE[self.tier]:
                raise GeneratedScientificRunnerError("fusion phase does not match scientific tier")
            if self.fusion.row_count != self.prediction_row_count:
                raise GeneratedScientificRunnerError(
                    "fusion and prediction capabilities have different row counts"
                )
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-scientific-tier-capabilities-v1\0", self.manifest()),
        )

    @property
    def training_split_role(self) -> SplitRole:
        return _TIER_ROLE[self.tier][1]

    @property
    def prediction_split_role(self) -> SplitRole:
        return _TIER_ROLE[self.tier][2]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "tier": self.tier.value,
            "fold_id": self.fold_id,
            "scientific_data_digest": self.scientific_data_digest,
            "training_data_digest": self.training_data_digest,
            "prediction_data_digest": self.prediction_data_digest,
            "training_split_token": self.training_split_token,
            "prediction_split_token": self.prediction_split_token,
            "training_split_role": self.training_split_role.value,
            "prediction_split_role": self.prediction_split_role.value,
            "training_features": self.training_features.manifest(),
            "training_targets": self.training_targets.manifest(),
            "training_user_groups": self.training_user_groups.manifest(),
            "prediction_features": self.prediction_features.manifest(),
            "prediction_row_count": self.prediction_row_count,
            "fusion": None if self.fusion is None else self.fusion.manifest(),
        }


@dataclass(frozen=True, slots=True)
class ProtectedScoringCapability:
    """Value-free scorer identity and the only trusted aggregate-scoring closure."""

    tier: ScientificTier
    fold_id: str | None
    scientific_data_digest: str
    scorer_digest: str
    alignment_digest: str
    row_count: int
    callback: ProtectedScoreCallback = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tier, ScientificTier) or self.tier not in _TIER_ROLE:
            raise GeneratedScientificRunnerError("scoring tier must be a scientific tier")
        if self.fold_id != _TIER_ROLE[self.tier][0]:
            raise GeneratedScientificRunnerError("scoring fold_id does not match its tier")
        for name in ("scientific_data_digest", "scorer_digest", "alignment_digest"):
            _digest(getattr(self, name), name)
        if type(self.row_count) is not int or self.row_count <= 0:
            raise GeneratedScientificRunnerError("scoring row_count must be positive")
        if not callable(self.callback):
            raise GeneratedScientificRunnerError("protected scoring callback must be callable")

    def __call__(self, scores: npt.NDArray[np.float64], /) -> ScoreResult:
        return self.callback(scores)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "tier": self.tier.value,
            "fold_id": self.fold_id,
            "scientific_data_digest": self.scientific_data_digest,
            "scorer_digest": self.scorer_digest,
            "alignment_digest": self.alignment_digest,
            "row_count": self.row_count,
            "labels_exposed": False,
        }


@dataclass(frozen=True, slots=True)
class FusionGridPointEvidence:
    """Protected aggregate evidence for one predeclared Fold B fusion grid point."""

    weights: tuple[float, float]
    metrics: OrganizerMetrics
    prediction_digest: str
    fusion_digest: str
    scorer_runtime_seconds: float

    def __post_init__(self) -> None:
        _fusion_weight(self.weights, "fusion point weights")
        if not isinstance(self.metrics, OrganizerMetrics):
            raise GeneratedScientificRunnerError("fusion point metrics must be OrganizerMetrics")
        _digest(self.prediction_digest, "fusion point prediction_digest")
        _digest(self.fusion_digest, "fusion point fusion_digest")
        if not math.isfinite(self.scorer_runtime_seconds) or self.scorer_runtime_seconds < 0.0:
            raise GeneratedScientificRunnerError(
                "fusion point scorer runtime must be finite and non-negative"
            )

    def manifest(self) -> dict[str, object]:
        return {
            "weights": list(self.weights),
            "metrics": {
                "GAUC": self.metrics.gauc,
                "nDCG@5": self.metrics.ndcg_at_5,
                "primary": self.metrics.primary,
            },
            "prediction_digest": self.prediction_digest,
            "fusion_digest": self.fusion_digest,
            "scorer_runtime_seconds": self.scorer_runtime_seconds,
        }


@dataclass(frozen=True, slots=True)
class FoldBFusionSelectionEvidence:
    """Atomic frozen result of scoring every fixed grid point on Fold B exactly once."""

    selector_digest: str
    points: tuple[FusionGridPointEvidence, ...]
    selected_weights: tuple[float, float]
    selected_prediction_digest: str
    selected_fusion_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.selector_digest, "fusion selector_digest")
        if tuple(point.weights for point in self.points) != FUSION_WEIGHT_GRID:
            raise GeneratedScientificRunnerError(
                "fusion selection must contain every fixed grid point in canonical order"
            )
        _fusion_weight(self.selected_weights, "selected fusion weights")
        selected = next(point for point in self.points if point.weights == self.selected_weights)
        if (
            selected.prediction_digest != self.selected_prediction_digest
            or selected.fusion_digest != self.selected_fusion_digest
        ):
            raise GeneratedScientificRunnerError(
                "selected fusion identities differ from their grid-point evidence"
            )
        expected_index, expected = max(
            enumerate(self.points),
            key=lambda item: (item[1].metrics.primary_decimal, -item[0]),
        )
        del expected_index
        if expected.weights != self.selected_weights:
            raise GeneratedScientificRunnerError(
                "selected fusion weight is not maximum primary with grid-order tie breaking"
            )
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-fold-b-fusion-selection-v1\0", self.manifest()),
        )

    def _point(self, weights: tuple[float, float]) -> FusionGridPointEvidence | None:
        return next((point for point in self.points if point.weights == weights), None)

    @property
    def standalone_primary(self) -> float | None:
        """The generated model's own primary, scored with the control weighted out.

        This is the only number that measures the candidate rather than the blend.  The primary
        that reaches selection, reflection and the report is the *selected* grid point, which is
        frequently mostly or entirely the official FM control, so a candidate that is far worse
        than the control still reports a score close to it.
        """

        point = self._point((1.0, 0.0))
        return None if point is None else float(point.metrics.primary)

    @property
    def control_primary(self) -> float | None:
        """The official FM control's own primary, i.e. the grid point that discards the model."""

        point = self._point((0.0, 1.0))
        return None if point is None else float(point.metrics.primary)

    @property
    def model_was_discarded(self) -> bool:
        """True when selection kept none of the generated model's ordering."""

        return self.selected_weights == (0.0, 1.0)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "selector_digest": self.selector_digest,
            "points": [point.manifest() for point in self.points],
            "selected_weights": list(self.selected_weights),
            "selected_prediction_digest": self.selected_prediction_digest,
            "selected_fusion_digest": self.selected_fusion_digest,
            "selection": "maximum_primary_then_grid_order",
        }


def _candidate_artifacts_manifest(artifacts: CandidateExecutionArtifacts) -> dict[str, object]:
    return {
        "output_validated": artifacts.output_validated,
        "diagnostic": artifacts.diagnostic,
        "closure_digest": artifacts.closure_digest,
        "entries": [
            {"role": role, "artifact": reference.manifest()}
            for role, reference in artifacts.entries
        ],
    }


@dataclass(frozen=True, slots=True)
class GeneratedScientificRunRecord:
    """Durable additive artifact projection for replay, continuation, and finalization."""

    request_digest: str
    evidence: ScientificRunEvidence
    source_snapshot_digest: str
    checkpoint: ArtifactRef
    raw_prediction: ArtifactRef
    replay_prediction: ArtifactRef
    scored_prediction: ArtifactRef
    raw_prediction_file_digest: str
    raw_prediction_logical_digest: str
    scored_prediction_digest: str
    train_artifacts: CandidateExecutionArtifacts
    prediction_artifacts: CandidateExecutionArtifacts
    replay_artifacts: CandidateExecutionArtifacts
    generated_replay_exact: bool
    fusion_selection: FoldBFusionSelectionEvidence | None = None
    fixed_fusion_result_digest: str | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "request_digest",
            "source_snapshot_digest",
            "raw_prediction_file_digest",
            "raw_prediction_logical_digest",
            "scored_prediction_digest",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.evidence, ScientificRunEvidence):
            raise GeneratedScientificRunnerError("record evidence must be ScientificRunEvidence")
        if self.evidence.request_digest != self.request_digest:
            raise GeneratedScientificRunnerError("record and evidence request identities differ")
        expected_kinds = (
            (self.checkpoint, ArtifactKind.CHECKPOINT, "checkpoint"),
            (self.raw_prediction, ArtifactKind.PREDICTION, "raw_prediction"),
            (self.replay_prediction, ArtifactKind.PREDICTION, "replay_prediction"),
            (self.scored_prediction, ArtifactKind.PREDICTION, "scored_prediction"),
        )
        for reference, kind, name in expected_kinds:
            if not isinstance(reference, ArtifactRef) or reference.kind is not kind:
                raise GeneratedScientificRunnerError(f"record {name} has the wrong artifact kind")
        if self.checkpoint.sha256 != self.evidence.identities.checkpoint_digest:
            raise GeneratedScientificRunnerError("record checkpoint and scientific evidence differ")
        if self.scored_prediction_digest != self.evidence.identities.prediction_digest:
            raise GeneratedScientificRunnerError(
                "record scored prediction and scientific evidence differ"
            )
        if self.raw_prediction.sha256 != self.replay_prediction.sha256:
            raise GeneratedScientificRunnerError("record raw prediction replay artifact differs")
        for name in ("train_artifacts", "prediction_artifacts", "replay_artifacts"):
            artifacts = getattr(self, name)
            if (
                not isinstance(artifacts, CandidateExecutionArtifacts)
                or not artifacts.output_validated
            ):
                raise GeneratedScientificRunnerError(
                    f"record {name} must be validated candidate execution artifacts"
                )
        try:
            recorded_checkpoint = self.train_artifacts.artifact("checkpoint")
            recorded_prediction = self.prediction_artifacts.artifact("prediction")
            recorded_replay = self.replay_artifacts.artifact("prediction")
        except KeyError as exc:
            raise GeneratedScientificRunnerError(
                "record execution artifacts omit checkpoint or prediction closure members"
            ) from exc
        if (
            recorded_checkpoint != self.checkpoint
            or recorded_prediction != self.raw_prediction
            or recorded_replay != self.replay_prediction
        ):
            raise GeneratedScientificRunnerError(
                "record top-level artifacts differ from execution artifact closures"
            )
        if type(self.generated_replay_exact) is not bool or not self.generated_replay_exact:
            raise GeneratedScientificRunnerError("record requires exact generated replay")
        if self.fusion_selection is not None and not isinstance(
            self.fusion_selection, FoldBFusionSelectionEvidence
        ):
            raise GeneratedScientificRunnerError(
                "record fusion_selection must be FoldBFusionSelectionEvidence or None"
            )
        if self.fixed_fusion_result_digest is not None:
            _digest(self.fixed_fusion_result_digest, "fixed_fusion_result_digest")
        if self.fusion_selection is not None and self.fixed_fusion_result_digest is not None:
            raise GeneratedScientificRunnerError(
                "record cannot contain both Fold B selection and a fixed fusion result"
            )
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-generated-scientific-run-record-v1\0", self.manifest()),
        )

    @property
    def frozen_fusion_weights(self) -> tuple[float, float] | None:
        return None if self.fusion_selection is None else self.fusion_selection.selected_weights

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "request_digest": self.request_digest,
            "evidence": self.evidence.manifest(),
            "source_snapshot_digest": self.source_snapshot_digest,
            "checkpoint": self.checkpoint.manifest(),
            "raw_prediction": self.raw_prediction.manifest(),
            "replay_prediction": self.replay_prediction.manifest(),
            "scored_prediction": self.scored_prediction.manifest(),
            "raw_prediction_file_digest": self.raw_prediction_file_digest,
            "raw_prediction_logical_digest": self.raw_prediction_logical_digest,
            "scored_prediction_digest": self.scored_prediction_digest,
            "train_artifacts": _candidate_artifacts_manifest(self.train_artifacts),
            "prediction_artifacts": _candidate_artifacts_manifest(self.prediction_artifacts),
            "replay_artifacts": _candidate_artifacts_manifest(self.replay_artifacts),
            "generated_replay_exact": self.generated_replay_exact,
            "fusion_selection": (
                None if self.fusion_selection is None else self.fusion_selection.manifest()
            ),
            "fixed_fusion_result_digest": self.fixed_fusion_result_digest,
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> GeneratedScientificRunRecord:
        """Strictly reconstruct one record without accepting unknown or coerced fields."""

        return _record_from_manifest(value)


def _artifact_from_json(value: object, name: str) -> ArtifactRef:
    mapping = _exact_mapping(
        value,
        name=name,
        keys={"schema_version", "algorithm", "sha256", "size_bytes", "kind"},
    )
    try:
        return ArtifactRef.from_manifest(mapping)
    except ValueError as exc:
        raise GeneratedScientificRunnerError(f"{name} artifact reference is invalid") from exc


def _candidate_artifacts_from_json(
    value: object,
    name: str,
) -> CandidateExecutionArtifacts:
    mapping = _exact_mapping(
        value,
        name=name,
        keys={"output_validated", "diagnostic", "closure_digest", "entries"},
    )
    diagnostic_value = mapping["diagnostic"]
    if diagnostic_value is not None and type(diagnostic_value) is not str:
        raise GeneratedScientificRunnerError(f"{name}.diagnostic must be text or null")
    entries: list[tuple[str, ArtifactRef]] = []
    for index, raw_entry in enumerate(_list(mapping["entries"], f"{name}.entries")):
        entry = _exact_mapping(
            raw_entry,
            name=f"{name}.entries[{index}]",
            keys={"role", "artifact"},
        )
        entries.append(
            (
                _string(entry["role"], f"{name}.entries[{index}].role"),
                _artifact_from_json(
                    entry["artifact"],
                    f"{name}.entries[{index}].artifact",
                ),
            )
        )
    artifacts = CandidateExecutionArtifacts(
        entries=tuple(entries),
        output_validated=_boolean(mapping["output_validated"], f"{name}.output_validated"),
        diagnostic=diagnostic_value,
    )
    if artifacts.closure_digest != _digest(mapping["closure_digest"], f"{name}.closure_digest"):
        raise GeneratedScientificRunnerError(f"{name} closure digest mismatch")
    return artifacts


def _metrics_from_json(value: object, name: str) -> OrganizerMetrics:
    mapping = _exact_mapping(
        value,
        name=name,
        keys={"GAUC", "nDCG@5", "primary"},
    )
    metrics = OrganizerMetrics(
        _float(mapping["GAUC"], f"{name}.GAUC"),
        _float(mapping["nDCG@5"], f"{name}.nDCG@5"),
    )
    if _float(mapping["primary"], f"{name}.primary") != metrics.primary:
        raise GeneratedScientificRunnerError(f"{name}.primary is not mean(GAUC, nDCG@5)")
    return metrics


def _scientific_evidence_from_json(value: object) -> ScientificRunEvidence:
    mapping = _exact_mapping(
        value,
        name="record.evidence",
        keys={
            "schema_version",
            "request_digest",
            "metrics",
            "gates",
            "identities",
            "resources",
            "replay_verified",
            "failure_fingerprint",
        },
    )
    if mapping["schema_version"] != 1:
        raise GeneratedScientificRunnerError("record.evidence schema_version must be 1")
    metrics_value = mapping["metrics"]
    metrics = (
        None
        if metrics_value is None
        else _metrics_from_json(metrics_value, "record.evidence.metrics")
    )
    gate_fields = set(GateEvidence.__dataclass_fields__)
    gates_mapping = _exact_mapping(
        mapping["gates"],
        name="record.evidence.gates",
        keys=gate_fields,
    )
    gates = GateEvidence(
        **{
            name: _boolean(gates_mapping[name], f"record.evidence.gates.{name}")
            for name in gate_fields
        }
    )
    identity_fields = set(RunIdentityEvidence.__dataclass_fields__)
    identity_mapping = _exact_mapping(
        mapping["identities"],
        name="record.evidence.identities",
        keys=identity_fields,
    )
    identities = RunIdentityEvidence(
        source_digest=_digest(identity_mapping["source_digest"], "source_digest"),
        parent_source_digest=_digest(
            identity_mapping["parent_source_digest"], "parent_source_digest"
        ),
        executable_diff_digest=_digest(
            identity_mapping["executable_diff_digest"], "executable_diff_digest"
        ),
        material_change_digest=_digest(
            identity_mapping["material_change_digest"], "material_change_digest"
        ),
        controller_attestation_digest=_digest(
            identity_mapping["controller_attestation_digest"],
            "controller_attestation_digest",
        ),
        config_digest=_digest(identity_mapping["config_digest"], "config_digest"),
        training_policy_digest=_digest(
            identity_mapping["training_policy_digest"], "training_policy_digest"
        ),
        data_digest=_digest(identity_mapping["data_digest"], "data_digest"),
        environment_digest=_digest(identity_mapping["environment_digest"], "environment_digest"),
        execution_digest=_digest(identity_mapping["execution_digest"], "execution_digest"),
        checkpoint_digest=_digest(identity_mapping["checkpoint_digest"], "checkpoint_digest"),
        prediction_digest=_digest(identity_mapping["prediction_digest"], "prediction_digest"),
        scorer_digest=_digest(identity_mapping["scorer_digest"], "scorer_digest"),
        artifact_closure_digest=_digest(
            identity_mapping["artifact_closure_digest"], "artifact_closure_digest"
        ),
    )
    resource_mapping = _exact_mapping(
        mapping["resources"],
        name="record.evidence.resources",
        keys={"wall_seconds", "peak_rss_bytes", "disk_bytes"},
    )
    resources = ResourceEvidence(
        wall_seconds=_float(resource_mapping["wall_seconds"], "resources.wall_seconds"),
        peak_rss_bytes=_integer(resource_mapping["peak_rss_bytes"], "resources.peak_rss_bytes"),
        disk_bytes=_integer(resource_mapping["disk_bytes"], "resources.disk_bytes"),
    )
    evidence = ScientificRunEvidence(
        request_digest=_digest(mapping["request_digest"], "evidence.request_digest"),
        metrics=metrics,
        gates=gates,
        identities=identities,
        resources=resources,
        replay_verified=_boolean(mapping["replay_verified"], "evidence.replay_verified"),
        failure_fingerprint=_optional_digest(
            mapping["failure_fingerprint"], "evidence.failure_fingerprint"
        ),
    )
    if _canonical_json(evidence.manifest()) != _canonical_json(mapping):
        raise GeneratedScientificRunnerError("record.evidence is not canonical")
    return evidence


def _fusion_point_from_json(value: object, index: int) -> FusionGridPointEvidence:
    name = f"record.fusion_selection.points[{index}]"
    mapping = _exact_mapping(
        value,
        name=name,
        keys={
            "weights",
            "metrics",
            "prediction_digest",
            "fusion_digest",
            "scorer_runtime_seconds",
        },
    )
    return FusionGridPointEvidence(
        weights=_weight_from_json(mapping["weights"], f"{name}.weights"),
        metrics=_metrics_from_json(mapping["metrics"], f"{name}.metrics"),
        prediction_digest=_digest(mapping["prediction_digest"], f"{name}.prediction_digest"),
        fusion_digest=_digest(mapping["fusion_digest"], f"{name}.fusion_digest"),
        scorer_runtime_seconds=_float(
            mapping["scorer_runtime_seconds"], f"{name}.scorer_runtime_seconds"
        ),
    )


def _fusion_selection_from_json(value: object) -> FoldBFusionSelectionEvidence:
    mapping = _exact_mapping(
        value,
        name="record.fusion_selection",
        keys={
            "schema_version",
            "selector_digest",
            "points",
            "selected_weights",
            "selected_prediction_digest",
            "selected_fusion_digest",
            "selection",
        },
    )
    if (
        mapping["schema_version"] != GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION
        or mapping["selection"] != "maximum_primary_then_grid_order"
    ):
        raise GeneratedScientificRunnerError("record.fusion_selection policy identity mismatch")
    points = tuple(
        _fusion_point_from_json(item, index)
        for index, item in enumerate(_list(mapping["points"], "record.fusion_selection.points"))
    )
    result = FoldBFusionSelectionEvidence(
        selector_digest=_digest(
            mapping["selector_digest"], "record.fusion_selection.selector_digest"
        ),
        points=points,
        selected_weights=_weight_from_json(
            mapping["selected_weights"], "record.fusion_selection.selected_weights"
        ),
        selected_prediction_digest=_digest(
            mapping["selected_prediction_digest"],
            "record.fusion_selection.selected_prediction_digest",
        ),
        selected_fusion_digest=_digest(
            mapping["selected_fusion_digest"],
            "record.fusion_selection.selected_fusion_digest",
        ),
    )
    if _canonical_json(result.manifest()) != _canonical_json(mapping):
        raise GeneratedScientificRunnerError("record.fusion_selection is not canonical")
    return result


def _record_from_manifest(value: Mapping[str, object]) -> GeneratedScientificRunRecord:
    mapping = _exact_mapping(
        value,
        name="generated scientific record",
        keys={
            "schema_version",
            "request_digest",
            "evidence",
            "source_snapshot_digest",
            "checkpoint",
            "raw_prediction",
            "replay_prediction",
            "scored_prediction",
            "raw_prediction_file_digest",
            "raw_prediction_logical_digest",
            "scored_prediction_digest",
            "train_artifacts",
            "prediction_artifacts",
            "replay_artifacts",
            "generated_replay_exact",
            "fusion_selection",
            "fixed_fusion_result_digest",
        },
    )
    if mapping["schema_version"] != GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION:
        raise GeneratedScientificRunnerError("generated scientific record schema_version mismatch")
    selection_value = mapping["fusion_selection"]
    result = GeneratedScientificRunRecord(
        request_digest=_digest(mapping["request_digest"], "record.request_digest"),
        evidence=_scientific_evidence_from_json(mapping["evidence"]),
        source_snapshot_digest=_digest(
            mapping["source_snapshot_digest"], "record.source_snapshot_digest"
        ),
        checkpoint=_artifact_from_json(mapping["checkpoint"], "record.checkpoint"),
        raw_prediction=_artifact_from_json(mapping["raw_prediction"], "record.raw_prediction"),
        replay_prediction=_artifact_from_json(
            mapping["replay_prediction"], "record.replay_prediction"
        ),
        scored_prediction=_artifact_from_json(
            mapping["scored_prediction"], "record.scored_prediction"
        ),
        raw_prediction_file_digest=_digest(
            mapping["raw_prediction_file_digest"], "record.raw_prediction_file_digest"
        ),
        raw_prediction_logical_digest=_digest(
            mapping["raw_prediction_logical_digest"], "record.raw_prediction_logical_digest"
        ),
        scored_prediction_digest=_digest(
            mapping["scored_prediction_digest"], "record.scored_prediction_digest"
        ),
        train_artifacts=_candidate_artifacts_from_json(
            mapping["train_artifacts"], "record.train_artifacts"
        ),
        prediction_artifacts=_candidate_artifacts_from_json(
            mapping["prediction_artifacts"], "record.prediction_artifacts"
        ),
        replay_artifacts=_candidate_artifacts_from_json(
            mapping["replay_artifacts"], "record.replay_artifacts"
        ),
        generated_replay_exact=_boolean(
            mapping["generated_replay_exact"], "record.generated_replay_exact"
        ),
        fusion_selection=(
            None if selection_value is None else _fusion_selection_from_json(selection_value)
        ),
        fixed_fusion_result_digest=_optional_digest(
            mapping["fixed_fusion_result_digest"], "record.fixed_fusion_result_digest"
        ),
    )
    if _canonical_json(result.manifest()) != _canonical_json(mapping):
        raise GeneratedScientificRunnerError("generated scientific record is not canonical")
    return result


class ScientificRunEvidenceRepository(Protocol):
    """Durable exact-request record cache; implementations must reject contradictions."""

    def load(self, request_digest: str) -> GeneratedScientificRunRecord | None: ...

    def commit(self, record: GeneratedScientificRunRecord) -> None: ...


class FileScientificRunEvidenceRepository:
    """Append-only canonical-JSON repository for restart-safe exact-request run records."""

    _MAX_RECORD_BYTES: Final = 16 * 1024 * 1024

    def __init__(self, root: Path | str) -> None:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise GeneratedScientificRunnerError("scientific record root must be a real directory")
        self.root = path.resolve(strict=True)

    def _path(self, request_digest: str) -> Path:
        return self.root / f"{_digest(request_digest, 'request_digest')}.json"

    @staticmethod
    def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise GeneratedScientificRunnerError(
                    f"scientific record JSON contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> object:
        raise GeneratedScientificRunnerError(
            f"scientific record JSON contains non-finite constant {value}"
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def load(self, request_digest: str) -> GeneratedScientificRunRecord | None:
        key = _digest(request_digest, "request_digest")
        path = self._path(key)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise GeneratedScientificRunnerError(
                "scientific record path must be a non-symlink regular file"
            )
        if metadata.st_size <= 0 or metadata.st_size > self._MAX_RECORD_BYTES:
            raise GeneratedScientificRunnerError("scientific record file size is invalid")
        try:
            payload = path.read_bytes()
            raw = json.loads(
                payload,
                object_pairs_hook=self._pairs,
                parse_constant=self._reject_constant,
            )
        except GeneratedScientificRunnerError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise GeneratedScientificRunnerError("scientific record JSON is malformed") from exc
        document = _exact_mapping(
            raw,
            name="scientific record document",
            keys={"schema_version", "record_digest", "record"},
        )
        if document["schema_version"] != GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION:
            raise GeneratedScientificRunnerError("scientific record document schema mismatch")
        if payload != _canonical_json(document):
            raise GeneratedScientificRunnerError("scientific record JSON is not canonical")
        record_value = _exact_mapping(
            document["record"],
            name="scientific record",
            keys={
                "schema_version",
                "request_digest",
                "evidence",
                "source_snapshot_digest",
                "checkpoint",
                "raw_prediction",
                "replay_prediction",
                "scored_prediction",
                "raw_prediction_file_digest",
                "raw_prediction_logical_digest",
                "scored_prediction_digest",
                "train_artifacts",
                "prediction_artifacts",
                "replay_artifacts",
                "generated_replay_exact",
                "fusion_selection",
                "fixed_fusion_result_digest",
            },
        )
        record = GeneratedScientificRunRecord.from_manifest(record_value)
        if record.request_digest != key:
            raise GeneratedScientificRunnerError("scientific record filename identity mismatch")
        if record.digest != _digest(document["record_digest"], "record_digest"):
            raise GeneratedScientificRunnerError("scientific record digest mismatch")
        return record

    def commit(self, record: GeneratedScientificRunRecord) -> None:
        if not isinstance(record, GeneratedScientificRunRecord):
            raise GeneratedScientificRunnerError(
                "scientific repository can commit only GeneratedScientificRunRecord"
            )
        target = self._path(record.request_digest)
        existing = self.load(record.request_digest)
        if existing is not None:
            if existing.digest != record.digest:
                raise GeneratedScientificRunnerError(
                    "contradictory scientific record already exists for request"
                )
            return
        payload = _canonical_json(
            {
                "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
                "record_digest": record.digest,
                "record": record.manifest(),
            }
        )
        if len(payload) > self._MAX_RECORD_BYTES:
            raise GeneratedScientificRunnerError("scientific record exceeds repository size limit")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{record.request_digest}.",
            suffix=".tmp",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                concurrent = self.load(record.request_digest)
                if concurrent is None or concurrent.digest != record.digest:
                    raise GeneratedScientificRunnerError(
                        "contradictory scientific record won a concurrent commit"
                    ) from None
            else:
                self._fsync_directory(self.root)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()


class CandidateJournalFactory(Protocol):
    """Create the store-owned journal for one exact generated action."""

    def __call__(
        self,
        *,
        request: ScientificRunRequest,
        action: CandidateAction,
        execution_id: str,
    ) -> CandidateExecutionJournal: ...


def _execution_id(request: ScientificRunRequest, label: str) -> str:
    prefix = f"scientific-{label}-"
    return prefix + request.digest[: 64 - len(prefix)]


def _run_manifest(run: GeneratedTrainRun | GeneratedPredictionRun) -> dict[str, object]:
    base: dict[str, object] = {
        "execution": run.execution.manifest(),
        "artifact_closure_digest": run.artifacts.closure_digest,
        "artifacts": [
            {"role": role, "artifact": reference.manifest()}
            for role, reference in run.artifacts.entries
        ],
    }
    if isinstance(run, GeneratedTrainRun):
        base.update(
            {
                "kind": "train",
                "checkpoint": run.checkpoint.manifest(),
                "checkpoint_digest": run.checkpoint_digest,
                "seed": run.seed,
            }
        )
    else:
        base.update(
            {
                "kind": "predict",
                "prediction": run.prediction.manifest(),
                "prediction_file_digest": run.prediction_file_digest,
                "logical_prediction_digest": run.logical_prediction_digest,
            }
        )
    return base


def _put_prediction_artifact(
    artifact_store: ArtifactStore,
    scores: npt.NDArray[np.float64],
) -> ArtifactRef:
    payload = io.BytesIO()
    np.save(payload, np.ascontiguousarray(scores, dtype="<f8"), allow_pickle=False)
    return artifact_store.put_bytes(payload.getvalue(), kind=ArtifactKind.PREDICTION)


def _validate_cached(
    record: object,
    request: ScientificRunRequest,
    artifact_store: ArtifactStore,
    identity: GeneratedCandidateIdentity,
) -> GeneratedScientificRunRecord:
    if not isinstance(record, GeneratedScientificRunRecord):
        raise GeneratedScientificRunnerError(
            "durable evidence repository returned a non-scientific run record"
        )
    evidence = record.evidence
    if evidence.request_digest != request.digest:
        raise GeneratedScientificRunnerError("durable evidence request identity mismatch")
    identities = evidence.identities
    for name in (
        "source_digest",
        "parent_source_digest",
        "executable_diff_digest",
        "material_change_digest",
        "controller_attestation_digest",
        "config_digest",
        "training_policy_digest",
        "data_digest",
        "environment_digest",
        "scorer_digest",
    ):
        if getattr(identities, name) != getattr(request, name):
            raise GeneratedScientificRunnerError(f"cached evidence {name} mismatch")
    _verify_record_source(record, identity, artifact_store)
    _verify_record_artifacts(record, artifact_store)
    return record


def _verify_record_source(
    record: GeneratedScientificRunRecord,
    identity: GeneratedCandidateIdentity,
    artifact_store: ArtifactStore,
) -> None:
    if (
        record.source_snapshot_digest != identity.source_snapshot.sha256
        or record.evidence.identities.source_digest != identity.source_digest
        or record.evidence.identities.config_digest != identity.config_digest
    ):
        raise GeneratedScientificRunnerError(
            "durable record generated source/config snapshot identity mismatch"
        )
    artifact_store.verify_directory(identity.source_snapshot)


def _verify_record_artifacts(
    record: GeneratedScientificRunRecord,
    artifact_store: ArtifactStore,
) -> None:
    scored_path = None
    for reference in (
        record.checkpoint,
        record.raw_prediction,
        record.replay_prediction,
        record.scored_prediction,
    ):
        path = artifact_store.verify(reference)
        if reference is record.scored_prediction:
            scored_path = path
    for artifacts in (
        record.train_artifacts,
        record.prediction_artifacts,
        record.replay_artifacts,
    ):
        for _, reference in artifacts.entries:
            artifact_store.verify(reference)
    if scored_path is None:  # defensive: the fixed tuple above always contains this reference.
        raise GeneratedScientificRunnerError("record scored prediction artifact is absent")
    try:
        stored = np.load(scored_path, allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise GeneratedScientificRunnerError(
            "record scored prediction artifact is not a safe NumPy array"
        ) from exc
    if not isinstance(stored, np.ndarray):
        if hasattr(stored, "close"):
            stored.close()
        raise GeneratedScientificRunnerError(
            "record scored prediction artifact must contain one NumPy array"
        )
    scores = _vector(stored, expected=int(stored.size), name="record scored prediction")
    if prediction_digest(scores) != record.scored_prediction_digest:
        raise GeneratedScientificRunnerError(
            "record scored prediction artifact and logical identity differ"
        )


def _score_result(
    scoring: ProtectedScoringCapability,
    request: ScientificRunRequest,
    scores: npt.NDArray[np.float64],
) -> tuple[ScoreResult, OrganizerMetrics]:
    result = scoring(scores)
    if not isinstance(result, ScoreResult):
        raise GeneratedScientificRunnerError("protected scorer must return ScoreResult")
    expected_prediction = prediction_digest(scores)
    if result.scorer_digest != request.scorer_digest:
        raise GeneratedScientificRunnerError("protected score used a different scorer identity")
    if result.prediction_digest != expected_prediction:
        raise GeneratedScientificRunnerError("protected score prediction identity mismatch")
    if result.rows != scoring.row_count:
        raise GeneratedScientificRunnerError("protected score row count mismatch")
    if type(result.users) is not int or result.users <= 0:
        raise GeneratedScientificRunnerError("protected score must include a positive user count")
    if not math.isfinite(result.runtime_seconds) or result.runtime_seconds < 0.0:
        raise GeneratedScientificRunnerError(
            "protected score runtime must be finite and non-negative"
        )
    metrics = OrganizerMetrics(result.gauc, result.ndcg_at_5)
    raw_primary = (result.gauc + result.ndcg_at_5) / 2.0
    encoded_primary = float(
        (np.float32(result.gauc) + np.float32(result.ndcg_at_5)) / np.float32(2.0)
    )
    if result.primary not in {raw_primary, encoded_primary}:
        raise GeneratedScientificRunnerError("protected score primary is not mean(GAUC, nDCG@5)")
    return result, metrics


class DurableGeneratedScientificRunner:
    """Callable scientific runner over generated train/predict/replay execution."""

    def __init__(
        self,
        *,
        executor: GeneratedCandidateExecutor,
        artifact_store: ArtifactStore,
        identity: GeneratedCandidateIdentity,
        capabilities: Mapping[ScientificTier, ScientificTierCapabilities],
        scoring_callbacks: Mapping[ScientificTier, ProtectedScoringCapability],
        journal_factory: CandidateJournalFactory,
        evidence_repository: ScientificRunEvidenceRepository,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if not isinstance(executor, GeneratedCandidateExecutor):
            raise GeneratedScientificRunnerError("executor must be GeneratedCandidateExecutor")
        if not isinstance(artifact_store, ArtifactStore):
            raise GeneratedScientificRunnerError("artifact_store must be ArtifactStore")
        if executor.artifact_store is not artifact_store:
            raise GeneratedScientificRunnerError(
                "executor and scientific runner artifact stores differ"
            )
        if not isinstance(identity, GeneratedCandidateIdentity):
            raise GeneratedScientificRunnerError("identity must be GeneratedCandidateIdentity")
        prepared = dict(capabilities)
        scoring = dict(scoring_callbacks)
        for tier, prepared_value in prepared.items():
            if tier is not prepared_value.tier:
                raise GeneratedScientificRunnerError("capability mapping key and tier differ")
        for tier, scoring_value in scoring.items():
            if tier is not scoring_value.tier:
                raise GeneratedScientificRunnerError("scoring mapping key and tier differ")
        if not callable(journal_factory):
            raise GeneratedScientificRunnerError("journal_factory must be callable")
        if not callable(getattr(evidence_repository, "load", None)) or not callable(
            getattr(evidence_repository, "commit", None)
        ):
            raise GeneratedScientificRunnerError(
                "evidence_repository must implement load and commit"
            )
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise GeneratedScientificRunnerError("cancel_event must be threading.Event or None")
        self._executor = executor
        self._artifact_store = artifact_store
        self._identity = identity
        self._capabilities = prepared
        self._scoring = scoring
        self._journal_factory = journal_factory
        self._repository = evidence_repository
        self._cancel_event = cancel_event

    def _raise_if_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise GeneratedScientificRunnerCancelledError(
                "trusted controller requested generated scientific cancellation"
            )

    def _prepared(
        self, request: ScientificRunRequest
    ) -> tuple[ScientificTierCapabilities, ProtectedScoringCapability]:
        if not isinstance(request, ScientificRunRequest):
            raise GeneratedScientificRunnerError("request must be ScientificRunRequest")
        if (
            request.source_digest != self._identity.source_digest
            or request.config_digest != self._identity.config_digest
        ):
            raise GeneratedScientificRunnerError(
                "scientific request source/config differs from generated identity"
            )
        capabilities = self._capabilities.get(request.tier)
        scoring = self._scoring.get(request.tier)
        if capabilities is None or scoring is None:
            raise GeneratedScientificRunnerError(
                f"scientific tier {request.tier.value} has no prepared capabilities and scorer"
            )
        if request.fold_id != capabilities.fold_id or request.fold_id != scoring.fold_id:
            raise GeneratedScientificRunnerError(
                "scientific request fold differs from prepared tier"
            )
        if (
            request.data_digest != capabilities.scientific_data_digest
            or request.data_digest != scoring.scientific_data_digest
        ):
            raise GeneratedScientificRunnerError(
                "scientific request data identity differs from prepared capabilities"
            )
        if request.scorer_digest != scoring.scorer_digest:
            raise GeneratedScientificRunnerError(
                "scientific request scorer differs from protected scoring capability"
            )
        if capabilities.prediction_row_count != scoring.row_count:
            raise GeneratedScientificRunnerError(
                "prepared prediction and protected scoring row counts differ"
            )
        # Fusion and protected scoring use intentionally domain-separated alignment digests.
        # Their common positional contract is the immutable tier row count; both exact digest
        # identities are retained in the aggregate artifact closure below.
        self._artifact_store.verify_directory(self._identity.source_snapshot)
        for reference in (
            capabilities.training_features,
            capabilities.training_targets,
            capabilities.training_user_groups,
            capabilities.prediction_features,
        ):
            self._artifact_store.verify(reference)
        return capabilities, scoring

    def load_record(self, request_digest: str) -> GeneratedScientificRunRecord | None:
        """Load and integrity-check the additive durable record without rerunning a candidate."""

        key = _digest(request_digest, "request_digest")
        record = self._repository.load(key)
        if record is None:
            return None
        if not isinstance(record, GeneratedScientificRunRecord) or record.request_digest != key:
            raise GeneratedScientificRunnerError("durable repository record identity mismatch")
        _verify_record_source(record, self._identity, self._artifact_store)
        _verify_record_artifacts(record, self._artifact_store)
        return record

    def __call__(self, request: ScientificRunRequest) -> ScientificRunEvidence:
        self._raise_if_cancelled()
        capabilities, scoring = self._prepared(request)
        cached = self._repository.load(request.digest)
        if cached is not None:
            return _validate_cached(
                cached,
                request,
                self._artifact_store,
                self._identity,
            ).evidence

        train_id = _execution_id(request, "train")
        train = self._executor.train(
            GeneratedTrainRequest(
                execution_id=train_id,
                identity=self._identity,
                split_role=capabilities.training_split_role,
                data_digest=capabilities.training_data_digest,
                split_token=capabilities.training_split_token,
                seed=request.seed,
                features=capabilities.training_features,
                targets=capabilities.training_targets,
                user_groups=capabilities.training_user_groups,
            ),
            journal=self._journal_factory(
                request=request,
                action=CandidateAction.TRAIN,
                execution_id=train_id,
            ),
            cancel_event=self._cancel_event,
        )
        self._raise_if_cancelled()
        if train.checkpoint_digest != train.checkpoint.sha256:
            raise GeneratedScientificRunnerError(
                "validated generated checkpoint differs from committed checkpoint artifact"
            )

        first_id = _execution_id(request, "predict")
        first = self._executor.predict(
            GeneratedPredictRequest(
                execution_id=first_id,
                identity=self._identity,
                split_role=capabilities.prediction_split_role,
                data_digest=capabilities.prediction_data_digest,
                split_token=capabilities.prediction_split_token,
                expected_count=capabilities.prediction_row_count,
                features=capabilities.prediction_features,
                checkpoint=train.checkpoint,
            ),
            journal=self._journal_factory(
                request=request,
                action=CandidateAction.PREDICT,
                execution_id=first_id,
            ),
            cancel_event=self._cancel_event,
        )
        self._raise_if_cancelled()
        replay_id = _execution_id(request, "replay")
        replay = self._executor.predict(
            GeneratedPredictRequest(
                execution_id=replay_id,
                identity=self._identity,
                split_role=capabilities.prediction_split_role,
                data_digest=capabilities.prediction_data_digest,
                split_token=capabilities.prediction_split_token,
                expected_count=capabilities.prediction_row_count,
                features=capabilities.prediction_features,
                checkpoint=train.checkpoint,
            ),
            journal=self._journal_factory(
                request=request,
                action=CandidateAction.PREDICT,
                execution_id=replay_id,
            ),
            cancel_event=self._cancel_event,
        )
        self._raise_if_cancelled()
        if (
            first.scores.tobytes(order="C") != replay.scores.tobytes(order="C")
            or first.prediction.sha256 != replay.prediction.sha256
            or first.prediction_file_digest != replay.prediction_file_digest
            or first.logical_prediction_digest != replay.logical_prediction_digest
        ):
            raise GeneratedScientificRunnerError("generated prediction replay is not byte exact")
        self._raise_if_cancelled()

        first_fusion: FusionResult | None = None
        replay_fusion: FusionResult | None = None
        fusion_selection: FoldBFusionSelectionEvidence | None = None
        if capabilities.fusion is None:
            scored = _vector(
                first.scores,
                expected=capabilities.prediction_row_count,
                name="generated predictions",
            )
            replay_scored = _vector(
                replay.scores,
                expected=capabilities.prediction_row_count,
                name="replayed generated predictions",
            )
            if scored.tobytes(order="C") != replay_scored.tobytes(order="C"):
                raise GeneratedScientificRunnerError("scored prediction replay is not byte exact")
            score, metrics = _score_result(scoring, request, scored)
            scorer_runtime_seconds = score.runtime_seconds
        elif isinstance(capabilities.fusion, TrustedFMRankFusion):
            first_fusion = capabilities.fusion.fuse(first.scores)
            replay_fusion = capabilities.fusion.fuse(replay.scores)
            scored = first_fusion.scores
            replay_scored = replay_fusion.scores
            if first_fusion.fusion_digest != replay_fusion.fusion_digest:
                raise GeneratedScientificRunnerError("rank-fusion replay identity differs")
            if scored.tobytes(order="C") != replay_scored.tobytes(order="C"):
                raise GeneratedScientificRunnerError("scored prediction replay is not byte exact")
            score, metrics = _score_result(scoring, request, scored)
            scorer_runtime_seconds = score.runtime_seconds
        else:
            selector = capabilities.fusion
            evaluated: list[
                tuple[FusionGridPointEvidence, ScoreResult, FusionResult, FusionResult]
            ] = []
            for weights in selector.weight_grid:
                self._raise_if_cancelled()
                candidate_fusion = selector.fuse(first.scores, weights)
                candidate_replay = selector.fuse(replay.scores, weights)
                if (
                    candidate_fusion.fusion_digest != candidate_replay.fusion_digest
                    or candidate_fusion.scores.tobytes(order="C")
                    != candidate_replay.scores.tobytes(order="C")
                ):
                    raise GeneratedScientificRunnerError(
                        "Fold B fusion grid replay is not byte exact"
                    )
                candidate_score, candidate_metrics = _score_result(
                    scoring,
                    request,
                    candidate_fusion.scores,
                )
                point = FusionGridPointEvidence(
                    weights=weights,
                    metrics=candidate_metrics,
                    prediction_digest=candidate_score.prediction_digest,
                    fusion_digest=candidate_fusion.fusion_digest,
                    scorer_runtime_seconds=candidate_score.runtime_seconds,
                )
                evaluated.append((point, candidate_score, candidate_fusion, candidate_replay))
            selected_index, selected = max(
                enumerate(evaluated),
                key=lambda item: (item[1][0].metrics.primary_decimal, -item[0]),
            )
            del selected_index
            selected_point, score, first_fusion, replay_fusion = selected
            metrics = selected_point.metrics
            scored = first_fusion.scores
            replay_scored = replay_fusion.scores
            fusion_selection = FoldBFusionSelectionEvidence(
                selector_digest=selector.digest,
                points=tuple(item[0] for item in evaluated),
                selected_weights=selected_point.weights,
                selected_prediction_digest=selected_point.prediction_digest,
                selected_fusion_digest=selected_point.fusion_digest,
            )
            scorer_runtime_seconds = math.fsum(item[0].scorer_runtime_seconds for item in evaluated)
        self._raise_if_cancelled()
        scored_prediction = _put_prediction_artifact(self._artifact_store, scored)
        closure_manifest = {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "request_digest": request.digest,
            "source_snapshot": self._identity.source_snapshot.manifest(),
            "source_digest": self._identity.source_digest,
            "config_digest": self._identity.config_digest,
            "capabilities": capabilities.manifest(),
            "scoring": scoring.manifest(),
            "train_artifact_closure_digest": train.artifacts.closure_digest,
            "predict_artifact_closure_digest": first.artifacts.closure_digest,
            "replay_artifact_closure_digest": replay.artifacts.closure_digest,
            "checkpoint": train.checkpoint.manifest(),
            "prediction": first.prediction.manifest(),
            "replay_prediction": replay.prediction.manifest(),
            "scored_prediction": scored_prediction.manifest(),
            "scored_prediction_digest": score.prediction_digest,
            "fusion_selection": (None if fusion_selection is None else fusion_selection.manifest()),
        }
        artifact_closure_digest = _manifest_digest(
            b"kuairand-generated-scientific-artifact-closure-v1\0",
            closure_manifest,
        )
        execution_manifest = {
            "schema_version": GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION,
            "request_digest": request.digest,
            "capabilities_digest": capabilities.digest,
            "artifact_closure_digest": artifact_closure_digest,
            "train": _run_manifest(train),
            "predict": _run_manifest(first),
            "replay": _run_manifest(replay),
            "generated_replay_exact": True,
            "scored_replay_exact": True,
            "fusion_policy_digest": (
                None if capabilities.fusion is None else capabilities.fusion.digest
            ),
            "fusion_result_digest": None if first_fusion is None else first_fusion.fusion_digest,
            "fusion_selection_digest": (
                None if fusion_selection is None else fusion_selection.digest
            ),
            "scorer_digest": score.scorer_digest,
            "prediction_digest": score.prediction_digest,
        }
        execution_digest = _manifest_digest(
            b"kuairand-generated-scientific-execution-v1\0",
            execution_manifest,
        )
        executions = (train.execution, first.execution, replay.execution)
        evidence = ScientificRunEvidence(
            request_digest=request.digest,
            metrics=metrics,
            gates=GateEvidence(),
            identities=RunIdentityEvidence(
                source_digest=request.source_digest,
                parent_source_digest=request.parent_source_digest,
                executable_diff_digest=request.executable_diff_digest,
                material_change_digest=request.material_change_digest,
                controller_attestation_digest=request.controller_attestation_digest,
                config_digest=request.config_digest,
                training_policy_digest=request.training_policy_digest,
                data_digest=request.data_digest,
                environment_digest=request.environment_digest,
                execution_digest=execution_digest,
                checkpoint_digest=train.checkpoint_digest,
                prediction_digest=score.prediction_digest,
                scorer_digest=score.scorer_digest,
                artifact_closure_digest=artifact_closure_digest,
            ),
            resources=ResourceEvidence(
                wall_seconds=math.fsum(item.wall_seconds for item in executions)
                + scorer_runtime_seconds,
                peak_rss_bytes=max(item.peak_tree_rss_bytes for item in executions),
                disk_bytes=max(item.peak_workspace_bytes for item in executions),
            ),
            replay_verified=True,
        )
        record = GeneratedScientificRunRecord(
            request_digest=request.digest,
            evidence=evidence,
            source_snapshot_digest=self._identity.source_snapshot.sha256,
            checkpoint=train.checkpoint,
            raw_prediction=first.prediction,
            replay_prediction=replay.prediction,
            scored_prediction=scored_prediction,
            raw_prediction_file_digest=first.prediction_file_digest,
            raw_prediction_logical_digest=first.logical_prediction_digest,
            scored_prediction_digest=score.prediction_digest,
            train_artifacts=train.artifacts,
            prediction_artifacts=first.artifacts,
            replay_artifacts=replay.artifacts,
            generated_replay_exact=True,
            fusion_selection=fusion_selection,
            fixed_fusion_result_digest=(
                first_fusion.fusion_digest
                if isinstance(capabilities.fusion, TrustedFMRankFusion) and first_fusion is not None
                else None
            ),
        )
        self._raise_if_cancelled()
        self._repository.commit(record)
        return _validate_cached(
            record,
            request,
            self._artifact_store,
            self._identity,
        ).evidence


__all__ = [
    "GENERATED_SCIENTIFIC_RUNNER_SCHEMA_VERSION",
    "CandidateJournalFactory",
    "DurableGeneratedScientificRunner",
    "FileScientificRunEvidenceRepository",
    "FoldBFusionSelectionEvidence",
    "FoldBFusionSelector",
    "FusionGridPointEvidence",
    "GeneratedScientificRunRecord",
    "GeneratedScientificRunnerCancelledError",
    "GeneratedScientificRunnerError",
    "ProtectedScoringCapability",
    "ScientificRunEvidenceRepository",
    "ScientificTierCapabilities",
    "TrustedFMRankFusion",
]

"""Declarative, label-free exact rank composition.

``RankGraph`` is data, not executable candidate code.  Its transform vocabulary, normalization
scope, tie policy, ordered row identity, member prediction identities, content digests, and weights
are all closed and validated before evaluation.  The evaluator only delegates to the existing pure
rank-fusion arithmetic; it has no scorer, label, runtime, or protected-query dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt

from kuairand_agent.candidates.fusion import TIE_POLICY, FusionError, fuse_ranked_members
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.domain.decisions import EvidenceStage
from kuairand_agent.domain.identity import PredictionId, canonical_json_sha256
from kuairand_agent.scoring.submission import prediction_digest

type Identity = int | str
type ScoreInput = Sequence[object] | npt.NDArray[np.generic]
type Float64Vector = npt.NDArray[np.float64]

RANK_GRAPH_SCHEMA_VERSION: Final = 1
RANK_GRAPH_TIE_POLICY: Final = TIE_POLICY
RANK_GRAPH_NORMALIZATION_SCOPE: Final = "within_user"


class RankGraphError(ValueError):
    """Raised when a graph or member vector violates exact declarative composition."""


class RankTransform(StrEnum):
    """Closed deterministic transform vocabulary for version-one rank graphs."""

    WITHIN_USER_DESCENDING_MIDRANK_PERCENTILE = "within_user_descending_midrank_percentile_v1"


def _identity(value: object, location: str) -> Identity:
    if type(value) is bool:
        raise RankGraphError(f"{location} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise RankGraphError(f"{location} must be an integer or non-empty string")


def _sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RankGraphError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _scores(value: ScoreInput, expected: int) -> Float64Vector:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RankGraphError("scores must be a one-dimensional finite numeric vector") from exc
    if raw.ndim != 1 or raw.size != expected or raw.dtype.kind not in "iuf":
        raise RankGraphError(
            f"scores must be a one-dimensional numeric vector of length {expected}"
        )
    try:
        scores = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RankGraphError("scores must be representable as float64") from exc
    if not np.isfinite(scores).all():
        raise RankGraphError("scores must contain only finite values")
    owned = np.array(scores, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class RankGraphRow:
    """One trusted canonical physical row used by every graph member."""

    row_id: int
    user_id: Identity
    video_id: Identity

    def __post_init__(self) -> None:
        if type(self.row_id) is not int or self.row_id < 0:
            raise RankGraphError("row_id must be a non-negative built-in integer")
        object.__setattr__(self, "user_id", _identity(self.user_id, "user_id"))
        object.__setattr__(self, "video_id", _identity(self.video_id, "video_id"))

    def manifest(self) -> dict[str, object]:
        """Return a type-tagged identity so integer and textual IDs cannot collide."""

        return {
            "row_id": self.row_id,
            "user_id": ["i" if type(self.user_id) is int else "s", self.user_id],
            "video_id": ["i" if type(self.video_id) is int else "s", self.video_id],
        }


@dataclass(frozen=True, slots=True)
class RankGraphMember:
    """One content-addressed input and its fixed transform/weight."""

    prediction_id: PredictionId
    prediction_sha256: str
    weight: float
    transform: RankTransform = RankTransform.WITHIN_USER_DESCENDING_MIDRANK_PERCENTILE

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, PredictionId):
            raise RankGraphError("prediction_id must be a PredictionId")
        object.__setattr__(
            self, "prediction_sha256", _sha256(self.prediction_sha256, "prediction_sha256")
        )
        if (
            type(self.weight) is not float
            or not math.isfinite(self.weight)
            or self.weight < 0.0
            or (self.weight == 0.0 and math.copysign(1.0, self.weight) < 0.0)
        ):
            raise RankGraphError("member weight must be a finite non-negative built-in float")
        if not isinstance(self.transform, RankTransform):
            raise RankGraphError("member transform is not allowlisted")

    def manifest(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id.value,
            "prediction_sha256": self.prediction_sha256,
            "weight": self.weight,
            "transform": self.transform.value,
        }


@dataclass(frozen=True, slots=True)
class RankGraph:
    """One immutable exact composition recipe valid at INNER, OUTER, or FINAL."""

    stage: EvidenceStage
    rows: tuple[RankGraphRow, ...]
    members: tuple[RankGraphMember, ...]
    schema_version: int = RANK_GRAPH_SCHEMA_VERSION
    normalization_scope: str = RANK_GRAPH_NORMALIZATION_SCOPE
    tie_policy: str = RANK_GRAPH_TIE_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != RANK_GRAPH_SCHEMA_VERSION:
            raise RankGraphError("rank graph schema_version must be 1")
        if not isinstance(self.stage, EvidenceStage):
            raise RankGraphError("stage must be INNER, OUTER, or FINAL")
        if type(self.rows) is not tuple or not self.rows:
            raise RankGraphError("rows must be a non-empty tuple")
        for expected, row in enumerate(self.rows):
            if not isinstance(row, RankGraphRow):
                raise RankGraphError("rows must contain RankGraphRow values")
            if row.row_id != expected:
                raise RankGraphError("row IDs must be zero-based, contiguous, and ordered")
        if type(self.members) is not tuple or len(self.members) < 2:
            raise RankGraphError("members must be a tuple containing at least two predictions")
        if any(not isinstance(member, RankGraphMember) for member in self.members):
            raise RankGraphError("members must contain RankGraphMember values")
        member_ids = tuple(member.prediction_id for member in self.members)
        if len(set(member_ids)) != len(member_ids):
            raise RankGraphError("rank graph member prediction IDs must be unique")
        if math.fsum(member.weight for member in self.members) != 1.0:
            raise RankGraphError("rank graph member weights must sum exactly to one")
        if self.normalization_scope != RANK_GRAPH_NORMALIZATION_SCOPE:
            raise RankGraphError("rank graph normalization_scope must be within_user")
        if self.tie_policy != RANK_GRAPH_TIE_POLICY:
            raise RankGraphError("rank graph tie_policy differs from the allowlisted policy")

    @property
    def ordered_row_ids(self) -> tuple[int, ...]:
        return tuple(row.row_id for row in self.rows)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "rows": [row.manifest() for row in self.rows],
            "members": [member.manifest() for member in self.members],
            "normalization_scope": self.normalization_scope,
            "tie_policy": self.tie_policy,
        }

    @property
    def digest(self) -> str:
        return canonical_json_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class PredictionVector:
    """One immutable content-addressed member supplied to a graph evaluator."""

    prediction_id: PredictionId
    stage: EvidenceStage
    rows: tuple[RankGraphRow, ...]
    scores: ScoreInput = field(repr=False)
    prediction_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, PredictionId):
            raise RankGraphError("prediction_id must be a PredictionId")
        if not isinstance(self.stage, EvidenceStage):
            raise RankGraphError("prediction stage must be INNER, OUTER, or FINAL")
        if type(self.rows) is not tuple or not self.rows:
            raise RankGraphError("prediction rows must be a non-empty tuple")
        if any(not isinstance(row, RankGraphRow) for row in self.rows):
            raise RankGraphError("prediction rows must contain RankGraphRow values")
        normalized = _scores(self.scores, len(self.rows))
        object.__setattr__(self, "scores", normalized)
        object.__setattr__(self, "prediction_sha256", prediction_digest(normalized))


@dataclass(frozen=True, slots=True)
class RankGraphResult:
    """Exact graph output with a new PredictionId bound to graph and vector bytes."""

    prediction_id: PredictionId
    prediction_sha256: str
    rank_graph_sha256: str
    fusion_sha256: str
    stage: EvidenceStage
    rows: tuple[RankGraphRow, ...]
    scores: Float64Vector = field(repr=False)


_PHASE_BY_STAGE: Final[Mapping[EvidenceStage, DataPhase]] = {
    EvidenceStage.INNER: DataPhase.INNER_VALID,
    EvidenceStage.OUTER: DataPhase.OUTER_VALID,
    EvidenceStage.FINAL: DataPhase.FINAL,
}


class RankGraphEvaluator:
    """Evaluate a closed RankGraph against an exact set of content-addressed vectors."""

    def evaluate(
        self,
        graph: RankGraph,
        predictions: Mapping[PredictionId, PredictionVector],
    ) -> RankGraphResult:
        if not isinstance(graph, RankGraph):
            raise RankGraphError("graph must be a RankGraph")
        if not isinstance(predictions, Mapping):
            raise RankGraphError("predictions must map PredictionId to PredictionVector")
        expected_ids = tuple(member.prediction_id for member in graph.members)
        if set(predictions) != set(expected_ids) or len(predictions) != len(expected_ids):
            raise RankGraphError("predictions must contain exactly the declared graph members")

        ordered_predictions: list[PredictionVector] = []
        for index, member in enumerate(graph.members):
            prediction = predictions[member.prediction_id]
            if not isinstance(prediction, PredictionVector):
                raise RankGraphError("predictions must contain PredictionVector values")
            if prediction.prediction_id != member.prediction_id:
                raise RankGraphError(f"member {index} prediction identity changed")
            if prediction.stage is not graph.stage:
                raise RankGraphError(f"member {index} stage differs from the graph")
            if prediction.rows != graph.rows:
                raise RankGraphError(f"member {index} ordered row alignment differs from the graph")
            if prediction.prediction_sha256 != member.prediction_sha256:
                raise RankGraphError(f"member {index} prediction content digest changed")
            ordered_predictions.append(prediction)

        try:
            fused = fuse_ranked_members(
                [row.user_id for row in graph.rows],
                [row.video_id for row in graph.rows],
                tuple(prediction.scores for prediction in ordered_predictions),
                weights=tuple(member.weight for member in graph.members),
                phase=_PHASE_BY_STAGE[graph.stage],
            )
        except FusionError as exc:
            raise RankGraphError(f"rank graph arithmetic rejected the graph: {exc}") from exc

        graph_sha256 = graph.digest
        output_id = PredictionId.from_rank_graph(
            ordered_row_ids=graph.ordered_row_ids,
            prediction_sha256=fused.prediction_digest,
            rank_graph_sha256=graph_sha256,
        )
        return RankGraphResult(
            prediction_id=output_id,
            prediction_sha256=fused.prediction_digest,
            rank_graph_sha256=graph_sha256,
            fusion_sha256=fused.fusion_digest,
            stage=graph.stage,
            rows=graph.rows,
            scores=fused.scores,
        )


__all__ = [
    "RANK_GRAPH_NORMALIZATION_SCOPE",
    "RANK_GRAPH_SCHEMA_VERSION",
    "RANK_GRAPH_TIE_POLICY",
    "PredictionVector",
    "RankGraph",
    "RankGraphError",
    "RankGraphEvaluator",
    "RankGraphMember",
    "RankGraphResult",
    "RankGraphRow",
    "RankTransform",
]

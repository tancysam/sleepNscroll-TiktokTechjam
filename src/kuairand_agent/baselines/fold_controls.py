"""Trusted official-FM controls for train-derived temporal folds.

This module is deliberately narrower than public-validation qualification.  It accepts only the
two frozen rolling-origin fold roles, binds row-level labels inside trusted scorer closures, and
returns the exact organizer NumPy FM artifacts needed for deterministic replay.  Public
validation and final-period dates are rejected before a label sequence is inspected.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import PredictionVector, StarterFMCheckpoint
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import (
    AggregateMetrics,
    StarterFMAdapter,
    StarterFMConfig,
    StarterFMRun,
    TrainingResources,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.folds import FOLD_A_SPEC, FOLD_B_SPEC, TemporalFoldSpec
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, ScoreResult, SplitIdentity

FOLD_CONTROL_SCHEMA_VERSION: Final = 1
_FOLD_SPECS: Final = {spec.name: spec for spec in (FOLD_A_SPEC, FOLD_B_SPEC)}

type LabelInput = Sequence[object] | npt.NDArray[np.generic] | None
type ScoreInput = Sequence[float] | npt.NDArray[np.generic]
type ScoreClosure = Callable[[ScoreInput], ScoreResult]


class FoldControlError(ValueError):
    """Raised when a fold control request crosses its trusted train-only boundary."""


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FoldControlError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _binary_labels(values: LabelInput, *, expected_count: int, role: str) -> tuple[int, ...]:
    if values is None:
        raise FoldControlError(f"{role} labels are absent; final-like inputs cannot be scored")
    try:
        labels = tuple(values)
    except TypeError as exc:
        raise FoldControlError(f"{role} labels must be an iterable binary vector") from exc
    if len(labels) != expected_count:
        raise FoldControlError(
            f"{role} labels must contain exactly {expected_count} rows; got {len(labels)}"
        )
    normalized: list[int] = []
    for index, value in enumerate(labels):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) not in (0, 1)
        ):
            raise FoldControlError(f"{role} labels[{index}] must be the integer 0 or 1")
        normalized.append(int(value))
    return tuple(normalized)


def _target_digest(inputs_digest: str, labels: npt.NDArray[np.int8]) -> str:
    digest = hashlib.sha256(b"kuairand-primary-training-targets-v1\0")
    digest.update(inputs_digest.encode("ascii"))
    digest.update(labels.astype("<i1", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class PrimaryTrainingTargets:
    """Immutable primary labels cryptographically bound to one canonical input sequence.

    The object implements the narrow target protocol consumed by :class:`StarterFMAdapter`.
    Construction snapshots and validates labels; callers cannot relabel another input table by
    replacing a digest string.
    """

    primary: npt.NDArray[np.int8] = field(repr=False)
    training_inputs_digest: str
    digest: str

    def __init__(self, training_inputs: CanonicalInputs, labels: LabelInput) -> None:
        if not isinstance(training_inputs, CanonicalInputs):
            raise FoldControlError("training_inputs must be CanonicalInputs")
        if len(training_inputs) <= 0:
            raise FoldControlError("training_inputs cannot be empty")
        normalized = _binary_labels(
            labels,
            expected_count=len(training_inputs),
            role="training",
        )
        array = np.frombuffer(bytes(normalized), dtype=np.int8)
        array.setflags(write=False)
        object.__setattr__(self, "primary", array)
        object.__setattr__(self, "training_inputs_digest", training_inputs.digest)
        object.__setattr__(self, "digest", _target_digest(training_inputs.digest, array))

    @classmethod
    def bind(cls, training_inputs: CanonicalInputs, labels: LabelInput) -> PrimaryTrainingTargets:
        """Name the security-relevant binding operation explicitly at call sites."""

        return cls(training_inputs, labels)

    @classmethod
    def from_inputs(
        cls,
        training_inputs: CanonicalInputs,
        labels: LabelInput,
    ) -> PrimaryTrainingTargets:
        """Readable alias for callers constructing a trusted prefix capability."""

        return cls.bind(training_inputs, labels)

    @property
    def row_count(self) -> int:
        return int(self.primary.size)

    def manifest(self) -> dict[str, object]:
        """Return value-free identity metadata suitable for controller evidence."""

        return {
            "schema_version": FOLD_CONTROL_SCHEMA_VERSION,
            "row_count": self.row_count,
            "training_inputs_digest": self.training_inputs_digest,
            "digest": self.digest,
            "target": "long_view",
            "dtype": "int8",
        }


def _validate_fold_query_role(fold_name: object, query_inputs: object) -> TemporalFoldSpec:
    # Validate the declared phase before touching the protected label argument.  Restricting the
    # role to A/B and its frozen date window makes outer-valid/final CanonicalInputs fail closed.
    if type(fold_name) is not str or fold_name not in _FOLD_SPECS:
        raise FoldControlError("fold_name must be train-derived fold 'A' or 'B'")
    if not isinstance(query_inputs, CanonicalInputs):
        raise FoldControlError("query_inputs must be CanonicalInputs")
    if len(query_inputs) <= 0:
        raise FoldControlError("query_inputs cannot be empty or final-like without outcomes")
    spec = _FOLD_SPECS[fold_name]
    if any(not spec.valid_start <= date <= spec.valid_end for date in query_inputs.date):
        raise FoldControlError(
            f"fold {fold_name} query dates must remain inside its official-train validation window"
        )
    return spec


@dataclass(frozen=True, slots=True)
class FoldScoringContext:
    """Value-free fold identity plus the only two label-bound scorer closures.

    Raw labels, :class:`Alignment`, and :class:`ProtectedScorer` are intentionally not fields on
    the returned object.  Only aggregate ``ScoreResult`` values can cross this boundary.
    ``__call__`` selects encoded-float32 semantics so the context directly satisfies the trusted
    starter-FM validation-scorer protocol.
    """

    fold_name: str
    fold_token: str
    query_inputs_digest: str
    query_alignment_digest: str
    row_count: int
    scorer_digest: str
    _score_closure: ScoreClosure = field(repr=False, compare=False)
    _encoded_score_closure: ScoreClosure = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.fold_name not in _FOLD_SPECS:
            raise FoldControlError("fold scoring context must represent fold A or B")
        for name, value in (
            ("fold_token", self.fold_token),
            ("query_inputs_digest", self.query_inputs_digest),
            ("query_alignment_digest", self.query_alignment_digest),
            ("scorer_digest", self.scorer_digest),
        ):
            _require_sha256(value, name)
        if type(self.row_count) is not int or self.row_count <= 0:
            raise FoldControlError("fold scoring row_count must be positive")
        if not callable(self._score_closure) or not callable(self._encoded_score_closure):
            raise FoldControlError("fold scoring closures must be callable")

    @property
    def validation_inputs_digest(self) -> str:
        """Compatibility identity required by the exact starter FM adapter."""

        return self.query_inputs_digest

    @property
    def split_name(self) -> str:
        return f"inner_fold_{self.fold_name}"

    def score(self, scores: ScoreInput, /) -> ScoreResult:
        """Score using the raw-integer label path for metric/metamorphic checks."""

        return self._score_closure(scores)

    def score_with_encoded_labels(self, scores: ScoreInput, /) -> ScoreResult:
        """Score using the organizer FM's exact encoded-``float32`` label path."""

        return self._encoded_score_closure(scores)

    def __call__(self, scores: npt.NDArray[np.float64], /) -> ScoreResult:
        return self.score_with_encoded_labels(scores)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": FOLD_CONTROL_SCHEMA_VERSION,
            "fold_name": self.fold_name,
            "split_name": self.split_name,
            "fold_token": self.fold_token,
            "query_inputs_digest": self.query_inputs_digest,
            "query_alignment_digest": self.query_alignment_digest,
            "row_count": self.row_count,
            "scorer_digest": self.scorer_digest,
            "labels_exposed": False,
        }


def build_fold_scoring_context(
    starter_dir: str | Path,
    fold_name: str,
    fold_token: str,
    query_inputs: CanonicalInputs,
    query_labels: LabelInput,
) -> FoldScoringContext:
    """Bind one official-train fold's labels to contiguous trusted organizer scoring.

    Role, token, and date-window checks happen before ``query_labels`` is iterated.  Consequently,
    passing public-validation/final inputs cannot reveal whether a supplied outcome object is
    present, well-formed, or maliciously lazy.
    """

    _validate_fold_query_role(fold_name, query_inputs)
    token = _require_sha256(fold_token, "fold_token")
    labels = _binary_labels(
        query_labels,
        expected_count=len(query_inputs),
        role=f"fold {fold_name} query",
    )
    split = SplitIdentity(
        name=f"inner_fold_{fold_name}",
        token=token,
        expected_count=len(query_inputs),
    )
    alignment = Alignment.from_ids(
        split=split,
        row_ids=tuple(range(len(query_inputs))),
        user_ids=query_inputs.user_id,
        video_ids=query_inputs.video_id,
    )
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)

    def score(scores: ScoreInput) -> ScoreResult:
        return scorer.score(
            alignment=alignment,
            split=split,
            labels=labels,
            scores=scores,
            expected_count=len(query_inputs),
        )

    def score_with_encoded_labels(scores: ScoreInput) -> ScoreResult:
        return scorer.score_with_encoded_labels(
            alignment=alignment,
            split=split,
            labels=labels,
            scores=scores,
            expected_count=len(query_inputs),
        )

    return FoldScoringContext(
        fold_name=fold_name,
        fold_token=token,
        query_inputs_digest=query_inputs.digest,
        query_alignment_digest=alignment.digest,
        row_count=len(query_inputs),
        scorer_digest=scorer.scorer_digest,
        _score_closure=score,
        _encoded_score_closure=score_with_encoded_labels,
    )


def _derived_fold_token(
    *,
    fold_name: str,
    prefix_inputs: CanonicalInputs,
    query_inputs: CanonicalInputs,
) -> str:
    payload = {
        "schema_version": FOLD_CONTROL_SCHEMA_VERSION,
        "source_split": "train",
        "fold_name": fold_name,
        "prefix_inputs_digest": prefix_inputs.digest,
        "query_inputs_digest": query_inputs.digest,
        "prefix_rows": len(prefix_inputs),
        "query_rows": len(query_inputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(b"kuairand-derived-fm-control-fold-v1\0" + encoded).hexdigest()


def _resolve_fold_role(
    *,
    prefix_inputs: object,
    query_inputs: object,
    declared_name: str | None,
) -> tuple[str, TemporalFoldSpec]:
    # Do not consult either target argument in this phase-classification function.
    if not isinstance(prefix_inputs, CanonicalInputs):
        raise FoldControlError("prefix_inputs must be CanonicalInputs")
    if len(prefix_inputs) <= 0:
        raise FoldControlError("prefix_inputs cannot be empty")
    if declared_name is None:
        if not isinstance(query_inputs, CanonicalInputs) or len(query_inputs) <= 0:
            raise FoldControlError("query_inputs must be non-empty CanonicalInputs")
        matches = tuple(
            spec.name
            for spec in _FOLD_SPECS.values()
            if all(spec.valid_start <= date <= spec.valid_end for date in query_inputs.date)
        )
        if len(matches) != 1:
            raise FoldControlError(
                "query inputs are not contained in one frozen official-train fold window"
            )
        name = matches[0]
    else:
        name = declared_name
    spec = _validate_fold_query_role(name, query_inputs)
    if any(not spec.train_start <= date <= spec.train_end for date in prefix_inputs.date):
        raise FoldControlError(
            f"fold {name} prefix dates must remain inside its official training-prefix window"
        )
    return name, spec


@dataclass(frozen=True, slots=True)
class FoldFMControlRun:
    """Exact fold-control result with replay material and value-free fold identities."""

    fold_name: str
    fold_token: str
    prefix_inputs_digest: str
    query_inputs_digest: str
    query_alignment_digest: str
    encoding: StarterEncoding = field(repr=False)
    training: StarterFMRun = field(repr=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.fold_name not in _FOLD_SPECS:
            raise FoldControlError("fold FM result must represent fold A or B")
        for name in (
            "fold_token",
            "prefix_inputs_digest",
            "query_inputs_digest",
            "query_alignment_digest",
        ):
            _require_sha256(getattr(self, name), name)
        if self.encoding.training_inputs_digest != self.prefix_inputs_digest:
            raise FoldControlError("fold FM encoding and prefix input identities differ")
        if self.training.train_inputs_digest != self.prefix_inputs_digest:
            raise FoldControlError("fold FM training and prefix input identities differ")
        if self.training.validation_inputs_digest != self.query_inputs_digest:
            raise FoldControlError("fold FM training and query input identities differ")
        encoded = json.dumps(
            self.logical_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        object.__setattr__(
            self,
            "digest",
            hashlib.sha256(b"kuairand-fold-fm-control-run-v1\0" + encoded).hexdigest(),
        )

    @property
    def checkpoint(self) -> StarterFMCheckpoint:
        return self.training.checkpoint

    @property
    def predictions(self) -> PredictionVector:
        return self.training.validation_predictions

    @property
    def aggregate_metrics(self) -> AggregateMetrics:
        return self.training.validation_metrics

    @property
    def metrics(self) -> AggregateMetrics:
        """Concise alias used by family-neutral campaign evaluators."""

        return self.aggregate_metrics

    @property
    def resources(self) -> TrainingResources:
        return self.training.resources

    @property
    def training_targets_digest(self) -> str:
        return self.training.training_targets_digest

    @property
    def encoding_digest(self) -> str:
        return self.encoding.digest

    @property
    def logical_digest(self) -> str:
        return self.digest

    @property
    def seed(self) -> int:
        return self.checkpoint.seed

    def replay_predictions(
        self,
        *,
        starter_dir: str | Path,
        query_inputs: CanonicalInputs,
    ) -> PredictionVector:
        """Require exact same-host query prediction bytes from the returned checkpoint."""

        if not isinstance(query_inputs, CanonicalInputs):
            raise FoldControlError("replay query_inputs must be CanonicalInputs")
        if query_inputs.digest != self.query_inputs_digest:
            raise FoldControlError("replay query inputs differ from the fold control")
        return StarterFMAdapter(
            starter_dir=starter_dir,
            config=StarterFMConfig(seed=self.seed),
        ).predict(
            checkpoint=self.checkpoint,
            encoding=self.encoding,
            inputs=query_inputs,
            expected_prediction_digest=self.predictions.digest,
        )

    def logical_manifest(self) -> dict[str, object]:
        return {
            "schema_version": FOLD_CONTROL_SCHEMA_VERSION,
            "fold_name": self.fold_name,
            "fold_token": self.fold_token,
            "prefix_inputs_digest": self.prefix_inputs_digest,
            "query_inputs_digest": self.query_inputs_digest,
            "query_alignment_digest": self.query_alignment_digest,
            "encoding_digest": self.encoding.digest,
            "training": self.training.logical_manifest(),
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self.logical_manifest(),
            "digest": self.digest,
            "resources": self.resources.manifest(),
        }


def run_fold_fm_control(
    prefix_inputs: CanonicalInputs,
    prefix_labels: LabelInput,
    query_inputs: CanonicalInputs,
    query_labels: LabelInput,
    starter_dir: str | Path,
    seed: int = 0,
    *,
    fold_name: str | None = None,
    fold_token: str | None = None,
) -> FoldFMControlRun:
    """Train the immutable official FM on a train prefix and score a train-derived fold.

    ``fold_name`` and ``fold_token`` may be supplied from a persisted
    :class:`~kuairand_agent.data.folds.TemporalFold`.  When omitted, the role is inferred from the
    frozen date windows and a target-independent token is derived from the two input identities.
    Supplying only one identity is rejected so campaign evidence cannot mix declared and derived
    provenance.
    """

    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise FoldControlError("seed must be a uint32-compatible integer")
    if (fold_name is None) != (fold_token is None):
        raise FoldControlError("fold_name and fold_token must be supplied together")
    resolved_name, _ = _resolve_fold_role(
        prefix_inputs=prefix_inputs,
        query_inputs=query_inputs,
        declared_name=fold_name,
    )
    resolved_token = (
        _derived_fold_token(
            fold_name=resolved_name,
            prefix_inputs=prefix_inputs,
            query_inputs=query_inputs,
        )
        if fold_token is None
        else _require_sha256(fold_token, "fold_token")
    )

    # Phase and chronology checks above deliberately precede both target bindings.
    targets = PrimaryTrainingTargets.bind(prefix_inputs, prefix_labels)
    scoring = build_fold_scoring_context(
        starter_dir,
        resolved_name,
        resolved_token,
        query_inputs,
        query_labels,
    )
    encoding = StarterEncoding.fit(prefix_inputs)
    training = StarterFMAdapter(
        starter_dir=starter_dir,
        config=StarterFMConfig(seed=seed),
    ).fit(
        encoding=encoding,
        train_inputs=prefix_inputs,
        train_targets=targets,
        validation_inputs=query_inputs,
        validation_scorer=scoring,
    )
    return FoldFMControlRun(
        fold_name=resolved_name,
        fold_token=resolved_token,
        prefix_inputs_digest=prefix_inputs.digest,
        query_inputs_digest=query_inputs.digest,
        query_alignment_digest=scoring.query_alignment_digest,
        encoding=encoding,
        training=training,
    )


__all__ = [
    "FOLD_CONTROL_SCHEMA_VERSION",
    "FoldControlError",
    "FoldFMControlRun",
    "FoldScoringContext",
    "PrimaryTrainingTargets",
    "build_fold_scoring_context",
    "run_fold_fm_control",
]

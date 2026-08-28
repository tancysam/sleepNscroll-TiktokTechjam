"""Validation-only random and item-popularity qualification rungs.

The production path consumes :class:`~kuairand_agent.data.canonical.CanonicalDataset` and binds
only its protected public-validation target to the hash-pinned organizer scorer.  It never imports
the organizer's eager ``data.load`` function and has no route to final-period outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self, overload

import numpy as np
import numpy.typing as npt

from kuairand_agent.contract import BENCHMARK_CONTRACT, PublishedScore, SplitName
from kuairand_agent.data.canonical import (
    CanonicalDataset,
    CanonicalInputs,
    ProtectedTargets,
    TrainingTargets,
)
from kuairand_agent.scoring.protected import (
    Alignment,
    ProtectedScorer,
    ScoreResult,
    SplitIdentity,
)

RUNG_SCHEMA_VERSION: Final = 1
RANDOM_SEEDS: Final = (0, 1, 2, 3, 4)
POPULARITY_PRIOR: Final = 20.0
FOUR_DECIMAL: Final = Decimal("0.0001")

type OrganizerRow = tuple[int, str, str, str, str, float, int]


class BaselineRungError(ValueError):
    """Raised when a validation rung or summary violates its frozen contract."""


class BaselineReferenceMismatch(RuntimeError):
    """Raised when a full-data rung misses its published four-decimal reference."""


class RungName(StrEnum):
    """The two non-trained organizer qualification rungs."""

    RANDOM = "random"
    ITEM_POPULARITY = "item_popularity"


def _digest(namespace: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(namespace.encode("ascii") + b"\0" + encoded).hexdigest()


def _round_four(value: float) -> float:
    return float(Decimal(str(value)).quantize(FOUR_DECIMAL, rounding=ROUND_HALF_UP))


@dataclass(frozen=True, slots=True)
class RungMetrics:
    """Exact aggregate metric triplet returned by the protected organizer scorer."""

    gauc: float
    ndcg_at_5: float
    primary: float

    def __post_init__(self) -> None:
        values = (self.gauc, self.ndcg_at_5, self.primary)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise BaselineRungError("rung metrics must be finite floats")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise BaselineRungError("rung metrics must be in [0, 1]")
        if not math.isclose(
            self.primary, (self.gauc + self.ndcg_at_5) / 2.0, rel_tol=0.0, abs_tol=1e-15
        ):
            raise BaselineRungError("rung primary must be the exact GAUC/nDCG@5 mean")

    @classmethod
    def from_score_result(cls, result: ScoreResult) -> Self:
        return cls(
            gauc=float(result.gauc),
            ndcg_at_5=float(result.ndcg_at_5),
            primary=float(result.primary),
        )

    @classmethod
    def from_published(cls, score: PublishedScore) -> Self:
        return cls(
            gauc=float(score.gauc),
            ndcg_at_5=float(score.ndcg_at_5),
            primary=float(score.primary),
        )

    def rounded_four(self) -> Self:
        """Return the deterministic reporting triplet used by hard reference gates."""

        gauc = _round_four(self.gauc)
        ndcg = _round_four(self.ndcg_at_5)
        primary = _round_four((gauc + ndcg) / 2.0)
        # Published primary is defined as the rounded mean of the published components.  Derive it
        # from rounded components so a tiny binary64 mean discrepancy cannot change the gate.
        return type(self)(gauc=gauc, ndcg_at_5=ndcg, primary=primary)

    def manifest(self) -> dict[str, float]:
        return {"GAUC": self.gauc, "nDCG@5": self.ndcg_at_5, "primary": self.primary}


def _reference(name: RungName) -> RungMetrics:
    for rung in BENCHMARK_CONTRACT.reference_rungs:
        if rung.name == name.value:
            return RungMetrics.from_published(rung.validation)
    raise AssertionError(f"benchmark contract lacks reference rung {name.value}")


@dataclass(frozen=True, slots=True)
class RungEvaluation:
    """One seed/run result with stable scientific identity and runtime diagnostics."""

    name: RungName
    seed: int | None
    metrics: RungMetrics
    users: int
    rows: int
    scorer_digest: str
    prediction_digest: str
    split_digest: str
    runtime_seconds: float = field(compare=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.name is RungName.RANDOM:
            if type(self.seed) is not int or self.seed not in RANDOM_SEEDS:
                raise BaselineRungError("random rung seed must be one of fixed seeds 0 through 4")
        elif self.seed is not None:
            raise BaselineRungError("item-popularity rung is deterministic and has no seed")
        if type(self.users) is not int or self.users <= 0:
            raise BaselineRungError("rung user count must be a positive integer")
        if type(self.rows) is not int or self.rows <= 0:
            raise BaselineRungError("rung row count must be a positive integer")
        for name, value in (
            ("scorer_digest", self.scorer_digest),
            ("prediction_digest", self.prediction_digest),
            ("split_digest", self.split_digest),
        ):
            if type(value) is not str or len(value) != 64:
                raise BaselineRungError(f"{name} must be a SHA-256 hex digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise BaselineRungError(f"{name} must be hexadecimal") from exc
        if type(self.runtime_seconds) is not float or not math.isfinite(self.runtime_seconds):
            raise BaselineRungError("rung runtime must be a finite float")
        if self.runtime_seconds < 0.0:
            raise BaselineRungError("rung runtime cannot be negative")
        object.__setattr__(
            self,
            "digest",
            _digest("kuairand-baseline-rung-evaluation-v1", self.logical_manifest()),
        )

    @classmethod
    def from_score_result(
        cls,
        *,
        name: RungName,
        seed: int | None,
        split_digest: str,
        result: ScoreResult,
    ) -> Self:
        return cls(
            name=name,
            seed=seed,
            metrics=RungMetrics.from_score_result(result),
            users=result.users,
            rows=result.rows,
            scorer_digest=result.scorer_digest,
            prediction_digest=result.prediction_digest,
            split_digest=split_digest,
            runtime_seconds=float(result.runtime_seconds),
        )

    def logical_manifest(self) -> dict[str, object]:
        """Return deterministic evidence; wall-clock runtime is deliberately separate."""

        return {
            "schema_version": RUNG_SCHEMA_VERSION,
            "name": self.name.value,
            "seed": self.seed,
            "metrics": self.metrics.manifest(),
            "users": self.users,
            "rows": self.rows,
            "scorer_digest": self.scorer_digest,
            "prediction_digest": self.prediction_digest,
            "split_digest": self.split_digest,
        }

    def manifest(self) -> dict[str, object]:
        manifest = self.logical_manifest()
        manifest["runtime_seconds"] = self.runtime_seconds
        manifest["digest"] = self.digest
        return manifest


@dataclass(frozen=True, slots=True)
class RungSummary:
    """Deterministic mean, rounded hard gate, and per-run evidence for one rung."""

    name: RungName
    evaluations: Sequence[RungEvaluation]
    reference_metrics: RungMetrics | None
    mean_metrics: RungMetrics = field(init=False)
    rounded_mean: RungMetrics = field(init=False)
    reference_passed: bool = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        if not evaluations or any(run.name is not self.name for run in evaluations):
            raise BaselineRungError("rung summary requires nonempty same-name evaluations")
        seeds = [run.seed for run in evaluations]
        if len(seeds) != len(set(seeds)):
            raise BaselineRungError("rung summary contains duplicate seed identities")
        mean = RungMetrics(
            gauc=float(statistics.fmean(run.metrics.gauc for run in evaluations)),
            ndcg_at_5=float(statistics.fmean(run.metrics.ndcg_at_5 for run in evaluations)),
            primary=float(statistics.fmean(run.metrics.primary for run in evaluations)),
        )
        rounded = mean.rounded_four()
        passed = self.reference_metrics is not None and rounded == self.reference_metrics
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "mean_metrics", mean)
        object.__setattr__(self, "rounded_mean", rounded)
        object.__setattr__(self, "reference_passed", passed)
        logical = {
            "schema_version": RUNG_SCHEMA_VERSION,
            "name": self.name.value,
            "evaluation_digests": [run.digest for run in evaluations],
            "mean_metrics": mean.manifest(),
            "rounded_mean": rounded.manifest(),
            "reference_metrics": (
                None if self.reference_metrics is None else self.reference_metrics.manifest()
            ),
            "reference_passed": passed,
        }
        object.__setattr__(self, "digest", _digest("kuairand-baseline-rung-summary-v1", logical))

    def logical_manifest(self) -> dict[str, object]:
        return {
            "schema_version": RUNG_SCHEMA_VERSION,
            "name": self.name.value,
            "evaluations": [
                run.logical_manifest() | {"digest": run.digest} for run in self.evaluations
            ],
            "mean_metrics": self.mean_metrics.manifest(),
            "rounded_mean": self.rounded_mean.manifest(),
            "reference_metrics": (
                None if self.reference_metrics is None else self.reference_metrics.manifest()
            ),
            "reference_passed": self.reference_passed,
            "digest": self.digest,
        }

    def manifest(self) -> dict[str, object]:
        manifest = self.logical_manifest()
        manifest["runtime_seconds"] = float(
            math.fsum(run.runtime_seconds for run in self.evaluations)
        )
        return manifest


@dataclass(frozen=True, slots=True)
class ValidationScoringContext:
    """Trusted public-validation alignment, protected labels, and pinned scorer."""

    split: SplitIdentity
    alignment: Alignment
    scorer: ProtectedScorer = field(repr=False, compare=False)
    _labels: tuple[int, ...] = field(repr=False)
    canonical_split_digest: str
    dataset_digest: str

    def __post_init__(self) -> None:
        if self.split.name != "outer_valid":
            raise BaselineRungError("baseline rung context must be outer_valid")
        if self.alignment.split != self.split or self.scorer.trusted_alignment != self.alignment:
            raise BaselineRungError("validation scorer and canonical alignment differ")
        if len(self._labels) != self.split.expected_count:
            raise BaselineRungError("protected validation label count differs from split identity")
        if self.split.token != self.canonical_split_digest:
            raise BaselineRungError(
                "validation split token must derive from canonical split digest"
            )
        for value in (self.canonical_split_digest, self.dataset_digest):
            if len(value) != 64:
                raise BaselineRungError("canonical validation identities must be SHA-256 digests")

    @property
    def row_count(self) -> int:
        return self.split.expected_count

    def score(self, scores: Sequence[float] | npt.NDArray[np.generic]) -> ScoreResult:
        """Enter the only explicit route from protected labels to the organizer scorer."""

        return self.scorer.score(
            alignment=self.alignment,
            split=self.split,
            labels=self._labels,
            scores=scores,
            expected_count=self.row_count,
        )

    def score_with_encoded_labels(
        self,
        scores: Sequence[float] | npt.NDArray[np.generic],
    ) -> ScoreResult:
        """Score with the organizer FM's exact encoded-``float32`` label semantics.

        ``baseline.run_fm`` receives labels from ``data.encode`` as ``float32``.  NumPy 2.x
        preserves that scalar dtype through part of the organizer nDCG calculation, so this
        explicit trusted route is required for byte-for-byte-compatible trainable-model
        qualification.  The protected labels remain bound inside this context.
        """

        return self.scorer.score_with_encoded_labels(
            alignment=self.alignment,
            split=self.split,
            labels=self._labels,
            scores=scores,
            expected_count=self.row_count,
        )


def build_validation_scoring_context(
    dataset: CanonicalDataset,
    starter_dir: str | Path,
) -> ValidationScoringContext:
    """Bind canonical public validation—not final evaluation—to the protected scorer."""

    if not isinstance(dataset, CanonicalDataset):
        raise BaselineRungError("baseline rungs require a CanonicalDataset")
    valid = dataset.valid
    if valid.name is not SplitName.VALID or not isinstance(valid.targets, ProtectedTargets):
        raise BaselineRungError("canonical public validation lacks protected long_view targets")
    if valid.row_count <= 0:
        raise BaselineRungError("public validation cannot be empty")
    split = SplitIdentity(
        name="outer_valid",
        token=valid.digest,
        expected_count=valid.row_count,
    )
    alignment = Alignment.from_ids(
        split=split,
        row_ids=valid.alignment.row_id,
        user_ids=valid.alignment.user_id,
        video_ids=valid.alignment.video_id,
    )
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)
    return ValidationScoringContext(
        split=split,
        alignment=alignment,
        scorer=scorer,
        _labels=valid.targets.reveal_for_scorer(),
        canonical_split_digest=valid.digest,
        dataset_digest=dataset.digest,
    )


def evaluate_random_validation(
    context: ValidationScoringContext,
    seed: int,
) -> RungEvaluation:
    """Score one fixed NumPy ``default_rng`` seed on public validation only."""

    if type(seed) is not int or seed not in RANDOM_SEEDS:
        raise BaselineRungError("random validation seed must be one of 0, 1, 2, 3, 4")
    scores = np.random.default_rng(seed).random(context.row_count)
    result = context.score(scores)
    return RungEvaluation.from_score_result(
        name=RungName.RANDOM,
        seed=seed,
        split_digest=context.canonical_split_digest,
        result=result,
    )


def evaluate_random_rungs(context: ValidationScoringContext) -> RungSummary:
    """Evaluate the complete, ordered seed-0-through-4 random reference rung."""

    evaluations = tuple(evaluate_random_validation(context, seed) for seed in RANDOM_SEEDS)
    return RungSummary(
        name=RungName.RANDOM,
        evaluations=evaluations,
        reference_metrics=_reference(RungName.RANDOM),
    )


def _popularity_scores(dataset: CanonicalDataset) -> npt.NDArray[np.float64]:
    train = dataset.train
    if train.name is not SplitName.TRAIN or not isinstance(train.targets, TrainingTargets):
        raise BaselineRungError("item popularity requires canonical training targets")
    labels = train.targets.long_view
    if not labels:
        raise BaselineRungError("item popularity cannot fit an empty training split")
    positives: dict[str, int] = {}
    impressions: dict[str, int] = {}
    for video_id, label in zip(train.inputs.video_id, labels, strict=True):
        impressions[video_id] = impressions.get(video_id, 0) + 1
        positives[video_id] = positives.get(video_id, 0) + label
    global_mean = math.fsum(labels) / len(labels)
    scores = np.empty(dataset.valid.row_count, dtype=np.float64)
    for index, video_id in enumerate(dataset.valid.inputs.video_id):
        count = impressions.get(video_id, 0)
        scores[index] = (
            (positives.get(video_id, 0) + POPULARITY_PRIOR * global_mean)
            / (count + POPULARITY_PRIOR)
            if count
            else global_mean
        )
    scores.setflags(write=False)
    return scores


def evaluate_popularity_validation(
    dataset: CanonicalDataset,
    context: ValidationScoringContext,
) -> RungSummary:
    """Evaluate the exact organizer item-rate smoother with fixed prior 20."""

    if context.dataset_digest != dataset.digest:
        raise BaselineRungError("popularity dataset differs from validation scoring context")
    result = context.score(_popularity_scores(dataset))
    evaluation = RungEvaluation.from_score_result(
        name=RungName.ITEM_POPULARITY,
        seed=None,
        split_digest=context.canonical_split_digest,
        result=result,
    )
    return RungSummary(
        name=RungName.ITEM_POPULARITY,
        evaluations=(evaluation,),
        reference_metrics=_reference(RungName.ITEM_POPULARITY),
    )


def require_reference_parity(summary: RungSummary) -> None:
    """Apply the non-negotiable four-decimal published-reference gate."""

    if summary.reference_metrics is None:
        raise BaselineReferenceMismatch(f"{summary.name.value} has no frozen reference")
    if not summary.reference_passed:
        raise BaselineReferenceMismatch(
            f"{summary.name.value} four-decimal mismatch: "
            f"observed={summary.rounded_mean.manifest()}, "
            f"expected={summary.reference_metrics.manifest()}"
        )


def select_best_seed(evaluations: Sequence[RungEvaluation]) -> RungEvaluation:
    """Choose highest primary, GAUC, nDCG@5, then the lowest seed deterministically."""

    runs = tuple(evaluations)
    if not runs:
        raise BaselineRungError("best-seed selection requires at least one run")
    if any(run.seed is None for run in runs):
        raise BaselineRungError("best-seed selection cannot contain unseeded runs")
    seeds = [run.seed for run in runs]
    if len(seeds) != len(set(seeds)):
        raise BaselineRungError("best-seed selection cannot contain duplicate seeds")
    return max(
        runs,
        key=lambda run: (
            run.metrics.primary,
            run.metrics.gauc,
            run.metrics.ndcg_at_5,
            -(run.seed if run.seed is not None else 0),
        ),
    )


@dataclass(frozen=True, slots=True)
class BaselineRungQualification:
    """Hard-gated random and popularity validation evidence."""

    random: RungSummary
    popularity: RungSummary
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.random.name is not RungName.RANDOM:
            raise BaselineRungError("qualification random summary has the wrong name")
        if self.popularity.name is not RungName.ITEM_POPULARITY:
            raise BaselineRungError("qualification popularity summary has the wrong name")
        require_reference_parity(self.random)
        require_reference_parity(self.popularity)
        object.__setattr__(
            self,
            "digest",
            _digest(
                "kuairand-baseline-rung-qualification-v1",
                {
                    "schema_version": RUNG_SCHEMA_VERSION,
                    "random_digest": self.random.digest,
                    "popularity_digest": self.popularity.digest,
                },
            ),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": RUNG_SCHEMA_VERSION,
            "random": self.random.manifest(),
            "item_popularity": self.popularity.manifest(),
            "digest": self.digest,
        }


def qualify_reference_rungs(
    dataset: CanonicalDataset,
    starter_dir: str | Path,
) -> BaselineRungQualification:
    """Run and hard-gate all non-trained public-validation qualification rungs."""

    context = build_validation_scoring_context(dataset, starter_dir)
    return BaselineRungQualification(
        random=evaluate_random_rungs(context),
        popularity=evaluate_popularity_validation(dataset, context),
    )


class _OrganizerRows(Sequence[OrganizerRow]):
    """Zero-copy organizer tuple view used only for immutable parity tests."""

    __slots__ = ("_inputs", "_labels", "_positions")

    def __init__(
        self,
        inputs: CanonicalInputs,
        labels: Sequence[int],
        positions: range,
    ) -> None:
        if len(inputs) != len(labels):
            raise BaselineRungError("organizer fixture inputs and labels differ in length")
        if positions.start < 0 or positions.stop > len(inputs):
            raise BaselineRungError("organizer fixture positions exceed their canonical split")
        self._inputs = inputs
        self._labels = labels
        self._positions = positions

    def __len__(self) -> int:
        return len(self._positions)

    def _row(self, item: int) -> OrganizerRow:
        position = self._positions[item]
        return (
            self._inputs.date[position],
            self._inputs.user_id[position],
            self._inputs.video_id[position],
            self._inputs.author_id[position],
            self._inputs.tab[position],
            self._inputs.duration_ms[position],
            self._labels[position],
        )

    @overload
    def __getitem__(self, item: int) -> OrganizerRow: ...

    @overload
    def __getitem__(self, item: slice) -> list[OrganizerRow]: ...

    def __getitem__(self, item: int | slice) -> OrganizerRow | list[OrganizerRow]:
        if isinstance(item, slice):
            return [self._row(index) for index in range(*item.indices(len(self)))]
        return self._row(item)

    def __iter__(self) -> Iterator[OrganizerRow]:
        for index in range(len(self)):
            yield self._row(index)


def organizer_validation_fixture(
    dataset: CanonicalDataset,
    *,
    placeholder_count: int = 1,
) -> Mapping[str, Sequence[OrganizerRow]]:
    """Adapt canonical train/valid rows for untouched organizer parity functions.

    The organizer's functions insist on evaluating a ``test`` entry.  This adapter deliberately
    supplies a harmless nonempty prefix copied from official *training* data; it never touches the
    canonical final split.  Production qualification ignores that placeholder result and uses the
    protected scorer exclusively.
    """

    if type(placeholder_count) is not int or placeholder_count <= 0:
        raise BaselineRungError("organizer placeholder_count must be a positive integer")
    if not isinstance(dataset.train.targets, TrainingTargets):
        raise BaselineRungError("organizer parity fixture requires training targets")
    if not isinstance(dataset.valid.targets, ProtectedTargets):
        raise BaselineRungError("organizer parity fixture requires protected validation targets")
    if placeholder_count > dataset.train.row_count:
        raise BaselineRungError("organizer placeholder exceeds the training split")
    train_labels = dataset.train.targets.long_view
    valid_labels = dataset.valid.targets.reveal_for_scorer()
    return MappingProxyType(
        {
            "train": _OrganizerRows(
                dataset.train.inputs, train_labels, range(dataset.train.row_count)
            ),
            "valid": _OrganizerRows(
                dataset.valid.inputs, valid_labels, range(dataset.valid.row_count)
            ),
            "test": _OrganizerRows(dataset.train.inputs, train_labels, range(placeholder_count)),
        }
    )


__all__ = [
    "FOUR_DECIMAL",
    "POPULARITY_PRIOR",
    "RANDOM_SEEDS",
    "RUNG_SCHEMA_VERSION",
    "BaselineReferenceMismatch",
    "BaselineRungError",
    "BaselineRungQualification",
    "RungEvaluation",
    "RungMetrics",
    "RungName",
    "RungSummary",
    "ValidationScoringContext",
    "build_validation_scoring_context",
    "evaluate_popularity_validation",
    "evaluate_random_rungs",
    "evaluate_random_validation",
    "organizer_validation_fixture",
    "qualify_reference_rungs",
    "require_reference_parity",
    "select_best_seed",
]

"""Bounded train-only analyses the research model may request for its next iteration.

Every finding that actually moved this project came from a person looking at data: the briefing's
slate statistics were wrong until someone measured the scored split, the rank-fusion defect was
found by auditing digests, and the fact that 86% of an ensembling gain lives in within-user rank
normalisation came from a probe. None of those were reachable by the agent, which sees only the
aggregates the controller decided in advance to compute.

This module lets the model ask a question instead. It names a query from a fixed vocabulary; the
trusted controller validates it, computes it over the *training* split alone, and returns scalars
into the next iteration's context. The model gains the ability to ask; it gains no authority. There
is no code execution, no filesystem access, and no path to a validation or final-period outcome --
the request is a typed enum plus validated feature names, not a program.

Three properties are enforced here rather than trusted:

* **Train-only.** The executor is handed the training feature matrix, training labels and training
  user ids. No validation or final-period array is reachable from this module.
* **Aggregate-only.** Every result is a scalar summarising at least ``_MIN_GROUP_ROWS`` rows, so no
  query can approach a row-level disclosure.
* **Named columns only.** Feature names are checked against the bundle the controller built, so a
  query cannot reference a column that does not exist or smuggle an expression.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kuairand_agent.research.context import AggregateRecord, AggregateScalar

# Matches the EDA floor: no reported number may summarise fewer rows than this.
_MIN_GROUP_ROWS: Final = 50
# A reflection may ask at most this many questions, so the capability cannot crowd out the
# briefing or turn one iteration into an unbounded survey.
MAX_ANALYSIS_REQUESTS: Final = 4
_MAX_BUCKETS: Final = 10
_MIN_BUCKETS: Final = 2


class AnalysisError(ValueError):
    """Raised when a requested analysis is not answerable within the frozen policy."""


class AnalysisKind(StrEnum):
    """The complete vocabulary of questions the model may ask.

    Each entry answers a question the campaign records have repeatedly left open. They are
    deliberately few: a larger surface is a larger thing to prove leakage-free, and none of the
    open questions in this project need one.
    """

    # "Does crossing these two columns carry signal I could not get from either alone?"
    WITHIN_USER_INTERACTION = "within_user_interaction"
    # "Is this feature's relationship to the label monotonic, or does it need a non-linear form?"
    LABEL_RATE_BY_BUCKET = "label_rate_by_bucket"
    # "Where in the slate-size distribution does this feature carry its signal?"
    SIGNAL_BY_SLATE_SIZE = "signal_by_slate_size"


@dataclass(frozen=True, slots=True)
class AnalysisQuery:
    """One validated request for a train-only aggregate."""

    kind: AnalysisKind
    feature: str
    second_feature: str | None = None
    buckets: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AnalysisKind):
            raise AnalysisError("analysis kind is not in the supported vocabulary")
        for name in ("feature", "second_feature"):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) is not str or not value or len(value) > 128:
                raise AnalysisError(f"analysis {name} must be a bounded non-empty string")
        if self.kind is AnalysisKind.WITHIN_USER_INTERACTION and self.second_feature is None:
            raise AnalysisError("within_user_interaction requires two features")
        if type(self.buckets) is not int or not _MIN_BUCKETS <= self.buckets <= _MAX_BUCKETS:
            raise AnalysisError(f"analysis buckets must be in [{_MIN_BUCKETS}, {_MAX_BUCKETS}]")

    def to_wire(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "feature": self.feature,
            "second_feature": self.second_feature,
            "buckets": self.buckets,
        }


def _user_codes(user_ids: Sequence[object]) -> tuple[NDArray[np.int64], int]:
    lookup: dict[object, int] = {}
    codes = np.empty(len(user_ids), dtype=np.int64)
    for index, value in enumerate(user_ids):
        code = lookup.get(value)
        if code is None:
            code = len(lookup)
            lookup[value] = code
        codes[index] = code
    return codes, len(lookup)


def _within_user_centred(
    column: NDArray[np.float64], codes: NDArray[np.int64], counts: NDArray[np.float64]
) -> NDArray[np.float64]:
    sums = np.bincount(codes, weights=column, minlength=counts.size)
    return column - (sums / counts)[codes]


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    left_energy = float(left @ left)
    right_energy = float(right @ right)
    if left_energy <= 0.0 or right_energy <= 0.0:
        return 0.0
    return float(left @ right) / float(np.sqrt(left_energy * right_energy))


@dataclass(frozen=True, slots=True)
class TrainAnalysisInputs:
    """The training arrays an analysis may read, and nothing else.

    Constructing this is the only way to reach the executor, so the train-only property is a
    property of the type rather than a convention a caller must remember.
    """

    feature_names: tuple[str, ...]
    feature_values: NDArray[np.float64]
    labels: NDArray[np.float64]
    user_ids: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.feature_values.ndim != 2:
            raise AnalysisError("training feature values must be two-dimensional")
        if self.feature_values.shape[1] != len(self.feature_names):
            raise AnalysisError("training features and names disagree on width")
        rows = self.feature_values.shape[0]
        if rows != self.labels.size or rows != len(self.user_ids):
            raise AnalysisError("training features, labels and user ids must align")

    def column(self, name: str) -> NDArray[np.float64]:
        try:
            index = self.feature_names.index(name)
        except ValueError:
            raise AnalysisError(f"unknown feature {name!r}") from None
        return np.asarray(self.feature_values[:, index], dtype=np.float64)


def run_analysis(inputs: TrainAnalysisInputs, query: AnalysisQuery) -> AggregateRecord:
    """Execute one validated query over the training split and return bounded scalars."""

    if inputs.feature_values.shape[0] < _MIN_GROUP_ROWS:
        raise AnalysisError("training split is too small to answer any analysis")
    codes, user_count = _user_codes(inputs.user_ids)
    counts = np.bincount(codes, minlength=user_count).astype(np.float64)
    target = _within_user_centred(inputs.labels, codes, counts)

    if query.kind is AnalysisKind.WITHIN_USER_INTERACTION:
        assert query.second_feature is not None  # enforced by AnalysisQuery
        left = inputs.column(query.feature)
        right = inputs.column(query.second_feature)
        values: dict[str, AggregateScalar] = {
            "feature_a": query.feature,
            "feature_b": query.second_feature,
            "within_user_corr_a": round(
                _correlation(_within_user_centred(left, codes, counts), target), 6
            ),
            "within_user_corr_b": round(
                _correlation(_within_user_centred(right, codes, counts), target), 6
            ),
            "within_user_corr_product": round(
                _correlation(_within_user_centred(left * right, codes, counts), target), 6
            ),
            "note": (
                "within_user_corr_product is the cross's own within-user correlation with "
                "long_view. It is worth building only if it exceeds both single-column "
                "correlations; a cross that does not is already captured by the columns you have."
            ),
        }
        return AggregateRecord("requested_within_user_interaction", values)

    if query.kind is AnalysisKind.LABEL_RATE_BY_BUCKET:
        column = inputs.column(query.feature)
        edges = np.quantile(column, np.linspace(0.0, 1.0, query.buckets + 1))
        assigned = np.clip(np.searchsorted(edges[1:-1], column, side="right"), 0, query.buckets - 1)
        values = {
            "feature": query.feature,
            "buckets": query.buckets,
        }
        for bucket in range(query.buckets):
            mask = assigned == bucket
            rows = int(mask.sum())
            if rows < _MIN_GROUP_ROWS:
                continue
            values[f"bucket_{bucket:02d}_rows"] = rows
            values[f"bucket_{bucket:02d}_label_rate"] = round(float(inputs.labels[mask].mean()), 6)
        values["note"] = (
            "Label rate per quantile bucket over training rows. A non-monotonic pattern means a "
            "linear term cannot capture this column and a bucketed or crossed form may."
        )
        return AggregateRecord("requested_label_rate_by_bucket", values)

    column = inputs.column(query.feature)
    centred = _within_user_centred(column, codes, counts)
    slate_sizes = counts[codes]
    edges = np.quantile(slate_sizes, np.linspace(0.0, 1.0, query.buckets + 1))
    assigned = np.clip(
        np.searchsorted(edges[1:-1], slate_sizes, side="right"), 0, query.buckets - 1
    )
    values = {
        "feature": query.feature,
        "buckets": query.buckets,
    }
    for bucket in range(query.buckets):
        mask = assigned == bucket
        rows = int(mask.sum())
        if rows < _MIN_GROUP_ROWS:
            continue
        values[f"bucket_{bucket:02d}_rows"] = rows
        values[f"bucket_{bucket:02d}_median_slate"] = float(np.median(slate_sizes[mask]))
        values[f"bucket_{bucket:02d}_within_user_corr"] = round(
            _correlation(centred[mask], target[mask]), 6
        )
    values["note"] = (
        "Within-user correlation with long_view, split by slate size. nDCG@5 weights every user "
        "equally and truncates at rank 5, while GAUC weights by positive count, so a column whose "
        "signal sits only in large slates helps GAUC and not nDCG@5."
    )
    return AggregateRecord("requested_signal_by_slate_size", values)


def run_requested_analyses(
    inputs: TrainAnalysisInputs,
    queries: Sequence[AnalysisQuery],
) -> tuple[AggregateRecord, ...]:
    """Answer each requested query, reporting failures as evidence rather than raising.

    A malformed or unanswerable question must not end a campaign: the model is told what went
    wrong so it can ask a better one next iteration.
    """

    records: list[AggregateRecord] = []
    for index, query in enumerate(queries[:MAX_ANALYSIS_REQUESTS], start=1):
        try:
            record = run_analysis(inputs, query)
        except AnalysisError as error:
            records.append(
                AggregateRecord(
                    f"requested_analysis_{index:02d}_failed",
                    {
                        "kind": query.kind.value,
                        "feature": query.feature,
                        "second_feature": query.second_feature,
                        "reason": str(error)[:200],
                    },
                )
            )
            continue
        records.append(
            AggregateRecord(f"requested_analysis_{index:02d}_{record.name}", dict(record.values))
        )
    return tuple(records)


__all__ = [
    "MAX_ANALYSIS_REQUESTS",
    "AnalysisError",
    "AnalysisKind",
    "AnalysisQuery",
    "TrainAnalysisInputs",
    "run_analysis",
    "run_requested_analyses",
]

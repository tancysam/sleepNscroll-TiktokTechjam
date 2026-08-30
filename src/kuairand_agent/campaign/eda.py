"""Train-only exploratory diagnostics the research model can actually form hypotheses from.

The agent proposes modelling changes without any ability to query the data, so whatever the
controller computes here is the entirety of what it knows. Until now that was about ten scalars --
row counts, cardinalities, duration percentiles -- which is not enough to motivate any hypothesis
except the one the briefing already names.

Everything here is computed from the training split alone: training inputs, the training feature
matrix, and training ``long_view``. No validation or final-period outcome is read, and the outputs
are aggregates over at least a minimum group size, never row-level values.

The diagnostics are deliberately *within-user*. This benchmark ranks each user's own impressions,
so a feature that is constant across a user's rows cannot reorder them no matter how strongly it
correlates with the label globally -- the organizers measured exactly that, finding purely
user-side first-order terms contribute nothing. A global correlation would therefore point the
agent at features that cannot help. Within-user centring measures the quantity that actually moves
GAUC and nDCG.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kuairand_agent.research.context import AggregateRecord

# Aggregates over fewer rows than this are not reported, so no diagnostic can approach a
# row-level disclosure of a small group.
_MIN_GROUP_ROWS: Final = 50
# Reported per direction. The full feature list is already in the method card; this names the ones
# worth reasoning about rather than restating all of them.
_TOP_FEATURES: Final = 12


class EdaError(ValueError):
    """Raised when diagnostics are requested over inconsistent training inputs."""


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


def within_user_feature_diagnostics(
    *,
    feature_names: Sequence[str],
    feature_values: NDArray[np.float64],
    labels: Sequence[int],
    user_ids: Sequence[object],
) -> tuple[AggregateRecord, ...]:
    """Rank features by how much they can actually reorder a user's own impressions.

    For each feature this reports the share of its variance that is *within* user, and its
    within-user correlation with ``long_view``. A feature with a within-user variance share near
    zero is constant across a user's rows and provably cannot change that user's ranking, which is
    the single most useful thing the agent can know before proposing a feature change.
    """

    values = np.asarray(feature_values, dtype=np.float64)
    if values.ndim != 2:
        raise EdaError("feature values must be two-dimensional")
    if values.shape[1] != len(feature_names):
        raise EdaError("feature values and names disagree on width")
    if values.shape[0] != len(labels) or values.shape[0] != len(user_ids):
        raise EdaError("features, labels and user ids must have identical row counts")
    if values.shape[0] < _MIN_GROUP_ROWS:
        return ()

    codes, user_count = _user_codes(user_ids)
    counts = np.bincount(codes, minlength=user_count).astype(np.float64)
    target = np.asarray(labels, dtype=np.float64)

    target_user_sum = np.bincount(codes, weights=target, minlength=user_count)
    target_centred = target - (target_user_sum / counts)[codes]
    target_energy = float(target_centred @ target_centred)

    rows: list[tuple[str, float, float]] = []
    for index, name in enumerate(feature_names):
        column = values[:, index]
        total_centred = column - column.mean()
        total_energy = float(total_centred @ total_centred)
        user_sum = np.bincount(codes, weights=column, minlength=user_count)
        centred = column - (user_sum / counts)[codes]
        within_energy = float(centred @ centred)
        # Share of variance available for reordering rows inside a user's own slate.
        within_share = 0.0 if total_energy <= 0.0 else within_energy / total_energy
        if within_energy <= 0.0 or target_energy <= 0.0:
            correlation = 0.0
        else:
            correlation = float(centred @ target_centred) / float(
                np.sqrt(within_energy * target_energy)
            )
        rows.append((name, within_share, correlation))

    ranked = sorted(rows, key=lambda item: -abs(item[2]))
    inert = [item for item in rows if item[1] < 1e-9]
    records = [
        AggregateRecord(
            "train_within_user_feature_signal",
            {
                "explanation": (
                    "within_user_corr is the correlation with long_view after removing each "
                    "user's own mean, so it measures what can reorder that user's impressions; "
                    "within_user_variance_share near 0 means the feature is constant within a "
                    "user and cannot affect ranking at all"
                ),
                "features_scored": len(rows),
                "rows": int(values.shape[0]),
                "users": user_count,
            },
        )
    ]
    for rank, (name, share, correlation) in enumerate(ranked[:_TOP_FEATURES], start=1):
        records.append(
            AggregateRecord(
                f"train_feature_signal_{rank:02d}",
                {
                    "feature": name,
                    "within_user_corr": round(correlation, 6),
                    "within_user_variance_share": round(share, 6),
                },
            )
        )
    if inert:
        records.append(
            AggregateRecord(
                "train_features_inert_within_user",
                {
                    "count": len(inert),
                    "features_csv": ",".join(name for name, _share, _corr in inert[:24]),
                    "note": (
                        "constant within each user, so these cannot reorder a user's own "
                        "impressions and contribute nothing to GAUC or nDCG on their own"
                    ),
                },
            )
        )
    return tuple(records)


def within_user_label_structure(
    *,
    labels: Sequence[int],
    user_ids: Sequence[object],
) -> tuple[AggregateRecord, ...]:
    """Describe the slate structure that decides which users can move the metrics at all."""

    if len(labels) != len(user_ids):
        raise EdaError("labels and user ids must have identical row counts")
    if len(labels) < _MIN_GROUP_ROWS:
        return ()

    codes, user_count = _user_codes(user_ids)
    counts = np.bincount(codes, minlength=user_count).astype(np.int64)
    positives = np.bincount(
        codes, weights=np.asarray(labels, dtype=np.float64), minlength=user_count
    )
    mixed = np.count_nonzero((positives > 0) & (positives < counts))
    zero = np.count_nonzero(positives == 0)
    allpos = np.count_nonzero(positives == counts)
    slate = counts.astype(np.float64)
    return (
        AggregateRecord(
            "train_within_user_label_structure",
            {
                "users": user_count,
                "mixed_label_users": int(mixed),
                "mixed_label_user_share": round(float(mixed) / user_count, 6),
                "zero_positive_users": int(zero),
                "all_positive_users": int(allpos),
                "slate_size_mean": round(float(slate.mean()), 4),
                "slate_size_p50": float(np.quantile(slate, 0.50)),
                "slate_size_p90": float(np.quantile(slate, 0.90)),
                "note": (
                    "only mixed-label users can move GAUC, and only they have movable nDCG, so "
                    "every gain must come from reordering within this subset"
                ),
            },
        ),
    )

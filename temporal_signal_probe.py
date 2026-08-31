"""Is there within-user ordering signal in time that the causal bundle does not carry?

Every measured lever in this project shaves variance on an existing ordering. This is the only
proposal on the table that could find NEW signal, so it is worth an afternoon before any more
modelling effort.

The 66-column causal bundle summarises an identity's engagement HISTORY. It carries exactly one
temporal column, ``date_offset_from_20220408``, which is constant across a user's rows on a given
day and therefore contributes nothing to a within-user ranking. Nothing in it describes WHEN
inside a day an impression happened, or WHERE in a session it sat.

Both are derivable from columns the organizers ship and the causal cutoff permits, because both
are properties of the impression itself rather than of its outcome:

- hour of day, and its cyclic encodings, from ``hourmin``
- position within the user's session, and time since the previous impression, from ``time_ms``

What is measured here is the WITHIN-USER correlation with ``long_view``: a feature constant across
a user's rows cannot reorder them however well it correlates globally, so the global number would
flatter every candidate.

CHOOSE THE REFERENCE CAREFULLY -- the first version of this probe did not, and reported the wrong
verdict. Against ``duration_ms`` the temporal features look 5x stronger and the answer reads
"live". But ``duration_ms`` is a weak raw column, not what the bundle is made of. Against a
strictly-past smoothed per-video rate, which is the bundle's own construction, the same temporal
features are 6x WEAKER and the answer is "flat". A comparison is only as honest as its baseline.

Read only, training rows only, no outcome from the scored period. Repository root, outside the
hash_source_tree slice.

Run:  python3 temporal_signal_probe.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
#: A user needs at least this many impressions, and both labels, to contribute an ordering.
_MIN_ROWS = 4
#: Session boundary. Two impressions further apart than this are treated as separate sessions.
_SESSION_GAP_MS = 30 * 60 * 1000


def _within_user_correlation(rows: dict[str, list[tuple[float, float]]]) -> tuple[float, int]:
    """Mean per-user Pearson correlation of a feature with the label, and the user count.

    Averaging per user is the point: it is the same weighting the within-user metrics apply, and
    it discards any between-user variation, which cannot reorder a slate.
    """

    total = 0.0
    counted = 0
    for pairs in rows.values():
        if len(pairs) < _MIN_ROWS:
            continue
        features = [value for value, _ in pairs]
        labels = [label for _, label in pairs]
        if len(set(labels)) < 2 or len(set(features)) < 2:
            continue
        feature_mean = sum(features) / len(features)
        label_mean = sum(labels) / len(labels)
        covariance = sum((value - feature_mean) * (label - label_mean) for value, label in pairs)
        feature_ss = math.fsum((value - feature_mean) ** 2 for value in features)
        label_ss = math.fsum((label - label_mean) ** 2 for label in labels)
        denominator = math.sqrt(feature_ss * label_ss)
        if denominator == 0.0:
            continue
        total += covariance / denominator
        counted += 1
    return (total / counted if counted else 0.0), counted


def main(data_dir: str) -> int:
    path = Path(data_dir) / _TRAIN_LOG
    per_user: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hourmin = int(row["hourmin"])
            per_user[row["user_id"]].append(
                {
                    "label": float(row["long_view"]),
                    "hour": float(hourmin // 100),
                    "minute_of_day": float((hourmin // 100) * 60 + hourmin % 100),
                    "time_ms": float(row["time_ms"]),
                    "duration_ms": float(row["duration_ms"]),
                    "video": row["video_id"],
                }
            )

    features: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: defaultdict(list)
        for name in (
            "hour_of_day",
            "hour_sin",
            "hour_cos",
            "session_position",
            "log_gap_since_previous",
            "duration_ms (raw reference)",
            "video_past_rate (bundle reference)",
        )
    }
    # A strictly-past, smoothed per-video long_view rate: the same construction as the bundle's
    # own causal columns, so the comparison is against real bundle strength rather than a weak
    # raw column. Accumulated in global time order, so no row ever sees its own outcome.
    ordered = sorted(
        (
            (float(item["time_ms"]), user, item)
            for user, items in per_user.items()
            for item in items
        ),
        key=lambda entry: entry[0],
    )
    prior = sum(float(item["label"]) for _, _, item in ordered) / len(ordered)
    seen: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for _, _, item in ordered:
        exposure, positive = seen[str(item["video"])]
        item["video_past_rate"] = (positive + 20.0 * prior) / (exposure + 20.0)
        seen[str(item["video"])] = [exposure + 1.0, positive + float(item["label"])]
    for user, impressions in per_user.items():
        impressions.sort(key=lambda item: float(item["time_ms"]))
        position = 0
        previous = None
        for item in impressions:
            gap = None if previous is None else float(item["time_ms"]) - previous
            position = 0 if gap is None or gap > _SESSION_GAP_MS else position + 1
            previous = float(item["time_ms"])
            label = float(item["label"])
            angle = 2.0 * math.pi * float(item["minute_of_day"]) / 1440.0
            features["hour_of_day"][user].append((float(item["hour"]), label))
            features["hour_sin"][user].append((math.sin(angle), label))
            features["hour_cos"][user].append((math.cos(angle), label))
            features["session_position"][user].append((float(position), label))
            features["log_gap_since_previous"][user].append(
                (math.log1p(max(gap or 0.0, 0.0) / 1000.0), label)
            )
            features["duration_ms (raw reference)"][user].append(
                (float(item["duration_ms"]), label)
            )
            features["video_past_rate (bundle reference)"][user].append(
                (float(item["video_past_rate"]), label)
            )

    print(f"users {len(per_user):,}   impressions {sum(len(v) for v in per_user.values()):,}")
    print(f"within-user mean correlation with long_view, users with >= {_MIN_ROWS} mixed rows\n")
    print(f"    {'feature':30s} {'corr':>9s} {'|corr|':>9s} {'users':>9s}")
    results = {}
    for name, rows in features.items():
        correlation, counted = _within_user_correlation(rows)
        results[name] = correlation
        print(f"    {name:30s} {correlation:>+9.4f} {abs(correlation):>9.4f} {counted:>9,}")

    reference = abs(results["video_past_rate (bundle reference)"])
    best = max((abs(value), name) for name, value in results.items() if "reference" not in name)
    print()
    print(f"    strongest temporal feature: {best[1]} at |{best[0]:.4f}|")
    print(
        f"    a strictly-past video rate, the bundle's own construction, sits at |{reference:.4f}|"
    )
    print()
    if best[0] < 0.25 * reference:
        print("    VERDICT: flat. Temporal position carries little within-user ordering signal")
        print("    next to a column the bundle already has. The plateau finding gains its last")
        print("    pillar and no iteration should be spent here.")
    else:
        print("    VERDICT: live. A temporal feature is comparable to an existing bundle column,")
        print("    so it is new ordering signal rather than variance shaving, and it is the only")
        print("    candidate mechanism on the table that could clear epsilon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data"))

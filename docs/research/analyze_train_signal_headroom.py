"""Read-only diagnostics for score-improvement signals in the official train period.

This script deliberately opens only the April 8--21 standard training log.  It never reads
public-validation or final-period rows, and it writes no data artifacts.  The output is a compact
JSON summary suitable for the score-stagnation research report.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median

BINARY_AUXILIARIES = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)
SEQUENCE_THRESHOLDS = (1, 5, 10, 20)
SESSION_GAP_SECONDS = (60, 300, 1_800, 7_200, 86_400)


def _phi(n11: int, n10: int, n01: int, n00: int) -> float | None:
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return None if denominator == 0 else (n11 * n00 - n10 * n01) / denominator


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile input cannot be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze(data_dir: Path) -> dict[str, object]:
    train_path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    row_count = 0
    long_view_positive = 0
    auxiliary_counts = {
        name: {"n11": 0, "n10": 0, "n01": 0, "n00": 0} for name in BINARY_AUXILIARIES
    }
    tab_counts: dict[str, Counter[str]] = defaultdict(Counter)
    hour_counts: Counter[int] = Counter()
    hour_matches_utc_plus_8 = 0
    threshold_disagreements = 0
    progress_sum = 0.0
    progress_below_threshold_sum = 0.0
    progress_below_threshold_count = 0
    histories: dict[str, list[tuple[int, str, str, int, int]]] = defaultdict(list)

    with train_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            user_id = row["user_id"]
            video_id = row["video_id"]
            tab = row["tab"]
            timestamp_ms = int(row["time_ms"])
            label = int(row["long_view"])
            click = int(row["is_click"])
            duration_ms = float(row["duration_ms"])
            play_time_ms = float(row["play_time_ms"])
            hourmin = int(row["hourmin"])

            long_view_positive += label
            hour_counts[hourmin] += 1
            utc_plus_8 = datetime.fromtimestamp(timestamp_ms / 1_000.0, UTC) + timedelta(hours=8)
            hour_matches_utc_plus_8 += int(hourmin == utc_plus_8.hour * 100)

            threshold = max(min(duration_ms, 18_000.0), 1.0)
            progress = min(max(play_time_ms / threshold, 0.0), 2.0)
            progress_sum += progress
            threshold_disagreements += int((play_time_ms >= threshold) != bool(label))
            if label == 0:
                progress_below_threshold_sum += progress
                progress_below_threshold_count += 1

            for name in BINARY_AUXILIARIES:
                auxiliary = int(row[name])
                auxiliary_counts[name][f"n{label}{auxiliary}"] += 1

            tab_counts[tab]["rows"] += 1
            tab_counts[tab]["long_view"] += label
            tab_counts[tab]["click"] += click
            histories[user_id].append((timestamp_ms, video_id, tab, label, click))

    user_lengths = [len(events) for events in histories.values()]
    sequence_coverage = Counter()
    gap_coverage = Counter()
    same_video_seen_before = 0
    same_tab_as_previous = 0
    rows_with_previous = 0
    nonmonotonic_input_users = 0

    for events in histories.values():
        if any(right[0] < left[0] for left, right in itertools.pairwise(events)):
            nonmonotonic_input_users += 1
        events.sort(key=lambda event: event[0])
        seen_videos: set[str] = set()
        for index, event in enumerate(events):
            timestamp_ms, video_id, tab, _, _ = event
            for threshold in SEQUENCE_THRESHOLDS:
                sequence_coverage[threshold] += int(index >= threshold)
            if video_id in seen_videos:
                same_video_seen_before += 1
            seen_videos.add(video_id)
            if index == 0:
                continue
            rows_with_previous += 1
            previous = events[index - 1]
            gap_seconds = max((timestamp_ms - previous[0]) / 1_000.0, 0.0)
            for threshold in SESSION_GAP_SECONDS:
                gap_coverage[threshold] += int(gap_seconds <= threshold)
            same_tab_as_previous += int(tab == previous[2])

    auxiliary_summary: dict[str, object] = {}
    for name, counts in auxiliary_counts.items():
        n11 = counts["n11"]
        n10 = counts["n10"]
        n01 = counts["n01"]
        n00 = counts["n00"]
        auxiliary_positive = n11 + n01
        auxiliary_summary[name] = {
            "positive_rate": _rate(auxiliary_positive, row_count),
            "p_auxiliary_given_long_view": _rate(n11, n11 + n10),
            "p_long_view_given_auxiliary": _rate(n11, auxiliary_positive),
            "phi_with_long_view": _phi(n11, n10, n01, n00),
            "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
        }

    return {
        "scope": {
            "member": train_path.name,
            "date_window": "2022-04-08 through 2022-04-21",
            "public_validation_or_final_rows_read": False,
        },
        "population": {
            "rows": row_count,
            "users": len(histories),
            "long_view_positive_rate": long_view_positive / row_count,
        },
        "auxiliary_targets": auxiliary_summary,
        "watch_time": {
            "mean_clipped_threshold_progress": progress_sum / row_count,
            "mean_progress_among_long_view_negatives": (
                progress_below_threshold_sum / progress_below_threshold_count
            ),
            "long_view_threshold_disagreements": threshold_disagreements,
        },
        "sequence_headroom": {
            "events_per_user": {
                "minimum": min(user_lengths),
                "p25": _quantile(user_lengths, 0.25),
                "median": median(user_lengths),
                "p75": _quantile(user_lengths, 0.75),
                "p90": _quantile(user_lengths, 0.90),
                "maximum": max(user_lengths),
            },
            "row_fraction_with_at_least_n_prior_events": {
                str(threshold): sequence_coverage[threshold] / row_count
                for threshold in SEQUENCE_THRESHOLDS
            },
            "row_fraction_with_prior_event_within_seconds": {
                str(threshold): gap_coverage[threshold] / row_count
                for threshold in SESSION_GAP_SECONDS
            },
            "row_fraction_with_previously_seen_video": same_video_seen_before / row_count,
            "same_tab_as_previous_fraction": _rate(same_tab_as_previous, rows_with_previous),
            "users_not_monotonic_in_physical_input_order": nonmonotonic_input_users,
        },
        "time_semantics": {
            "unique_hourmin_values": sorted(hour_counts),
            "hourmin_matches_floor_hour_from_time_ms_at_utc_plus_8_fraction": (
                hour_matches_utc_plus_8 / row_count
            ),
        },
        "tab_slices": {
            tab: {
                "rows": counts["rows"],
                "row_share": counts["rows"] / row_count,
                "long_view_rate": counts["long_view"] / counts["rows"],
                "click_rate": counts["click"] / counts["rows"],
            }
            for tab, counts in sorted(tab_counts.items(), key=lambda item: int(item[0]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.data_dir), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

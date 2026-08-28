from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from kuairand_agent.scoring.submission import (
    HEADER,
    AlignmentRow,
    SubmissionError,
    compare_protected_metrics,
    compare_within_user_order,
    compare_within_user_top5,
    prediction_digest,
    read_submission,
    submission_digest,
    validate_alignment,
    validate_submission,
    within_user_order,
    write_submission,
)


def alignment() -> list[AlignmentRow]:
    return [
        AlignmentRow(0, "user-a", "duplicate-video"),
        AlignmentRow(1, "user-b", "video-b"),
        AlignmentRow(2, "user-a", "duplicate-video"),
        AlignmentRow(3, "user-a", "video-c"),
    ]


def test_high_precision_write_read_round_trip_and_digests(tmp_path: Path) -> None:
    rows = alignment()
    scores = np.array(
        [
            np.nextafter(0.123456, np.inf),
            -0.0,
            np.nextafter(0.123456, -np.inf),
            np.finfo(np.float64).tiny,
        ],
        dtype=np.float64,
    )
    path = tmp_path / "submission.csv"

    artifact = write_submission(path, rows, scores)
    checked = read_submission(path, rows)

    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(HEADER)
    assert artifact.row_count == len(rows)
    assert artifact.round_trip_identity is True
    assert artifact.within_user_order_preserved is True
    assert artifact.top5_preserved is True
    assert artifact.protected_metrics_preserved is None
    assert checked.scores.tobytes() == scores.tobytes()
    assert artifact.prediction_digest == checked.prediction_digest == prediction_digest(scores)
    assert artifact.submission_digest == submission_digest(path)
    assert artifact.submission_digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not checked.scores.flags.writeable


def test_writer_is_byte_deterministic_and_repr_round_trips_binary64(tmp_path: Path) -> None:
    rows = [AlignmentRow(index, "u", f"v{index}") for index in range(6)]
    scores = np.array(
        [
            np.finfo(np.float64).max,
            np.nextafter(1.0, 2.0),
            np.nextafter(1.0, 0.0),
            np.nextafter(0.0, 1.0),
            -0.0,
            -np.finfo(np.float64).max,
        ],
        dtype=np.float64,
    )
    first = write_submission(tmp_path / "first.csv", rows, scores)
    second = write_submission(tmp_path / "second.csv", rows, scores)

    assert first.submission_digest == second.submission_digest
    assert first.path.read_bytes() == second.path.read_bytes()
    assert read_submission(first.path, rows).scores.tobytes() == scores.tobytes()


def test_duplicate_user_video_pairs_remain_distinct_positional_rows(tmp_path: Path) -> None:
    rows = alignment()
    scores = [0.1, 0.2, 0.9, 0.3]
    path = tmp_path / "duplicates.csv"
    write_submission(path, rows, scores)

    records = path.read_text(encoding="utf-8").splitlines()
    assert records[1].split(",")[:3] == ["0", "user-a", "duplicate-video"]
    assert records[3].split(",")[:3] == ["2", "user-a", "duplicate-video"]
    assert within_user_order(rows, scores)["user-a"] == (2, 3, 0)


def test_audited_six_significant_digit_tie_regression_is_prevented(tmp_path: Path) -> None:
    rows = [AlignmentRow(index, "one-user", f"video-{index}") for index in range(6)]
    # Rows 4 and 5 straddle rank five.  The later row is genuinely larger, but the immutable
    # starter writer's .6g formatting turns both into 0.123456; stable tie handling then admits
    # physical row 4 instead and changes nDCG whenever their labels differ.
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1234564, 0.12345649], dtype=np.float64)
    rounded = np.array([float(f"{score:.6g}") for score in scores], dtype=np.float64)

    assert rounded[4] == rounded[5]
    assert compare_within_user_order(rows, scores, rounded) is False
    assert compare_within_user_top5(rows, scores, rounded) is False

    artifact = write_submission(tmp_path / "precise.csv", rows, scores)
    assert artifact.round_trip_identity is True
    assert artifact.top5_preserved is True
    assert compare_within_user_top5(rows, scores, artifact.scores) is True


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("row_id,user_id,video_id,score\n", "truncated"),
        (
            "row_id,user_id,video_id,score\n"
            "0,user-a,duplicate-video,0.1\n"
            "1,user-b,video-b,0.2\n"
            "2,user-a,duplicate-video,0.3\n",
            "truncated",
        ),
        (
            "row_id,user_id,video_id,score\n"
            "0,user-a,duplicate-video,0.1\n"
            "1,user-b,video-b,0.2\n"
            "2,user-a,duplicate-video,0.3\n"
            "3,user-a,video-c,0.4\n"
            "4,user-x,video-x,0.5\n",
            "extra row",
        ),
        (
            "user_id,row_id,video_id,score\nuser-a,0,duplicate-video,0.1\n",
            "header",
        ),
        (
            "row_id,user_id,video_id,score\n00,user-a,duplicate-video,0.1\n",
            "canonical '0'",
        ),
        (
            "row_id,user_id,video_id,score\n+0,user-a,duplicate-video,0.1\n",
            "canonical '0'",
        ),
        (
            "row_id,user_id,video_id,score\n 0,user-a,duplicate-video,0.1\n",
            "canonical '0'",
        ),
        (
            "row_id,user_id,video_id,score\n0,user-a,duplicate-video,nan\n",
            "finite",
        ),
        (
            "row_id,user_id,video_id,score\n0,user-a,duplicate-video,inf\n",
            "finite",
        ),
        (
            "row_id,user_id,video_id,score\n0,user-a,duplicate-video,-Infinity\n",
            "finite",
        ),
        (
            "row_id,user_id,video_id,score\n0,user-a,duplicate-video,nope\n",
            "not a float64",
        ),
        (
            "row_id,user_id,video_id,score\n0,user-a,duplicate-video,0.1,unexpected\n",
            "5 fields",
        ),
        (
            "row_id,user_id,video_id,score\n"
            "0,user-a,duplicate-video,0.1\n"
            "1,user-b,video-b,0.2\n"
            "2,user-a,other-video,0.3\n",
            "alignment mismatch",
        ),
    ],
)
def test_malformed_submission_is_rejected(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(SubmissionError, match=message):
        read_submission(path, alignment())


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([0.1, 0.2], "count mismatch"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], "count mismatch"),
        ([0.1, 0.2, np.nan, 0.4], "finite"),
        ([0.1, np.inf, 0.3, 0.4], "finite"),
        ([0.1, "0.2", 0.3, 0.4], "real number"),
        ([0.1, True, 0.3, 0.4], "real number"),
    ],
)
def test_writer_rejects_bad_prediction_vectors(
    tmp_path: Path, scores: list[object], message: str
) -> None:
    with pytest.raises(SubmissionError, match=message):
        write_submission(tmp_path / "bad.csv", alignment(), scores)
    assert not (tmp_path / "bad.csv").exists()


def test_writer_refuses_overwrite_and_preserves_existing_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "submission.csv"
    destination.write_bytes(b"user-owned-existing-submission\n")

    with pytest.raises(SubmissionError, match="refusing to overwrite"):
        write_submission(destination, alignment(), [0.1, 0.2, 0.3, 0.4])

    assert destination.read_bytes() == b"user-owned-existing-submission\n"
    assert not tuple(tmp_path.glob(".submission.csv.*.tmp"))


def test_writer_refuses_existing_symlink_without_changing_target(tmp_path: Path) -> None:
    target = tmp_path / "target.csv"
    target.write_bytes(b"external-target\n")
    destination = tmp_path / "submission.csv"
    destination.symlink_to(target)

    with pytest.raises(SubmissionError, match="refusing to overwrite"):
        write_submission(destination, alignment(), [0.1, 0.2, 0.3, 0.4])

    assert destination.is_symlink()
    assert target.read_bytes() == b"external-target\n"


def test_non_contiguous_or_invalid_alignment_is_rejected() -> None:
    with pytest.raises(SubmissionError, match="contiguous"):
        validate_alignment([AlignmentRow(0, "u", "v"), AlignmentRow(2, "u", "v")])
    with pytest.raises(SubmissionError, match="at least one"):
        validate_alignment([])
    with pytest.raises(SubmissionError, match="built-in int"):
        AlignmentRow(True, "u", "v")
    with pytest.raises(SubmissionError, match="user_id"):
        AlignmentRow(0, "", "v")


def test_protected_metric_parity_is_optional_and_checked(tmp_path: Path) -> None:
    rows = alignment()
    scores = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float64)
    calls = 0

    def metric_evaluator(values: NDArray[np.float64]) -> Mapping[str, float]:
        nonlocal calls
        calls += 1
        assert not values.flags.writeable
        return {
            "GAUC": float(values[2] > values[0]),
            "nDCG@5": float(np.mean(values)),
            "primary": (float(values[2] > values[0]) + float(np.mean(values))) / 2,
            "runtime_seconds": float(calls),  # deliberately ignored
        }

    artifact = write_submission(
        tmp_path / "metric-checked.csv",
        rows,
        scores,
        protected_metric_evaluator=metric_evaluator,
    )
    assert artifact.protected_metrics_preserved is True
    # Staged and final read-back each compare before/after.
    assert calls == 4
    assert compare_protected_metrics(scores, scores.copy(), metric_evaluator)


def test_validation_rejects_changed_order_and_metrics(tmp_path: Path) -> None:
    rows = alignment()
    reference = np.array([0.9, 0.2, 0.1, 0.3], dtype=np.float64)
    altered = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float64)
    path = tmp_path / "altered.csv"
    write_submission(path, rows, altered)

    with pytest.raises(SubmissionError, match="ordering or top-five"):
        validate_submission(path, rows, reference_scores=reference)

    def metrics(values: NDArray[np.float64]) -> Mapping[str, float]:
        return {"GAUC": float(values[0]), "nDCG@5": 0.5, "primary": 0.5}

    assert not compare_protected_metrics(reference, altered, metrics)


def test_csv_user_and_video_alignment_is_exact_string_equality(tmp_path: Path) -> None:
    rows = [AlignmentRow(0, "001", "0007")]
    path = tmp_path / "numeric-looking-ids.csv"
    path.write_text("row_id,user_id,video_id,score\n0,1,7,0.5\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="alignment mismatch"):
        read_submission(path, rows)

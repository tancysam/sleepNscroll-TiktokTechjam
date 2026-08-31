from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kuairand_agent.data.canonical import (
    APPROVED_AUXILIARY_TARGETS,
    BINARY_AUXILIARY_TARGETS,
    LOG_HEADER,
    OUTCOME_FIELDS,
    PRIMARY_TARGET,
    STANDARD_LOG_FILENAMES,
    VIDEO_BASIC_FILENAME,
    VIDEO_BASIC_HEADER,
    CanonicalDataError,
    CanonicalFinalSplit,
    CanonicalInputs,
    ProtectedTargets,
    TrainingTargets,
    load_canonical_dataset,
)


def test_canonical_v1_identity_remains_compatible_with_feature_extensions() -> None:
    normal = CanonicalInputs(
        user_id=("1", "2"),
        video_id=("10", "20"),
        date=(20220408, 20220409),
        duration_ms=(10_000.0, 20_000.0),
        tab=("0", "1"),
        author_id=("100", "200"),
        time_ms=(1, 2),
        video_type=("NORMAL", "NORMAL"),
    )
    extended = CanonicalInputs(
        user_id=("1", "2"),
        video_id=("10", "20"),
        date=(20220408, 20220409),
        duration_ms=(10_000.0, 20_000.0),
        tab=("0", "1"),
        author_id=("100", "200"),
        time_ms=(1, 2),
        video_type=("NORMAL", "AD"),
    )

    assert normal.video_type != extended.video_type
    assert normal.digest == extended.digest


def _log_row(**overrides: str) -> list[str]:
    values = {
        "user_id": "001",
        "video_id": "010",
        "date": "20220408",
        "hourmin": "400",
        "time_ms": "1650000000000",
        "is_click": "1",
        "is_like": "0",
        "is_follow": "0",
        "is_comment": "0",
        "is_forward": "0",
        "is_hate": "0",
        "long_view": "1",
        "play_time_ms": "20000",
        "duration_ms": "30000",
        "profile_stay_time": "0",
        "comment_stay_time": "0",
        "is_profile_enter": "0",
        "is_rand": "0",
        "tab": "01",
    }
    values.update(overrides)
    return [values[name] for name in LOG_HEADER]


def _video_row(video_id: str, author_id: str) -> list[str]:
    values = {
        "video_id": video_id,
        "author_id": author_id,
        "video_type": "NORMAL",
        "upload_dt": "2020-01-01",
        "upload_type": "ShortImport",
        "visible_status": "1",
        "video_duration": "30000.0",
        "server_width": "720",
        "server_height": "1280",
        "music_id": "123",
        "music_type": "4",
        "tag": "12,65",
    }
    return [values[name] for name in VIDEO_BASIC_HEADER]


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_dataset(
    root: Path,
    *,
    first_rows: list[list[str]],
    second_rows: list[list[str]],
    videos: list[list[str]] | None = None,
) -> None:
    root.mkdir()
    _write_csv(
        root / VIDEO_BASIC_FILENAME,
        VIDEO_BASIC_HEADER,
        videos if videos is not None else [_video_row("010", "090")],
    )
    _write_csv(root / STANDARD_LOG_FILENAMES[0], LOG_HEADER, first_rows)
    _write_csv(root / STANDARD_LOG_FILENAMES[1], LOG_HEADER, second_rows)


def _poison_outcomes(prefix: str) -> dict[str, str]:
    return {name: f"{prefix}-{name}" for name in OUTCOME_FIELDS}


def test_canonical_order_duplicate_pairs_alignment_and_phase_target_boundary(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    first = [
        _log_row(date="20220408", time_ms="1"),
        _log_row(date="20220421", time_ms="2", long_view="0"),
    ]
    valid_aux_poison = {name: f"must-not-parse-valid-{name}" for name in APPROVED_AUXILIARY_TARGETS}
    second = [
        _log_row(
            user_id="002",
            video_id="099",
            date="20220422",
            time_ms="3",
            long_view="1",
            **valid_aux_poison,
        ),
        _log_row(date="20220429", time_ms="4", **_poison_outcomes("final-poison")),
        # A synthetic late occurrence proves file order, not sorting, is canonical identity.
        _log_row(user_id="003", date="20220408", time_ms="5", long_view="0"),
    ]
    _write_dataset(data, first_rows=first, second_rows=second)

    dataset = load_canonical_dataset(data)

    assert dataset.train.alignment.row_id == (0, 1, 2)
    assert dataset.train.inputs.user_id == ("001", "001", "003")
    assert dataset.train.inputs.date == (20220408, 20220421, 20220408)
    assert dataset.train.inputs.tab == ("01", "01", "01")
    assert dataset.train.alignment.user_id == dataset.train.inputs.user_id
    assert dataset.train.alignment.video_id == ("010", "010", "010")
    assert dataset.train.alignment.row_id[:2] == (0, 1)  # repeated pair remains two rows
    assert dataset.train.inputs.author_id == ("090", "090", "090")
    assert dataset.train.inputs.video_type == ("NORMAL", "NORMAL", "NORMAL")

    assert isinstance(dataset.train.targets, TrainingTargets)
    assert dataset.train.targets.long_view == (1, 0, 0)
    assert isinstance(dataset.valid.targets, ProtectedTargets)
    assert dataset.valid.targets.reveal_for_scorer() == (1,)
    assert dataset.valid.inputs.author_id == ("UNK",)
    assert dataset.valid.inputs.video_type == ("UNKNOWN",)
    assert isinstance(dataset.final, CanonicalFinalSplit)
    assert dataset.test is dataset.final

    assert set(dataset.train.targets.column_names) == {PRIMARY_TARGET, *APPROVED_AUXILIARY_TARGETS}
    assert dataset.valid.outcome_trace.parsed_fields == (PRIMARY_TARGET,)
    assert dataset.valid.outcome_trace.skipped_cell_count == len(APPROVED_AUXILIARY_TARGETS)
    assert dataset.final.outcome_trace.parsed_fields == ()
    assert dataset.final.outcome_trace.skipped_cell_count == len(OUTCOME_FIELDS)
    assert dataset.final.outcome_trace.manifest()["skipped_values_recorded"] is False
    assert not hasattr(dataset.final, "final_targets")
    assert set(dataset.final.inputs.field_names).isdisjoint(OUTCOME_FIELDS)


def test_final_outcome_bytes_are_not_decoded_and_semantic_digests_are_invariant(
    tmp_path: Path,
) -> None:
    first_rows = [_log_row(date="20220408")]
    valid_values = _poison_outcomes("valid-skipped")
    valid_values[PRIMARY_TARGET] = "0"
    valid = _log_row(date="20220422", **valid_values)
    # Restore the only public-validation outcome that is legally parsed.
    valid[LOG_HEADER.index(PRIMARY_TARGET)] = "0"
    final_a = _log_row(date="20220429", **_poison_outcomes("FINAL_A_SENTINEL"))
    final_b = _log_row(date="20220429", **_poison_outcomes("FINAL_B_SENTINEL"))
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(left, first_rows=first_rows, second_rows=[valid, final_a])
    _write_dataset(right, first_rows=first_rows, second_rows=[valid, final_b])

    # Make a skipped final-outcome token invalid UTF-8.  The binary selected-cell projection must
    # still load because it never decodes that field.
    right_log = right / STANDARD_LOG_FILENAMES[1]
    raw = right_log.read_bytes()
    raw = raw.replace(b"FINAL_B_SENTINEL-is_click", b"\xff\xfe")
    right_log.write_bytes(raw)

    first = load_canonical_dataset(left)
    second = load_canonical_dataset(right)

    assert first.final.inputs == second.final.inputs
    assert first.final.alignment == second.final.alignment
    assert first.final.digest == second.final.digest
    assert first.digest == second.digest
    assert first.manifest() == second.manifest()
    rendered = repr(second.final) + repr(second.final.outcome_trace.manifest())
    assert "FINAL_A_SENTINEL" not in rendered
    assert "FINAL_B_SENTINEL" not in rendered


@pytest.mark.parametrize(
    ("split_date", "bad_field"),
    [
        ("20220408", "is_click"),
        ("20220408", "long_view"),
        ("20220422", "long_view"),
    ],
)
def test_malformed_accessible_targets_fail_closed(
    tmp_path: Path, split_date: str, bad_field: str
) -> None:
    data = tmp_path / f"data-{split_date}-{bad_field}"
    row = _log_row(date=split_date, **{bad_field: "poison-secret-value"})
    first = [row] if split_date <= "20220421" else []
    second = [] if first else [row]
    _write_dataset(data, first_rows=first, second_rows=second)

    with pytest.raises(CanonicalDataError, match=bad_field) as raised:
        load_canonical_dataset(data)
    assert "poison-secret-value" not in str(raised.value)


def test_public_auxiliary_poison_is_skipped_but_primary_is_protected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    auxiliary_poison = {name: "not-even-utf8-semantics" for name in APPROVED_AUXILIARY_TARGETS}
    _write_dataset(
        data,
        first_rows=[_log_row(date="20220408")],
        second_rows=[_log_row(date="20220422", long_view="1", **auxiliary_poison)],
    )
    dataset = load_canonical_dataset(data)
    assert isinstance(dataset.valid.targets, ProtectedTargets)
    assert dataset.valid.targets.reveal_for_scorer() == (1,)


def test_exact_headers_and_unique_author_mapping_fail_closed(tmp_path: Path) -> None:
    wrong_header = tmp_path / "wrong-header"
    _write_dataset(
        wrong_header,
        first_rows=[_log_row()],
        second_rows=[],
    )
    path = wrong_header / STANDARD_LOG_FILENAMES[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("user_id,video_id", "video_id,user_id")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CanonicalDataError, match="header mismatch"):
        load_canonical_dataset(wrong_header)

    duplicate_author = tmp_path / "duplicate-author"
    _write_dataset(
        duplicate_author,
        first_rows=[_log_row()],
        second_rows=[],
        videos=[_video_row("010", "090"), _video_row("010", "091")],
    )
    with pytest.raises(CanonicalDataError, match="duplicates video_id"):
        load_canonical_dataset(duplicate_author)


def test_two_clean_builds_are_logically_identical_and_safe_input_change_is_not(
    tmp_path: Path,
) -> None:
    left = tmp_path / "one" / "data"
    right = tmp_path / "two" / "data"
    changed = tmp_path / "three" / "data"
    left.parent.mkdir()
    right.parent.mkdir()
    changed.parent.mkdir()
    rows = [_log_row(date="20220408")]
    second = [_log_row(date="20220422", long_view="0"), _log_row(date="20220429")]
    _write_dataset(left, first_rows=rows, second_rows=second)
    _write_dataset(right, first_rows=rows, second_rows=second)
    _write_dataset(
        changed,
        first_rows=rows,
        second_rows=[_log_row(date="20220422", long_view="0"), _log_row(date="20220429", tab="2")],
    )

    first = load_canonical_dataset(left)
    replay = load_canonical_dataset(right)
    safe_change = load_canonical_dataset(changed)
    assert first.manifest() == replay.manifest()
    assert first.digest == replay.digest
    assert first.final.digest != safe_change.final.digest


def test_published_headers_are_frozen_in_exact_archive_order() -> None:
    assert LOG_HEADER == (
        "user_id",
        "video_id",
        "date",
        "hourmin",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
        "is_rand",
        "tab",
    )
    assert VIDEO_BASIC_HEADER == (
        "video_id",
        "author_id",
        "video_type",
        "upload_dt",
        "upload_type",
        "visible_status",
        "video_duration",
        "server_width",
        "server_height",
        "music_id",
        "music_type",
        "tag",
    )
    assert BINARY_AUXILIARY_TARGETS == (
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "is_profile_enter",
    )

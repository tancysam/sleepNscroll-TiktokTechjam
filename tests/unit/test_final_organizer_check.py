from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kuairand_agent.contract import verify_starter_kit
from kuairand_agent.data.canonical import LOG_HEADER, OUTCOME_FIELDS, VIDEO_BASIC_HEADER
from kuairand_agent.finalization.organizer_check import (
    OrganizerCheckError,
    check_final_submission,
)

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


def _log_row(
    user_id: bytes,
    video_id: bytes,
    date: int,
    *,
    outcome_token: bytes = b"0",
) -> bytes:
    cells = [b"0"] * len(LOG_HEADER)
    values = {
        "user_id": user_id,
        "video_id": video_id,
        "date": str(date).encode("ascii"),
        "hourmin": b"1200",
        "time_ms": b"1",
        "duration_ms": b"1000",
        "is_rand": b"0",
        "tab": b"1",
    }
    for name, value in values.items():
        cells[LOG_HEADER.index(name)] = value
    for name in OUTCOME_FIELDS:
        cells[LOG_HEADER.index(name)] = outcome_token
    return b",".join(cells) + b"\n"


def _video_row(video_id: str) -> str:
    values = {
        "video_id": video_id,
        "author_id": f"author-{video_id}",
        "video_type": "NORMAL",
        "upload_dt": "2022-01-01",
        "upload_type": "1",
        "visible_status": "1",
        "video_duration": "1000",
        "server_width": "720",
        "server_height": "1280",
        "music_id": "music",
        "music_type": "1",
        "tag": "tag",
    }
    return ",".join(values[name] for name in VIDEO_BASIC_HEADER)


def _make_data_view(root: Path, final_outcome_token: bytes) -> Path:
    data = root / "data"
    data.mkdir(parents=True)
    header = ",".join(LOG_HEADER).encode("ascii") + b"\n"
    (data / "log_standard_4_08_to_4_21_pure.csv").write_bytes(
        header + _log_row(b"train", b"video-train", 20220408)
    )
    (data / "log_standard_4_22_to_5_08_pure.csv").write_bytes(
        header
        + _log_row(b"valid", b"video-valid", 20220422)
        + _log_row(
            b"test-a",
            b"video-test-a",
            20220429,
            outcome_token=final_outcome_token,
        )
        + _log_row(
            b"test-b",
            b"video-test-b",
            20220508,
            outcome_token=final_outcome_token,
        )
    )
    videos = ["video-train", "video-valid", "video-test-a", "video-test-b"]
    (data / "video_features_basic_pure.csv").write_text(
        ",".join(VIDEO_BASIC_HEADER)
        + "\n"
        + "\n".join(_video_row(video_id) for video_id in videos)
        + "\n",
        encoding="utf-8",
    )
    return data


def _submission(path: Path, *, valid: bool = True) -> Path:
    if valid:
        body = (
            "row_id,user_id,video_id,score\n"
            "0,test-a,video-test-a,0.12345678901234566\n"
            "1,test-b,video-test-b,0.9876543210987654\n"
        )
    else:
        body = "row_id,user_id,video_id,score\n0,wrong,alignment,0.5\n"
    path.write_text(body, encoding="utf-8")
    return path


def test_untouched_checker_runs_check_only_against_private_masked_view(tmp_path: Path) -> None:
    data = _make_data_view(tmp_path / "dataset", b"\xff\x80")
    submission = _submission(tmp_path / "submission.csv")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    starter_before = verify_starter_kit(STARTER)

    evidence = check_final_submission(
        submission,
        data_dir=data,
        starter_dir=STARTER,
        scratch_dir=scratch,
    )

    assert evidence.checker_returncode == 0
    assert evidence.checker_command[-1] == "--check"
    assert "--score" not in evidence.checker_command
    assert evidence.masked_view.final_rows_masked == 2
    assert evidence.masked_view.final_outcome_cells_replaced == 2 * len(OUTCOME_FIELDS)
    assert "split=test" in evidence.checker_stdout
    assert evidence.checker_stderr == ""
    assert evidence.submission_sha256 == hashlib.sha256(submission.read_bytes()).hexdigest()
    assert evidence.starter_manifest_sha256 == starter_before.manifest_sha256
    assert verify_starter_kit(STARTER).manifest_sha256 == starter_before.manifest_sha256
    assert tuple(scratch.iterdir()) == ()


def test_relative_starter_path_is_bound_before_private_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _make_data_view(tmp_path / "dataset", b"\xff")
    submission = _submission(tmp_path / "submission.csv")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(ROOT)

    evidence = check_final_submission(
        submission,
        data_dir=data,
        starter_dir=Path("kuairand-starter-kit"),
        scratch_dir=scratch,
    )

    assert evidence.checker_returncode == 0
    assert tuple(scratch.iterdir()) == ()


def test_checker_evidence_is_independent_of_opaque_final_outcome_bytes(tmp_path: Path) -> None:
    first_data = _make_data_view(tmp_path / "first", b"\xff")
    second_data = _make_data_view(tmp_path / "second", b"\x80\x81\xfe")
    submission = _submission(tmp_path / "submission.csv")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    first = check_final_submission(
        submission,
        data_dir=first_data,
        starter_dir=STARTER,
        scratch_dir=scratch,
    )
    second = check_final_submission(
        submission,
        data_dir=second_data,
        starter_dir=STARTER,
        scratch_dir=scratch,
    )

    assert first == second
    trace = first.masked_view.manifest()["final_outcome_isolation"]
    assert isinstance(trace, dict)
    assert trace["outcome_cells_sliced"] == 0
    assert trace["outcome_cells_decoded"] == 0
    assert trace["outcome_cells_hashed"] == 0
    assert trace["outcome_cells_scored"] == 0
    assert tuple(scratch.iterdir()) == ()


def test_checker_rejection_is_bounded_and_private_view_is_cleaned(tmp_path: Path) -> None:
    data = _make_data_view(tmp_path / "dataset", b"\xff")
    submission = _submission(tmp_path / "bad.csv", valid=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(OrganizerCheckError, match="organizer checker rejected"):
        check_final_submission(
            submission,
            data_dir=data,
            starter_dir=STARTER,
            scratch_dir=scratch,
        )

    assert tuple(scratch.iterdir()) == ()
    verify_starter_kit(STARTER)


def test_masking_fault_cleans_private_view_before_checker_launch(tmp_path: Path) -> None:
    data = _make_data_view(tmp_path / "dataset", b"\xff")
    late = data / "log_standard_4_22_to_5_08_pure.csv"
    late.write_bytes(late.read_bytes() + b"not,enough,fields\n")
    submission = _submission(tmp_path / "submission.csv")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(OrganizerCheckError, match="wrong number of CSV fields"):
        check_final_submission(
            submission,
            data_dir=data,
            starter_dir=STARTER,
            scratch_dir=scratch,
        )

    assert tuple(scratch.iterdir()) == ()


def test_checker_rejects_symlinked_required_data_file(tmp_path: Path) -> None:
    data = _make_data_view(tmp_path / "dataset", b"\xff")
    basic = data / "video_features_basic_pure.csv"
    real = data / "basic-real.csv"
    basic.rename(real)
    basic.symlink_to(real)
    submission = _submission(tmp_path / "submission.csv")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(OrganizerCheckError, match="regular non-symlink"):
        check_final_submission(
            submission,
            data_dir=data,
            starter_dir=STARTER,
            scratch_dir=scratch,
        )

    assert tuple(scratch.iterdir()) == ()

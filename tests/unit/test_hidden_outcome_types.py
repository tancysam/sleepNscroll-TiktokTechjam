from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import (
    LOG_HEADER,
    OUTCOME_FIELDS,
    STANDARD_LOG_FILENAMES,
    VIDEO_BASIC_FILENAME,
    VIDEO_BASIC_HEADER,
    CanonicalAlignment,
    CanonicalDataset,
    CanonicalFinalSplit,
    CanonicalInputs,
    CanonicalSplit,
    CanonicalTrainingSplit,
    CanonicalValidationSplit,
    OutcomeAccessTrace,
    TrainingTargets,
    load_canonical_dataset,
)
from kuairand_agent.data.capabilities import DataPhase, build_final_inputs
from kuairand_agent.evaluation.protected import ProtectedLabels, ProtectedResult


def _inputs(*, date: int) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=("1",),
        video_id=("10",),
        date=(date,),
        duration_ms=(1_000.0,),
        tab=("0",),
        author_id=("100",),
        time_ms=(1,),
    )


def _alignment(name: SplitName, inputs: CanonicalInputs) -> CanonicalAlignment:
    return CanonicalAlignment(
        split=name,
        row_id=(0,),
        user_id=inputs.user_id,
        video_id=inputs.video_id,
    )


def _trace(name: SplitName, parsed: tuple[str, ...]) -> OutcomeAccessTrace:
    return OutcomeAccessTrace(
        split=name,
        row_count=1,
        parsed_fields=parsed,
        skipped_fields=tuple(field for field in OUTCOME_FIELDS if field not in parsed),
    )


def _training_targets() -> TrainingTargets:
    auxiliary = tuple(field for field in OUTCOME_FIELDS if field != "long_view")
    return TrainingTargets(
        {
            "long_view": (1,),
            **{name: (0.0,) for name in auxiliary},
        }
    )


def test_phase_specific_splits_make_final_outcomes_structurally_unrepresentable() -> None:
    train_inputs = _inputs(date=20220408)
    valid_inputs = _inputs(date=20220422)
    final_inputs = _inputs(date=20220429)
    train = CanonicalTrainingSplit(
        name=SplitName.TRAIN,
        inputs=train_inputs,
        alignment=_alignment(SplitName.TRAIN, train_inputs),
        outcome_trace=_trace(SplitName.TRAIN, OUTCOME_FIELDS),
        targets=_training_targets(),
    )
    valid = CanonicalValidationSplit(
        name=SplitName.VALID,
        inputs=valid_inputs,
        alignment=_alignment(SplitName.VALID, valid_inputs),
        outcome_trace=_trace(SplitName.VALID, ("long_view",)),
        targets=ProtectedLabels((0,)),
    )
    final = CanonicalFinalSplit(
        name=SplitName.TEST,
        inputs=final_inputs,
        alignment=_alignment(SplitName.TEST, final_inputs),
        outcome_trace=_trace(SplitName.TEST, ()),
    )

    assert {field.name for field in dataclasses.fields(final)} == {
        "name",
        "inputs",
        "alignment",
        "outcome_trace",
        "digest",
    }
    assert not hasattr(final, "targets")
    assert "target_access" not in final.manifest()
    assert "target_digest" not in final.manifest()
    dataset = CanonicalDataset(train, valid, final, "a" * 64)
    assert isinstance(dataset.train, CanonicalTrainingSplit)
    assert isinstance(dataset.valid, CanonicalValidationSplit)
    assert isinstance(dataset.final, CanonicalFinalSplit)


def test_legacy_split_adapter_erases_final_target_shape_at_dataset_seam() -> None:
    final_inputs = _inputs(date=20220429)
    legacy = CanonicalSplit(
        name=SplitName.TEST,
        inputs=final_inputs,
        alignment=_alignment(SplitName.TEST, final_inputs),
        targets=None,
        outcome_trace=_trace(SplitName.TEST, ()),
    )
    typed = legacy.to_phase_split()

    assert isinstance(typed, CanonicalFinalSplit)
    assert not hasattr(typed, "targets")
    final_capability = build_final_inputs(typed)
    assert final_capability.phase is DataPhase.FINAL
    assert tuple(final_capability.columns) == (
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_ms",
    )


def _row(*, date: str, outcome: str) -> list[str]:
    values = {
        "user_id": "1",
        "video_id": "10",
        "date": date,
        "hourmin": "400",
        "time_ms": "1",
        "is_click": outcome,
        "is_like": outcome,
        "is_follow": outcome,
        "is_comment": outcome,
        "is_forward": outcome,
        "is_hate": outcome,
        "long_view": outcome,
        "play_time_ms": outcome,
        "duration_ms": "1000",
        "profile_stay_time": outcome,
        "comment_stay_time": outcome,
        "is_profile_enter": outcome,
        "is_rand": "0",
        "tab": "0",
    }
    return [values[name] for name in LOG_HEADER]


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_final_outcome_bytes_stay_undecoded_and_out_of_the_schema(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    video = {
        "video_id": "10",
        "author_id": "100",
        "video_type": "NORMAL",
        "upload_dt": "2020-01-01",
        "upload_type": "ShortImport",
        "visible_status": "1",
        "video_duration": "1000",
        "server_width": "720",
        "server_height": "1280",
        "music_id": "1",
        "music_type": "1",
        "tag": "1",
    }
    _write_csv(
        data / VIDEO_BASIC_FILENAME,
        VIDEO_BASIC_HEADER,
        [[video[name] for name in VIDEO_BASIC_HEADER]],
    )
    _write_csv(data / STANDARD_LOG_FILENAMES[0], LOG_HEADER, [_row(date="20220408", outcome="1")])
    late = data / STANDARD_LOG_FILENAMES[1]
    _write_csv(late, LOG_HEADER, [_row(date="20220429", outcome="FINAL_SECRET")])
    late.write_bytes(late.read_bytes().replace(b"FINAL_SECRET", b"\xff\xfe"))

    dataset = load_canonical_dataset(data)

    assert isinstance(dataset.final, CanonicalFinalSplit)
    assert not hasattr(dataset.final, "targets")
    assert dataset.final.outcome_trace.parsed_fields == ()
    assert dataset.final.outcome_trace.skipped_cell_count == len(OUTCOME_FIELDS)
    assert "FINAL_SECRET" not in repr(dataset.final)


def test_protected_types_expose_only_aggregate_evidence_across_evaluation_seam() -> None:
    labels = ProtectedLabels((0, 1))
    result = ProtectedResult(
        gauc=0.6,
        ndcg_at_5=0.8,
        primary=0.7,
        users=1,
        rows=2,
        scorer_digest="a" * 64,
        prediction_digest="b" * 64,
        runtime_seconds=0.1,
    )

    assert labels.__class__.__module__ == "kuairand_agent.evaluation.protected"
    assert result.__class__.__module__ == "kuairand_agent.evaluation.protected"
    assert not hasattr(labels, "column")
    assert not hasattr(labels, "values")
    assert not hasattr(result, "labels")
    assert not hasattr(result, "targets")
    assert set(result.as_dict()) == {
        "GAUC",
        "nDCG@5",
        "primary",
        "users",
        "rows",
        "scorer_digest",
        "prediction_digest",
        "runtime_seconds",
    }

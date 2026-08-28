from __future__ import annotations

import copy
import json
from typing import cast

import pytest

from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import (
    APPROVED_AUXILIARY_TARGETS,
    BINARY_AUXILIARY_TARGETS,
    PRIMARY_TARGET,
    CanonicalAlignment,
    CanonicalInputs,
    CanonicalSplit,
    OutcomeAccessTrace,
    ProtectedTargets,
    TrainingTargets,
)
from kuairand_agent.data.folds import (
    FOLD_A_SPEC,
    FOLD_B_SPEC,
    TemporalFoldError,
    TemporalFoldSet,
    build_temporal_folds,
)


def _training_split(
    *,
    labels: tuple[int, ...] | None = None,
    omitted_dates: frozenset[int] = frozenset(),
) -> CanonicalSplit:
    dates: list[int] = []
    for date in range(20220408, 20220422):
        if date not in omitted_dates:
            dates.extend((date, date))
    row_count = len(dates)
    actual_labels = labels if labels is not None else tuple(index % 2 for index in range(row_count))
    if len(actual_labels) != row_count:
        raise AssertionError("fixture label count differs from fixture rows")
    inputs = CanonicalInputs(
        user_id=("001",) * row_count,
        video_id=tuple(f"{index:03d}" for index in range(row_count)),
        date=tuple(dates),
        duration_ms=(30_000.0,) * row_count,
        tab=("1",) * row_count,
        author_id=("090",) * row_count,
        time_ms=tuple(range(1, row_count + 1)),
    )
    alignment = CanonicalAlignment(
        split=SplitName.TRAIN,
        row_id=tuple(range(row_count)),
        user_id=inputs.user_id,
        video_id=inputs.video_id,
    )
    target_columns: dict[str, tuple[int | float, ...]] = {PRIMARY_TARGET: actual_labels}
    for name in APPROVED_AUXILIARY_TARGETS:
        if name in BINARY_AUXILIARY_TARGETS:
            target_columns[name] = actual_labels
        else:
            target_columns[name] = tuple(float(value) for value in actual_labels)
    targets = TrainingTargets(target_columns)
    return CanonicalSplit(
        name=SplitName.TRAIN,
        inputs=inputs,
        alignment=alignment,
        targets=targets,
        outcome_trace=OutcomeAccessTrace(
            split=SplitName.TRAIN,
            row_count=row_count,
            parsed_fields=(PRIMARY_TARGET, *APPROVED_AUXILIARY_TARGETS),
            skipped_fields=(),
        ),
    )


def test_exact_frozen_a_b_date_windows_and_canonical_positions() -> None:
    folds = build_temporal_folds(_training_split())

    assert FOLD_A_SPEC.train_dates == tuple(range(20220408, 20220416))
    assert FOLD_A_SPEC.valid_dates == (20220416, 20220417, 20220418)
    assert FOLD_B_SPEC.train_dates == tuple(range(20220408, 20220419))
    assert FOLD_B_SPEC.valid_dates == (20220419, 20220420, 20220421)
    assert folds.fold_a.train_positions == tuple(range(16))
    assert folds.fold_a.valid_positions == tuple(range(16, 22))
    assert folds.fold_b.train_positions == tuple(range(22))
    assert folds.fold_b.valid_positions == tuple(range(22, 28))
    assert folds.fold_a.train_indices == folds.fold_a.train_positions
    assert folds.fold_b.valid_indices == folds.fold_b.valid_positions


def test_fold_positions_and_digests_are_target_independent() -> None:
    original = _training_split()
    assert isinstance(original.targets, TrainingTargets)
    changed_labels = tuple(1 - value for value in original.targets.long_view)
    changed = _training_split(labels=changed_labels)

    first = build_temporal_folds(original)
    second = build_temporal_folds(changed)
    assert original.targets is not None
    assert changed.targets is not None
    assert original.targets.digest != changed.targets.digest
    assert original.digest != changed.digest
    assert first.manifest() == second.manifest()
    assert first.digest == second.digest


def test_restart_round_trip_preserves_exact_manifest_and_digest() -> None:
    original = build_temporal_folds(_training_split())
    persisted = json.loads(json.dumps(original.manifest(), sort_keys=True))
    restored = TemporalFoldSet.from_manifest(persisted)
    assert restored == original
    assert restored.manifest() == original.manifest()
    assert restored.digest == original.digest

    tampered = copy.deepcopy(persisted)
    tampered["folds"][0]["valid_positions"].append(27)
    with pytest.raises(TemporalFoldError, match=r"digest mismatch|overlap"):
        TemporalFoldSet.from_manifest(tampered)


def test_empty_window_fails_but_zero_row_boundary_date_keeps_frozen_spec() -> None:
    # The official archive has zero retained standard-log rows on 2022-04-08.  The interval is
    # still frozen at that inclusive date; it does not imply artificial per-day row support.
    boundary_gap = build_temporal_folds(_training_split(omitted_dates=frozenset({20220408})))
    assert boundary_gap.fold_a.spec.train_start == 20220408

    with pytest.raises(TemporalFoldError, match="valid_positions cannot be empty"):
        build_temporal_folds(
            _training_split(omitted_dates=frozenset({20220416, 20220417, 20220418}))
        )


def test_strict_mixed_label_positive_and_slate_support_fails_closed() -> None:
    unsupported = _training_split(labels=(0,) * 28)
    with pytest.raises(TemporalFoldError, match=r"support.*positives"):
        build_temporal_folds(unsupported)

    # Position construction itself remains date-only and is available for deterministic audit.
    folds = build_temporal_folds(unsupported, validate_support=False)
    assert folds.fold_a.valid_positions == tuple(range(16, 22))
    assert folds.fold_b.valid_positions == tuple(range(22, 28))


def test_non_train_split_and_manifest_boundary_drift_are_rejected() -> None:
    train = _training_split()
    assert isinstance(train.targets, TrainingTargets)
    wrong_name = CanonicalSplit(
        name=SplitName.VALID,
        inputs=train.inputs,
        alignment=CanonicalAlignment(
            split=SplitName.VALID,
            row_id=train.alignment.row_id,
            user_id=train.alignment.user_id,
            video_id=train.alignment.video_id,
        ),
        targets=ProtectedTargets(train.targets.long_view),
        outcome_trace=OutcomeAccessTrace(
            split=SplitName.VALID,
            row_count=train.row_count,
            parsed_fields=(PRIMARY_TARGET,),
            skipped_fields=APPROVED_AUXILIARY_TARGETS,
        ),
    )
    with pytest.raises(TemporalFoldError, match="train split"):
        build_temporal_folds(wrong_name)

    manifest = build_temporal_folds(train).manifest()
    changed = copy.deepcopy(manifest)
    changed_folds = cast(list[dict[str, object]], changed["folds"])
    changed_spec = cast(dict[str, object], changed_folds[0]["spec"])
    changed_valid = cast(dict[str, object], changed_spec["valid"])
    changed_valid["start"] = 20220415
    with pytest.raises(TemporalFoldError, match=r"ordered|frozen"):
        TemporalFoldSet.from_manifest(changed)

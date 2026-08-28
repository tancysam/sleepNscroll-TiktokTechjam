from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import overload

import numpy as np
import pytest

from kuairand_agent.baselines.fold_controls import (
    FoldControlError,
    PrimaryTrainingTargets,
    build_fold_scoring_context,
    run_fold_fm_control,
)
from kuairand_agent.baselines.organizer import load_verified_organizer
from kuairand_agent.data.canonical import CanonicalInputs

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


def _inputs(prefix: str, dates: tuple[int, ...], *, start_time: int = 0) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u{index // 2}" for index in range(len(dates))),
        video_id=tuple(f"{prefix}-v{index % 5}" for index in range(len(dates))),
        date=dates,
        duration_ms=tuple(float(800 + index * 101) for index in range(len(dates))),
        tab=tuple(str(index % 3) for index in range(len(dates))),
        author_id=tuple(f"a{index % 4}" for index in range(len(dates))),
        time_ms=tuple(start_time + index for index in range(len(dates))),
    )


def _fold_a() -> tuple[CanonicalInputs, tuple[int, ...], CanonicalInputs, tuple[int, ...]]:
    prefix = _inputs("prefix", tuple(20220408 + index % 8 for index in range(16)))
    query = _inputs(
        "query", (20220416, 20220416, 20220417, 20220417, 20220418, 20220418), start_time=100
    )
    return prefix, tuple(index % 2 for index in range(16)), query, (1, 0, 1, 0, 0, 1)


class _ExplodingLabels(Sequence[object]):
    """A sentinel proving invalid phases fail before any target access."""

    def __len__(self) -> int:
        raise AssertionError("labels were inspected")

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        raise AssertionError(f"labels[{index}] was inspected")


def _raw_rows(inputs: CanonicalInputs, labels: tuple[int, ...]) -> list[list[object]]:
    return [
        [
            inputs.date[index],
            inputs.user_id[index],
            inputs.video_id[index],
            inputs.author_id[index],
            inputs.tab[index],
            inputs.duration_ms[index],
            labels[index],
        ]
        for index in range(len(inputs))
    ]


def test_primary_targets_are_read_only_and_bound_to_exact_canonical_inputs() -> None:
    prefix, labels, _, _ = _fold_a()
    targets = PrimaryTrainingTargets.bind(prefix, labels)
    numpy_targets = PrimaryTrainingTargets.from_inputs(prefix, np.asarray(labels, dtype=np.int8))
    changed = _inputs("changed", tuple(prefix.date), start_time=10_000)
    changed_targets = PrimaryTrainingTargets(changed, labels)

    assert targets.row_count == len(prefix)
    assert targets.training_inputs_digest == prefix.digest
    assert targets.primary.dtype == np.dtype("int8")
    assert not targets.primary.flags.writeable
    assert targets.digest != changed_targets.digest
    assert numpy_targets.digest == targets.digest
    assert set(targets.manifest()) == {
        "schema_version",
        "row_count",
        "training_inputs_digest",
        "digest",
        "target",
        "dtype",
    }
    assert "primary" not in targets.manifest()
    with pytest.raises(ValueError):
        targets.primary[0] = 0


@pytest.mark.parametrize(
    "labels",
    [
        (1, 0),
        tuple([1, 0] * 7 + [1, 2]),
        tuple([1, 0] * 7 + [1, True]),
        tuple([1, 0] * 7 + [1, 0.0]),
        None,
    ],
)
def test_primary_targets_reject_malformed_or_absent_labels(labels: object) -> None:
    prefix, _, _, _ = _fold_a()
    with pytest.raises(FoldControlError):
        PrimaryTrainingTargets(prefix, labels)  # type: ignore[arg-type]


def test_fold_context_exposes_bound_scoring_not_labels_or_alignment() -> None:
    _, _, query, labels = _fold_a()
    context = build_fold_scoring_context(STARTER, "A", "a" * 64, query, labels)
    scores = np.linspace(-1.0, 1.0, len(query), dtype=np.float64)

    integer_result = context.score(scores)
    encoded_result = context.score_with_encoded_labels(scores)

    assert context.fold_name == "A"
    assert context.split_name == "inner_fold_A"
    assert context.fold_token == "a" * 64
    assert context.validation_inputs_digest == query.digest
    assert integer_result.rows == encoded_result.rows == len(query)
    assert not hasattr(context, "labels")
    assert not hasattr(context, "_labels")
    assert not hasattr(context, "alignment")
    assert not hasattr(context, "scorer")
    assert context.manifest()["labels_exposed"] is False


def test_fold_context_positive_affine_score_transform_preserves_metrics_only() -> None:
    _, _, query, labels = _fold_a()
    context = build_fold_scoring_context(STARTER, "A", "b" * 64, query, labels)
    scores = np.asarray((0.1, 0.4, -0.2, 0.7, -0.5, 0.9), dtype=np.float64)

    original = context.score_with_encoded_labels(scores)
    transformed = context.score_with_encoded_labels(scores * 13.0 - 4.0)

    assert transformed.gauc == original.gauc
    assert transformed.ndcg_at_5 == original.ndcg_at_5
    assert transformed.primary == original.primary
    assert transformed.prediction_digest != original.prediction_digest


@pytest.mark.parametrize(
    ("fold_name", "dates", "token"),
    [
        ("outer_valid", (20220422, 20220422), "a" * 64),
        ("final", (20220429, 20220429), "a" * 64),
        ("A", (20220422, 20220422), "a" * 64),
        ("B", (20220429, 20220429), "a" * 64),
        ("A", (20220416, 20220416), "not-a-token"),
    ],
)
def test_fold_context_rejects_non_train_roles_before_inspecting_labels(
    fold_name: str,
    dates: tuple[int, ...],
    token: str,
) -> None:
    query = _inputs("forbidden", dates)
    with pytest.raises(FoldControlError):
        build_fold_scoring_context(STARTER, fold_name, token, query, _ExplodingLabels())


def test_fold_context_rejects_final_like_absent_outcomes() -> None:
    _, _, query, _ = _fold_a()
    with pytest.raises(FoldControlError, match="labels are absent"):
        build_fold_scoring_context(STARTER, "A", "c" * 64, query, None)


def test_fold_fm_control_is_deterministic_and_exactly_replayable() -> None:
    prefix, prefix_labels, query, query_labels = _fold_a()

    first = run_fold_fm_control(
        prefix,
        prefix_labels,
        query,
        query_labels,
        STARTER,
        seed=2,
    )
    second = run_fold_fm_control(
        prefix,
        prefix_labels,
        query,
        query_labels,
        STARTER,
        seed=2,
    )
    replay = first.replay_predictions(starter_dir=STARTER, query_inputs=query)

    assert first.fold_name == "A"
    assert first.prefix_inputs_digest == prefix.digest
    assert first.query_inputs_digest == query.digest
    assert first.checkpoint.digest == second.checkpoint.digest
    assert first.predictions.digest == second.predictions.digest == replay.digest
    assert first.predictions.scores.tobytes() == replay.scores.tobytes()
    assert first.training.logical_digest == second.training.logical_digest
    assert first.digest == second.digest
    assert first.resources.device == "cpu"
    assert first.resources.precision == "float32"


def test_fold_fm_control_accepts_persisted_fold_identity_for_fold_b() -> None:
    prefix = _inputs("b-prefix", tuple(20220408 + index % 11 for index in range(22)))
    query = _inputs(
        "b-query", (20220419, 20220419, 20220420, 20220420, 20220421, 20220421), start_time=100
    )
    control = run_fold_fm_control(
        prefix,
        tuple(index % 2 for index in range(len(prefix))),
        query,
        (1, 0, 0, 1, 1, 0),
        STARTER,
        fold_name="B",
        fold_token="d" * 64,
    )

    assert control.fold_name == "B"
    assert control.fold_token == "d" * 64
    assert control.aggregate_metrics.primary == float(
        (np.float32(control.aggregate_metrics.gauc) + np.float32(control.aggregate_metrics.ndcg5))
        / 2.0
    )


def test_fold_fm_control_matches_untouched_organizer_run_fm() -> None:
    prefix, prefix_labels, query, query_labels = _fold_a()
    adapted = run_fold_fm_control(
        prefix,
        prefix_labels,
        query,
        query_labels,
        STARTER,
        seed=0,
    )
    organizer = load_verified_organizer(STARTER)
    rows = {
        "train": _raw_rows(prefix, prefix_labels),
        "valid": _raw_rows(query, query_labels),
        # The untouched function requires a nonempty test entry.  Reusing this synthetic
        # train-derived fold does not touch canonical public-validation or final outcomes.
        "test": _raw_rows(query, query_labels),
    }
    expected = organizer.baseline.run_fm(rows, seed=0, verbose=False)["valid"]

    assert adapted.aggregate_metrics.gauc == expected["GAUC"]
    assert adapted.aggregate_metrics.ndcg_at_5 == expected["nDCG@5"]
    assert adapted.aggregate_metrics.primary == expected["primary"]


def test_fold_fm_control_rejects_public_or_final_dates_before_any_label_access() -> None:
    prefix = _inputs("prefix", (20220408, 20220408))
    public_query = _inputs("public", (20220422, 20220422), start_time=100)
    final_query = _inputs("final", (20220429, 20220429), start_time=200)

    for forbidden in (public_query, final_query):
        with pytest.raises(FoldControlError, match="not contained"):
            run_fold_fm_control(
                prefix,
                _ExplodingLabels(),
                forbidden,
                _ExplodingLabels(),
                STARTER,
            )


def test_fold_fm_control_rejects_mixed_or_mislabeled_fold_roles() -> None:
    prefix, prefix_labels, query, query_labels = _fold_a()
    mixed = _inputs("mixed", (20220418, 20220419))

    with pytest.raises(FoldControlError, match="not contained"):
        run_fold_fm_control(prefix, prefix_labels, mixed, query_labels[:2], STARTER)
    with pytest.raises(FoldControlError, match="query dates"):
        run_fold_fm_control(
            prefix,
            prefix_labels,
            query,
            query_labels,
            STARTER,
            fold_name="B",
            fold_token="e" * 64,
        )
    with pytest.raises(FoldControlError, match="supplied together"):
        run_fold_fm_control(
            prefix,
            prefix_labels,
            query,
            query_labels,
            STARTER,
            fold_name="A",
        )


def test_query_label_mutation_cannot_change_target_free_fold_identity() -> None:
    prefix, prefix_labels, query, query_labels = _fold_a()
    mutated = tuple(1 - label for label in query_labels)

    first = run_fold_fm_control(prefix, prefix_labels, query, query_labels, STARTER)
    second = run_fold_fm_control(prefix, prefix_labels, query, mutated, STARTER)

    assert first.fold_token == second.fold_token
    assert first.query_inputs_digest == second.query_inputs_digest
    assert first.query_alignment_digest == second.query_alignment_digest
    assert first.encoding.digest == second.encoding.digest
    # Query labels can legitimately alter inner early stopping, but never the input/fold identity.
    assert first.training.training_targets_digest == second.training.training_targets_digest

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.candidates.observed_pair_objectives import (
    DURATION_SECONDS_POSITION,
    ObservedPairObjectiveError,
    prepare_duration_pair_ablation,
)
from kuairand_agent.candidates.pairwise import GAUCPairSampler
from kuairand_agent.data.capabilities import DataPhase

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "candidate_seed"))

from candidate_seed.reference_observed_pair_objectives import (  # noqa: E402
    prepare_reference_duration_pair_ablation,
)


def _features(durations: list[float]) -> np.ndarray:
    values = np.zeros((len(durations), DURATION_SECONDS_POSITION + 1), dtype="<f8")
    values[:, DURATION_SECONDS_POSITION] = durations
    return values


def test_duration_ablation_preserves_exact_control_and_replaces_half_within_bucket() -> None:
    groups = np.asarray([10, 10, 10, 10, 20, 20, 20, 20], dtype=np.int64)
    targets = np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    features = _features([4.0, 12.0, 3.0, 11.0, 20.0, 61.0, 19.0, 65.0])

    batch = prepare_duration_pair_ablation(
        features,
        groups,
        targets,
        pair_count=128,
        seed=17,
        phase=DataPhase.INNER_TRAIN,
    )
    exact_control = GAUCPairSampler(
        groups,
        targets,
        phase=DataPhase.INNER_TRAIN,
    ).sample(128, seed=17)

    np.testing.assert_array_equal(batch.control_positive_indices, exact_control.positive_indices)
    np.testing.assert_array_equal(batch.control_negative_indices, exact_control.negative_indices)
    assert batch.intervention_mask.tolist() == ([False, True] * 64)
    assert batch.intervention_pair_count == 64
    assert batch.conditioned_eligible_group_count == 4
    assert batch.conditioned_eligible_positive_count == 4
    np.testing.assert_array_equal(
        batch.treatment_positive_indices[~batch.intervention_mask],
        batch.control_positive_indices[~batch.intervention_mask],
    )
    np.testing.assert_array_equal(
        batch.treatment_negative_indices[~batch.intervention_mask],
        batch.control_negative_indices[~batch.intervention_mask],
    )
    conditioned_positive = batch.treatment_positive_indices[batch.intervention_mask]
    conditioned_negative = batch.treatment_negative_indices[batch.intervention_mask]
    np.testing.assert_array_equal(targets[conditioned_positive], np.ones(64, dtype=np.int8))
    np.testing.assert_array_equal(targets[conditioned_negative], np.zeros(64, dtype=np.int8))
    np.testing.assert_array_equal(groups[conditioned_positive], groups[conditioned_negative])
    duration_codes = np.searchsorted(
        np.asarray([5.0, 10.0, 18.0, 30.0, 60.0]),
        features[:, DURATION_SECONDS_POSITION],
        side="right",
    )
    np.testing.assert_array_equal(
        duration_codes[conditioned_positive],
        duration_codes[conditioned_negative],
    )
    for array in (
        batch.control_positive_indices,
        batch.control_negative_indices,
        batch.treatment_positive_indices,
        batch.treatment_negative_indices,
        batch.intervention_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_duration_bucket_boundaries_are_exact_and_seed_replay_is_stable() -> None:
    durations = [0.0, 5.0, 10.0, 18.0, 30.0, 60.0]
    groups = np.repeat(np.arange(len(durations), dtype=np.int64), 2)
    targets = np.tile(np.asarray([1, 0], dtype=np.int8), len(durations))
    features = _features([value for duration in durations for value in (duration, duration)])

    first = prepare_duration_pair_ablation(
        features,
        groups,
        targets,
        pair_count=240,
        seed=91,
        phase=DataPhase.TRAIN,
    )
    replay = prepare_duration_pair_ablation(
        features.copy(order="C"),
        groups.copy(),
        targets.copy(),
        pair_count=240,
        seed=91,
        phase=DataPhase.TRAIN,
    )
    changed_seed = prepare_duration_pair_ablation(
        features,
        groups,
        targets,
        pair_count=240,
        seed=92,
        phase=DataPhase.TRAIN,
    )

    for name in (
        "control_positive_indices",
        "control_negative_indices",
        "treatment_positive_indices",
        "treatment_negative_indices",
        "intervention_mask",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(replay, name))
    assert first.treatment_seed == replay.treatment_seed
    assert first.treatment_seed != first.seed
    assert bool(
        np.logical_or(
            first.treatment_positive_indices != changed_seed.treatment_positive_indices,
            first.treatment_negative_indices != changed_seed.treatment_negative_indices,
        ).any()
    )
    conditioned = first.intervention_mask
    np.testing.assert_array_equal(
        groups[first.treatment_positive_indices[conditioned]],
        groups[first.treatment_negative_indices[conditioned]],
    )


def test_duration_ablation_fails_closed_for_nontraining_or_unusable_inputs() -> None:
    with pytest.raises(ObservedPairObjectiveError, match="train or inner_train"):
        prepare_duration_pair_ablation(
            object(),
            np.asarray([1, 1]),
            np.asarray([1, 0]),
            pair_count=2,
            seed=0,
            phase=DataPhase.OUTER_VALID,
        )

    groups = np.asarray([1, 1, 1, 1], dtype=np.int64)
    targets = np.asarray([1, 1, 0, 0], dtype=np.int8)
    with pytest.raises(ObservedPairObjectiveError, match="same-user bucket"):
        prepare_duration_pair_ablation(
            _features([4.0, 4.0, 20.0, 20.0]),
            groups,
            targets,
            pair_count=32,
            seed=0,
            phase=DataPhase.INNER_TRAIN,
        )

    for invalid_count in (0, 3, 1_000_002, True):
        with pytest.raises(ObservedPairObjectiveError, match="pair_count"):
            prepare_duration_pair_ablation(
                _features([4.0, 4.0]),
                np.asarray([1, 1]),
                np.asarray([1, 0]),
                pair_count=invalid_count,
                seed=0,
                phase=DataPhase.TRAIN,
            )


@pytest.mark.parametrize(
    ("features", "groups", "targets", "seed", "match"),
    (
        (
            np.zeros((2, DURATION_SECONDS_POSITION + 1), dtype=np.float32),
            np.asarray([1, 1]),
            np.asarray([1, 0]),
            0,
            "little-endian float64",
        ),
        (
            _features([-1.0, -1.0]),
            np.asarray([1, 1]),
            np.asarray([1, 0]),
            0,
            "non-negative",
        ),
        (
            _features([4.0, 4.0]),
            np.asarray([True, True]),
            np.asarray([1, 0]),
            0,
            "non-boolean numeric",
        ),
        (
            _features([4.0, 4.0]),
            np.asarray([1, 1]),
            np.asarray([1]),
            0,
            "aligned",
        ),
        (
            _features([4.0, 4.0]),
            np.asarray([1, 1]),
            np.asarray([1, 0]),
            True,
            "unsigned 32-bit",
        ),
    ),
)
def test_duration_ablation_rejects_malformed_numeric_inputs(
    features: np.ndarray,
    groups: np.ndarray,
    targets: np.ndarray,
    seed: object,
    match: str,
) -> None:
    with pytest.raises(ObservedPairObjectiveError, match=match):
        prepare_duration_pair_ablation(
            features,
            groups,
            targets,
            pair_count=32,
            seed=seed,  # type: ignore[arg-type]
            phase=DataPhase.TRAIN,
        )


def test_candidate_protected_mirror_matches_controller_pair_bytes() -> None:
    groups = np.asarray([8, 8, 8, 8, 9, 9, 9, 9], dtype=np.int64)
    targets = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    features = _features([4.0, 3.0, 22.0, 25.0, 10.0, 12.0, 61.0, 65.0])

    controller = prepare_duration_pair_ablation(
        features,
        groups,
        targets,
        pair_count=512,
        seed=20260830,
        phase=DataPhase.INNER_TRAIN,
    )
    candidate = prepare_reference_duration_pair_ablation(
        features,
        groups,
        targets,
        pair_count=512,
        seed=20260830,
    )

    for name in (
        "control_positive_indices",
        "control_negative_indices",
        "treatment_positive_indices",
        "treatment_negative_indices",
        "intervention_mask",
    ):
        np.testing.assert_array_equal(getattr(candidate, name), getattr(controller, name))
    assert candidate.seed == controller.seed
    assert candidate.treatment_seed == controller.treatment_seed
    assert candidate.conditioned_eligible_group_count == controller.conditioned_eligible_group_count
    assert (
        candidate.conditioned_eligible_positive_count
        == controller.conditioned_eligible_positive_count
    )


def test_conditioned_branch_matches_independent_positive_ticket_oracle() -> None:
    groups = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    targets = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.int8)
    features = _features([4.0, 3.0, 2.0, 20.0, 22.0, 25.0])

    batch = prepare_duration_pair_ablation(
        features,
        groups,
        targets,
        pair_count=200_000,
        seed=73,
        phase=DataPhase.TRAIN,
    )
    pairs, counts = np.unique(
        np.column_stack(
            (
                batch.treatment_positive_indices[batch.intervention_mask],
                batch.treatment_negative_indices[batch.intervention_mask],
            )
        ),
        axis=0,
        return_counts=True,
    )
    observed = {
        (int(pair[0]), int(pair[1])): int(count) / batch.intervention_pair_count
        for pair, count in zip(pairs, counts, strict=True)
    }

    assert set(observed) == {(0, 2), (1, 2), (3, 4), (3, 5)}
    assert observed[(0, 2)] == pytest.approx(1.0 / 3.0, abs=0.005)
    assert observed[(1, 2)] == pytest.approx(1.0 / 3.0, abs=0.005)
    assert observed[(3, 4)] == pytest.approx(1.0 / 6.0, abs=0.005)
    assert observed[(3, 5)] == pytest.approx(1.0 / 6.0, abs=0.005)

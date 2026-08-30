from __future__ import annotations

import hashlib

import numpy as np

from candidate_seed.reference_pairwise_fm import (
    REFERENCE_FEATURE_POSITIONS,
    _fit_reference_pairwise_fm,
    reference_pairwise_fm_scores,
    sample_reference_logged_pairs,
)
from kuairand_agent.candidates.pairwise_fm import (
    EncodedFMInputs,
    PairwiseFMAdapter,
    PairwiseFMConfig,
    PairwiseFMTrainingData,
)
from kuairand_agent.data.capabilities import DataPhase


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_protected_logged_pair_sampler_is_deterministic_aligned_and_bounded() -> None:
    user_groups = np.asarray([4, 4, 4, 9, 9, 12, 12, 12, 12], dtype=np.int64)
    targets = np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=np.float64)

    positive, negative = sample_reference_logged_pairs(
        user_groups,
        targets,
        pair_count=4096,
        seed=17,
    )
    repeated_positive, repeated_negative = sample_reference_logged_pairs(
        user_groups,
        targets,
        pair_count=4096,
        seed=17,
    )
    other_positive, other_negative = sample_reference_logged_pairs(
        user_groups,
        targets,
        pair_count=4096,
        seed=18,
    )

    np.testing.assert_array_equal(positive, repeated_positive)
    np.testing.assert_array_equal(negative, repeated_negative)
    np.testing.assert_array_equal(targets[positive], np.ones(positive.size))
    np.testing.assert_array_equal(targets[negative], np.zeros(negative.size))
    np.testing.assert_array_equal(user_groups[positive], user_groups[negative])
    assert positive.dtype == np.dtype("<i8")
    assert negative.dtype == np.dtype("<i8")
    assert bool(np.logical_or(positive != other_positive, negative != other_negative).any())


def test_protected_candidate_primitive_matches_controller_reference_bytes() -> None:
    user_groups = np.repeat(np.arange(12, dtype=np.int64), 6)
    within_user = np.tile(np.arange(6, dtype=np.int32), 12)
    targets = np.ascontiguousarray(within_user >= 3, dtype=np.int8)
    encoded = np.column_stack(
        (
            user_groups.astype(np.int32),
            np.int32(13) + (within_user % 5),
            np.int32(20) + (user_groups.astype(np.int32) % 7),
            np.int32(28) + (within_user % 3),
            np.int32(32) + (within_user % 6),
        )
    ).astype(np.int32)
    total_dim = int(np.max(encoded)) + 2
    features = np.zeros((encoded.shape[0], 82), dtype="<f8")
    features[:, REFERENCE_FEATURE_POSITIONS] = encoded

    candidate_state = _fit_reference_pairwise_fm(
        features,
        targets.astype(np.float64),
        user_groups,
        pairs_per_epoch=8192,
        epochs=1,
    )
    encoding_digest = _digest(b"candidate-reference-encoding")
    train_inputs_digest = _digest(b"candidate-reference-train-inputs")
    training_targets_digest = _digest(b"candidate-reference-targets")
    controller_inputs = EncodedFMInputs(
        values=np.ascontiguousarray(encoded),
        phase=DataPhase.INNER_TRAIN,
        inputs_digest=train_inputs_digest,
        encoding_digest=encoding_digest,
        total_dim=total_dim,
    )
    controller_training = PairwiseFMTrainingData(
        inputs=controller_inputs,
        labels=targets,
        user_ids=tuple(f"user-{value}" for value in user_groups),
        training_targets_digest=training_targets_digest,
        target_inputs_digest=train_inputs_digest,
    )
    controller = PairwiseFMAdapter(
        source_digest=_digest(b"candidate-reference-source"),
        config=PairwiseFMConfig(
            seed=20260830,
            learning_rate=0.001,
            l2=0.000001,
            pair_batch_size=8192,
            pairs_per_epoch=8192,
            max_epochs=1,
        ),
    )
    controller_run = controller.fit(controller_training)

    np.testing.assert_array_equal(
        candidate_state["reference_factors"],
        controller_run.checkpoint.V,
    )
    np.testing.assert_array_equal(
        candidate_state["reference_linear"],
        controller_run.checkpoint.W,
    )
    assert float(candidate_state["reference_final_pairwise_loss"]) == (
        controller_run.trace[-1].mean_pairwise_loss
    )

    candidate_scores = reference_pairwise_fm_scores(features, candidate_state)
    query_inputs = EncodedFMInputs(
        values=np.ascontiguousarray(encoded),
        phase=DataPhase.INNER_VALID,
        inputs_digest=_digest(b"candidate-reference-query-inputs"),
        encoding_digest=encoding_digest,
        total_dim=total_dim,
    )
    controller_scores = controller.predict(controller_run.checkpoint, query_inputs).scores
    np.testing.assert_array_equal(candidate_scores, controller_scores)

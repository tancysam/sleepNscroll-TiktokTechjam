from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.organizer import load_verified_organizer
from kuairand_agent.candidates.pairwise_fm import (
    EncodedFMInputs,
    PairwiseFMAdapter,
    PairwiseFMConfig,
    PairwiseFMError,
    PairwiseFMRun,
    PairwiseFMTrainingData,
    initialize_pairwise_fm,
    pairwise_fm_batch_loss_and_gradients,
    pairwise_fm_scores,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.capabilities import DataPhase

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
SOURCE_DIGEST = "a" * 64


def _inputs(prefix: str, rows: int, *, start_time: int = 0) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u{index // 2}" for index in range(rows)),
        video_id=tuple(f"{prefix}-v{index}" for index in range(rows)),
        date=tuple(20220408 for _ in range(rows)),
        duration_ms=tuple(float(1000 + index * 137) for index in range(rows)),
        tab=tuple(str(index % 2) for index in range(rows)),
        author_id=tuple(f"a{index % 3}" for index in range(rows)),
        time_ms=tuple(start_time + index for index in range(rows)),
    )


@dataclass(frozen=True)
class _Targets:
    phase: DataPhase
    primary: npt.NDArray[np.int8]
    training_inputs_digest: str
    digest: str = "d" * 64

    @property
    def row_count(self) -> int:
        return int(self.primary.size)


class _ForbiddenTargets:
    phase = DataPhase.OUTER_VALID
    training_inputs_digest = "0" * 64
    digest = "d" * 64
    row_count = 2
    inspected = False

    def __array__(self, *_: object, **__: object) -> npt.NDArray[np.int8]:
        type(self).inspected = True
        raise AssertionError("non-training outcomes must not be inspected")


def _fit(
    *,
    seed: int = 7,
    epochs: int = 2,
    pair_batch_size: int = 4,
    pairs_per_epoch: int = 6,
    device_metadata: str | None = None,
) -> tuple[PairwiseFMRun, StarterEncoding, CanonicalInputs]:
    train = _inputs("train", 12)
    encoding = StarterEncoding.fit(train)
    targets = _Targets(
        DataPhase.INNER_TRAIN,
        np.asarray([1, 0] * 6, dtype=np.int8),
        train.digest,
    )
    encoded = EncodedFMInputs.from_encoding(
        encoding,
        train,
        phase=DataPhase.INNER_TRAIN,
    )
    training = PairwiseFMTrainingData(
        inputs=encoded,
        labels=targets.primary,
        user_ids=train.user_id,
        training_targets_digest=targets.digest,
        target_inputs_digest=targets.training_inputs_digest,
    )
    run = PairwiseFMAdapter(
        source_digest=SOURCE_DIGEST,
        config=PairwiseFMConfig(
            seed=seed,
            max_epochs=epochs,
            pair_batch_size=pair_batch_size,
            pairs_per_epoch=pairs_per_epoch,
            device_metadata=device_metadata,
        ),
    ).fit(training)
    return run, encoding, train


def test_pairwise_fm_batch_loss_and_dense_gradients_match_worked_golden() -> None:
    positive = np.asarray([[0, 2, 4, 6, 8]], dtype=np.int32)
    negative = np.asarray([[0, 3, 4, 6, 8]], dtype=np.int32)
    factors = np.zeros((9, 16), dtype=np.float32)
    factors[0, 0] = np.float32(0.2)
    factors[2, 0] = np.float32(0.3)
    factors[3, 0] = np.float32(-0.4)
    linear = np.zeros(9, dtype=np.float32)

    result = pairwise_fm_batch_loss_and_gradients(
        positive,
        negative,
        V=factors,
        W=linear,
        l2=0.0,
    )

    # Scores are accumulated with the pointwise-control float32 FM arithmetic.
    assert result.loss == pytest.approx(0.6255951820599558, abs=1e-15)
    assert result.pair_count == 1
    assert result.V_gradient.dtype == np.float32
    assert result.W_gradient.dtype == np.float32
    assert not result.V_gradient.flags.writeable
    assert not result.W_gradient.flags.writeable
    np.testing.assert_allclose(
        result.W_gradient[[0, 2, 3, 4, 6, 8]],
        [0, -0.46505705, 0.46505705, 0, 0, 0],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.V_gradient[[0, 2, 3, 4, 6, 8], 0],
        [-0.32553995, -0.09301141, 0.09301141, -0.32553995, -0.32553995, -0.32553995],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_array_equal(result.V_gradient[:, 1:], np.zeros((9, 15), np.float32))


@pytest.mark.parametrize(
    ("positive", "negative", "V", "W", "message"),
    [
        (
            np.asarray([[0, 1, 2, 3]], dtype=np.int32),
            np.asarray([[0, 1, 2, 3]], dtype=np.int32),
            np.zeros((4, 16), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            "shape.*N, 5",
        ),
        (
            np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64),
            np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32),
            np.zeros((5, 16), dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            "int32",
        ),
        (
            np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32),
            np.asarray([[0, 1, 2, 3, 5]], dtype=np.int32),
            np.zeros((5, 16), dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            "encoded IDs",
        ),
        (
            np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32),
            np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32),
            np.full((5, 16), np.nan, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            "finite",
        ),
    ],
)
def test_pairwise_fm_gradient_rejects_invalid_shapes_dtypes_ids_and_nonfinite_state(
    positive: np.ndarray,
    negative: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(PairwiseFMError, match=message):
        pairwise_fm_batch_loss_and_gradients(positive, negative, V=V, W=W, l2=0.0)


def test_dense_l2_changes_gradients_but_not_reported_pairwise_data_loss() -> None:
    positive = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32)
    negative = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32)
    factors = np.full((5, 16), np.float32(0.25), dtype=np.float32)
    linear = np.full(5, np.float32(-0.5), dtype=np.float32)

    unregularized = pairwise_fm_batch_loss_and_gradients(
        positive, negative, V=factors, W=linear, l2=0.0
    )
    regularized = pairwise_fm_batch_loss_and_gradients(
        positive, negative, V=factors, W=linear, l2=0.1
    )

    assert regularized.loss == unregularized.loss == pytest.approx(float(np.log(2.0)))
    np.testing.assert_allclose(
        regularized.V_gradient - unregularized.V_gradient,
        np.float32(0.1) * factors,
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        regularized.W_gradient - unregularized.W_gradient,
        np.float32(0.1) * linear,
        rtol=0,
        atol=2e-8,
    )


def test_initialization_and_score_reduction_are_byte_exact_with_verified_pointwise_control() -> (
    None
):
    organizer = load_verified_organizer(STARTER)
    control = organizer.baseline.FM(37, k=16, lr=0.001, l2=1e-6, seed=91)
    candidate = initialize_pairwise_fm(total_dim=37, seed=91)
    encoded = np.random.default_rng(14).integers(0, 37, size=(10_000, 5), dtype=np.int32)

    np.testing.assert_array_equal(candidate.V, control.V)
    np.testing.assert_array_equal(candidate.W, control.W)
    assert candidate.b.tobytes() == np.asarray(control.b, dtype=np.float32).tobytes()
    np.testing.assert_array_equal(
        pairwise_fm_scores(encoded, V=candidate.V, W=candidate.W, b=candidate.b),
        control.logits(encoded)[0],
    )


def test_config_freezes_control_dimensions_and_declares_the_single_intervention() -> None:
    config = PairwiseFMConfig(seed=4, device_metadata="mps-available-not-used")
    manifest = config.manifest()

    assert config.factor_dim == 16
    assert manifest["execution_device"] == "cpu"
    assert manifest["device_metadata"] == "mps-available-not-used"
    assert manifest["epoch_selection"] == "trusted_inner_fold_controller_only"
    assert manifest["pointwise_control"] == {
        "fields": ["user_id", "video_id", "author_id", "tab", "dur_bucket"],
        "factor_dim": 16,
        "initializer": "numpy.default_rng(seed).normal(0,0.01).astype(float32)",
        "optimizer": "dense_adam(beta1=0.9,beta2=0.999,epsilon=1e-8)",
        "precision": "float32",
        "changed_mechanism": "gauc_aligned_same_user_pairwise_logistic_objective",
    }
    with pytest.raises(TypeError):
        PairwiseFMConfig(factor_dim=8)  # type: ignore[call-arg]
    for kwargs in (
        {"seed": True},
        {"pair_batch_size": 0},
        {"pairs_per_epoch": 1_000_001},
        {"max_epochs": 0},
        {"learning_rate": float("nan")},
        {"l2": -1.0},
        {"device_metadata": "bad\nmetadata"},
    ):
        with pytest.raises(PairwiseFMError):
            PairwiseFMConfig(**kwargs)


def test_seeded_training_is_exactly_replayable_and_batches_include_partial_tail() -> None:
    first, _, _ = _fit()
    replay, _, _ = _fit()
    other_seed, _, _ = _fit(seed=8)

    assert first.logical_digest == replay.logical_digest
    assert first.checkpoint.digest == replay.checkpoint.digest
    np.testing.assert_array_equal(first.checkpoint.V, replay.checkpoint.V)
    np.testing.assert_array_equal(first.checkpoint.W, replay.checkpoint.W)
    assert first.checkpoint.digest != other_seed.checkpoint.digest
    assert [epoch.batch_sizes for epoch in first.trace] == [(4, 2), (4, 2)]
    assert [epoch.optimizer_steps for epoch in first.trace] == [2, 4]
    assert [epoch.sampled_pairs for epoch in first.trace] == [6, 12]
    assert first.checkpoint.optimizer_steps == 4
    assert first.checkpoint.sampled_pairs == 12
    assert first.checkpoint.b.tobytes() == b"\x00\x00\x00\x00"
    assert not first.checkpoint.V.flags.writeable
    assert not first.checkpoint.W.flags.writeable


def test_optional_device_metadata_never_changes_cpu_numeric_training() -> None:
    ordinary, _, _ = _fit(device_metadata=None)
    annotated, _, _ = _fit(device_metadata="local-mps-present")

    np.testing.assert_array_equal(ordinary.checkpoint.V, annotated.checkpoint.V)
    np.testing.assert_array_equal(ordinary.checkpoint.W, annotated.checkpoint.W)
    assert ordinary.config_digest != annotated.config_digest
    assert ordinary.checkpoint.digest != annotated.checkpoint.digest


def test_label_free_prediction_replays_exactly_for_all_inference_phases() -> None:
    run, encoding, _ = _fit()
    adapter = PairwiseFMAdapter(
        source_digest=SOURCE_DIGEST,
        config=PairwiseFMConfig(
            seed=7,
            max_epochs=2,
            pair_batch_size=4,
            pairs_per_epoch=6,
        ),
    )
    query = _inputs("query", 5, start_time=100)

    for phase in (DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL):
        encoded_query = EncodedFMInputs.from_encoding(encoding, query, phase=phase)
        first = adapter.predict(run.checkpoint, encoded_query)
        replay = adapter.predict(
            run.checkpoint,
            encoded_query,
            expected_prediction_digest=first.prediction_digest,
        )
        np.testing.assert_array_equal(first.scores, replay.scores)
        assert first.phase is phase
        assert first.checkpoint_digest == run.checkpoint.digest
        assert first.inputs_digest == query.digest
        assert not first.scores.flags.writeable

    with pytest.raises(PairwiseFMError, match="expected prediction digest"):
        adapter.predict(
            run.checkpoint,
            EncodedFMInputs.from_encoding(encoding, query, phase=DataPhase.FINAL),
            expected_prediction_digest="0" * 64,
        )
    for phase in (DataPhase.TRAIN, DataPhase.INNER_TRAIN):
        with pytest.raises(PairwiseFMError, match="label-free inference"):
            adapter.predict(
                run.checkpoint,
                EncodedFMInputs.from_encoding(encoding, query, phase=phase),
            )


def test_fit_rejects_nontraining_phase_before_inspecting_outcomes() -> None:
    _ForbiddenTargets.inspected = False
    train = _inputs("train", 2)
    encoded = EncodedFMInputs.from_encoding(
        StarterEncoding.fit(train),
        train,
        phase=DataPhase.OUTER_VALID,
    )

    with pytest.raises(PairwiseFMError, match="inner_train"):
        PairwiseFMTrainingData(
            inputs=encoded,
            labels=_ForbiddenTargets(),
            user_ids=train.user_id,
            training_targets_digest=_ForbiddenTargets.digest,
            target_inputs_digest=_ForbiddenTargets.training_inputs_digest,
        )
    assert not _ForbiddenTargets.inspected


def test_fit_rejects_misaligned_targets_and_encoding_and_invalid_labels() -> None:
    train = _inputs("train", 4)
    encoding = StarterEncoding.fit(train)
    encoded = EncodedFMInputs.from_encoding(
        encoding,
        train,
        phase=DataPhase.INNER_TRAIN,
    )
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)

    with pytest.raises(PairwiseFMError, match="not aligned"):
        PairwiseFMTrainingData(
            inputs=encoded,
            labels=labels,
            user_ids=train.user_id,
            training_targets_digest="d" * 64,
            target_inputs_digest="0" * 64,
        )
    other = _inputs("other", 4)
    other_encoding = StarterEncoding.fit(other)
    other_encoded = EncodedFMInputs.from_encoding(
        other_encoding,
        train,
        phase=DataPhase.INNER_TRAIN,
    )
    assert other_encoded.encoding_digest != encoded.encoding_digest
    with pytest.raises(PairwiseFMError, match="binary"):
        PairwiseFMTrainingData(
            inputs=encoded,
            labels=np.asarray([1, 0, 2, 0], dtype=np.int8),
            user_ids=train.user_id,
            training_targets_digest="d" * 64,
            target_inputs_digest=train.digest,
        )


def test_fit_and_predict_signatures_exclude_public_or_final_outcomes_and_scorers() -> None:
    assert tuple(inspect.signature(PairwiseFMAdapter.fit).parameters) == (
        "self",
        "training",
    )
    assert tuple(inspect.signature(PairwiseFMAdapter.predict).parameters) == (
        "self",
        "checkpoint",
        "inputs",
        "expected_prediction_digest",
    )
    signatures = (
        str(inspect.signature(PairwiseFMAdapter.fit))
        + str(inspect.signature(PairwiseFMAdapter.predict))
    ).lower()
    assert "validation_scorer" not in signatures
    assert "public" not in signatures
    assert "final_targets" not in signatures


def test_candidate_result_manifest_is_replay_complete_and_contains_no_metric_claim() -> None:
    run, _, _ = _fit()
    manifest = run.candidate_result_manifest()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {str(key).lower() for key in value} | set().union(
                *(keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert manifest["candidate_family"] == "pairwise_fm"
    assert manifest["checkpoint_digest"] == run.checkpoint.digest
    assert manifest["logical_digest"] == run.logical_digest
    assert manifest["training_objective"] == "gauc_aligned_pairwise_logistic"
    assert {"gauc", "ndcg@5", "primary", "validation_metric"}.isdisjoint(keys(manifest))
    assert json.dumps(manifest, sort_keys=True, allow_nan=False)

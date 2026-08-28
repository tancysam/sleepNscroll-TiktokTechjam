from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from kuairand_agent.candidates.neural import (
    HybridLossConfig,
    NeuralArchitecture,
    NeuralCeilings,
    NeuralFeatureBatch,
    NeuralFitConfig,
    NeuralModelConfig,
    NeuralPairIndices,
    NeuralPerformanceEvidence,
    NeuralPrimitiveError,
    NeuralTrainingTargets,
    assess_device_parity,
    assess_neural_eligibility,
    build_neural_model,
    dcnv2_cross_step,
    deepfm_interaction,
    fit_neural,
    hybrid_binary_pairwise_loss,
    predict_neural,
)
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import ScoreResult


def _torch() -> Any:
    return pytest.importorskip("torch")


def test_deepfm_interaction_matches_worked_second_order_equation() -> None:
    torch = _torch()
    embeddings = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, -1.0], [2.0, 2.0], [-3.0, 1.0]],
        ]
    )

    interaction = deepfm_interaction(embeddings)

    # Per latent dimension, sum all pairwise field products.  Row one is
    # (1*3 + 1*5 + 3*5) + (2*4 + 2*6 + 4*6) = 67.
    # Row two is (1*2 + -1*2) + (1*-3 + -1*1) + (2*-3 + 2*1) = -8.
    torch.testing.assert_close(interaction, torch.tensor([67.0, -8.0]), rtol=0.0, atol=0.0)


def test_dcnv2_cross_step_matches_worked_full_matrix_equation() -> None:
    torch = _torch()
    x0 = torch.tensor([[2.0, 3.0]])
    current = torch.tensor([[5.0, 7.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([1.0, -1.0])

    crossed = dcnv2_cross_step(x0, current, weight, bias)

    # W @ current + b = [6, 6], hence x0 * [6, 6] + current = [17, 25].
    torch.testing.assert_close(crossed, torch.tensor([[17.0, 25.0]]), rtol=0.0, atol=0.0)


def test_neural_config_manifest_is_immutable_canonical_and_digest_bound() -> None:
    config = NeuralModelConfig(
        architecture=NeuralArchitecture.DEEPFM,
        categorical_cardinalities=(7, 11, 5),
        dense_feature_count=2,
        embedding_dim=4,
        hidden_dims=(12, 6),
        cross_layers=2,
        dropout=0.0,
    )

    manifest = config.manifest()

    assert manifest.as_dict() == {
        "schema_version": 1,
        "architecture": "deepfm",
        "categorical_cardinalities": [7, 11, 5],
        "dense_feature_count": 2,
        "embedding_dim": 4,
        "hidden_dims": [12, 6],
        "cross_layers": 2,
        "dropout": 0.0,
    }
    assert len(manifest.digest) == 64
    assert config.digest == manifest.digest
    with pytest.raises(FrozenInstanceError):
        manifest.embedding_dim = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"categorical_cardinalities": (1, 3)},
        {"embedding_dim": 0},
        {"dense_feature_count": -1},
        {"hidden_dims": ()},
        {"cross_layers": 0},
        {"dropout": float("nan")},
    ],
)
def test_neural_config_rejects_unbounded_or_nonfinite_dimensions(kwargs: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {
        "architecture": NeuralArchitecture.DCNV2,
        "categorical_cardinalities": (7, 11),
        "dense_feature_count": 2,
    }
    defaults.update(kwargs)
    with pytest.raises(NeuralPrimitiveError):
        NeuralModelConfig(**defaults)


def test_models_share_the_control_backbone_and_return_one_logit_per_impression() -> None:
    torch = _torch()
    base: dict[str, Any] = {
        "categorical_cardinalities": (7, 11, 5),
        "dense_feature_count": 2,
        "embedding_dim": 3,
        "hidden_dims": (8, 4),
        "cross_layers": 2,
    }
    categorical = torch.tensor([[1, 2, 3], [4, 5, 1]], dtype=torch.int64)
    dense = torch.tensor([[0.5, -1.0], [2.0, 3.0]], dtype=torch.float32)

    control = build_neural_model(
        NeuralModelConfig(architecture=NeuralArchitecture.CONTROL, **base), seed=19
    )
    deepfm = build_neural_model(
        NeuralModelConfig(architecture=NeuralArchitecture.DEEPFM, **base), seed=19
    )
    dcnv2 = build_neural_model(
        NeuralModelConfig(architecture=NeuralArchitecture.DCNV2, **base), seed=19
    )

    for built in (control, deepfm, dcnv2):
        assert built.module(categorical, dense).shape == (2,)
        assert built.device.actual_device == "cpu"
        assert built.device.deterministic_algorithms
    assert control.backbone_parameter_count == deepfm.backbone_parameter_count
    assert control.backbone_parameter_count == dcnv2.backbone_parameter_count
    assert control.parameter_count == deepfm.parameter_count
    assert dcnv2.parameter_count > control.parameter_count


def test_cpu_model_construction_and_forward_are_exactly_seed_replayable() -> None:
    torch = _torch()
    config = NeuralModelConfig(
        architecture=NeuralArchitecture.DCNV2,
        categorical_cardinalities=(4, 6),
        dense_feature_count=1,
        embedding_dim=2,
        hidden_dims=(5,),
        cross_layers=1,
    )
    categorical = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    dense = torch.tensor([[0.25], [-0.5]], dtype=torch.float32)

    first = build_neural_model(config, seed=923)
    replay = build_neural_model(config, seed=923)

    torch.testing.assert_close(
        first.module(categorical, dense),
        replay.module(categorical, dense),
        rtol=0.0,
        atol=0.0,
    )


def test_hybrid_loss_matches_worked_pointwise_and_pairwise_case() -> None:
    torch = _torch()

    loss = hybrid_binary_pairwise_loss(
        pointwise_logits=torch.tensor([0.0, 0.0]),
        targets=torch.tensor([1.0, 0.0]),
        positive_logits=torch.tensor([0.0]),
        negative_logits=torch.tensor([0.0]),
        phase=DataPhase.INNER_TRAIN,
        config=HybridLossConfig(pointwise_weight=0.25, pairwise_weight=0.75),
    )

    assert float(loss.item()) == pytest.approx(0.6931471805599453, abs=1e-7)


def test_hybrid_loss_masks_missing_targets_and_rejects_invalid_active_values() -> None:
    torch = _torch()
    logits = torch.tensor([0.0, float("nan")])
    targets = torch.tensor([1.0, float("nan")])

    masked = hybrid_binary_pairwise_loss(
        pointwise_logits=logits,
        targets=targets,
        positive_logits=torch.tensor([1.0]),
        negative_logits=torch.tensor([0.0]),
        mask=torch.tensor([True, False]),
        phase=DataPhase.TRAIN,
    )

    assert bool(torch.isfinite(masked).item())
    with pytest.raises(NeuralPrimitiveError, match="boolean"):
        hybrid_binary_pairwise_loss(
            pointwise_logits=torch.tensor([0.0]),
            targets=torch.tensor([1.0]),
            positive_logits=torch.tensor([1.0]),
            negative_logits=torch.tensor([0.0]),
            mask=torch.tensor([1]),
            phase=DataPhase.TRAIN,
        )
    with pytest.raises(NeuralPrimitiveError, match="binary"):
        hybrid_binary_pairwise_loss(
            pointwise_logits=torch.tensor([0.0]),
            targets=torch.tensor([2.0]),
            positive_logits=torch.tensor([1.0]),
            negative_logits=torch.tensor([0.0]),
            phase=DataPhase.TRAIN,
        )
    with pytest.raises(NeuralPrimitiveError, match="only for train or inner_train"):
        hybrid_binary_pairwise_loss(
            pointwise_logits=torch.tensor([0.0]),
            targets=torch.tensor([1.0]),
            positive_logits=torch.tensor([1.0]),
            negative_logits=torch.tensor([0.0]),
            phase=DataPhase.OUTER_VALID,
        )


def test_neural_eligibility_enforces_every_strict_ceiling_and_nearest_rank_p95() -> None:
    ceilings = NeuralCeilings(
        max_parameters=100,
        max_p95_epoch_seconds=2.0,
        min_examples_per_second=50.0,
        max_checkpoint_bytes=1_000,
    )
    passing = NeuralPerformanceEvidence(
        parameter_count=100,
        epoch_seconds=(1.0, 2.0, 1.5),
        examples_per_second=(50.0, 80.0, 60.0),
        checkpoint_bytes=1_000,
    )

    accepted = assess_neural_eligibility(passing, ceilings)

    assert accepted.passed
    assert accepted.reasons == ()
    assert passing.p95_epoch_seconds == 2.0
    assert passing.minimum_examples_per_second == 50.0

    rejected = assess_neural_eligibility(
        NeuralPerformanceEvidence(
            parameter_count=101,
            epoch_seconds=(2.01,),
            examples_per_second=(49.0,),
            checkpoint_bytes=1_001,
        ),
        ceilings,
    )
    assert not rejected.passed
    assert rejected.reasons == (
        "parameter_count_exceeds_ceiling",
        "p95_epoch_seconds_exceeds_ceiling",
        "throughput_below_floor",
        "checkpoint_bytes_exceeds_ceiling",
    )


def test_device_parity_uses_a_declared_finite_absolute_tolerance() -> None:
    exact = assess_device_parity([0.0, 1.0], [0.0, 1.0], tolerance=1e-4)
    near = assess_device_parity([0.0, 1.0], [0.00005, 0.99995], tolerance=1e-4)
    far = assess_device_parity([0.0, 1.0], [0.001, 1.0], tolerance=1e-4)

    assert exact.within_tolerance and exact.max_absolute_difference == 0.0
    assert near.within_tolerance
    assert not far.within_tolerance
    with pytest.raises(NeuralPrimitiveError, match="finite"):
        assess_device_parity([0.0], [float("nan")], tolerance=1e-4)


class _ArrayAccessTrap:
    def __array__(self, dtype: object = None, copy: object = None) -> object:
        raise AssertionError("unauthorized target content was inspected")


def _tiny_features(phase: DataPhase) -> NeuralFeatureBatch:
    return NeuralFeatureBatch(
        phase=phase,
        categorical=np.asarray(
            [[0, 0], [1, 1], [2, 2], [3, 3], [0, 4], [1, 5], [2, 0], [3, 1]],
            dtype=np.int64,
        ),
        dense=np.asarray(
            [[0.0], [1.0], [0.2], [0.8], [0.1], [0.9], [0.3], [0.7]],
            dtype=np.float32,
        ),
    )


def test_target_phase_is_rejected_before_any_unauthorized_array_access() -> None:
    with pytest.raises(NeuralPrimitiveError, match="only for train or inner_train"):
        NeuralTrainingTargets(phase=DataPhase.FINAL, values=_ArrayAccessTrap())


def test_fit_and_label_free_prediction_have_exact_cpu_checkpoint_replay() -> None:
    _torch()
    model_config = NeuralModelConfig(
        architecture=NeuralArchitecture.DEEPFM,
        categorical_cardinalities=(4, 6),
        dense_feature_count=1,
        embedding_dim=2,
        hidden_dims=(6,),
        cross_layers=1,
    )
    fit_config = NeuralFitConfig(
        seed=73,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        loss=HybridLossConfig(pointwise_weight=0.5, pairwise_weight=0.5),
    )
    train = _tiny_features(DataPhase.INNER_TRAIN)
    targets = NeuralTrainingTargets(
        phase=DataPhase.INNER_TRAIN,
        values=np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
    )
    pairs = NeuralPairIndices(
        positive_indices=np.asarray([0, 2, 4, 6]),
        negative_indices=np.asarray([1, 3, 5, 7]),
    )
    inner_valid = _tiny_features(DataPhase.INNER_VALID)
    scored_predictions: list[np.ndarray] = []

    def aggregate_scorer(scores: np.ndarray) -> ScoreResult:
        assert not scores.flags.writeable
        scored_predictions.append(scores)
        return ScoreResult(
            gauc=0.6,
            ndcg_at_5=0.5,
            primary=0.55,
            users=4,
            rows=len(scores),
            scorer_digest="s" * 64,
            prediction_digest="p" * 64,
            runtime_seconds=0.01,
        )

    ceilings = NeuralCeilings(
        max_parameters=10_000,
        max_p95_epoch_seconds=10.0,
        min_examples_per_second=0.1,
        max_checkpoint_bytes=1_000_000,
    )
    first = fit_neural(
        model_config,
        train,
        targets,
        pairs,
        fit_config=fit_config,
        ceilings=ceilings,
        inner_valid_features=inner_valid,
        inner_valid_scorer=aggregate_scorer,
    )
    replay = fit_neural(
        model_config,
        train,
        targets,
        pairs,
        fit_config=fit_config,
        ceilings=ceilings,
        inner_valid_features=inner_valid,
        inner_valid_scorer=aggregate_scorer,
    )

    assert first.state_digest == replay.state_digest
    assert first.digest == replay.digest
    assert first.result.eligibility.passed
    assert first.result.inner_valid_primary == 0.55
    assert first.result.training_target_digest == targets.digest
    assert first.result.training_feature_digest == train.digest
    assert len(scored_predictions) == 2

    final_features = _tiny_features(DataPhase.FINAL)
    first_prediction = predict_neural(first, final_features, requested_device="cpu")
    replay_prediction = predict_neural(replay, final_features, requested_device="cpu")
    np.testing.assert_array_equal(first_prediction.scores, replay_prediction.scores)
    assert first_prediction.prediction_digest == replay_prediction.prediction_digest
    assert first_prediction.phase is DataPhase.FINAL
    assert not first_prediction.scores.flags.writeable


def test_fit_rejects_non_inner_validation_scorers_and_inconsistent_pairs() -> None:
    model_config = NeuralModelConfig(
        architecture=NeuralArchitecture.CONTROL,
        categorical_cardinalities=(4, 6),
        dense_feature_count=1,
        hidden_dims=(4,),
    )
    fit_config = NeuralFitConfig(seed=1, epochs=1, batch_size=8, learning_rate=0.01)
    targets = NeuralTrainingTargets(
        phase=DataPhase.TRAIN,
        values=np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
    )
    invalid_pairs = NeuralPairIndices(positive_indices=[1], negative_indices=[0])
    ceilings = NeuralCeilings(10_000, 10.0, 0.1, 1_000_000)

    with pytest.raises(NeuralPrimitiveError, match="positive pair indices"):
        fit_neural(
            model_config,
            _tiny_features(DataPhase.TRAIN),
            targets,
            invalid_pairs,
            fit_config=fit_config,
            ceilings=ceilings,
        )
    with pytest.raises(NeuralPrimitiveError, match="inner_valid"):
        fit_neural(
            model_config,
            _tiny_features(DataPhase.TRAIN),
            targets,
            NeuralPairIndices([0], [1]),
            fit_config=fit_config,
            ceilings=ceilings,
            inner_valid_features=_tiny_features(DataPhase.OUTER_VALID),
            inner_valid_scorer=lambda _scores: ScoreResult(
                0.5, 0.5, 0.5, 1, 8, "s" * 64, "p" * 64, 0.01
            ),
        )


def test_mps_checkpoint_prediction_parity_with_cpu_when_backend_is_available() -> None:
    torch = _torch()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available in this execution environment")
    model_config = NeuralModelConfig(
        architecture=NeuralArchitecture.DEEPFM,
        categorical_cardinalities=(4, 6),
        dense_feature_count=1,
        embedding_dim=2,
        hidden_dims=(4,),
        cross_layers=1,
    )
    checkpoint = fit_neural(
        model_config,
        _tiny_features(DataPhase.TRAIN),
        NeuralTrainingTargets(
            phase=DataPhase.TRAIN,
            values=np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
        ),
        NeuralPairIndices([0, 2, 4, 6], [1, 3, 5, 7]),
        fit_config=NeuralFitConfig(seed=5, epochs=1, batch_size=4, learning_rate=0.01),
        ceilings=NeuralCeilings(10_000, 10.0, 0.1, 1_000_000),
    )
    query = _tiny_features(DataPhase.INNER_VALID)

    cpu = predict_neural(checkpoint, query, requested_device="cpu")
    mps = predict_neural(checkpoint, query, requested_device="mps")
    parity = assess_device_parity(cpu.scores, mps.scores, tolerance=1e-4)

    assert parity.within_tolerance
    assert mps.device.actual_device == "mps"

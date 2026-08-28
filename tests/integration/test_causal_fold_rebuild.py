from __future__ import annotations

import numpy as np

from kuairand_agent.data.causal_features import (
    AggregateSpec,
    BuildIdentity,
    CausalInputs,
    OutcomeEvents,
    build_causal_feature_pair,
)

SPEC = (AggregateSpec(name="video", key_fields=("video_id",), smoothing=2.0),)


def _identity(fold: str) -> BuildIdentity:
    return BuildIdentity(
        dataset="synthetic-kuairand-pure-v1",
        split=fold,
        field_policy="safe-fields-v1",
        builder_source="causal-builder-test-source-v1",
    )


def test_each_temporal_fold_rebuilds_from_only_its_authorized_training_prefix() -> None:
    # Fold A stops after t=20.  Its t=30 and t=40 holdout rows are both queried
    # from that one terminal state, so the t=30 outcome cannot roll forward.
    fold_a = build_causal_feature_pair(
        prefix_inputs=CausalInputs(time_ms=(10, 20), fields={"video_id": ("a", "a")}),
        prefix_outcomes=OutcomeEvents(long_view=(1, 0)),
        specs=SPEC,
        identity=_identity("fold-a"),
        query_inputs=CausalInputs(time_ms=(30, 40), fields={"video_id": ("a", "a")}),
    )

    # Fold B is a distinct later rebuild where t=30 has become an authorized
    # prefix label.  It may therefore affect the later t=40 query.
    fold_b = build_causal_feature_pair(
        prefix_inputs=CausalInputs(time_ms=(10, 20, 30), fields={"video_id": ("a", "a", "a")}),
        prefix_outcomes=OutcomeEvents(long_view=(1, 0, 1)),
        specs=SPEC,
        identity=_identity("fold-b"),
        query_inputs=CausalInputs(time_ms=(40,), fields={"video_id": ("a",)}),
    )

    assert fold_a.query is not None
    assert fold_b.query is not None
    np.testing.assert_array_equal(fold_a.query.values[0], fold_a.query.values[1])
    np.testing.assert_allclose(fold_a.query.values[0], (0.5, 2.0, 1.0, 0.5))
    np.testing.assert_allclose(fold_b.query.values[0], (2.0 / 3.0, 3.0, 2.0, 2.0 / 3.0))
    np.testing.assert_array_equal(fold_a.prefix.values, fold_b.prefix.values[:2])
    assert fold_a.cache_key != fold_b.cache_key

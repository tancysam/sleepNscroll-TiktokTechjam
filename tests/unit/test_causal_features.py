from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from kuairand_agent.data.canonical import OUTCOME_FIELDS
from kuairand_agent.data.causal_features import (
    AggregateSpec,
    BuildIdentity,
    CausalFeatureCache,
    CausalFeatureCacheError,
    CausalFeatureError,
    CausalFeaturePair,
    CausalInputs,
    OutcomeEvents,
    build_causal_feature_pair,
)

IDENTITY = BuildIdentity(
    dataset="synthetic-kuairand-pure-v1",
    split="synthetic-inner-fold-v1",
    field_policy="safe-fields-v1",
    builder_source="causal-builder-test-source-v1",
)


def test_strict_past_equal_time_features_are_scattered_to_canonical_order() -> None:
    inputs = CausalInputs(
        time_ms=(30, 10, 20, 20, 40),
        fields={"video_id": ("a", "a", "a", "b", "b")},
    )
    outcomes = OutcomeEvents(long_view=(0, 1, 0, 1, 1))

    pair = build_causal_feature_pair(
        prefix_inputs=inputs,
        prefix_outcomes=outcomes,
        specs=(AggregateSpec(name="video", key_fields=("video_id",), smoothing=2.0),),
        identity=IDENTITY,
    )

    assert pair.prefix.feature_names == (
        "global__long_view_prior",
        "video__exposure",
        "video__positive",
        "video__smoothed_rate",
    )
    np.testing.assert_allclose(
        pair.prefix.values,
        np.array(
            [
                [2.0 / 3.0, 2.0, 1.0, 7.0 / 12.0],
                [0.5, 0.0, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.5, 1.0, 1.0, 2.0 / 3.0],
            ]
        ),
    )
    assert pair.query is None


def _one_key_inputs(times: tuple[int, ...], keys: tuple[object, ...]) -> CausalInputs:
    return CausalInputs(time_ms=times, fields={"video_id": keys})


def _one_key_build(
    labels: tuple[int, ...], *, inputs: CausalInputs | None = None
) -> CausalFeaturePair:
    selected = inputs or _one_key_inputs((10, 20, 30, 40), ("a", "a", "a", "a"))
    return build_causal_feature_pair(
        prefix_inputs=selected,
        prefix_outcomes=OutcomeEvents(long_view=labels),
        specs=(AggregateSpec(name="video", key_fields=("video_id",), smoothing=2.0),),
        identity=IDENTITY,
    )


def test_current_and_future_outcome_mutations_cannot_change_current_or_past_rows() -> None:
    original = _one_key_build((1, 0, 1, 0))
    current_changed = _one_key_build((1, 0, 0, 0))
    final_changed = _one_key_build((1, 0, 1, 1))

    # Row 2's own label is invisible to row 2, while the authorized later row
    # can observe that event after its timestamp has passed.
    np.testing.assert_array_equal(original.prefix.values[:3], current_changed.prefix.values[:3])
    assert not np.array_equal(original.prefix.values[3], current_changed.prefix.values[3])
    np.testing.assert_array_equal(original.prefix.values, final_changed.prefix.values)


def test_permuting_simultaneous_rows_only_permutes_their_canonical_outputs() -> None:
    times = (10, 20, 20, 30, 30)
    videos = ("a", "a", "b", "a", "b")
    labels = (1, 0, 1, 1, 0)
    original = _one_key_build(labels, inputs=_one_key_inputs(times, videos))

    permutation = np.array([0, 2, 1, 4, 3], dtype=np.int64)
    permuted = _one_key_build(
        tuple(labels[int(index)] for index in permutation),
        inputs=_one_key_inputs(
            tuple(times[int(index)] for index in permutation),
            tuple(videos[int(index)] for index in permutation),
        ),
    )
    inverse = np.argsort(permutation)

    np.testing.assert_array_equal(permuted.prefix.values[inverse], original.prefix.values)


def test_ordered_typed_composite_keys_cannot_collide() -> None:
    inputs = CausalInputs(
        time_ms=(10, 20, 30, 40, 50, 60),
        fields={
            "left": ("a|b", "a", 1, "1", True, "a|b"),
            "right": ("c", "b|c", "x", "x", "x", "c"),
        },
    )
    pair = build_causal_feature_pair(
        prefix_inputs=inputs,
        prefix_outcomes=OutcomeEvents(long_view=(1, 0, 1, 0, 1, 0)),
        specs=(AggregateSpec(name="pair", key_fields=("left", "right"), smoothing=1.0),),
        identity=IDENTITY,
    )

    # Delimiter-like strings, int/string equality, and bool/int equality all
    # remain distinct categories.  Only the exact first tuple repeats.
    np.testing.assert_array_equal(pair.prefix.values[:, 1], (0, 0, 0, 0, 0, 1))
    assert pair.prefix.values[5, 2] == 1.0


def test_frozen_query_uses_terminal_prefix_state_without_query_updates() -> None:
    prefix = _one_key_inputs((10, 20), ("a", "a"))
    query = _one_key_inputs((30, 40, 50), ("a", "a", "unseen"))
    pair = build_causal_feature_pair(
        prefix_inputs=prefix,
        prefix_outcomes=OutcomeEvents(long_view=(1, 0)),
        specs=(AggregateSpec(name="video", key_fields=("video_id",), smoothing=2.0),),
        identity=IDENTITY,
        query_inputs=query,
    )

    assert pair.query is not None
    np.testing.assert_array_equal(pair.query.values[0], pair.query.values[1])
    np.testing.assert_allclose(pair.query.values[0], (0.5, 2.0, 1.0, 0.5))
    np.testing.assert_allclose(pair.query.values[2], (0.5, 0.0, 0.0, 0.5))


def test_query_outcomes_are_structurally_absent_and_cannot_affect_features() -> None:
    prefix = _one_key_inputs((10, 20), ("a", "a"))
    query = _one_key_inputs((30, 40), ("a", "a"))
    labels = [0, 1]
    first = build_causal_feature_pair(
        prefix_inputs=prefix,
        prefix_outcomes=OutcomeEvents(long_view=(1, 0)),
        specs=(AggregateSpec(key_fields=("video_id",)),),
        identity=IDENTITY,
        query_inputs=query,
    )
    labels[:] = [1, 0]
    second = build_causal_feature_pair(
        prefix_inputs=prefix,
        prefix_outcomes=OutcomeEvents(long_view=(1, 0)),
        specs=(AggregateSpec(key_fields=("video_id",)),),
        identity=IDENTITY,
        query_inputs=query,
    )

    assert first.logical_digest == second.logical_digest
    unsafe_call = cast(Callable[..., object], build_causal_feature_pair)
    with pytest.raises(TypeError, match="query_outcomes"):
        unsafe_call(
            prefix_inputs=prefix,
            prefix_outcomes=OutcomeEvents(long_view=(1, 0)),
            specs=(AggregateSpec(key_fields=("video_id",)),),
            identity=IDENTITY,
            query_inputs=query,
            query_outcomes=labels,
        )


def test_inputs_and_feature_artifacts_snapshot_mutable_sources() -> None:
    times = [10, 20]
    videos = ["a", "a"]
    labels = [1, 0]
    inputs = CausalInputs(time_ms=times, fields={"video_id": videos})
    outcomes = OutcomeEvents(long_view=labels)
    pair = build_causal_feature_pair(
        prefix_inputs=inputs,
        prefix_outcomes=outcomes,
        specs=(AggregateSpec(key_fields=("video_id",)),),
        identity=IDENTITY,
    )
    before = pair.prefix.values.copy()
    times[0] = 999
    videos[0] = "changed"
    labels[0] = 0

    assert inputs.time_ms == (10, 20)
    assert inputs.fields["video_id"] == ("a", "a")
    assert outcomes.long_view == (1, 0)
    np.testing.assert_array_equal(pair.prefix.values, before)
    with pytest.raises(ValueError, match="read-only"):
        pair.prefix.values[0, 0] = 9.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        pair.prefix.values.setflags(write=True)


def test_logical_identities_cover_safe_content_labels_specs_and_build_identity(
    tmp_path: Path,
) -> None:
    prefix = _one_key_inputs((10, 20), ("a", "a"))
    query = _one_key_inputs((30,), ("a",))
    kwargs: dict[str, Any] = {
        "prefix_inputs": prefix,
        "prefix_outcomes": OutcomeEvents(long_view=(1, 0)),
        "specs": (AggregateSpec(key_fields=("video_id",)),),
        "identity": IDENTITY,
        "query_inputs": query,
    }
    first = build_causal_feature_pair(**kwargs, cache_dir=tmp_path / "one")
    second = build_causal_feature_pair(**kwargs, cache_dir=tmp_path / "two")

    assert first.manifest() == second.manifest()
    assert first.logical_digest == second.logical_digest
    assert first.cache_key == second.cache_key
    assert (
        first.cache_key
        != build_causal_feature_pair(
            **{**kwargs, "prefix_outcomes": OutcomeEvents(long_view=(0, 0))}
        ).cache_key
    )
    assert (
        first.cache_key
        != build_causal_feature_pair(
            **{**kwargs, "query_inputs": _one_key_inputs((30,), ("b",))}
        ).cache_key
    )
    assert (
        first.cache_key
        != build_causal_feature_pair(
            **{**kwargs, "specs": (AggregateSpec(key_fields=("video_id",), smoothing=3.0),)}
        ).cache_key
    )
    assert (
        first.cache_key
        != build_causal_feature_pair(
            **{**kwargs, "identity": dataclasses.replace(IDENTITY, split="another-fold")}
        ).cache_key
    )


def test_input_digest_is_independent_of_mapping_insertion_order() -> None:
    first = CausalInputs(
        time_ms=(10, 20),
        fields={"video_id": (1, 2), "author_id": (3, 4)},
    )
    second = CausalInputs(
        time_ms=(10, 20),
        fields={"author_id": (3, 4), "video_id": (1, 2)},
    )

    assert first.logical_digest == second.logical_digest
    assert tuple(first.fields) == tuple(second.fields) == ("author_id", "video_id")


def test_cache_hit_is_logically_identical_and_does_not_rewrite_artifacts(tmp_path: Path) -> None:
    cache = CausalFeatureCache(tmp_path / "cache")
    kwargs: dict[str, Any] = {
        "prefix_inputs": _one_key_inputs((10, 20), ("a", "a")),
        "prefix_outcomes": OutcomeEvents(long_view=(1, 0)),
        "specs": (AggregateSpec(key_fields=("video_id",)),),
        "identity": IDENTITY,
        "query_inputs": _one_key_inputs((30,), ("a",)),
    }
    cold = cache.build_or_load(**kwargs)
    artifacts = sorted(cache.cache_dir.iterdir())
    before = {path.name: path.read_bytes() for path in artifacts}
    warm = cache.build_or_load(**kwargs)

    assert {path.name: path.read_bytes() for path in artifacts} == before
    assert warm.manifest() == cold.manifest()
    assert [path.suffix for path in artifacts] == [".json", ".npz"]
    assert not list(cache.cache_dir.glob("*.tmp"))


@pytest.mark.parametrize("corrupt_suffix", [".json", ".npz"])
def test_corrupt_committed_cache_hit_is_rejected(tmp_path: Path, corrupt_suffix: str) -> None:
    cache = CausalFeatureCache(tmp_path / "cache")
    kwargs: dict[str, Any] = {
        "prefix_inputs": _one_key_inputs((10, 20), ("a", "a")),
        "prefix_outcomes": OutcomeEvents(long_view=(1, 0)),
        "specs": (AggregateSpec(key_fields=("video_id",)),),
        "identity": IDENTITY,
    }
    pair = cache.build_or_load(**kwargs)
    (cache.cache_dir / f"{pair.cache_key}{corrupt_suffix}").write_bytes(b"corrupt")

    with pytest.raises(CausalFeatureCacheError, match="corrupt"):
        cache.build_or_load(**kwargs)


def test_incomplete_cache_commit_is_rejected(tmp_path: Path) -> None:
    cache = CausalFeatureCache(tmp_path / "cache")
    kwargs: dict[str, Any] = {
        "prefix_inputs": _one_key_inputs((10, 20), ("a", "a")),
        "prefix_outcomes": OutcomeEvents(long_view=(1, 0)),
        "specs": (AggregateSpec(key_fields=("video_id",)),),
        "identity": IDENTITY,
    }
    pair = cache.build_or_load(**kwargs)
    (cache.cache_dir / f"{pair.cache_key}.json").unlink()

    with pytest.raises(CausalFeatureCacheError, match="incomplete"):
        cache.build_or_load(**kwargs)


@pytest.mark.parametrize(
    ("time_ms", "fields", "message"),
    [
        ((True,), {"video_id": ("a",)}, "integral"),
        ((1.5,), {"video_id": ("a",)}, "integral"),
        ((2**63,), {"video_id": ("a",)}, "int64"),
        ((1,), {}, "safe field"),
        ((1, 2), {"video_id": ("a",)}, "expected 2"),
        ((1,), {"row_id": (0,)}, "row_id"),
        ((1,), {"long_view": (1,)}, "outcome"),
        ((1,), {"is_click": (1,)}, "outcome"),
        ((1,), {"profile_stay_time": (1,)}, "outcome"),
        ((1,), {"video_id": (float("nan"),)}, "finite"),
        ((1,), {"video_id": (object(),)}, "must be str"),
    ],
)
def test_invalid_causal_inputs_fail_closed(
    time_ms: tuple[object, ...], fields: dict[str, tuple[object, ...]], message: str
) -> None:
    with pytest.raises(CausalFeatureError, match=message):
        CausalInputs(time_ms=time_ms, fields=fields)


@pytest.mark.parametrize("outcome_name", OUTCOME_FIELDS)
def test_public_aggregate_spec_rejects_every_canonical_outcome_name(
    outcome_name: str,
) -> None:
    """Every canonical same-row outcome is blocked at the public feature seam."""

    with pytest.raises(CausalFeatureError, match="outcome"):
        AggregateSpec(key_fields=(outcome_name,))


@pytest.mark.parametrize("labels", [(), (True,), (-1,), (2,), (1.0,), ("1",)])
def test_only_binary_integral_prefix_long_view_is_authorized(
    labels: tuple[object, ...],
) -> None:
    with pytest.raises(CausalFeatureError, match=r"binary|at least one"):
        OutcomeEvents(long_view=labels)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AggregateSpec(key_fields=()),
        lambda: AggregateSpec(key_fields=("video_id", "video_id")),
        lambda: AggregateSpec(key_fields=("row_id",)),
        lambda: AggregateSpec(key_fields=("long_view",)),
        lambda: AggregateSpec(key_fields=("video_id",), name="bad-name"),
        lambda: AggregateSpec(key_fields=("video_id",), smoothing=0.0),
        lambda: AggregateSpec(key_fields=("video_id",), smoothing=-1.0),
        lambda: AggregateSpec(key_fields=("video_id",), smoothing=float("nan")),
        lambda: AggregateSpec(key_fields=("video_id",), smoothing=cast(Any, True)),
        lambda: AggregateSpec(key_fields=("video_id",), initial_prior=-0.1),
        lambda: AggregateSpec(key_fields=("video_id",), initial_prior=1.1),
    ],
)
def test_invalid_aggregate_specs_fail_closed(factory: Callable[[], AggregateSpec]) -> None:
    with pytest.raises(CausalFeatureError):
        factory()


def test_invalid_request_shapes_fields_priors_and_fold_boundaries_fail_closed() -> None:
    prefix = _one_key_inputs((10, 20), ("a", "a"))
    outcomes = OutcomeEvents(long_view=(1, 0))
    with pytest.raises(CausalFeatureError, match="equal row counts"):
        build_causal_feature_pair(
            prefix_inputs=prefix,
            prefix_outcomes=OutcomeEvents(long_view=(1,)),
            specs=(AggregateSpec(key_fields=("video_id",)),),
            identity=IDENTITY,
        )
    with pytest.raises(CausalFeatureError, match="missing aggregate"):
        build_causal_feature_pair(
            prefix_inputs=prefix,
            prefix_outcomes=outcomes,
            specs=(AggregateSpec(key_fields=("author_id",)),),
            identity=IDENTITY,
        )
    with pytest.raises(CausalFeatureError, match=r"max\(prefix time_ms\) < min"):
        build_causal_feature_pair(
            prefix_inputs=prefix,
            prefix_outcomes=outcomes,
            specs=(AggregateSpec(key_fields=("video_id",)),),
            identity=IDENTITY,
            query_inputs=_one_key_inputs((20, 30), ("a", "a")),
        )
    with pytest.raises(CausalFeatureError, match="global initial_prior"):
        build_causal_feature_pair(
            prefix_inputs=prefix,
            prefix_outcomes=outcomes,
            specs=(
                AggregateSpec(name="one", key_fields=("video_id",), initial_prior=0.5),
                AggregateSpec(name="two", key_fields=("video_id",), initial_prior=0.4),
            ),
            identity=IDENTITY,
        )
    with pytest.raises(CausalFeatureError, match="at least one AggregateSpec"):
        build_causal_feature_pair(
            prefix_inputs=prefix,
            prefix_outcomes=outcomes,
            specs=(),
            identity=IDENTITY,
        )
    with pytest.raises(CausalFeatureError, match="aggregate names must be unique"):
        build_causal_feature_pair(
            prefix_inputs=prefix,
            prefix_outcomes=outcomes,
            specs=(
                AggregateSpec(name="duplicate", key_fields=("video_id",)),
                AggregateSpec(name="duplicate", key_fields=("video_id",), smoothing=3.0),
            ),
            identity=IDENTITY,
        )


@pytest.mark.parametrize(
    "identity_factory",
    [
        lambda: BuildIdentity("", "split", "policy", "source"),
        lambda: BuildIdentity("data", "bad\nsplit", "policy", "source"),
        lambda: BuildIdentity("data", "split", "policy", cast(Any, 3)),
    ],
)
def test_invalid_build_identities_fail_closed(
    identity_factory: Callable[[], BuildIdentity],
) -> None:
    with pytest.raises(CausalFeatureError):
        identity_factory()


def test_row_identity_never_appears_in_candidate_feature_schema() -> None:
    inputs = CausalInputs(
        time_ms=(10, 20),
        fields={"user_id": (1, 1), "video_id": (2, 3)},
    )
    pair = build_causal_feature_pair(
        prefix_inputs=inputs,
        prefix_outcomes=OutcomeEvents(long_view=(1, 0)),
        specs=(
            AggregateSpec(key_fields=("user_id",)),
            AggregateSpec(key_fields=("user_id", "video_id")),
        ),
        identity=IDENTITY,
    )

    assert all("row_id" not in name for name in pair.prefix.feature_names)
    assert pair.prefix.feature_count == 7

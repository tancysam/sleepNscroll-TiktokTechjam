from __future__ import annotations

from dataclasses import replace

import pytest

from kuairand_agent.evaluation.resampling import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HolmBootstrapDecision,
    OneSidedLowerBound,
    PairedUserClusterBootstrap,
    PromotionEvidenceError,
    holm_correct_bootstrap,
    holm_step_down,
)


def _result(delta: float) -> PairedUserClusterBootstrap:
    bound = OneSidedLowerBound(point_delta=delta, lower_bound=delta)
    return PairedUserClusterBootstrap(
        gauc=bound,
        ndcg_at_5=bound,
        primary=bound,
        clusters=10,
        rows=20,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        alignment_sha256="a" * 64,
        primary_replicates=(delta,) * BOOTSTRAP_RESAMPLES,
    )


def test_holm_step_down_uses_fixed_two_finalist_thresholds_and_adjusted_p_values() -> None:
    decisions = holm_step_down({"second": 0.04, "first": 0.02})

    assert [(item.finalist_id, item.alpha_threshold, item.rejected) for item in decisions] == [
        ("first", 0.025, True),
        ("second", 0.05, True),
    ]
    assert decisions[0].adjusted_p_value == pytest.approx(0.04)
    assert decisions[1].adjusted_p_value == pytest.approx(0.04)


def test_holm_stops_after_the_first_non_rejection() -> None:
    decisions = holm_step_down({"first": 0.03, "second": 0.04})
    assert decisions[0].rejected is False
    assert decisions[1].rejected is False


def test_holm_bootstrap_requires_strictly_material_adjusted_lower_bound() -> None:
    strong = _result(0.010)
    exact_margin = _result(0.002)

    decisions = holm_correct_bootstrap({"strong": strong, "exact-margin": exact_margin})
    by_id: dict[str, HolmBootstrapDecision] = {item.finalist_id: item for item in decisions}

    assert by_id["strong"].alpha_threshold == 0.025
    assert by_id["strong"].adjusted_lower_bound == 0.010
    assert by_id["strong"].materially_confirmed
    assert not by_id["exact-margin"].materially_confirmed


def test_holm_rejects_more_than_two_frozen_finalists_or_invalid_p_values() -> None:
    with pytest.raises(PromotionEvidenceError, match="more than two"):
        holm_step_down({"a": 0.01, "b": 0.02, "c": 0.03})
    with pytest.raises(PromotionEvidenceError, match=r"in \[0, 1\]"):
        holm_step_down({"a": 1.1})


def test_holm_family_requires_the_same_exact_cluster_alignment() -> None:
    first = _result(0.01)
    second = replace(_result(0.02), alignment_sha256="b" * 64)

    with pytest.raises(PromotionEvidenceError, match="same exact cluster"):
        holm_correct_bootstrap({"first": first, "second": second})

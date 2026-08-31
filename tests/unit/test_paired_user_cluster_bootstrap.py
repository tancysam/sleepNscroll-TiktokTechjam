from __future__ import annotations

from dataclasses import replace

import pytest

from kuairand_agent.evaluation.promotion import PROMOTION_POLICY_V1
from kuairand_agent.evaluation.resampling import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    PromotionEvidenceError,
    UserClusterMetric,
    paired_user_cluster_bootstrap,
)


def _paired_clusters() -> tuple[tuple[UserClusterMetric, ...], tuple[UserClusterMetric, ...]]:
    # Canonical rows are interleaved by user.  Each cluster keeps its own physical row order while
    # cluster order follows first appearance in the canonical vector.
    rows = ((0, 4), (1, 3), (2, 5), (6, 7))
    fallback_gauc = (0.60, 0.70, 0.50, 0.80)
    candidate_gauc = (0.61, 0.72, 0.49, 0.83)
    fallback_ndcg = (0.50, 0.40, 0.70, 0.60)
    candidate_ndcg = (0.52, 0.39, 0.73, 0.64)
    denominators = (1, 2, 1, 3)
    candidate = tuple(
        UserClusterMetric(
            cluster_id=f"u-{index}",
            row_ids=row_ids,
            gauc_numerator=candidate_gauc[index] * denominators[index],
            gauc_denominator=denominators[index],
            ndcg_at_5=candidate_ndcg[index],
        )
        for index, row_ids in enumerate(rows)
    )
    fallback = tuple(
        UserClusterMetric(
            cluster_id=f"u-{index}",
            row_ids=row_ids,
            gauc_numerator=fallback_gauc[index] * denominators[index],
            gauc_denominator=denominators[index],
            ndcg_at_5=fallback_ndcg[index],
        )
        for index, row_ids in enumerate(rows)
    )
    return candidate, fallback


def test_paired_user_cluster_bootstrap_is_frozen_and_reproducible() -> None:
    candidate, fallback = _paired_clusters()

    first = paired_user_cluster_bootstrap(candidate, fallback)
    replay = paired_user_cluster_bootstrap(candidate, fallback)

    assert first.resamples == BOOTSTRAP_RESAMPLES == PROMOTION_POLICY_V1.resample_count == 10_000
    assert first.seed == BOOTSTRAP_SEED == PROMOTION_POLICY_V1.bootstrap_seed == 20_260_831
    assert first.primary.confidence_level == CONFIDENCE_LEVEL == 0.95
    assert first.clusters == 4
    assert first.rows == 8
    assert first.primary_replicates == replay.primary_replicates
    assert first.primary == replay.primary
    assert first.gauc.point_delta == pytest.approx(0.13 / 7.0)
    assert first.ndcg_at_5.point_delta == pytest.approx(0.02)
    assert first.primary.point_delta == pytest.approx(((0.13 / 7.0) + 0.02) / 2.0)
    assert first.primary.lower_bound > 0.005
    assert len(first.primary_replicates) == 10_000


@pytest.mark.parametrize(
    "candidate_edit,fallback_edit,error",
    [
        (lambda values: values[:-1], lambda values: values, "same cluster count"),
        (
            lambda values: values,
            lambda values: (values[1], values[0], *values[2:]),
            "ordered cluster identity",
        ),
        (
            lambda values: values,
            lambda values: (replace(values[0], row_ids=(0, 7)), *values[1:]),
            "mismatched ordered row identity",
        ),
        (
            lambda values: (replace(values[0], gauc_denominator=2), *values[1:]),
            lambda values: values,
            "mismatched GAUC eligibility",
        ),
    ],
)
def test_missing_reordered_or_misaligned_clusters_are_hard_evidence_failures(
    candidate_edit: object,
    fallback_edit: object,
    error: str,
) -> None:
    candidate, fallback = _paired_clusters()
    edit_candidate = candidate_edit
    edit_fallback = fallback_edit
    assert callable(edit_candidate)
    assert callable(edit_fallback)

    with pytest.raises(PromotionEvidenceError, match=error):
        paired_user_cluster_bootstrap(edit_candidate(candidate), edit_fallback(fallback))


def test_cluster_rows_must_cover_the_exact_canonical_population() -> None:
    candidate, fallback = _paired_clusters()
    candidate = (replace(candidate[0], row_ids=(0, 8)), *candidate[1:])
    fallback = (replace(fallback[0], row_ids=(0, 8)), *fallback[1:])

    with pytest.raises(PromotionEvidenceError, match="every zero-based canonical row"):
        paired_user_cluster_bootstrap(candidate, fallback)


def test_ineligible_or_invalid_user_cluster_is_not_silently_dropped() -> None:
    with pytest.raises(PromotionEvidenceError, match="positive for every eligible"):
        UserClusterMetric(
            cluster_id="degenerate",
            row_ids=(0,),
            gauc_numerator=0.0,
            gauc_denominator=0,
            ndcg_at_5=0.0,
        )

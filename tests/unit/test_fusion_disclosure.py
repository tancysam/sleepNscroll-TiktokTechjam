"""The generated model's own score must remain separable from the blend that gets reported.

Every generated prediction is rank-fused with the official FM control, and the reported primary is
the selected blend's.  Four iterations across runs 09, 10 and 14 recorded a primary bit-identical
to the control because the selector chose weight 0.0 for the model, and each was read downstream as
"matched the baseline" rather than "was rejected".  These tests pin the fields that tell them
apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest

from kuairand_agent.campaign.full_campaign_runtime import _fusion_disclosure
from kuairand_agent.campaign.generated_scientific_runner import (
    FoldBFusionSelectionEvidence,
    FusionGridPointEvidence,
    GeneratedScientificRunnerError,
    GeneratedScientificRunRecord,
)
from kuairand_agent.campaign.scientific import ScientificTier
from kuairand_agent.campaign.selector import OrganizerMetrics
from kuairand_agent.candidates.fusion import FUSION_WEIGHT_GRID

_CONTROL_GAUC = 0.6583324670791626
_CONTROL_NDCG = 0.49251559376716614


type _Grid = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]

# Both fixtures below are the literal (GAUC, nDCG@5) grids recorded by run 14 on the Fold B
# screen.  They are used verbatim because the ordering they produce is the point: the control's
# GAUC is the highest of the five in BOTH cases, and the 25/75 blend wins the first one only on
# nDCG@5.  A fixture that varies GAUC alone cannot reproduce that and silently tests nothing.
_MODEL_WEAKER_BLEND_WINS: _Grid = (
    (0.6502764225006104, 0.4904730021953583),
    (0.6522281169891357, 0.4911235272884369),
    (0.6568991541862488, 0.49232783913612366),
    (0.6583007574081421, 0.49257364869117737),
    (_CONTROL_GAUC, _CONTROL_NDCG),
)
_MODEL_DISCARDED: _Grid = (
    (0.6473813056945801, 0.4902079105377197),
    (0.649784505367279, 0.4906582236289978),
    (0.6564492583274841, 0.4926203787326813),
    (0.6579196453094482, 0.49256250262260437),
    (_CONTROL_GAUC, _CONTROL_NDCG),
)


def _metrics(gauc: float, ndcg: float = _CONTROL_NDCG) -> OrganizerMetrics:
    return OrganizerMetrics(gauc=gauc, ndcg_at_5=ndcg)


def _selection(grid: _Grid) -> FoldBFusionSelectionEvidence:
    """Build one Fold B selection over the exact fixed grid, model weight descending."""

    points = tuple(
        FusionGridPointEvidence(
            weights=weights,
            metrics=_metrics(gauc, ndcg),
            prediction_digest=f"{index:02d}" * 32,
            fusion_digest=f"{index + 10:02d}" * 32,
            scorer_runtime_seconds=0.5,
        )
        for index, (weights, (gauc, ndcg)) in enumerate(zip(FUSION_WEIGHT_GRID, grid, strict=True))
    )
    best = max(enumerate(points), key=lambda item: (item[1].metrics.primary_decimal, -item[0]))[1]
    return FoldBFusionSelectionEvidence(
        selector_digest="ab" * 32,
        points=points,
        selected_weights=best.weights,
        selected_prediction_digest=best.prediction_digest,
        selected_fusion_digest=best.fusion_digest,
    )


def _records(
    selection: FoldBFusionSelectionEvidence,
) -> Mapping[tuple[ScientificTier, int], GeneratedScientificRunRecord]:
    """Bind one Fold B selection under the key the disclosure reads.

    A real ``GeneratedScientificRunRecord`` carries a full artifact closure and scientific
    evidence.  ``_fusion_disclosure`` reads exactly one attribute off it, so a stand-in keeps this
    test on the disclosure logic instead of on record construction.
    """

    holder = SimpleNamespace(fusion_selection=selection)
    return cast(
        "Mapping[tuple[ScientificTier, int], GeneratedScientificRunRecord]",
        {(ScientificTier.FOLD_B_SCREEN, 0): holder},
    )


def test_standalone_and_control_points_are_readable_from_the_grid() -> None:
    selection = _selection(_MODEL_WEAKER_BLEND_WINS)
    standalone_gauc, standalone_ndcg = _MODEL_WEAKER_BLEND_WINS[0]

    # The reported primary was 0.5754372, a hair above the control's 0.5754240, which reads as a
    # tie.  The model alone scored 0.5703747, which is 6 sigma below the control at sigma 0.0008.
    assert selection.selected_weights == (0.25, 0.75)
    assert selection.standalone_primary == pytest.approx((standalone_gauc + standalone_ndcg) / 2)
    assert selection.control_primary == pytest.approx((_CONTROL_GAUC + _CONTROL_NDCG) / 2)
    assert selection.standalone_primary is not None
    assert selection.control_primary is not None
    assert selection.control_primary - selection.standalone_primary > 0.004
    assert not selection.model_was_discarded

    disclosure = _fusion_disclosure(_records(selection))
    assert disclosure["fusion_weights_selected"] == "0.25 model, 0.75 official FM control"
    note = disclosure["fusion_note"]
    assert isinstance(note, str)
    assert "WEAKER than the control" in note


def test_a_discarded_model_is_reported_as_a_rejection_not_a_tie() -> None:
    selection = _selection(_MODEL_DISCARDED)
    standalone_gauc, standalone_ndcg = _MODEL_DISCARDED[0]

    assert selection.selected_weights == (0.0, 1.0)
    assert selection.model_was_discarded

    disclosure = _fusion_disclosure(_records(selection))
    assert disclosure["candidate_standalone_primary"] == pytest.approx(
        (standalone_gauc + standalone_ndcg) / 2, abs=5e-8
    )
    # The reported primary here is the control's own score, bit for bit.  Nothing downstream may
    # describe that as a tie.
    assert disclosure["fold_b_control_primary"] == pytest.approx(
        (_CONTROL_GAUC + _CONTROL_NDCG) / 2, abs=5e-8
    )
    note = disclosure["fusion_note"]
    assert isinstance(note, str)
    assert "NOT a tie" in note


def test_disclosure_is_empty_rather_than_wrong_when_no_fold_b_selection_exists() -> None:
    disclosure = _fusion_disclosure({})

    assert disclosure == {
        "candidate_standalone_primary": None,
        "fold_b_control_primary": None,
        "fusion_weights_selected": None,
        "fusion_model_discarded": None,
        "fusion_rank_correlation_with_control": None,
        "fusion_note": None,
    }


def test_selection_still_refuses_a_grid_that_is_not_the_frozen_one() -> None:
    points = tuple(
        FusionGridPointEvidence(
            weights=weights,
            metrics=_metrics(_CONTROL_GAUC),
            prediction_digest=f"{index:02d}" * 32,
            fusion_digest=f"{index + 10:02d}" * 32,
            scorer_runtime_seconds=0.5,
        )
        for index, weights in enumerate(FUSION_WEIGHT_GRID[:-1])
    )
    with pytest.raises(GeneratedScientificRunnerError):
        FoldBFusionSelectionEvidence(
            selector_digest="ab" * 32,
            points=points,
            selected_weights=(1.0, 0.0),
            selected_prediction_digest="00" * 32,
            selected_fusion_digest="10" * 32,
        )


def test_the_screen_margin_is_a_measured_sigma_not_zero() -> None:
    """At 0.0 a candidate promoted on a +0.00004 Fold B delta and re-based the incumbent.

    That is an eighth of the measured seed sigma (docs/RESULTS.md 3.4, 0.000316), so the incumbent
    chain was a random walk spending Fold A launches on noise.
    """

    from kuairand_agent.campaign.full_campaign_runtime import FOLD_B_SCREEN_MARGIN

    assert FOLD_B_SCREEN_MARGIN == 0.000316
    # The churn it stops.
    assert FOLD_B_SCREEN_MARGIN > 0.00004
    # And what it does NOT stop: the two largest Fold B gains on record clear this margin and
    # still lost Fold A by more than they won. This margin is not a fix for Fold B overfitting.
    assert FOLD_B_SCREEN_MARGIN < 0.000590
    assert FOLD_B_SCREEN_MARGIN < 0.000344


def test_the_standalone_tolerance_admits_the_best_candidate_and_rejects_the_worst() -> None:
    """The Fold B screen reads the FUSED primary, so weak models passed on a rescued blend.

    sol5 iteration-01 scored 0.570702 standalone against a control of 0.575424 -- 4.7 sigma down --
    and still cleared a screen tightened to one sigma, because its 25/75 blend beat the parent by
    0.000153. It then spent a Fold A launch and three outer seeds.

    An outright standalone win is the wrong bar: it would reject every candidate this project has
    produced, including run 16 at 0.00089 below the control, which is the structure the briefing
    now recommends building on.
    """

    from kuairand_agent.contract import MEASURED_SEED_SIGMA, STANDALONE_TOLERANCE

    assert STANDALONE_TOLERANCE == 6 * MEASURED_SEED_SIGMA
    assert STANDALONE_TOLERANCE < 0.004722, "sol5 iteration-01 must be rejected"
    assert STANDALONE_TOLERANCE > 0.000890, "run 16, the best on record, must be admitted"


def test_rank_correlation_separates_rebuilt_the_control_from_diverse_but_weak() -> None:
    """One scalar, and no train-only analysis request can reach it."""

    import numpy as np

    from kuairand_agent.candidates.fusion import _rank_correlation

    ordering = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    assert _rank_correlation(ordering, ordering) == 1.0, "must not exceed 1.0 by float error"
    assert _rank_correlation(ordering, 1.0 - ordering) == -1.0
    assert _rank_correlation(ordering, np.ones(5)) is None, "a constant member has no correlation"


def test_the_prompt_prices_trades_against_the_measured_sigma() -> None:
    """0.0008 alone made the model discount real effects: it is 2.5x the sigma we measure."""

    from kuairand_agent.contract import MEASURED_SEED_SIGMA, PUBLISHED_SEED_SIGMA
    from kuairand_agent.research.prompts import instructions_for
    from kuairand_agent.research.schemas import ResearchOperation

    rendered = instructions_for(ResearchOperation.PROPOSE)
    assert f"{MEASURED_SEED_SIGMA:g}" in rendered
    assert f"{PUBLISHED_SEED_SIGMA:g}" in rendered
    assert "fusion_rank_correlation_with_control" in rendered


def test_the_prompt_states_the_real_thread_budget() -> None:
    """Stated by hand as one, understating the budget four-fold against the shipped runner."""

    import tomllib
    from pathlib import Path

    from kuairand_agent.contract import CANDIDATE_THREADS
    from kuairand_agent.research.prompts import instructions_for
    from kuairand_agent.research.schemas import ResearchOperation

    config = tomllib.loads(Path("configs/full-pure.toml").read_text(encoding="utf-8"))
    assert config["runner"]["threads"] == CANDIDATE_THREADS, (
        "constant must track the shipped config"
    )
    assert f"{CANDIDATE_THREADS} CPU threads" in instructions_for(ResearchOperation.PROPOSE)


def test_reflect_is_told_to_spend_every_analysis_request() -> None:
    """They are the only free data access the model has, and none has ever been used."""

    from kuairand_agent.research.prompts import instructions_for
    from kuairand_agent.research.schemas import ResearchOperation

    assert "Spend all" in instructions_for(ResearchOperation.REFLECT)

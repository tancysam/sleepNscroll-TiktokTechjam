"""What a campaign remembers about its own runs must be enough to diagnose them.

The record the proposer reads carried the mean of GAUC and nDCG@5 and nothing else. A candidate
that gained 0.0090 GAUC and lost 0.0205 nDCG@5 -- measured repeatedly on this benchmark -- was
therefore remembered as one negative scalar, which reads as "that direction failed" rather than
"the ranking objective worked and the top-5 behaviour did not". The second reading is the one an
engineer makes from the same run, and it points at a fix instead of an abandonment.

The configuration was missing for the same reason. Candidates now run their own internal grids
over decay half-lives, round counts and regularisation, and that search was discarded with the run
directory: a later campaign learned that a family scored some number, never at what setting, so it
could not take a configuration that worked and push it further.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import kuairand_agent.campaign.full_campaign_runtime as runtime
from kuairand_agent.campaign.selector import OrganizerMetrics


def _run(gauc: float, ndcg: float) -> SimpleNamespace:
    return SimpleNamespace(metrics=OrganizerMetrics(gauc=gauc, ndcg_at_5=ndcg))


def _incumbent(fold_metrics: dict[str, OrganizerMetrics]) -> SimpleNamespace:
    return SimpleNamespace(inner_by_fold=tuple(fold_metrics.items()))


def test_opposite_metric_movements_are_recorded_separately() -> None:
    """The exact signature measured on this benchmark: GAUC up, nDCG@5 down further."""

    candidate = SimpleNamespace(runs=(_run(0.667923, 0.536363), _run(0.667476, 0.535341)))
    incumbent = _incumbent(
        {
            "B": OrganizerMetrics(gauc=0.658332, ndcg_at_5=0.492516),
            "A": OrganizerMetrics(gauc=0.658576, ndcg_at_5=0.556815),
        }
    )

    values = runtime._decomposed_metrics(cast(Any, candidate), cast(Any, incumbent))

    # Fold A is the confirmation run, where the trade is visible in both directions at once.
    assert float(cast(float, values["fold_A_delta_gauc"])) > 0.0, "the GAUC gain must survive"
    assert float(cast(float, values["fold_A_delta_ndcg_at_5"])) < 0.0, "the loss must too"
    assert float(cast(float, values["fold_A_delta_primary"])) < 0.0
    # Without this the two halves are indistinguishable from a uniformly worse candidate.
    assert values["fold_A_candidate_gauc"] == 0.667476
    assert values["fold_A_candidate_ndcg_at_5"] == 0.535341
    assert "opposite directions" in str(values["metric_decomposition_note"])


def test_each_fold_is_reported_separately_so_replication_is_visible() -> None:
    candidate = SimpleNamespace(runs=(_run(0.66, 0.50), _run(0.67, 0.55)))
    incumbent = _incumbent(
        {
            "B": OrganizerMetrics(gauc=0.65, ndcg_at_5=0.49),
            "A": OrganizerMetrics(gauc=0.65, ndcg_at_5=0.54),
        }
    )

    values = runtime._decomposed_metrics(cast(Any, candidate), cast(Any, incumbent))

    for fold in ("A", "B"):
        for suffix in ("candidate_gauc", "candidate_ndcg_at_5", "delta_primary"):
            assert f"fold_{fold}_{suffix}" in values


def test_a_candidate_that_never_scored_records_no_metrics() -> None:
    """A crashed branch must not be given fabricated decomposition."""

    crashed = SimpleNamespace(runs=(SimpleNamespace(metrics=None),))
    incumbent = _incumbent({"B": OrganizerMetrics(gauc=0.65, ndcg_at_5=0.49)})

    assert runtime._decomposed_metrics(cast(Any, crashed), cast(Any, incumbent)) == {}
    assert runtime._decomposed_metrics(None, cast(Any, incumbent)) == {}


def test_the_tuning_that_produced_a_score_is_recorded() -> None:
    """Verbatim from the campaign that produced this session's best measured delta."""

    config = {
        "candidate_family": "causal_identity_lambdarank",
        "l2": 8.0,
        "num_rounds": 450,
        "min_data_in_leaf": 300,
        "lambdarank_truncation_level": 5,
    }
    materialized = SimpleNamespace(file=lambda path: SimpleNamespace(content=json.dumps(config)))

    rendered = runtime._bounded_config_json(runtime._candidate_config(materialized))

    assert rendered is not None
    assert json.loads(rendered) == config


def test_an_unreadable_or_oversized_config_is_absent_not_fatal() -> None:
    """Config is evidence, never a failure, and is bounded before it reaches a prompt."""

    def raises(_path: str) -> object:
        raise RuntimeError("no config.json in this candidate")

    assert runtime._candidate_config(SimpleNamespace(file=raises)) is None
    assert runtime._candidate_config(SimpleNamespace()) is None
    assert (
        runtime._candidate_config(SimpleNamespace(file=lambda _p: SimpleNamespace(content="{[")))
        is None
    )
    assert runtime._bounded_config_json(None) is None
    assert runtime._bounded_config_json({}) is None
    assert runtime._bounded_config_json({"k": "x" * 4000}) is None

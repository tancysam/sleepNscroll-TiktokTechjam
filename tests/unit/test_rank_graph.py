from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kuairand_agent.candidates.fusion import TIE_POLICY
from kuairand_agent.domain.decisions import EvidenceStage
from kuairand_agent.domain.identity import PredictionId
from kuairand_agent.finalization.rank_graph import (
    PredictionVector,
    RankGraph,
    RankGraphError,
    RankGraphEvaluator,
    RankGraphMember,
    RankGraphRow,
)
from kuairand_agent.scoring.submission import prediction_digest


def _prediction_id(character: str) -> PredictionId:
    return PredictionId(character * 64)


def _fixture(
    stage: EvidenceStage = EvidenceStage.FINAL,
) -> tuple[RankGraph, dict[PredictionId, PredictionVector]]:
    rows = (
        RankGraphRow(0, "u", "dup"),
        RankGraphRow(1, "u", "middle"),
        RankGraphRow(2, "u", "dup"),
        RankGraphRow(3, "v", "x"),
        RankGraphRow(4, "v", "x"),
    )
    first_scores = [3.0, 2.0, 1.0, 9.0, 9.0]
    second_scores = [1.0, 2.0, 3.0, 0.0, 1.0]
    first_id = _prediction_id("a")
    second_id = _prediction_id("b")
    graph = RankGraph(
        stage=stage,
        rows=rows,
        members=(
            RankGraphMember(first_id, prediction_digest(first_scores), 0.75),
            RankGraphMember(second_id, prediction_digest(second_scores), 0.25),
        ),
    )
    predictions = {
        first_id: PredictionVector(first_id, stage, rows, first_scores),
        second_id: PredictionVector(second_id, stage, rows, second_scores),
    }
    return graph, predictions


@pytest.mark.parametrize("stage", tuple(EvidenceStage))
def test_rank_graph_is_exact_at_inner_outer_and_final(stage: EvidenceStage) -> None:
    graph, predictions = _fixture(stage)

    result = RankGraphEvaluator().evaluate(graph, predictions)

    np.testing.assert_array_equal(result.scores, [0.75, 0.5, 0.25, 0.375, 0.625])
    assert result.prediction_sha256 == prediction_digest(result.scores)
    assert result.prediction_id == PredictionId.from_rank_graph(
        ordered_row_ids=(0, 1, 2, 3, 4),
        prediction_sha256=result.prediction_sha256,
        rank_graph_sha256=graph.digest,
    )
    assert result.rank_graph_sha256 == graph.digest
    assert result.stage is stage
    assert result.rows == graph.rows
    assert not result.scores.flags.writeable


def test_rank_graph_member_order_and_weight_positions_are_semantic() -> None:
    graph, predictions = _fixture()
    first = RankGraphEvaluator().evaluate(graph, predictions)
    reversed_graph = replace(
        graph,
        members=(
            replace(graph.members[1], weight=0.75),
            replace(graph.members[0], weight=0.25),
        ),
    )

    reversed_result = RankGraphEvaluator().evaluate(reversed_graph, predictions)

    assert not np.array_equal(first.scores, reversed_result.scores)
    assert first.rank_graph_sha256 != reversed_result.rank_graph_sha256
    assert first.prediction_id != reversed_result.prediction_id


def test_rank_graph_rejects_row_reorder_content_change_and_member_set_drift() -> None:
    graph, predictions = _fixture()
    first_id = graph.members[0].prediction_id
    changed_rows = list(graph.rows)
    changed_rows[0] = RankGraphRow(0, "different-user", "dup")
    predictions[first_id] = PredictionVector(
        first_id,
        EvidenceStage.FINAL,
        tuple(changed_rows),
        [3.0, 2.0, 1.0, 9.0, 9.0],
    )

    with pytest.raises(RankGraphError, match="ordered row alignment"):
        RankGraphEvaluator().evaluate(graph, predictions)

    graph, predictions = _fixture()
    predictions[first_id] = PredictionVector(
        first_id,
        EvidenceStage.FINAL,
        graph.rows,
        [3.0, 2.0, 1.0, 9.0, 8.0],
    )
    with pytest.raises(RankGraphError, match="content digest changed"):
        RankGraphEvaluator().evaluate(graph, predictions)

    graph, predictions = _fixture()
    predictions.pop(graph.members[0].prediction_id)
    with pytest.raises(RankGraphError, match="exactly the declared"):
        RankGraphEvaluator().evaluate(graph, predictions)


def test_rank_graph_schema_weights_ties_and_rows_are_closed() -> None:
    graph, _ = _fixture()
    assert graph.tie_policy == TIE_POLICY
    with pytest.raises(RankGraphError, match="sum exactly"):
        replace(graph, members=(replace(graph.members[0], weight=0.70), graph.members[1]))
    with pytest.raises(RankGraphError, match="tie_policy"):
        replace(graph, tie_policy="row-order-wins")
    with pytest.raises(RankGraphError, match="zero-based"):
        replace(graph, rows=(RankGraphRow(1, "u", "x"), *graph.rows[1:]))

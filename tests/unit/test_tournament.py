from __future__ import annotations

from dataclasses import replace

import pytest

from kuairand_agent.domain.identity import ExperimentId, FamilyId
from kuairand_agent.search.tournament import (
    InnerMetrics,
    MatchedControlTournament,
    MatchedEvaluationContext,
    ParetoArchive,
    ScientificResult,
    TournamentDisposition,
    TournamentError,
    TournamentEvidence,
)


def _experiment(character: str) -> ExperimentId:
    return ExperimentId(character * 64)


def _family(character: str) -> FamilyId:
    return FamilyId(character * 64)


_CONTEXT = MatchedEvaluationContext(
    row_set_sha256="f" * 64,
    fold_protocol="temporal-ab-v1",
    fidelity="fold-b",
    seeds=(0,),
)


def _evidence(
    identity: str,
    *,
    family: str,
    score: float,
    parent: ExperimentId | None = None,
    complementarity: float = 0.5,
    stability: float = 0.8,
    runtime: float = 10.0,
    memory: float = 100.0,
) -> TournamentEvidence:
    return TournamentEvidence(
        experiment_id=_experiment(identity),
        family_id=_family(family),
        parent_experiment_id=parent,
        context=_CONTEXT,
        result=ScientificResult.COMPLETED,
        metrics=InnerMetrics(primary=score, gauc=score, ndcg5=score),
        complementarity=complementarity,
        stability=stability,
        runtime_seconds=runtime,
        peak_memory_mb=memory,
    )


def test_candidate_must_beat_matched_parent_and_official_fallback() -> None:
    fallback = _evidence("a", family="1", score=0.600)
    parent = _evidence("b", family="2", score=0.601)
    candidate = _evidence("c", family="3", score=0.604, parent=parent.experiment_id)
    tournament = MatchedControlTournament(practical_inner_margin=0.002)

    decision = tournament.assess(
        candidate=candidate,
        parent=parent,
        official_fallback=fallback,
    )

    assert decision.disposition is TournamentDisposition.ADVANCE
    assert decision.matched_parent_delta == pytest.approx(0.003)
    assert decision.fallback_delta == pytest.approx(0.004)
    assert decision.protected_query_allowed
    assert decision.archive_admitted


def test_submaterial_candidate_cannot_spend_protected_query() -> None:
    fallback = _evidence("a", family="1", score=0.600)
    parent = _evidence("b", family="2", score=0.601)
    candidate = _evidence("c", family="3", score=0.602, parent=parent.experiment_id)
    decision = MatchedControlTournament(practical_inner_margin=0.002).assess(
        candidate=candidate,
        parent=parent,
        official_fallback=fallback,
    )
    assert decision.disposition is TournamentDisposition.SCIENTIFIC_REJECTION
    assert not decision.protected_query_allowed


def test_infrastructure_failure_is_retryable_not_scientific_loss() -> None:
    parent = _evidence("b", family="2", score=0.601)
    failed = TournamentEvidence(
        experiment_id=_experiment("c"),
        family_id=_family("3"),
        parent_experiment_id=parent.experiment_id,
        context=_CONTEXT,
        result=ScientificResult.INFRASTRUCTURE_FAILURE,
        metrics=None,
        complementarity=None,
        stability=None,
        runtime_seconds=1.0,
        peak_memory_mb=10.0,
        diagnostic="worker exited",
    )
    decision = MatchedControlTournament().assess(
        candidate=failed,
        parent=parent,
        official_fallback=_evidence("a", family="1", score=0.600),
    )
    assert decision.disposition is TournamentDisposition.INFRASTRUCTURE_RETRY
    assert not decision.protected_query_allowed


def test_unmatched_fold_or_parent_is_rejected() -> None:
    parent = _evidence("b", family="2", score=0.601)
    candidate = _evidence("c", family="3", score=0.604, parent=_experiment("d"))
    with pytest.raises(TournamentError, match="declared matched parent"):
        MatchedControlTournament().assess(
            candidate=candidate,
            parent=parent,
            official_fallback=_evidence("a", family="1", score=0.600),
        )

    candidate = replace(candidate, parent_experiment_id=parent.experiment_id)
    other_context = replace(_CONTEXT, seeds=(1,))
    with pytest.raises(TournamentError, match="contexts are not matched"):
        MatchedControlTournament().assess(
            candidate=replace(candidate, context=other_context),
            parent=parent,
            official_fallback=_evidence("a", family="1", score=0.600),
        )


def test_pareto_archive_preserves_diversity_and_rejects_dominated_entry() -> None:
    archive = ParetoArchive(max_slots=3, max_per_family=1)
    first = _evidence("a", family="1", score=0.61, runtime=20.0)
    same_family_worse = _evidence("b", family="1", score=0.60, runtime=30.0)
    diverse = _evidence("c", family="2", score=0.605, complementarity=0.9, runtime=40.0)

    assert archive.add(first)
    assert not archive.add(same_family_worse)
    assert archive.add(diverse)
    assert tuple(value.experiment_id for value in archive.entries()) == (
        first.experiment_id,
        diverse.experiment_id,
    )

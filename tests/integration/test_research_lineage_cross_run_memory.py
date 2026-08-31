"""Cross-run memory: a fresh campaign inherits typed evidence from earlier ones.

These tests exercise the actual wiring in ``full_campaign_runtime`` — not just the standalone
:class:`~kuairand_agent.campaign.store.ResearchLineageLedger` (covered in
``test_research_lineage_ledger.py``) — to prove that seeding actually changes behavior, and that
it never contaminates this campaign's own accounting.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from kuairand_agent.campaign import full_campaign_runtime as runtime
from kuairand_agent.campaign.scientific import CampaignStopReason, ScientificCampaignResult
from kuairand_agent.campaign.selector import IncumbentEvidence, OrganizerMetrics
from kuairand_agent.campaign.store import ResearchLineageLedger
from kuairand_agent.execution.artifacts import ArtifactRef, ArtifactStore
from kuairand_agent.research.schemas import Proposal, Reflection, RequiredField
from tests.integration.test_full_campaign_runtime import _breadth_fixture, _typed_branch_rejection

_BENCHMARK = "b" * 64
_STARTER = "c" * 64
_SOURCE = "d" * 64


class _StubLineage:
    """Stand-in for ``LiveResearchLineage`` exposing only what admission recording reads."""

    def __init__(self, *, candidate_id: str, proposal: object) -> None:
        self.candidate_id = candidate_id
        self.proposal = proposal


_EVALUATION = "e" * 64


def test_prior_root_failure_totals_trip_the_circuit_breaker_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family already rejected twice in earlier campaigns closes on its very first repeat here."""

    opaque = cast(Any, object())
    fingerprint = _typed_branch_rejection(
        candidate_id="candidate-01", root_code="reserved_filename", root_subject="baseline.py"
    ).root_failure.fingerprint

    def prepare(**kwargs: object) -> object:
        raise _typed_branch_rejection(
            candidate_id=f"candidate-{cast(int, kwargs['scientific_iteration']):02d}",
            root_code="reserved_filename",
            root_subject="baseline.py",
        )

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="fresh-campaign",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=10,
        safe_context_factory=cast(Any, lambda records: opaque),
        continue_check=lambda: True,
        prior_root_failure_totals=MappingProxyType({fingerprint: 2}),
    )

    assert prepared.status == "repeated_pre_admission_failure"
    assert prepared.branches_attempted == 1
    # Only this campaign's own branch is in the reported evidence — the two prior-campaign
    # rejections are not double-counted into this run's own accounting.
    assert len(prepared.rejected_records) == 1


def test_without_prior_totals_the_same_sequence_needs_three_local_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    attempts: list[int] = []

    def prepare(**kwargs: object) -> object:
        attempts.append(cast(int, kwargs["scientific_iteration"]))
        raise _typed_branch_rejection(
            candidate_id=f"candidate-{len(attempts):02d}",
            root_code="reserved_filename",
            root_subject="baseline.py",
        )

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="fresh-campaign",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=10,
        safe_context_factory=cast(Any, lambda records: opaque),
        continue_check=lambda: True,
    )

    assert prepared.status == "repeated_pre_admission_failure"
    assert len(attempts) == 3


def test_rejection_is_durably_recorded_and_visible_only_as_advisory_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    ledger_path = tmp_path / "lineage.sqlite3"
    seen_context_sizes: list[int] = []

    def prepare(**kwargs: object) -> object:
        raise _typed_branch_rejection(
            candidate_id="candidate-01", root_code="reserved_filename", root_subject="baseline.py"
        )

    def safe_context_factory(records: tuple[object, ...]) -> object:
        seen_context_sizes.append(len(records))
        return opaque

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="run-a",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=3,
        safe_context_factory=cast(Any, safe_context_factory),
        continue_check=lambda: True,
        benchmark_digest=_BENCHMARK,
        starter_digest=_STARTER,
        source_digest=_SOURCE,
        lineage_ledger_path=ledger_path,
        evaluation_digest=_EVALUATION,
        prior_advisory_records=(),
    )

    assert prepared.status == "repeated_pre_admission_failure"
    # The durable ledger now has the evidence for the *next* campaign to inherit.
    ledger = ResearchLineageLedger.open(ledger_path)
    try:
        summary = ledger.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_SOURCE
        )
        assert dict(summary.proposal_family_rejection_totals) == {"pairwise": 3}
        assert all(event.campaign_id == "run-a" for event in summary.recent_events)
    finally:
        ledger.close()

    # This run's own rejected_records account for exactly its own three branches; nothing from
    # the ledger read leaked into what gets reported as *this* campaign's history.
    assert len(prepared.rejected_records) == 3


def test_prepare_live_lineage_portfolio_does_not_itself_record_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission (with real fold metrics) is recorded by the caller after training completes.

    ``_prepare_live_lineage_portfolio`` only prepares the lineage; it never runs
    ``run_scientific_campaign``, so it has no fold evidence to attach. It must not write a
    metrics-free placeholder row either — the caller (``run_provider_free_campaign`` or
    ``_run_autonomous_followups``) owns the one durable admission event per accepted candidate.
    """
    from kuairand_agent.research.schemas import Proposal, RequiredField

    opaque = cast(Any, object())
    ledger_path = tmp_path / "lineage.sqlite3"
    proposal = Proposal(
        proposal_id="listwise-v1",
        hypothesis="A listwise ranking objective improves within-user ordering.",
        mechanism="Fit a listwise ranking objective over logged impressions.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="official-fm-fallback-seed-4",
        principal_change="Replace the pointwise objective with listwise ranking.",
        files_expected=("candidate.py", "config.json"),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:user_id",
                "inference_input",
                "Group rows by user for listwise ranking.",
            ),
        ),
        objective="listwise ranking objective",
        sampling="all logged impressions",
        grouping="user impression groups",
        weighting="uniform rows",
        causal_cutoff="No response-derived feature is used.",
        estimated_runtime_seconds=30,
        estimated_memory_mb=256,
        smoke_plan="Fit on two users and predict four rows.",
        inner_fold_plan="Screen seed 0 on Fold B.",
        falsification_criteria="Reject without a Fold B improvement.",
        promotion_criteria="Require positive mean across A and B.",
        maximum_repairs=1,
        rollback_parent_id="official-fm-fallback-seed-4",
        attributions=("listwise ranking literature",),
    )
    admitted = _StubLineage(candidate_id="candidate-01", proposal=proposal)

    def prepare(**kwargs: object) -> object:
        return admitted

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="run-a",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=3,
        safe_context_factory=cast(Any, lambda records: opaque),
        continue_check=lambda: True,
        benchmark_digest=_BENCHMARK,
        starter_digest=_STARTER,
        source_digest=_SOURCE,
        lineage_ledger_path=ledger_path,
        evaluation_digest=_EVALUATION,
    )

    assert prepared.status == "accepted"
    # No rejection occurred either, so nothing should have created the ledger file at all.
    assert not ledger_path.exists()


def test_autonomous_followup_records_real_fold_metrics_against_the_actual_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After training, the ledger sees this candidate's real result against its real parent.

    The parent's own fold metrics must come from the incumbent as it stood *before* this
    attempt — never the fixed official-FM qualification numbers alone — since a run's own
    incumbent can already be a previously promoted generated candidate.
    """

    (
        request,
        _conv,
        first_result,
        first_reflection,
        transcript,
        runtime_template,
        first_lineage,
    ) = _breadth_fixture(tmp_path, proposal_breadth=1)
    config = runtime_template.config
    fallback = first_result.fallback
    ledger_path = tmp_path / "lineage.sqlite3"

    # The current parent is NOT the plain official FM: Fold A already differs from Fold B,
    # simulating a campaign whose incumbent is itself an earlier promoted candidate.
    current_parent = IncumbentEvidence(
        candidate_id="previously-promoted-candidate",
        inner_by_fold=(
            ("A", OrganizerMetrics(gauc=0.6580043435096741, ndcg_at_5=0.5562537312507629)),
            ("B", OrganizerMetrics(gauc=0.6583324670791626, ndcg_at_5=0.49251559376716614)),
        ),
        outer_by_seed=fallback.outer_by_seed,
        evidence_receipt_digest=fallback.evidence_receipt_digest,
        replayable=True,
        eligible=True,
        official_fm=True,
    )
    first_result = ScientificCampaignResult(
        config_digest=config.digest,
        fallback=fallback,
        incumbent=current_parent,
        candidates=(),
        public_feedback=(),
        convergence=first_result.convergence,
        launches_used=first_result.launches_used,
        elapsed_seconds=first_result.elapsed_seconds,
        stop_reason=first_result.stop_reason,
    )

    proposal = Proposal(
        proposal_id="pairwise-v1",
        hypothesis="A pairwise objective aligned to GAUC improves within-user ordering.",
        mechanism="Fit a pairwise ranking objective over logged impressions.",
        expected_metric_effects=("GAUC",),
        parent_candidate_id="previously-promoted-candidate",
        principal_change="Replace the pointwise objective with pairwise ranking.",
        files_expected=("candidate.py", "config.json"),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:user_id",
                "inference_input",
                "Group rows by user for pairwise ranking.",
            ),
        ),
        objective="pairwise ranking objective",
        sampling="all logged impressions",
        grouping="user impression groups",
        weighting="uniform rows",
        causal_cutoff="No response-derived feature is used.",
        estimated_runtime_seconds=30,
        estimated_memory_mb=256,
        smoke_plan="Fit on two users and predict four rows.",
        inner_fold_plan="Screen seed 0 on Fold B.",
        falsification_criteria="Reject without a Fold B improvement.",
        promotion_criteria="Require positive mean across A and B.",
        maximum_repairs=1,
        rollback_parent_id="previously-promoted-candidate",
        attributions=("pairwise ranking literature",),
    )

    def prepare_lineage(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            candidate_id="live-candidate-2",
            parent=kwargs["parent"],
            proposal=proposal,
            materialized=object(),
        )

    def not_promoted_with_metrics(**kwargs: object) -> ScientificCampaignResult:
        candidates = cast(tuple[Any, ...], kwargs["candidates"])
        candidate = candidates[0]
        run_b = SimpleNamespace(
            digest="1" * 64,
            metrics=OrganizerMetrics(gauc=0.6582522392272949, ndcg_at_5=0.4926654100418091),
        )
        run_a = SimpleNamespace(
            digest="2" * 64,
            metrics=OrganizerMetrics(gauc=0.6579948663711548, ndcg_at_5=0.5567957162857056),
        )
        own_result = SimpleNamespace(
            candidate=candidate,
            outcome=SimpleNamespace(value="inner_rejected"),
            runs=(run_b, run_a),
            reason="inner_mean_not_positive",
            remedy=None,
        )
        return ScientificCampaignResult(
            config_digest=config.digest,
            fallback=fallback,
            incumbent=current_parent,  # not promoted: incumbent is unchanged
            candidates=cast(Any, (own_result,)),
            public_feedback=(),
            convergence=cast(Any, kwargs["initial_convergence"]).update_after_iteration(None),
            launches_used=cast(int, kwargs["initial_launches_used"]) + 2,
            elapsed_seconds=float(cast(float, kwargs["initial_elapsed_seconds"])) + 1.0,
            stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
        )

    def reflect(**kwargs: object) -> tuple[str, str, ArtifactRef, Reflection]:
        return (
            "reflection-request-2",
            "reflection-response-2",
            transcript,
            Reflection(
                response_id="reflection-2",
                summary="No material gain.",
                recommendation="propose_next",
                lessons=("Try a different mechanism.",),
            ),
        )

    monkeypatch.setattr(runtime, "_safe_context", lambda **_kwargs: object())
    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare_lineage)
    monkeypatch.setattr(
        runtime,
        "_ensure_lineage_ledger",
        lambda **kwargs: (object(), f"iteration-{kwargs['scientific_iteration']:02d}"),
    )
    monkeypatch.setattr(
        runtime,
        "_generated_scientific_candidate",
        lambda **kwargs: SimpleNamespace(candidate_id=kwargs["candidate_id"]),
    )
    monkeypatch.setattr(runtime, "_open_outer_ledger", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        runtime, "DurableScientificLedgerAdapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(runtime, "run_scientific_campaign", not_promoted_with_metrics)
    monkeypatch.setattr(runtime, "_candidate_selection", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_reflect", reflect)

    runtime._run_autonomous_followups(
        request=cast(Any, request),
        data=cast(Any, object()),
        runtime_template=runtime_template,
        scientific_config=config,
        fallback=fallback,
        outer_ledger_path=tmp_path / "outer.sqlite3",
        candidate_limits=cast(Any, object()),
        dataset_digest="d" * 64,
        context_evidence=cast(Any, object()),
        validation_inputs=cast(Any, object()),
        final_inputs=cast(Any, object()),
        research_model=cast(Any, object()),
        first_lineage=cast(Any, first_lineage),
        first_result=first_result,
        first_selection=None,
        first_reflection=first_reflection,
        first_reflection_evidence=("request-1", "response-1", transcript),
        prior_records=(),
        lineage_ledger_path=ledger_path,
        evaluation_digest=_EVALUATION,
    )

    ledger = ResearchLineageLedger.open(ledger_path)
    try:
        summary = ledger.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_SOURCE
        )
        assert len(summary.recent_events) >= 1
        event = summary.recent_events[0]
        assert event.outcome == "admitted"
        assert event.promoted is False
        assert event.inner_fold_a is not None and event.inner_fold_b is not None
        assert event.inner_fold_a.primary == pytest.approx(0.6073952913284302)
        assert event.inner_fold_b.primary == pytest.approx(0.575458824634552)
        # The parent reference is the *actual* current incumbent's own fold metrics (their exact
        # arithmetic mean), not the fixed official-FM qualification numbers.
        assert event.parent_fold_a_primary == pytest.approx(0.6071290373802185)
        assert event.parent_fold_b_primary == pytest.approx(
            (0.6583324670791626 + 0.49251559376716614) / 2
        )
    finally:
        ledger.close()


def _parent_incumbent() -> IncumbentEvidence:
    from tests.unit.test_scientific_campaign import _fallback

    return _fallback()


def _assert_fold_pairs_consistent(metrics: runtime._LineageAdmissionMetrics) -> None:
    """Each fold's (inner metrics, parent reference) pair must travel together; ``promoted``
    must be null unless *both* folds are present -- the exact invariant the ledger's CHECK
    constraint enforces. Fold A and Fold B are otherwise independent of each other."""

    assert (metrics.inner_fold_a is None) == (metrics.parent_fold_a_primary is None)
    assert (metrics.inner_fold_b is None) == (metrics.parent_fold_b_primary is None)
    if metrics.promoted is not None:
        assert metrics.inner_fold_a is not None
        assert metrics.inner_fold_b is not None


def test_lineage_admission_metrics_never_returns_a_partial_mix() -> None:
    """Regression: each fold pair is all-or-nothing, and ``promoted`` requires both folds.

    This is a direct, exhaustive test of every branch in ``_lineage_admission_metrics``. A real
    live campaign crashed because an earlier version of this function returned a real
    ``promoted=False`` alongside null fold metrics for a screen-rejected candidate, violating the
    ledger's CHECK constraint. A *screen-rejected* candidate (real Fold B evidence, no Fold A) is
    legitimate and must be preserved, not discarded -- that's the whole point of this function.
    """

    parent = _parent_incumbent()

    # Branch 1: no matching candidate in result.candidates at all (e.g. budget-rejected before
    # any processing).
    result_no_candidate = cast(Any, SimpleNamespace(incumbent=parent, candidates=()))
    metrics = runtime._lineage_admission_metrics(
        parent_incumbent=parent, result=result_no_candidate, candidate_id="candidate-01"
    )
    _assert_fold_pairs_consistent(metrics)
    assert metrics.inner_fold_a is None and metrics.inner_fold_b is None
    assert metrics.promoted is None

    # Branch 2: candidate found, but zero runs at all (e.g. BUDGET_REJECTED before any launch).
    own_result_no_runs = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="candidate-01"), runs=()
    )
    result_no_runs = cast(Any, SimpleNamespace(incumbent=parent, candidates=(own_result_no_runs,)))
    metrics = runtime._lineage_admission_metrics(
        parent_incumbent=parent, result=result_no_runs, candidate_id="candidate-01"
    )
    _assert_fold_pairs_consistent(metrics)
    assert metrics.inner_fold_a is None and metrics.inner_fold_b is None
    assert metrics.promoted is None

    # Branch 3: SCREEN_REJECTED -- exactly one run (Fold B), no Fold A. The real Fold B evidence
    # must be preserved on its own; this is the exact shape that crashed the live run before the
    # fix, and the exact shape this whole feature exists to stop discarding.
    own_result_screen_rejected = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="candidate-01"),
        runs=(
            SimpleNamespace(
                digest="1" * 64,
                metrics=OrganizerMetrics(gauc=0.6, ndcg_at_5=0.5),
            ),
        ),
    )
    result_screen_rejected = cast(
        Any, SimpleNamespace(incumbent=parent, candidates=(own_result_screen_rejected,))
    )
    metrics = runtime._lineage_admission_metrics(
        parent_incumbent=parent, result=result_screen_rejected, candidate_id="candidate-01"
    )
    _assert_fold_pairs_consistent(metrics)
    assert metrics.inner_fold_b is not None
    assert metrics.inner_fold_a is None
    assert metrics.promoted is None

    # Branch 4: two runs present, but Fold A has no metrics (CALLBACK_FAILED on the second run).
    # Fold B's real evidence must still be preserved even though Fold A cannot be.
    own_result_missing_fold_a = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="candidate-01"),
        runs=(
            SimpleNamespace(
                digest="1" * 64,
                metrics=OrganizerMetrics(gauc=0.6, ndcg_at_5=0.5),
            ),
            SimpleNamespace(digest="2" * 64, metrics=None),
        ),
    )
    result_missing_fold_a = cast(
        Any, SimpleNamespace(incumbent=parent, candidates=(own_result_missing_fold_a,))
    )
    metrics = runtime._lineage_admission_metrics(
        parent_incumbent=parent, result=result_missing_fold_a, candidate_id="candidate-01"
    )
    _assert_fold_pairs_consistent(metrics)
    assert metrics.inner_fold_b is not None
    assert metrics.inner_fold_a is None
    assert metrics.promoted is None

    # Branch 5: full metrics on both sides -- the only case where `promoted` can be populated.
    own_result_complete = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="candidate-01"),
        runs=(
            SimpleNamespace(
                digest="1" * 64,
                metrics=OrganizerMetrics(gauc=0.66, ndcg_at_5=0.53),
            ),
            SimpleNamespace(
                digest="2" * 64,
                metrics=OrganizerMetrics(gauc=0.67, ndcg_at_5=0.54),
            ),
        ),
    )
    result_complete = cast(
        Any, SimpleNamespace(incumbent=parent, candidates=(own_result_complete,))
    )
    metrics = runtime._lineage_admission_metrics(
        parent_incumbent=parent, result=result_complete, candidate_id="candidate-01"
    )
    _assert_fold_pairs_consistent(metrics)
    assert metrics.promoted is False
    assert metrics.inner_fold_a is not None
    assert metrics.inner_fold_b is not None


def test_lineage_ledger_write_survives_an_incomplete_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end regression: an autonomous iteration whose candidate has no fold evidence must
    still record a valid (fully-null-metrics) ledger row instead of crashing the campaign."""

    (
        request,
        _conv,
        first_result,
        first_reflection,
        transcript,
        runtime_template,
        first_lineage,
    ) = _breadth_fixture(tmp_path, proposal_breadth=1)
    config = runtime_template.config
    fallback = first_result.fallback
    ledger_path = tmp_path / "lineage.sqlite3"

    proposal = Proposal(
        proposal_id="pairwise-v1",
        hypothesis="A pairwise objective aligned to GAUC improves within-user ordering.",
        mechanism="Fit a pairwise ranking objective over logged impressions.",
        expected_metric_effects=("GAUC",),
        parent_candidate_id="official-fm",
        principal_change="Replace the pointwise objective with pairwise ranking.",
        files_expected=("candidate.py", "config.json"),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:user_id",
                "inference_input",
                "Group rows by user for pairwise ranking.",
            ),
        ),
        objective="pairwise ranking objective",
        sampling="all logged impressions",
        grouping="user impression groups",
        weighting="uniform rows",
        causal_cutoff="No response-derived feature is used.",
        estimated_runtime_seconds=30,
        estimated_memory_mb=256,
        smoke_plan="Fit on two users and predict four rows.",
        inner_fold_plan="Screen seed 0 on Fold B.",
        falsification_criteria="Reject without a Fold B improvement.",
        promotion_criteria="Require positive mean across A and B.",
        maximum_repairs=1,
        rollback_parent_id="official-fm",
        attributions=("pairwise ranking literature",),
    )

    def prepare_lineage(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            candidate_id="live-candidate-2",
            parent=kwargs["parent"],
            proposal=proposal,
            materialized=object(),
        )

    def screen_rejected(**kwargs: object) -> ScientificCampaignResult:
        candidates = cast(tuple[Any, ...], kwargs["candidates"])
        candidate = candidates[0]
        own_result = SimpleNamespace(
            candidate=candidate,
            outcome=SimpleNamespace(value="screen_rejected"),
            runs=(SimpleNamespace(digest="1" * 64, metrics=None),),
            reason="fold_b_screen_failed",
            remedy=None,
        )
        return ScientificCampaignResult(
            config_digest=config.digest,
            fallback=fallback,
            incumbent=fallback,  # unchanged: this candidate never had a chance to be promoted
            candidates=cast(Any, (own_result,)),
            public_feedback=(),
            convergence=cast(Any, kwargs["initial_convergence"]).update_after_iteration(None),
            launches_used=cast(int, kwargs["initial_launches_used"]) + 1,
            elapsed_seconds=float(kwargs["initial_elapsed_seconds"]) + 1.0,  # type: ignore[arg-type]
            stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
        )

    def reflect(**kwargs: object) -> tuple[str, str, ArtifactRef, Reflection]:
        return (
            "reflection-request-2",
            "reflection-response-2",
            transcript,
            Reflection(
                response_id="reflection-2",
                summary="Screened out before Fold A.",
                recommendation="propose_next",
                lessons=("Try a different mechanism.",),
            ),
        )

    monkeypatch.setattr(runtime, "_safe_context", lambda **_kwargs: object())
    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare_lineage)
    monkeypatch.setattr(
        runtime,
        "_ensure_lineage_ledger",
        lambda **kwargs: (object(), f"iteration-{kwargs['scientific_iteration']:02d}"),
    )
    monkeypatch.setattr(
        runtime,
        "_generated_scientific_candidate",
        lambda **kwargs: SimpleNamespace(candidate_id=kwargs["candidate_id"]),
    )
    monkeypatch.setattr(runtime, "_open_outer_ledger", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        runtime, "DurableScientificLedgerAdapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(runtime, "run_scientific_campaign", screen_rejected)
    monkeypatch.setattr(runtime, "_candidate_selection", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_reflect", reflect)

    # Must not raise -- this is exactly the call that crashed the live campaign.
    runtime._run_autonomous_followups(
        request=cast(Any, request),
        data=cast(Any, object()),
        runtime_template=runtime_template,
        scientific_config=config,
        fallback=fallback,
        outer_ledger_path=tmp_path / "outer.sqlite3",
        candidate_limits=cast(Any, object()),
        dataset_digest="d" * 64,
        context_evidence=cast(Any, object()),
        validation_inputs=cast(Any, object()),
        final_inputs=cast(Any, object()),
        research_model=cast(Any, object()),
        first_lineage=cast(Any, first_lineage),
        first_result=first_result,
        first_selection=None,
        first_reflection=first_reflection,
        first_reflection_evidence=("request-1", "response-1", transcript),
        prior_records=(),
        lineage_ledger_path=ledger_path,
        evaluation_digest=_EVALUATION,
    )

    ledger = ResearchLineageLedger.open(ledger_path)
    try:
        summary = ledger.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_SOURCE
        )
        assert len(summary.recent_events) >= 1
        event = summary.recent_events[0]
        assert event.outcome == "admitted"
        assert event.inner_fold_a is None
        assert event.inner_fold_b is None
        assert event.parent_fold_a_primary is None
        assert event.parent_fold_b_primary is None
        assert event.promoted is None
    finally:
        ledger.close()


def test_lineage_ledger_preserves_a_screen_rejected_candidates_real_fold_b_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate that fails the Fold B screen still has a real, worth-keeping Fold B result.

    Requiring both folds before recording anything (the original design) would silently discard
    this — the most common outcome once ``screen_margin`` is enforced for real — leaving memory
    blind to exactly the data point it most needs.
    """

    (
        request,
        _conv,
        first_result,
        first_reflection,
        transcript,
        runtime_template,
        first_lineage,
    ) = _breadth_fixture(tmp_path, proposal_breadth=1)
    config = runtime_template.config
    fallback = first_result.fallback
    ledger_path = tmp_path / "lineage.sqlite3"

    proposal = Proposal(
        proposal_id="listwise-v1",
        hypothesis="A listwise objective improves within-user ordering.",
        mechanism="Fit a listwise ranking objective over logged impressions.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="official-fm",
        principal_change="Replace the pointwise objective with listwise ranking.",
        files_expected=("candidate.py", "config.json"),
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:user_id",
                "inference_input",
                "Group rows by user for listwise ranking.",
            ),
        ),
        objective="listwise ranking objective",
        sampling="all logged impressions",
        grouping="user impression groups",
        weighting="uniform rows",
        causal_cutoff="No response-derived feature is used.",
        estimated_runtime_seconds=30,
        estimated_memory_mb=256,
        smoke_plan="Fit on two users and predict four rows.",
        inner_fold_plan="Screen seed 0 on Fold B.",
        falsification_criteria="Reject without a Fold B improvement.",
        promotion_criteria="Require positive mean across A and B.",
        maximum_repairs=1,
        rollback_parent_id="official-fm",
        attributions=("listwise ranking literature",),
    )

    def prepare_lineage(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            candidate_id="live-candidate-2",
            parent=kwargs["parent"],
            proposal=proposal,
            materialized=object(),
        )

    def screen_rejected_with_real_metrics(**kwargs: object) -> ScientificCampaignResult:
        candidates = cast(tuple[Any, ...], kwargs["candidates"])
        candidate = candidates[0]
        own_result = SimpleNamespace(
            candidate=candidate,
            outcome=SimpleNamespace(value="screen_rejected"),
            runs=(
                SimpleNamespace(
                    digest="1" * 64,
                    metrics=OrganizerMetrics(gauc=0.64, ndcg_at_5=0.48),
                ),
            ),
            reason="fold_b_screen_failed",
            remedy=None,
        )
        return ScientificCampaignResult(
            config_digest=config.digest,
            fallback=fallback,
            incumbent=fallback,
            candidates=cast(Any, (own_result,)),
            public_feedback=(),
            convergence=cast(Any, kwargs["initial_convergence"]).update_after_iteration(None),
            launches_used=cast(int, kwargs["initial_launches_used"]) + 1,
            elapsed_seconds=float(kwargs["initial_elapsed_seconds"]) + 1.0,  # type: ignore[arg-type]
            stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
        )

    def reflect(**kwargs: object) -> tuple[str, str, ArtifactRef, Reflection]:
        return (
            "reflection-request-2",
            "reflection-response-2",
            transcript,
            Reflection(
                response_id="reflection-2",
                summary="Screened out before Fold A.",
                recommendation="propose_next",
                lessons=("Try a different mechanism.",),
            ),
        )

    monkeypatch.setattr(runtime, "_safe_context", lambda **_kwargs: object())
    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare_lineage)
    monkeypatch.setattr(
        runtime,
        "_ensure_lineage_ledger",
        lambda **kwargs: (object(), f"iteration-{kwargs['scientific_iteration']:02d}"),
    )
    monkeypatch.setattr(
        runtime,
        "_generated_scientific_candidate",
        lambda **kwargs: SimpleNamespace(candidate_id=kwargs["candidate_id"]),
    )
    monkeypatch.setattr(runtime, "_open_outer_ledger", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        runtime, "DurableScientificLedgerAdapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(runtime, "run_scientific_campaign", screen_rejected_with_real_metrics)
    monkeypatch.setattr(runtime, "_candidate_selection", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_reflect", reflect)

    runtime._run_autonomous_followups(
        request=cast(Any, request),
        data=cast(Any, object()),
        runtime_template=runtime_template,
        scientific_config=config,
        fallback=fallback,
        outer_ledger_path=tmp_path / "outer.sqlite3",
        candidate_limits=cast(Any, object()),
        dataset_digest="d" * 64,
        context_evidence=cast(Any, object()),
        validation_inputs=cast(Any, object()),
        final_inputs=cast(Any, object()),
        research_model=cast(Any, object()),
        first_lineage=cast(Any, first_lineage),
        first_result=first_result,
        first_selection=None,
        first_reflection=first_reflection,
        first_reflection_evidence=("request-1", "response-1", transcript),
        prior_records=(),
        lineage_ledger_path=ledger_path,
        evaluation_digest=_EVALUATION,
    )

    ledger = ResearchLineageLedger.open(ledger_path)
    try:
        summary = ledger.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_SOURCE
        )
        assert len(summary.recent_events) >= 1
        event = summary.recent_events[0]
        assert event.outcome == "admitted"
        assert event.inner_fold_b is not None
        assert event.inner_fold_b.gauc == pytest.approx(0.64)
        assert event.parent_fold_b_primary is not None
        assert event.inner_fold_a is None
        assert event.parent_fold_a_primary is None
        assert event.promoted is None
    finally:
        ledger.close()

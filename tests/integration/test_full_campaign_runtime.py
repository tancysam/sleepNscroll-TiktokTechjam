from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kuairand_agent.campaign import full_campaign_runtime as runtime
from kuairand_agent.campaign.controller import (
    CAMPAIGN_DATABASE_NAME,
    CampaignCreateRequest,
    CampaignEngine,
)
from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.full_campaign import (
    FullCampaignCancelled,
    FullCampaignError,
    FullCampaignOutcome,
    FullCampaignOutcomeRepository,
    FullCampaignProgressLedger,
    FullCampaignStage,
    prepare_campaign_data_plane,
)
from kuairand_agent.campaign.scientific import (
    CampaignStopReason,
    CandidateOutcome,
    ScientificCampaignConfig,
    ScientificCampaignResult,
    ScientificTier,
)
from kuairand_agent.campaign.selector import IncumbentEvidence, OrganizerMetrics
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactRef, ArtifactStore
from kuairand_agent.research.context import AggregateRecord
from kuairand_agent.research.production import (
    LiveResearchBranchRejected,
    ResearchFailureObservation,
)
from kuairand_agent.research.schemas import Reflection, canonical_json_bytes
from tests.integration.test_campaign_controller import FakeClock, build_request
from tests.unit.test_full_campaign import _dataset
from tests.unit.test_scientific_campaign import _config, _fallback

ROOT = Path(__file__).resolve().parents[2]


def _deployment_gate_fixture(
    *,
    representative_primary: float,
    fallback_primary: float = 0.6,
    other_candidate_primaries: tuple[float, float] | None = None,
) -> tuple[object, object, object, object]:
    candidate_id = "generated-candidate"
    representative_metrics = OrganizerMetrics(
        representative_primary,
        representative_primary,
    )
    other_primaries = (
        (representative_primary, representative_primary)
        if other_candidate_primaries is None
        else other_candidate_primaries
    )
    representative_evidence = SimpleNamespace(metrics=representative_metrics)
    candidate_result = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id=candidate_id),
        outcome=CandidateOutcome.PROMOTED_CONFIRMED,
        runs=(representative_evidence,),
        selection=SimpleNamespace(
            selected_candidate_id=candidate_id,
            challenger_candidate_id=candidate_id,
        ),
    )
    result = SimpleNamespace(
        fallback=SimpleNamespace(
            candidate_id="official-fm-fallback-seed-4",
            official_fm=True,
            replayable=True,
            eligible=True,
            outer_by_seed=tuple(
                SimpleNamespace(seed=seed, metrics=OrganizerMetrics(0.6, 0.6))
                for seed in (0, 1, 2)
            ),
        ),
        incumbent=SimpleNamespace(
            candidate_id=candidate_id,
            official_fm=False,
            replayable=True,
            eligible=True,
            outer_by_seed=(
                SimpleNamespace(seed=0, metrics=representative_metrics),
                *(
                    SimpleNamespace(seed=seed, metrics=OrganizerMetrics(primary, primary))
                    for seed, primary in zip((1, 2), other_primaries, strict=True)
                ),
            ),
        ),
        candidates=(candidate_result,),
    )
    representative_record = SimpleNamespace(evidence=representative_evidence)
    qualification = SimpleNamespace(
        fallback=SimpleNamespace(
            seed=4,
            metrics=SimpleNamespace(gauc=fallback_primary, ndcg_at_5=fallback_primary),
        )
    )
    return result, candidate_result, representative_record, qualification


@pytest.mark.parametrize(
    ("representative_primary", "expected"),
    (
        (0.602, False),
        (0.6020000000000001, True),
    ),
)
def test_candidate_artifact_deployment_gate_is_strict_at_material_boundary(
    representative_primary: float,
    expected: bool,
) -> None:
    result, candidate_result, representative_record, qualification = (
        _deployment_gate_fixture(representative_primary=representative_primary)
    )

    assert (
        runtime._candidate_artifact_clears_deployment_gate(
            result=cast(Any, result),
            candidate_result=cast(Any, candidate_result),
            representative_record=cast(Any, representative_record),
            qualification=cast(Any, qualification),
        )
        is expected
    )


def test_research_incumbent_with_tiny_mean_gain_keeps_seed_four_deployment_fallback() -> None:
    # The three candidate seeds average to +0.0000422497590383 over the matched FM controls,
    # while the exact representative seed-0 artifact remains below immutable fallback seed 4.
    result, candidate_result, representative_record, qualification = _deployment_gate_fixture(
        representative_primary=0.599,
        other_candidate_primaries=(0.6005, 0.6006267492771149),
    )

    assert cast(Any, result).incumbent.candidate_id == "generated-candidate"
    assert float(
        sum(item.metrics.primary_decimal for item in cast(Any, result).incumbent.outer_by_seed)
        / 3
        - OrganizerMetrics(0.6, 0.6).primary_decimal
    ) == pytest.approx(
        0.0000422497590383
    )
    assert not runtime._candidate_artifact_clears_deployment_gate(
        result=cast(Any, result),
        candidate_result=cast(Any, candidate_result),
        representative_record=cast(Any, representative_record),
        qualification=cast(Any, qualification),
    )
    candidate_runtime = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="generated-candidate"),
        records={(ScientificTier.OUTER_MATCHED_SEED, 0): representative_record},
    )
    assert (
        runtime._candidate_selection(
            experiment_id="iteration-01",
            result=cast(Any, result),
            runtime=cast(Any, candidate_runtime),
            qualification=cast(Any, qualification),
            features=cast(Any, object()),
            feature_artifacts=cast(Any, object()),
            dataset_digest="d" * 64,
            validation_inputs=cast(Any, object()),
            final_inputs=cast(Any, object()),
            limits=cast(Any, object()),
        )
        is None
    )


def test_candidate_artifact_deployment_gate_fails_closed_on_identity_mismatch() -> None:
    result, candidate_result, representative_record, qualification = _deployment_gate_fixture(
        representative_primary=0.603
    )
    cast(Any, qualification).fallback.seed = 0

    assert not runtime._candidate_artifact_clears_deployment_gate(
        result=cast(Any, result),
        candidate_result=cast(Any, candidate_result),
        representative_record=cast(Any, representative_record),
        qualification=cast(Any, qualification),
    )


def test_unconfirmed_research_result_cannot_pass_artifact_deployment_gate() -> None:
    result, candidate_result, representative_record, qualification = _deployment_gate_fixture(
        representative_primary=0.603
    )
    cast(Any, candidate_result).outcome = CandidateOutcome.PROMOTED_UNCONFIRMED

    assert not runtime._candidate_artifact_clears_deployment_gate(
        result=cast(Any, result),
        candidate_result=cast(Any, candidate_result),
        representative_record=cast(Any, representative_record),
        qualification=cast(Any, qualification),
    )


def _typed_branch_rejection(
    *,
    candidate_id: str,
    root_code: str,
    root_subject: str,
    terminal_code: str = "declared_symbol_unchanged",
    terminal_subject: str = "main",
    family: str = "pairwise",
    signature_seed: str = "1",
) -> LiveResearchBranchRejected:
    root = ResearchFailureObservation.create(
        stage="materialization",
        category="static_policy",
        code=root_code,
        subject=root_subject,
        diagnostic=f"root {root_code}: {root_subject}",
    )
    terminal = ResearchFailureObservation.create(
        stage="materiality",
        category="materiality",
        code=terminal_code,
        subject=terminal_subject,
        diagnostic=f"terminal {terminal_code}: {terminal_subject}",
    )
    return LiveResearchBranchRejected(
        failed_candidate_id=candidate_id,
        repairs_attempted=1,
        diagnostic=terminal.diagnostic,
        root_failure=root,
        terminal_failure=terminal,
        proposal_family=family,
        proposal_signature=hashlib.sha256(signature_seed.encode("ascii")).hexdigest(),
    )


def test_initial_live_lineage_portfolio_skips_an_exhausted_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    attempted: list[int] = []
    context_record_counts: list[int] = []

    def prepare(**kwargs: object) -> object:
        iteration = cast(int, kwargs["scientific_iteration"])
        attempted.append(iteration)
        if iteration == 1:
            raise LiveResearchBranchRejected(
                failed_candidate_id="candidate-01-repair-1",
                repairs_attempted=1,
                diagnostic="declared material symbol did not change executable source",
            )
        return SimpleNamespace(candidate_id="candidate-02")

    def safe_context(records: tuple[object, ...]) -> object:
        context_record_counts.append(len(records))
        return opaque

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="initial-live-portfolio",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=3,
        safe_context_factory=cast(Any, safe_context),
        continue_check=lambda: True,
    )

    assert prepared is not None
    assert prepared.scientific_iteration == 2
    assert prepared.lineage is not None
    assert prepared.lineage.candidate_id == "candidate-02"
    assert len(prepared.rejected_records) == 1
    assert attempted == [1, 2]
    assert context_record_counts == [0, 1]


def test_initial_live_lineage_portfolio_stops_on_third_identical_root_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    attempts: list[int] = []
    contexts: list[tuple[object, ...]] = []

    def prepare(**kwargs: object) -> object:
        iteration = cast(int, kwargs["scientific_iteration"])
        attempts.append(iteration)
        raise _typed_branch_rejection(
            candidate_id=f"candidate-{iteration:02d}",
            root_code="reserved_filename",
            root_subject="baseline.py",
            family=("pairwise" if iteration < 3 else "listwise"),
            signature_seed=str(iteration),
        )

    def safe_context(records: tuple[object, ...]) -> object:
        contexts.append(records)
        return opaque

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="repeated-root-failure",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=10,
        safe_context_factory=cast(Any, safe_context),
        continue_check=lambda: True,
    )

    assert prepared.status == "repeated_pre_admission_failure"
    assert prepared.lineage is None
    assert prepared.safe_context is None
    assert prepared.scientific_iteration is None
    assert prepared.branches_attempted == 3
    assert attempts == [1, 2, 3]
    assert [len(items) for items in contexts] == [0, 1, 2]
    assert len(prepared.rejected_records) == 3
    first, second, third = prepared.rejected_records
    assert first.values["root_failure_total_count"] == 1
    assert first.values["root_failure_consecutive_count"] == 1
    assert second.values["root_failure_total_count"] == 2
    assert second.values["root_failure_consecutive_count"] == 2
    assert second.values["proposal_family_blocked"] is True
    assert third.values["root_failure_total_count"] == 3
    assert third.values["root_failure_consecutive_count"] == 3


def test_initial_live_lineage_portfolio_resets_only_consecutive_root_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    roots = (
        ("reserved_filename", "baseline.py", "pairwise"),
        ("reserved_filename", "baseline.py", "listwise"),
        ("unsupported_suffix", ".csv", "calibration"),
        ("reserved_filename", "baseline.py", "optimization"),
    )

    def prepare(**kwargs: object) -> object:
        iteration = cast(int, kwargs["scientific_iteration"])
        code, subject, family = roots[iteration - 1]
        raise _typed_branch_rejection(
            candidate_id=f"candidate-{iteration:02d}",
            root_code=code,
            root_subject=subject,
            family=family,
            signature_seed=str(iteration),
        )

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="nonconsecutive-root-failure",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=10,
        safe_context_factory=lambda _records: cast(Any, opaque),
        continue_check=lambda: True,
    )

    assert prepared.status == "repeated_pre_admission_failure"
    assert prepared.branches_attempted == 4
    assert prepared.rejected_records[-1].values["root_failure_total_count"] == 3
    assert prepared.rejected_records[-1].values["root_failure_consecutive_count"] == 1


def test_initial_live_lineage_portfolio_exposes_second_same_family_as_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    seen_contexts: list[tuple[AggregateRecord, ...]] = []

    def prepare(**kwargs: object) -> object:
        iteration = cast(int, kwargs["scientific_iteration"])
        if iteration < 3:
            raise _typed_branch_rejection(
                candidate_id=f"candidate-{iteration:02d}",
                root_code=("reserved_filename" if iteration == 1 else "unsupported_suffix"),
                root_subject=("baseline.py" if iteration == 1 else ".csv"),
                family="pairwise",
                signature_seed=str(iteration),
            )
        return SimpleNamespace(candidate_id="candidate-03")

    def safe_context(records: tuple[AggregateRecord, ...]) -> object:
        seen_contexts.append(records)
        return opaque

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    prepared = runtime._prepare_live_lineage_portfolio(
        campaign_id="same-family-block",
        parent=opaque,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=5,
        safe_context_factory=cast(Any, safe_context),
        continue_check=lambda: True,
    )

    assert prepared.status == "accepted"
    assert prepared.branches_attempted == 3
    assert len(seen_contexts) == 3
    assert seen_contexts[-1][-1].values["proposal_family"] == "pairwise"
    assert seen_contexts[-1][-1].values["proposal_family_attempt_count"] == 2
    assert seen_contexts[-1][-1].values["proposal_family_blocked"] is True


def test_initial_live_lineage_portfolio_resumes_persisted_rejections_without_duplicate_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    generated_root = tmp_path / "generated"
    provider_calls: list[int] = []

    def prepare(**kwargs: object) -> object:
        iteration = cast(int, kwargs["scientific_iteration"])
        provider_calls.append(iteration)
        raise _typed_branch_rejection(
            candidate_id=f"candidate-{iteration:02d}",
            root_code="reserved_filename",
            root_subject="baseline.py",
            family=("pairwise" if iteration < 3 else "listwise"),
            signature_seed=str(iteration),
        )

    continue_calls = 0

    def interrupt_after_one_rejection() -> bool:
        nonlocal continue_calls
        continue_calls += 1
        if continue_calls == 1:
            return True
        raise FullCampaignCancelled("synthetic interruption after durable rejection")

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    with pytest.raises(FullCampaignCancelled, match="synthetic interruption"):
        runtime._prepare_live_lineage_portfolio(
            campaign_id="durable-rejection-resume",
            parent=opaque,
            generated_root=generated_root,
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            model=opaque,
            provider="openai",
            maximum_iterations=10,
            safe_context_factory=cast(Any, lambda _records: opaque),
            continue_check=interrupt_after_one_rejection,
        )

    resumed_context_sizes: list[int] = []

    def resumed_safe_context(records: tuple[AggregateRecord, ...]) -> object:
        resumed_context_sizes.append(len(records))
        return opaque

    resumed = runtime._prepare_live_lineage_portfolio(
        campaign_id="durable-rejection-resume",
        parent=opaque,
        generated_root=generated_root,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=10,
        safe_context_factory=cast(Any, resumed_safe_context),
        continue_check=lambda: True,
    )

    assert resumed.status == "repeated_pre_admission_failure"
    assert resumed.branches_attempted == 3
    assert provider_calls == [1, 2, 3]
    assert resumed_context_sizes == [1, 2]
    assert len(resumed.rejected_records) == 3


def test_initial_live_lineage_portfolio_rejects_tampered_rejection_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = cast(Any, object())
    generated_root = tmp_path / "generated"
    continue_calls = 0

    def continue_once() -> bool:
        nonlocal continue_calls
        continue_calls += 1
        return continue_calls == 1

    monkeypatch.setattr(
        runtime,
        "prepare_or_rehydrate_live_lineage",
        lambda **_kwargs: (_ for _ in ()).throw(
            _typed_branch_rejection(
                candidate_id="candidate-01",
                root_code="reserved_filename",
                root_subject="baseline.py",
            )
        ),
    )
    stopped = runtime._prepare_live_lineage_portfolio(
        campaign_id="tampered-rejection-resume",
        parent=opaque,
        generated_root=generated_root,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=opaque,
        provider="openai",
        maximum_iterations=10,
        safe_context_factory=cast(Any, lambda _records: opaque),
        continue_check=continue_once,
    )
    assert stopped.status == "admission_closed"

    journal_entry = generated_root / runtime._REJECTION_JOURNAL_DIR / "rejection-01.json"
    decoded = json.loads(journal_entry.read_bytes())
    decoded["record"]["values"]["root_failure_code"] = "tampered"
    journal_entry.chmod(0o600)
    journal_entry.write_bytes(canonical_json_bytes(decoded))

    with pytest.raises(FullCampaignError, match="journal digest is inconsistent"):
        runtime._prepare_live_lineage_portfolio(
            campaign_id="tampered-rejection-resume",
            parent=opaque,
            generated_root=generated_root,
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            model=opaque,
            provider="openai",
            maximum_iterations=10,
            safe_context_factory=cast(Any, lambda _records: opaque),
            continue_check=lambda: True,
        )


def test_autonomous_followup_driver_keeps_proposing_until_exact_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the live outer loop with provider/model seams replaced by typed fakes."""

    config = _config()
    fallback = _fallback()
    first_convergence = ConvergenceState.initial(fallback.outer_by_seed[0].metrics.primary)
    first_convergence = first_convergence.update_after_iteration(None)
    first_result = ScientificCampaignResult(
        config_digest=config.digest,
        fallback=fallback,
        incumbent=fallback,
        candidates=(),
        public_feedback=(),
        convergence=first_convergence,
        launches_used=config.launches_already_used + 1,
        elapsed_seconds=1.0,
        stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
    )
    first_reflection = Reflection(
        response_id="reflection-1",
        summary="The first branch did not materially improve the incumbent.",
        recommendation="propose_next",
        lessons=("Try another bounded hypothesis.",),
    )

    class FakeStore:
        def __init__(self) -> None:
            self.revision = 0
            self.convergence_manifests: list[dict[str, object]] = []

        def snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(revision=self.revision)

        def set_convergence_state(
            self,
            manifest: dict[str, object],
            *,
            expected_revision: int,
            reason: str,
        ) -> None:
            assert expected_revision == self.revision
            assert reason.startswith("persist autonomous scientific convergence cursor")
            self.convergence_manifests.append(manifest)
            self.revision += 1

    class FakeEngine:
        def __init__(self) -> None:
            self.status_calls = 0

        def status(self, _run_dir: Path) -> SimpleNamespace:
            self.status_calls += 1
            return SimpleNamespace(outer_queries_remaining=6)

    store = FakeStore()
    engine = FakeEngine()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    transcript = artifacts.put_bytes(b"{}", kind=ArtifactKind.LOG)
    parent = SimpleNamespace(candidate_id="seed-parent")
    first_lineage = SimpleNamespace(
        candidate_id="live-candidate-1",
        parent=parent,
        materialized=object(),
    )
    opaque = cast(Any, object())
    runtime_template = runtime._ScientificRuntime(
        engine=cast(Any, engine),
        run_dir=tmp_path,
        campaign_store=cast(Any, store),
        artifacts=artifacts,
        executor=opaque,
        lineage=cast(Any, first_lineage),
        candidate=cast(Any, SimpleNamespace(candidate_id="live-candidate-1")),
        experiment_id="iteration-01",
        scientific_iteration=1,
        config=config,
        feature_artifacts=opaque,
        features=opaque,
        fold_a=opaque,
        fold_b=opaque,
        fold_a_query_inputs=opaque,
        fold_b_query_inputs=opaque,
        fold_a_scorer=opaque,
        fold_b_scorer=opaque,
        outer_scorer=cast(
            Any,
            SimpleNamespace(scorer=SimpleNamespace(scorer_digest="a" * 64)),
        ),
        outer_admission=None,
        qualification=opaque,
        repository=opaque,
        evidence_registry={},
        cancel_event=None,
        records={},
    )
    request = SimpleNamespace(
        campaign_id="autonomous-fake-provider",
        config=SimpleNamespace(
            validation=SimpleNamespace(outer_promotion_limit=6),
            research=SimpleNamespace(provider="openai"),
        ),
    )
    prepared_iterations: list[int] = []
    cursor_history: list[tuple[int, int]] = []
    context_record_counts: list[int] = []

    def prepare_lineage(**kwargs: object) -> SimpleNamespace:
        iteration = cast(int, kwargs["scientific_iteration"])
        prepared_iterations.append(iteration)
        if iteration == 2:
            raise LiveResearchBranchRejected(
                failed_candidate_id="live-candidate-2-repair-1",
                repairs_attempted=1,
                diagnostic="declared material symbols did not change executable source",
            )
        return SimpleNamespace(
            candidate_id=f"live-candidate-{iteration}",
            parent=kwargs["parent"],
            materialized=object(),
        )

    def continue_campaign(**kwargs: object) -> ScientificCampaignResult:
        prior = cast(ConvergenceState, kwargs["initial_convergence"])
        launches = cast(int, kwargs["initial_launches_used"])
        cursor_history.append((prior.completed_iterations, launches))
        next_convergence = prior.update_after_iteration(None)
        return ScientificCampaignResult(
            config_digest=config.digest,
            fallback=fallback,
            incumbent=fallback,
            candidates=(),
            public_feedback=(),
            convergence=next_convergence,
            launches_used=launches + 1,
            elapsed_seconds=float(cast(float, kwargs["initial_elapsed_seconds"])) + 1.0,
            stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
        )

    def reflect(**kwargs: object) -> tuple[str, str, ArtifactRef, Reflection]:
        iteration = cast(int, kwargs["scientific_iteration"])
        return (
            f"reflection-request-{iteration}",
            f"reflection-response-{iteration}",
            transcript,
            Reflection(
                response_id=f"reflection-{iteration}",
                summary="No material gain; continue until the frozen patience is reached.",
                recommendation="propose_next",
                lessons=("Preserve the incumbent and continue.",),
            ),
        )

    def safe_context(**kwargs: object) -> object:
        context_record_counts.append(len(cast(tuple[object, ...], kwargs["campaign_records"])))
        return opaque

    monkeypatch.setattr(runtime, "_safe_context", safe_context)
    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare_lineage)
    monkeypatch.setattr(
        runtime,
        "_ensure_lineage_ledger",
        lambda **kwargs: (opaque, f"iteration-{kwargs['scientific_iteration']:02d}"),
    )
    monkeypatch.setattr(
        runtime,
        "_generated_scientific_candidate",
        lambda **kwargs: SimpleNamespace(candidate_id=kwargs["candidate_id"]),
    )
    fake_project_ledger = SimpleNamespace(projection=lambda: opaque)
    monkeypatch.setattr(
        runtime,
        "_open_outer_ledger",
        lambda *_args, **_kwargs: nullcontext(fake_project_ledger),
    )
    monkeypatch.setattr(runtime, "DurableScientificLedgerAdapter", lambda *_args, **_kwargs: opaque)
    monkeypatch.setattr(
        runtime,
        "ScoringReceiptBook",
        SimpleNamespace(from_projection=lambda _projection: opaque),
    )
    monkeypatch.setattr(
        runtime,
        "ReceiptAwareOuterEvaluationLedger",
        lambda *_args, **_kwargs: opaque,
    )
    monkeypatch.setattr(runtime, "run_scientific_campaign", continue_campaign)
    monkeypatch.setattr(runtime, "_candidate_selection", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_reflect", reflect)

    result = runtime._run_autonomous_followups(
        request=cast(Any, request),
        data=opaque,
        runtime_template=runtime_template,
        scientific_config=config,
        fallback=fallback,
        outer_ledger_path=tmp_path / "outer.sqlite3",
        candidate_limits=opaque,
        dataset_digest="d" * 64,
        context_evidence=opaque,
        validation_inputs=opaque,
        final_inputs=opaque,
        research_model=opaque,
        first_lineage=cast(Any, first_lineage),
        first_result=first_result,
        first_selection=None,
        first_reflection=first_reflection,
        first_reflection_evidence=("request-1", "response-1", transcript),
        prior_records=(),
    )

    assert prepared_iterations == [2, 3, 4]
    assert cursor_history == [(1, 7), (2, 8)]
    assert context_record_counts == [1, 2, 3]
    assert result.iterations_completed == 4
    assert result.result.convergence.completed_iterations == 3
    assert result.result.convergence.should_stop is True
    assert result.result.launches_used == 9
    assert result.result.stop_reason is CampaignStopReason.CONVERGED
    assert len(result.rejected_records) == 1
    assert result.rejected_records[0].values["root_failure_code"] == "branch_rejected"
    assert len(store.convergence_manifests) == 2
    assert engine.status_calls == 3


def test_production_runtime_preserves_active_interpreter_at_both_child_seams() -> None:
    fold_source = inspect.getsource(runtime._fold_control)
    campaign_source = inspect.getsource(runtime.run_provider_free_campaign)

    assert fold_source.count("interpreter=active_python_interpreter()") == 1
    assert campaign_source.count("interpreter=active_python_interpreter()") == 1
    assert "Path(sys.executable).resolve" not in fold_source
    assert "Path(sys.executable).resolve" not in campaign_source


class _Qualification:
    def __init__(
        self,
        *,
        request: CampaignCreateRequest,
        validation_rows: int,
        final_rows: int,
    ) -> None:
        metrics = OrganizerMetrics(0.62, 0.58)
        self.root = request.qualification_run_dir
        self.manifest_digest = request.qualification_manifest_digest
        self.qualification_input_digest = "4" * 64
        self.benchmark_digest = request.benchmark_digest
        self.canonical_digest = request.dataset_manifest_digest
        self.audit_digest = "5" * 64
        self.starter_manifest_digest = request.starter_manifest_digest
        self.scorer_digest = STARTER_FILE_SHA256["evaluate.py"]
        self.validation_row_count = validation_rows
        self.final_row_count = final_rows
        self.outer_runs = tuple(SimpleNamespace(seed=seed, metrics=metrics) for seed in (0, 1, 2))
        self.fallback = SimpleNamespace(manifest_digest=request.fallback.manifest_digest)

    def outer_seed(self, seed: int) -> object:
        return next(item for item in self.outer_runs if item.seed == seed)


def _fold(name: str) -> object:
    metrics = OrganizerMetrics(0.61 if name == "A" else 0.60, 0.59)
    return SimpleNamespace(
        control=SimpleNamespace(metrics=metrics),
        evidence=SimpleNamespace(digest=hashlib.sha256(f"fold-{name}".encode()).hexdigest()),
    )


def test_provider_free_runtime_closes_fallback_and_exactly_retries_without_final_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)
    engine.create(request)
    canonical = _dataset()
    prepared = prepare_campaign_data_plane(
        canonical,
        expected_dataset_digest=canonical.digest,
    )
    public_dataset = SimpleNamespace(
        digest=request.dataset_manifest_digest,
        valid=canonical.valid,
        final=canonical.final,
    )
    qualification = _Qualification(
        request=request,
        validation_rows=canonical.valid.row_count,
        final_rows=canonical.final.row_count,
    )

    monkeypatch.setattr(runtime, "_validate_locked_runtime", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_resolve_directory",
        lambda _root, _value, name: (
            ROOT / "candidate_templates" / "lambdarank"
            if "template" in name
            else ROOT / "kuairand-starter-kit"
            if "starter" in name
            else tmp_path
        ),
    )
    monkeypatch.setattr(
        runtime,
        "verify_starter_kit",
        lambda path: SimpleNamespace(
            root=path,
            manifest_sha256=request.starter_manifest_digest,
        ),
    )
    monkeypatch.setattr(runtime, "load_canonical_dataset", lambda _path: public_dataset)
    monkeypatch.setattr(runtime, "prepare_campaign_data_plane", lambda *_args, **_kw: prepared)
    monkeypatch.setattr(
        runtime,
        "load_official_fm_qualification",
        lambda *_args, **_kwargs: qualification,
    )
    monkeypatch.setattr(
        runtime,
        "_fold_control",
        lambda *, fold_name, **_kwargs: _fold(fold_name),
    )

    def close_with_fallback(
        *,
        config: ScientificCampaignConfig,
        fallback: IncumbentEvidence,
        **_kwargs: object,
    ) -> ScientificCampaignResult:
        return ScientificCampaignResult(
            config_digest=config.digest,
            fallback=fallback,
            incumbent=fallback,
            candidates=(),
            public_feedback=(),
            convergence=ConvergenceState.initial(
                sum(item.metrics.primary for item in fallback.outer_by_seed) / 3
            ),
            launches_used=config.launches_already_used,
            elapsed_seconds=config.elapsed_seconds_at_start,
            stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
        )

    monkeypatch.setattr(runtime, "run_scientific_campaign", close_with_fallback)

    def retained_loader(
        run_dir: Path,
        *,
        engine: CampaignEngine | None = None,
    ) -> FullCampaignOutcome:
        del engine
        return FullCampaignOutcomeRepository(
            run_dir=run_dir,
            artifact_store=ArtifactStore(run_dir / "artifacts"),
            progress=FullCampaignProgressLedger(
                run_dir / "production" / "progress",
                create=False,
            ),
        ).load(request_digest=request.digest)

    monkeypatch.setattr(runtime, "load_full_campaign_outcome", retained_loader)

    first = runtime.run_provider_free_campaign(
        request.run_dir,
        project_root=ROOT,
        engine=engine,
        outer_ledger_path=tmp_path / "outer-ledger.sqlite3",
    )
    progress = FullCampaignProgressLedger(
        request.run_dir / "production" / "progress",
        create=False,
    )
    first_checkpoints = progress.checkpoints()
    first_deadlines = tuple((request.run_dir / "controller" / "deadline").iterdir())
    with CampaignStore.open(
        request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=request.campaign_id,
    ) as store:
        assert store.snapshot().launches_used == 6
        assert tuple(item.launch_number for item in store.launches()) == tuple(range(1, 7))

    second = runtime.run_provider_free_campaign(
        request.run_dir,
        project_root=ROOT,
        engine=engine,
        outer_ledger_path=tmp_path / "outer-ledger.sqlite3",
    )

    assert first == second
    assert first.finalization_required
    assert first.fallback_preserved
    assert first.selection is None
    assert first.scientific_result_digest is not None
    assert first.reflection_transcript is not None
    assert canonical.final.targets is None
    assert canonical.final.outcome_trace.parsed_cell_count == 0
    assert tuple(item.stage for item in first_checkpoints) == tuple(FullCampaignStage)
    science = next(
        item for item in first_checkpoints if item.stage is FullCampaignStage.SCIENCE_COMPLETE
    )
    assert science.evidence["research_stage_counts"] == {
        "branches_attempted": 1,
        "proposal_responses_accepted": 0,
        "implementation_responses_accepted": 0,
        "repair_responses_accepted": 0,
        "branches_rejected_pre_execution": 0,
        "candidates_admitted": 1,
        "training_started": 0,
        "inner_evaluations_completed": 0,
        "outer_evaluations_completed": 0,
    }
    assert progress.checkpoints() == first_checkpoints
    assert tuple((request.run_dir / "controller" / "deadline").iterdir()) == first_deadlines

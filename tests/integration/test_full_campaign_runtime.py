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
)
from kuairand_agent.campaign.selector import IncumbentEvidence, OrganizerMetrics
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactRef, ArtifactStore
from kuairand_agent.research.context import AggregateRecord
from kuairand_agent.research.interface import ResearchModelError
from kuairand_agent.research.production import (
    LiveResearchBranchRejected,
    ResearchFailureObservation,
)
from kuairand_agent.research.schemas import Reflection, canonical_json_bytes
from tests.integration.test_campaign_controller import FakeClock, build_request
from tests.unit.test_full_campaign import _dataset
from tests.unit.test_scientific_campaign import _config, _fallback

ROOT = Path(__file__).resolve().parents[2]


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


def _drive_autonomous_followups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepare_lineage_hook: Any = None,
    reflect_hook: Any = None,
    deadline: Any = None,
) -> SimpleNamespace:
    """Exercise the live outer loop with provider/model seams replaced by typed fakes.

    ``prepare_lineage_hook`` and ``reflect_hook`` are called with the scientific iteration
    before the corresponding fake returns, so a test can make either seam raise.
    ``deadline`` replaces what the engine reports for the real clock.
    """

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
            self.deadline_calls = 0

        def status(self, _run_dir: Path) -> SimpleNamespace:
            self.status_calls += 1
            return SimpleNamespace(outer_queries_remaining=6)

        def inspect_deadline(self, _run_dir: Path) -> SimpleNamespace:
            # The loop now stops on the real clock rather than a counter that cannot see
            # provider latency, so the fake must answer that question too.
            self.deadline_calls += 1
            if deadline is not None:
                observed: SimpleNamespace = deadline(self.deadline_calls)
                return observed
            return SimpleNamespace(
                finalization_reserve_active=False,
                hard_expired=False,
            )

    store = FakeStore()
    engine = FakeEngine()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    transcript = artifacts.put_bytes(b"{}", kind=ArtifactKind.LOG)
    parent = SimpleNamespace(candidate_id="seed-parent")
    first_lineage = SimpleNamespace(
        candidate_id="live-candidate-1",
        parent=parent,
        materialized=object(),
        proposal=SimpleNamespace(
            objective="stub ranking objective",
            mechanism="stub mechanism",
            principal_change="stub principal change",
        ),
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
        if prepare_lineage_hook is not None:
            prepare_lineage_hook(iteration)
        return SimpleNamespace(
            candidate_id=f"live-candidate-{iteration}",
            parent=kwargs["parent"],
            materialized=object(),
            proposal=SimpleNamespace(
                objective="stub ranking objective",
                mechanism="stub mechanism",
                principal_change="stub principal change",
            ),
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
        if reflect_hook is not None:
            reflect_hook(iteration)
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
    monkeypatch.setattr(runtime, "_open_outer_ledger", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(runtime, "DurableScientificLedgerAdapter", lambda *_args, **_kwargs: opaque)
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

    return SimpleNamespace(
        result=result,
        prepared_iterations=prepared_iterations,
        cursor_history=cursor_history,
        context_record_counts=context_record_counts,
        store=store,
        engine=engine,
    )


def test_autonomous_followup_driver_keeps_proposing_until_exact_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected branch costs one iteration; the loop continues to exact convergence."""

    def reject_the_second_branch(iteration: int) -> None:
        if iteration == 2:
            raise LiveResearchBranchRejected(
                failed_candidate_id="live-candidate-2-repair-1",
                repairs_attempted=1,
                diagnostic="declared material symbols did not change executable source",
            )

    driven = _drive_autonomous_followups(
        tmp_path,
        monkeypatch,
        prepare_lineage_hook=reject_the_second_branch,
    )
    result = driven.result

    assert driven.prepared_iterations == [2, 3, 4]
    assert driven.cursor_history == [(1, 7), (2, 8)]
    assert driven.context_record_counts == [1, 2, 3]
    assert result.iterations_completed == 4
    assert result.result.convergence.completed_iterations == 3
    assert result.result.convergence.should_stop is True
    assert result.result.launches_used == 9
    assert result.result.stop_reason is CampaignStopReason.CONVERGED
    assert len(result.rejected_records) == 1
    assert result.rejected_records[0].values["root_failure_code"] == "branch_rejected"
    assert len(driven.store.convergence_manifests) == 2
    assert driven.engine.status_calls == 3


@pytest.mark.parametrize(
    ("code", "expected_reason"),
    [
        ("deadline", CampaignStopReason.FINALIZATION_RESERVE),
        ("http", CampaignStopReason.CANDIDATES_EXHAUSTED),
        (None, CampaignStopReason.CANDIDATES_EXHAUSTED),
    ],
)
def test_autonomous_followup_driver_closes_research_when_the_provider_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str | None,
    expected_reason: CampaignStopReason,
) -> None:
    """A provider failure must cost the branch, not the campaign.

    Only ``LiveResearchBranchRejected`` used to be caught here, so the deadline error the provider
    raises by design once the finalization reserve opens escaped this loop, escaped the single try
    block in the CLI, and left five hours of research with no bundle written.  The loop must now
    return normally with a terminal reason, and a scheduled deadline must be reported as the
    reserve rather than as candidate exhaustion.
    """

    failure = ResearchModelError("provider refused the implementation request")
    if code is not None:
        failure.code = SimpleNamespace(value=code)  # type: ignore[attr-defined]

    def fail_the_second_branch(iteration: int) -> None:
        if iteration == 2:
            raise failure

    driven = _drive_autonomous_followups(
        tmp_path,
        monkeypatch,
        prepare_lineage_hook=fail_the_second_branch,
    )
    result = driven.result

    assert driven.prepared_iterations == [2]
    assert result.result.stop_reason is expected_reason
    closures = [
        record
        for record in result.rejected_records
        if record.values["branch_outcome"] == "research_closed_by_provider_failure"
    ]
    assert len(closures) == 1
    assert closures[0].values["operation"] == "lineage"
    assert closures[0].values["failure_type"] == "ResearchModelError"
    assert closures[0].values["failure_code"] == (code or "")


def test_reflection_degrades_to_a_single_line_when_the_provider_fails() -> None:
    """Reflection is commentary, so losing it must not lose the campaign.

    The call into ``reflection_model.reflect`` was the one provider seam in the research loop with
    no guard at all.  The substitute must also survive the report writer: ``schemas._text`` permits
    newlines that ``report._text`` rejects, and a transport error message is a plausible source of
    one, so the diagnostic is collapsed here rather than at the point of rendering.
    """

    failure = ResearchModelError("provider returned\n  a multi-line\ttransport diagnostic")
    reflection = runtime._unavailable_reflection(failure, scientific_iteration=3)

    assert reflection.response_id == "reflection-unavailable-03"
    assert reflection.recommendation == "close_branch"
    assert "\n" not in reflection.summary
    assert "provider returned a multi-line transport diagnostic" in reflection.summary
    assert reflection.lessons


def test_reflection_degradation_never_produces_an_empty_summary() -> None:
    """An exception can stringify to nothing, which would render as an unexplained blank."""

    reflection = runtime._unavailable_reflection(ResearchModelError(), scientific_iteration=1)

    assert reflection.summary.endswith("ResearchModelError")


def test_reflection_seam_uses_the_degrading_helper() -> None:
    """Guard the wiring, since the helper above is only useful if the seam actually calls it."""

    source = inspect.getsource(runtime._reflect)

    assert "except ResearchModelError as exc:" in source
    assert "_unavailable_reflection(exc, scientific_iteration=scientific_iteration)" in source


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("finalization_reserve_active", CampaignStopReason.FINALIZATION_RESERVE),
        ("hard_expired", CampaignStopReason.HARD_DEADLINE),
    ],
)
def test_autonomous_followup_driver_stops_on_the_real_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected_reason: CampaignStopReason,
) -> None:
    """The scientific counter cannot see provider time, so the engine clock must be consulted.

    ``scientific.py`` advances ``elapsed_seconds`` only by candidate subprocess wall time.  Summing
    the provider journals of one real overnight run gives 4011 s of latency the counter never saw,
    which is more than the entire finalization reserve.
    """

    def clock(call: int) -> SimpleNamespace:
        expired = call >= 2
        return SimpleNamespace(
            finalization_reserve_active=expired and field == "finalization_reserve_active",
            hard_expired=expired and field == "hard_expired",
        )

    driven = _drive_autonomous_followups(tmp_path, monkeypatch, deadline=clock)

    assert driven.result.result.stop_reason is expected_reason
    assert driven.prepared_iterations == [2]


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


def _runs(primaries: tuple[float, ...]) -> SimpleNamespace:
    """A candidate result carrying `primaries` in the tier order the campaign produces."""

    return SimpleNamespace(
        runs=tuple(
            SimpleNamespace(metrics=OrganizerMetrics(gauc=primary, ndcg_at_5=primary))
            for primary in primaries
        )
    )


def test_measured_primary_reports_the_outer_seed_mean_and_its_delta() -> None:
    """Runs arrive as Fold B screen, Fold A confirmation, then the matched outer seeds."""

    incumbent = _fallback(0.61)

    primary, delta, tier = runtime._measured_primary(
        cast(Any, _runs((0.57, 0.60, 0.601, 0.602, 0.603))),
        incumbent,
    )

    assert tier == "outer_matched_seed"
    assert primary == pytest.approx(0.602, abs=1e-9)
    assert delta == pytest.approx(-0.008, abs=1e-9)


def test_measured_primary_falls_back_to_the_inner_tier_before_outer_promotion() -> None:
    incumbent = _fallback(0.61)

    primary, delta, tier = runtime._measured_primary(cast(Any, _runs((0.58,))), incumbent)

    assert tier == "inner_fold"
    assert primary == pytest.approx(0.58, abs=1e-9)
    assert delta == pytest.approx(-0.03, abs=1e-9)


def test_measured_primary_claims_nothing_when_no_run_produced_metrics() -> None:
    incumbent = _fallback(0.61)

    assert runtime._measured_primary(None, incumbent) == (None, None, None)
    assert runtime._measured_primary(
        cast(Any, SimpleNamespace(runs=(SimpleNamespace(metrics=None),))),
        incumbent,
    ) == (None, None, None)


def test_iteration_record_carries_the_tested_direction_and_the_reflection_lessons() -> None:
    """The proposer sees only these records, so a tested direction must be legible in them."""

    config = _config()
    fallback = _fallback(0.61)
    result = ScientificCampaignResult(
        config_digest=config.digest,
        fallback=fallback,
        incumbent=fallback,
        candidates=(),
        public_feedback=(),
        convergence=ConvergenceState.initial(0.61),
        launches_used=config.launches_already_used + 1,
        elapsed_seconds=1.0,
        stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
    )
    reflection = Reflection(
        response_id="reflection-1",
        summary="The pairwise branch did not beat the incumbent.",
        recommendation="propose_next",
        lessons=("Objective alone is inert.", "Capacity is the binding constraint."),
    )
    proposal = SimpleNamespace(
        objective="within-user pairwise softplus ranking objective",
        mechanism="sample a positive then a negative from the same user",
        principal_change="replace pointwise log loss",
    )

    record = runtime._iteration_record(
        result,
        reflection,
        scientific_iteration=2,
        proposal=cast(Any, proposal),
    )

    values = record.values
    assert values["proposal_family"] == "pairwise"
    assert values["proposal_objective"] == proposal.objective
    assert values["proposal_principal_change"] == proposal.principal_change
    assert "Capacity is the binding constraint." in cast(str, values["reflection_lessons"])
    # Without a proposal the record claims no direction rather than fabricating one.
    bare = runtime._iteration_record(result, reflection, scientific_iteration=2)
    assert bare.values["proposal_family"] is None
    # A branch that ran is never flagged as an execution failure.
    assert values["execution_failed"] is False
    assert values["execution_failure_note"] is None


def test_a_crashed_candidate_is_recorded_as_scoreless_rather_than_as_a_baseline_tie() -> None:
    """A crash must not read to the next proposer as a result.

    ``_reflect`` substitutes the fallback's seed-0 metrics when a run produced none, because
    ``ExperimentResultSummary`` requires three finite metrics. In run 10 that made two candidates
    that raised IndexError inside train_model look like they had tied the baseline, so nothing in
    the loop had any reason to avoid repeating the defect.
    """

    config = _config()
    fallback = _fallback(0.61)
    crashed = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="candidate-01-crashed"),
        outcome=CandidateOutcome.CALLBACK_FAILED,
        runs=(),
        reason="callback_failed:CandidateExecutionError",
    )
    result = ScientificCampaignResult(
        config_digest=config.digest,
        fallback=fallback,
        incumbent=fallback,
        candidates=(cast(Any, crashed),),
        public_feedback=(),
        convergence=ConvergenceState.initial(0.61),
        launches_used=config.launches_already_used + 1,
        elapsed_seconds=1.0,
        stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
    )
    reflection = Reflection(
        response_id="reflection-1",
        summary="Reported metrics matched the official FM seed 0 reference.",
        recommendation="propose_next",
        lessons=("Numbers matched the reference exactly.",),
    )

    values = runtime._iteration_record(result, reflection, scientific_iteration=1).values

    assert values["execution_failed"] is True
    assert values["candidate_primary"] is None
    assert values["delta_vs_incumbent"] is None
    assert "NO measured score" in cast(str, values["execution_failure_note"])

from __future__ import annotations

import hashlib
import inspect
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
    FullCampaignOutcome,
    FullCampaignOutcomeRepository,
    FullCampaignProgressLedger,
    FullCampaignStage,
    prepare_campaign_data_plane,
)
from kuairand_agent.campaign.scientific import (
    CampaignStopReason,
    ScientificCampaignConfig,
    ScientificCampaignResult,
)
from kuairand_agent.campaign.selector import IncumbentEvidence, OrganizerMetrics
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactRef, ArtifactStore
from kuairand_agent.research.schemas import Reflection
from tests.integration.test_campaign_controller import FakeClock, build_request
from tests.unit.test_full_campaign import _dataset
from tests.unit.test_scientific_campaign import _config, _fallback

ROOT = Path(__file__).resolve().parents[2]


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

    def prepare_lineage(**kwargs: object) -> SimpleNamespace:
        iteration = cast(int, kwargs["scientific_iteration"])
        prepared_iterations.append(iteration)
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

    monkeypatch.setattr(runtime, "_safe_context", lambda **_kwargs: opaque)
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
    )

    assert prepared_iterations == [2, 3]
    assert cursor_history == [(1, 7), (2, 8)]
    assert result.iterations_completed == 3
    assert result.result.convergence.completed_iterations == 3
    assert result.result.convergence.should_stop is True
    assert result.result.launches_used == 9
    assert result.result.stop_reason is CampaignStopReason.CONVERGED
    assert len(store.convergence_manifests) == 2
    assert engine.status_calls == 2


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
    assert progress.checkpoints() == first_checkpoints
    assert tuple((request.run_dir / "controller" / "deadline").iterdir()) == first_deadlines

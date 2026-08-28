from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from kuairand_agent.campaign import full_campaign_runtime as runtime
from kuairand_agent.campaign.budgets import LaunchCategory
from kuairand_agent.campaign.controller import (
    CAMPAIGN_DATABASE_NAME,
    CampaignCreateRequest,
    CampaignEngine,
)
from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.full_campaign import (
    FullCampaignCancelled,
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
from kuairand_agent.execution.artifacts import ArtifactStore
from tests.integration.test_campaign_controller import FakeClock, build_request
from tests.unit.test_full_campaign import _dataset

ROOT = Path(__file__).resolve().parents[2]

type _FoldHook = Callable[[str], None]
type _ScienceHook = Callable[
    [ScientificCampaignConfig, IncumbentEvidence], ScientificCampaignResult
]


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


def _fallback_result(
    config: ScientificCampaignConfig,
    fallback: IncumbentEvidence,
    *,
    launches_used: int | None = None,
) -> ScientificCampaignResult:
    mean_primary = sum(item.metrics.primary for item in fallback.outer_by_seed) / len(
        fallback.outer_by_seed
    )
    return ScientificCampaignResult(
        config_digest=config.digest,
        fallback=fallback,
        incumbent=fallback,
        candidates=(),
        public_feedback=(),
        convergence=ConvergenceState.initial(mean_primary),
        launches_used=(config.launches_already_used if launches_used is None else launches_used),
        elapsed_seconds=config.elapsed_seconds_at_start,
        stop_reason=CampaignStopReason.CANDIDATES_EXHAUSTED,
    )


@dataclass(slots=True)
class _Harness:
    request: CampaignCreateRequest
    clock: FakeClock
    engine: CampaignEngine
    outer_ledger_path: Path
    fold_calls: list[str]
    science_configs: list[ScientificCampaignConfig]

    def run(self, *, cancel_event: Event | None = None) -> FullCampaignOutcome:
        return runtime.run_provider_free_campaign(
            self.request.run_dir,
            project_root=ROOT,
            engine=self.engine,
            outer_ledger_path=self.outer_ledger_path,
            cancel_event=cancel_event,
        )

    def progress(self) -> FullCampaignProgressLedger:
        return FullCampaignProgressLedger(
            self.request.run_dir / "production" / "progress",
            create=False,
        )


def _install_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fold_hook: _FoldHook | None = None,
    science_hook: _ScienceHook | None = None,
) -> _Harness:
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
    fold_calls: list[str] = []
    science_configs: list[ScientificCampaignConfig] = []

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
    monkeypatch.setattr(
        runtime,
        "prepare_campaign_data_plane",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        runtime,
        "load_official_fm_qualification",
        lambda *_args, **_kwargs: qualification,
    )

    def fold_control(*, fold_name: str, **_kwargs: object) -> object:
        fold_calls.append(fold_name)
        if fold_hook is not None:
            fold_hook(fold_name)
        return _fold(fold_name)

    monkeypatch.setattr(runtime, "_fold_control", fold_control)

    def scientific_campaign(
        *,
        config: ScientificCampaignConfig,
        fallback: IncumbentEvidence,
        **_kwargs: object,
    ) -> ScientificCampaignResult:
        science_configs.append(config)
        if science_hook is not None:
            return science_hook(config, fallback)
        return _fallback_result(config, fallback)

    monkeypatch.setattr(runtime, "run_scientific_campaign", scientific_campaign)

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
    return _Harness(
        request=request,
        clock=clock,
        engine=engine,
        outer_ledger_path=tmp_path / "outer-ledger.sqlite3",
        fold_calls=fold_calls,
        science_configs=science_configs,
    )


def test_finalization_reserve_before_fold_b_admits_no_fold_or_scientific_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(tmp_path, monkeypatch)
    harness.clock.advance(3_600)

    outcome = harness.run()

    assert harness.fold_calls == []
    assert harness.science_configs == []
    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.selection is None
    assert outcome.scientific_result_digest is None
    assert outcome.reflection_transcript is None
    assert outcome.launches_used == 6
    checkpoints = harness.progress().checkpoints()
    assert tuple(item.stage for item in checkpoints) == tuple(FullCampaignStage)
    lineage = next(item for item in checkpoints if item.stage is FullCampaignStage.LINEAGE_READY)
    assert lineage.evidence["reason"] == "finalization_reserve_active_before_fold_B"
    assert harness.engine.status(harness.request.run_dir).finalization_required


def test_finalization_reserve_after_fold_b_skips_fold_a_and_closes_with_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_holder: list[FakeClock] = []

    def enter_reserve_after_fold_b(fold_name: str) -> None:
        if fold_name == "B":
            clock_holder[0].advance(3_600)

    harness = _install_harness(
        tmp_path,
        monkeypatch,
        fold_hook=enter_reserve_after_fold_b,
    )
    clock_holder.append(harness.clock)

    outcome = harness.run()

    assert harness.fold_calls == ["B"]
    assert harness.science_configs == []
    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.selection is None
    assert outcome.scientific_result_digest is None
    assert outcome.reflection_transcript is None
    assert outcome.launches_used == 6
    fold_checkpoint = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.FOLD_CONTROLS_READY
    )
    assert fold_checkpoint.evidence["reason"] == (
        "finalization_reserve_active_between_fold_B_and_fold_A"
    )
    assert fold_checkpoint.evidence["fold_b_status"] == "completed"
    assert fold_checkpoint.evidence["fold_a_status"] == "not_started"


def test_cancel_set_by_scientific_callback_publishes_no_success_or_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = Event()

    def cancel_during_science(
        config: ScientificCampaignConfig,
        fallback: IncumbentEvidence,
    ) -> ScientificCampaignResult:
        cancellation.set()
        return _fallback_result(config, fallback)

    harness = _install_harness(
        tmp_path,
        monkeypatch,
        science_hook=cancel_during_science,
    )

    with pytest.raises(FullCampaignCancelled, match="cancelled"):
        harness.run(cancel_event=cancellation)

    assert cancellation.is_set()
    assert len(harness.science_configs) == 1
    checkpoints = harness.progress().checkpoints()
    assert tuple(item.stage for item in checkpoints) == (
        FullCampaignStage.DATA_PREPARED,
        FullCampaignStage.QUALIFICATION_VERIFIED,
        FullCampaignStage.FOLD_CONTROLS_READY,
        FullCampaignStage.FEATURES_READY,
        FullCampaignStage.LINEAGE_READY,
    )
    assert all(item.stage is not FullCampaignStage.SCIENCE_COMPLETE for item in checkpoints)
    assert all(item.stage is not FullCampaignStage.REFLECTED for item in checkpoints)
    assert all(item.stage is not FullCampaignStage.FINALIZATION_REQUIRED for item in checkpoints)
    assert not harness.engine.status(harness.request.run_dir).finalization_required
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        experiment = store.experiment("iteration-01")
        assert experiment is not None
        assert experiment.status == "RUNNING"


class _SyntheticCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("categories", "checkpoint_name"),
    (
        pytest.param(
            (LaunchCategory.DIVERSE_INNER_SCREEN,),
            "candidate_fold_B_screen",
            id="candidate-fold-B-before-reserve",
        ),
        pytest.param(
            (
                LaunchCategory.DIVERSE_INNER_SCREEN,
                LaunchCategory.TEMPORAL_FOLD_CONFIRMATION,
            ),
            "candidate_fold_A_confirmation",
            id="candidate-fold-A-before-reserve",
        ),
    ),
)
def test_inner_candidate_crash_cannot_authorize_new_research_inside_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    categories: tuple[LaunchCategory, ...],
    checkpoint_name: str,
) -> None:
    def crash_before_outer_reservation(
        _config: ScientificCampaignConfig,
        _fallback: IncumbentEvidence,
    ) -> ScientificCampaignResult:
        raise _SyntheticCrash(checkpoint_name)

    harness = _install_harness(
        tmp_path,
        monkeypatch,
        science_hook=crash_before_outer_reservation,
    )
    with pytest.raises(_SyntheticCrash, match=checkpoint_name):
        harness.run()

    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        campaign_id=harness.request.campaign_id,
    ) as store:
        for ordinal, category in enumerate(categories, start=1):
            store.reserve_launch(
                launch_id=f"inner-candidate-crash-{ordinal}",
                reservation_key=f"inner-candidate-crash:{checkpoint_name}:{ordinal}",
                category=category.value,
                purpose=f"simulate completed {checkpoint_name} launch {ordinal}",
                expected_revision=store.snapshot().revision,
                experiment_id="iteration-01",
                scientific_iteration=1,
                seed=0,
                metadata={"simulated_checkpoint": checkpoint_name},
            )

    harness.clock.advance(3_600)
    outcome = harness.run()

    assert len(harness.science_configs) == 1
    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.scientific_result_digest is None
    assert outcome.selection is None
    assert outcome.launches_used == 6 + len(categories)
    science = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.SCIENCE_COMPLETE
    )
    assert science.evidence["reason"] == "finalization_reserve_active_before_fold_B"
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        assert store.snapshot().launches_used == 6 + len(categories)


@pytest.mark.parametrize(
    ("categories", "checkpoint_name"),
    (
        pytest.param(
            (LaunchCategory.DIVERSE_INNER_SCREEN,),
            "fold_B_screen",
            id="crash-after-screen",
        ),
        pytest.param(
            (
                LaunchCategory.DIVERSE_INNER_SCREEN,
                LaunchCategory.TEMPORAL_FOLD_CONFIRMATION,
            ),
            "fold_A_confirmation",
            id="crash-after-fold-A",
        ),
        pytest.param(
            (
                LaunchCategory.DIVERSE_INNER_SCREEN,
                LaunchCategory.TEMPORAL_FOLD_CONFIRMATION,
                LaunchCategory.DISTINCT_OUTER_PROMOTION,
            ),
            "outer_promotion",
            id="crash-after-outer",
        ),
    ),
)
def test_retry_freezes_scientific_start_cursor_and_does_not_double_count_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    categories: tuple[LaunchCategory, ...],
    checkpoint_name: str,
) -> None:
    call_count = 0

    def crash_then_close(
        config: ScientificCampaignConfig,
        fallback: IncumbentEvidence,
    ) -> ScientificCampaignResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _SyntheticCrash(checkpoint_name)
        return _fallback_result(
            config,
            fallback,
            launches_used=config.launches_already_used + len(categories),
        )

    harness = _install_harness(
        tmp_path,
        monkeypatch,
        science_hook=crash_then_close,
    )

    with pytest.raises(_SyntheticCrash, match=checkpoint_name):
        harness.run()

    first = harness.science_configs[0]
    assert first.launches_already_used == 6
    assert harness.progress().checkpoints()[-1].stage is FullCampaignStage.LINEAGE_READY

    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        campaign_id=harness.request.campaign_id,
    ) as store:
        for ordinal, category in enumerate(categories, start=1):
            store.reserve_launch(
                launch_id=f"synthetic-crash-{ordinal}",
                reservation_key=f"synthetic-crash:{checkpoint_name}:{ordinal}",
                category=category.value,
                purpose=f"simulate released {checkpoint_name} launch {ordinal}",
                expected_revision=store.snapshot().revision,
                experiment_id="iteration-01",
                scientific_iteration=1,
                seed=0,
                metadata={"simulated_checkpoint": checkpoint_name},
            )
        assert store.snapshot().launches_used == 6 + len(categories)

    harness.clock.advance(30)
    outcome = harness.run()

    assert len(harness.science_configs) == 2
    second = harness.science_configs[1]
    assert second.launches_already_used == first.launches_already_used == 6
    assert second.elapsed_seconds_at_start == first.elapsed_seconds_at_start
    assert second.wall_clock_seconds == first.wall_clock_seconds
    assert second.finalization_reserve_seconds == first.finalization_reserve_seconds
    assert second.digest == first.digest
    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.launches_used == 6 + len(categories)
    with CampaignStore.open(
        harness.request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=harness.request.campaign_id,
    ) as store:
        launches = store.launches()
        assert store.snapshot().launches_used == 6 + len(categories)
        assert tuple(item.launch_number for item in launches) == tuple(
            range(1, 7 + len(categories))
        )

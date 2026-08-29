from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

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
from kuairand_agent.config import (
    OpenAIResearchConfig,
    OpenAITokenPricingConfig,
    ResearchConfig,
)
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.research.factory import (
    AvailableResearchProvider,
    ProviderUnavailableCode,
    ProviderUnavailableDiagnostic,
)
from kuairand_agent.research.production import (
    LiveResearchBranchRejected,
    ResearchFailureObservation,
)
from kuairand_agent.research.provider import (
    OpenAIFailoverModel,
    OpenAIProviderError,
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    ProviderErrorCode,
    ResponsesTransportError,
    TokenPricing,
    TransportRequest,
    TransportResponse,
)
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
    research_config: ResearchConfig | None = None,
    research_provider: AvailableResearchProvider | None = None,
) -> _Harness:
    request = build_request(tmp_path)
    if research_config is not None:
        request = replace(
            request,
            config=replace(request.config, schema_version=2, research=research_config),
        )
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
            else ROOT / "candidate_seed"
            if "candidate seed" in name
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
    if research_provider is not None:
        monkeypatch.setattr(
            runtime,
            "select_research_provider",
            lambda *_args, **_kwargs: research_provider,
        )

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


@dataclass(frozen=True, slots=True)
class _UnavailableTransport:
    def send(self, _request: TransportRequest) -> TransportResponse:
        raise ResponsesTransportError("synthetic transport outage")


class _NeverCalledResearchModel:
    def propose(self, _request: object) -> object:
        raise AssertionError("proposal model should be replaced at the lineage seam")

    def implement(self, _request: object) -> object:
        raise AssertionError("implementation model should be replaced at the lineage seam")

    def repair(self, _request: object) -> object:
        raise AssertionError("repair model should be replaced at the lineage seam")

    def reflect(self, _request: object) -> object:
        raise AssertionError("reflection model should be replaced at the lineage seam")


def _research_config(*, key_env: str, max_transport_retries: int = 0) -> ResearchConfig:
    return ResearchConfig(
        provider="openai",
        max_repairs_per_experiment=2,
        run_kind="autonomous",
        openai=OpenAIResearchConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            reasoning_effort="high",
            api_key_env=key_env,
            timeout_seconds=1.0,
            max_response_bytes=1_048_576,
            max_output_tokens=4096,
            max_malformed_retries=1,
            max_transport_retries=max_transport_retries,
            pricing=OpenAITokenPricingConfig(
                input_usd_per_million="1",
                cached_input_usd_per_million="0.1",
                output_usd_per_million="2",
            ),
        ),
    )


def _pre_admission_rejection(iteration: int) -> LiveResearchBranchRejected:
    root = ResearchFailureObservation.create(
        stage="materialization",
        category="static_policy",
        code="reserved_filename",
        subject="baseline.py",
        diagnostic="reserved candidate filename is forbidden: 'baseline.py'",
    )
    terminal = ResearchFailureObservation.create(
        stage="materiality",
        category="materiality",
        code="declared_symbol_unchanged",
        subject="main",
        diagnostic="declared material symbol did not change executable source: main",
    )
    return LiveResearchBranchRejected(
        failed_candidate_id=f"candidate-{iteration:02d}",
        repairs_attempted=1,
        diagnostic=terminal.diagnostic,
        root_failure=root,
        terminal_failure=terminal,
        proposal_family=("pairwise" if iteration < 3 else "listwise"),
        proposal_signature=hashlib.sha256(f"proposal-{iteration}".encode()).hexdigest(),
    )


def test_repeated_pre_admission_failure_commits_truthful_rejection_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_env = "KUAIRAND_TEST_REPEATED_FAILURE_KEY"
    monkeypatch.setenv(key_env, "sk-test-repeated-failure")
    harness = _install_harness(
        tmp_path,
        monkeypatch,
        research_config=_research_config(key_env=key_env),
        research_provider=AvailableResearchProvider(
            provider="openai",
            model=cast(Any, _NeverCalledResearchModel()),
            live_provider_used=True,
        ),
    )

    def reject(**kwargs: object) -> object:
        raise _pre_admission_rejection(cast(int, kwargs["scientific_iteration"]))

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", reject)

    outcome = harness.run()

    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.selection is None
    assert harness.science_configs == []
    checkpoints = harness.progress().checkpoints()
    science = next(item for item in checkpoints if item.stage is FullCampaignStage.SCIENCE_COMPLETE)
    assert science.evidence["portfolio_cap_reason"] == "repeated_pre_admission_failure"
    assert science.evidence["portfolio_count"] == 3
    assert science.evidence["research_stage_counts"] == {
        "branches_attempted": 3,
        "proposal_responses_accepted": 0,
        "implementation_responses_accepted": 0,
        "repair_responses_accepted": 0,
        "branches_rejected_pre_execution": 3,
        "candidates_admitted": 0,
        "training_started": 0,
        "inner_evaluations_completed": 0,
        "outer_evaluations_completed": 0,
    }
    summary = cast(
        Mapping[str, Any],
        science.evidence["research_rejection_summary"],
    )
    assert summary["branches_rejected_pre_execution"] == 3
    assert summary["root_counts"] == [
        {
            "fingerprint": _pre_admission_rejection(1).root_failure.fingerprint,
            "stage": "materialization",
            "category": "static_policy",
            "code": "reserved_filename",
            "subject": "baseline.py",
            "count": 3,
        }
    ]
    assert summary["terminal_counts"][0]["count"] == 3
    rejection_ledger = cast(Mapping[str, Any], science.evidence["research_rejection_ledger"])
    assert rejection_ledger["kind"] == "log"


def test_initial_portfolio_provider_failure_preserves_prior_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque = object()

    def prepare(**kwargs: object) -> object:
        iteration = cast(int, kwargs["scientific_iteration"])
        if iteration == 1:
            raise _pre_admission_rejection(iteration)
        raise OpenAIProviderError(
            ProviderErrorCode.TRANSPORT,
            "synthetic provider failure",
        )

    monkeypatch.setattr(runtime, "prepare_or_rehydrate_live_lineage", prepare)
    preparation = runtime._prepare_live_lineage_portfolio(
        campaign_id="provider-after-rejection",
        parent=cast(Any, opaque),
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        model=cast(Any, opaque),
        provider="openai",
        maximum_iterations=5,
        safe_context_factory=cast(Any, lambda _records: opaque),
        continue_check=lambda: True,
    )

    assert preparation.status == "provider_unavailable"
    assert preparation.branches_attempted == 2
    assert preparation.provider_error is not None
    assert len(preparation.rejected_records) == 1
    assert preparation.rejected_records[0].values["root_failure_code"] == "reserved_filename"


def test_provider_failure_after_rejection_commits_the_accumulated_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_env = "KUAIRAND_TEST_PROVIDER_AFTER_REJECTION_KEY"
    monkeypatch.setenv(key_env, "sk-test-provider-after-rejection")
    model = OpenAIResponsesModel(
        OpenAIResponsesConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            reasoning_effort="high",
            pricing=TokenPricing(
                input_usd_per_million="1",
                cached_input_usd_per_million="0.1",
                output_usd_per_million="2",
            ),
            api_key_env=key_env,
            timeout_seconds=1.0,
            max_response_bytes=1_048_576,
            max_output_tokens=4096,
            max_malformed_retries=1,
            max_transport_retries=0,
        ),
        transport=_UnavailableTransport(),
    )
    harness = _install_harness(
        tmp_path,
        monkeypatch,
        research_config=_research_config(key_env=key_env),
        research_provider=AvailableResearchProvider(
            provider="openai",
            model=model,
            live_provider_used=True,
        ),
    )
    rejection = _pre_admission_rejection(1)
    record = runtime._rejected_lineage_record(rejection, scientific_iteration=1)

    monkeypatch.setattr(
        runtime,
        "_prepare_live_lineage_portfolio",
        lambda **_kwargs: runtime._LiveLineagePreparation(
            status="provider_unavailable",
            lineage=None,
            safe_context=None,
            scientific_iteration=None,
            rejected_records=(record,),
            branches_attempted=2,
            provider_error=OpenAIProviderError(
                ProviderErrorCode.TRANSPORT,
                "synthetic provider failure after one rejected branch",
                attempts=1,
            ),
        ),
    )

    outcome = harness.run()

    assert outcome.finalization_required
    reflected = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.REFLECTED
    )
    assert reflected.evidence["portfolio_cap_reason"] == "runtime_provider_unavailable"
    assert reflected.evidence["portfolio_count"] == 2
    stage_counts = cast(Mapping[str, Any], reflected.evidence["research_stage_counts"])
    assert stage_counts["branches_attempted"] == 2
    assert stage_counts["branches_rejected_pre_execution"] == 1
    summary = cast(Mapping[str, Any], reflected.evidence["research_rejection_summary"])
    assert summary["branches_rejected_pre_execution"] == 1
    assert cast(list[Mapping[str, Any]], summary["root_counts"])[0]["code"] == ("reserved_filename")


def test_initial_live_provider_retry_exhaustion_closes_with_durable_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_env = "KUAIRAND_TEST_OPENAI_KEY"
    monkeypatch.setenv(key_env, "sk-test-provider-unavailable")
    pricing = TokenPricing(
        input_usd_per_million="1",
        cached_input_usd_per_million="0.1",
        output_usd_per_million="2",
    )
    model = OpenAIResponsesModel(
        OpenAIResponsesConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            reasoning_effort="high",
            pricing=pricing,
            api_key_env=key_env,
            timeout_seconds=1.0,
            max_response_bytes=1_048_576,
            max_output_tokens=4096,
            max_malformed_retries=1,
            max_transport_retries=2,
        ),
        transport=_UnavailableTransport(),
    )
    research_config = ResearchConfig(
        provider="openai",
        max_repairs_per_experiment=2,
        run_kind="autonomous",
        openai=OpenAIResearchConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            reasoning_effort="high",
            api_key_env=key_env,
            timeout_seconds=1.0,
            max_response_bytes=1_048_576,
            max_output_tokens=4096,
            max_malformed_retries=1,
            max_transport_retries=2,
            pricing=OpenAITokenPricingConfig(
                input_usd_per_million="1",
                cached_input_usd_per_million="0.1",
                output_usd_per_million="2",
            ),
        ),
    )
    harness = _install_harness(
        tmp_path,
        monkeypatch,
        research_config=research_config,
        research_provider=AvailableResearchProvider(
            provider="openai",
            model=model,
            live_provider_used=True,
        ),
    )

    outcome = harness.run()

    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert outcome.selection is None
    assert outcome.scientific_result_digest is None
    assert outcome.reflection_transcript is None
    assert outcome.launches_used == 6
    assert [item.outcome for item in model.transcripts] == ["failed", "failed", "failed"]
    checkpoints = harness.progress().checkpoints()
    lineage = next(item for item in checkpoints if item.stage is FullCampaignStage.LINEAGE_READY)
    assert lineage.evidence["generated_lineage_status"] == "provider_retry_exhausted"
    assert lineage.evidence["provider_diagnostic"] == {
        "attempts": 3,
        "category": "provider_unavailable",
        "code": "transport",
        "credential_env": key_env,
        "credential_envs": [key_env],
        "message": "The live research provider failed after its bounded retry policy.",
        "operation": "propose",
        "provider": "openai",
        "provider_failures": [],
        "retryable": True,
        "status_code": None,
    }
    reflected = next(item for item in checkpoints if item.stage is FullCampaignStage.REFLECTED)
    assert reflected.evidence["portfolio_cap_reason"] == "runtime_provider_unavailable"
    assert reflected.evidence["portfolio_count"] == 1
    reflected_usage = cast(Mapping[str, Any], reflected.evidence["provider_usage"])
    assert reflected_usage["transcript_count"] == 3
    assert reflected.evidence["research_stage_counts"] == {
        "branches_attempted": 1,
        "proposal_responses_accepted": 0,
        "implementation_responses_accepted": 0,
        "repair_responses_accepted": 0,
        "branches_rejected_pre_execution": 0,
        "candidates_admitted": 0,
        "training_started": 0,
        "inner_evaluations_completed": 0,
        "outer_evaluations_completed": 0,
    }
    reflected_summary = cast(Mapping[str, Any], reflected.evidence["research_rejection_summary"])
    assert reflected_summary["branches_rejected_pre_execution"] == 0
    assert harness.science_configs == []


def test_runtime_provider_factory_shares_authoritative_remaining_research_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_env = "KUAIRAND_TEST_FACTORY_DEADLINE_KEY"
    monkeypatch.setenv(key_env, "sk-test-factory-deadline")
    harness = _install_harness(
        tmp_path,
        monkeypatch,
        research_config=_research_config(key_env=key_env),
    )
    observed_models: list[OpenAIResponsesModel] = []

    def select_provider(
        _research: object,
        *,
        openai_model_factory: Callable[[OpenAIResponsesConfig, object], object],
        **_kwargs: object,
    ) -> ProviderUnavailableDiagnostic:
        config = OpenAIResponsesConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            reasoning_effort="high",
            pricing=TokenPricing(
                input_usd_per_million="1",
                cached_input_usd_per_million="0.1",
                output_usd_per_million="2",
            ),
            api_key_env=key_env,
            timeout_seconds=1.0,
            max_response_bytes=1_048_576,
            max_output_tokens=4096,
            max_malformed_retries=1,
            max_transport_retries=0,
        )
        for _ in range(2):
            observed_models.append(cast(OpenAIResponsesModel, openai_model_factory(config, None)))
        return ProviderUnavailableDiagnostic(
            provider="openai",
            code=ProviderUnavailableCode.INITIALIZATION_FAILED,
            message="Synthetic provider closure after adapter construction.",
            retryable=True,
            credential_env=key_env,
        )

    monkeypatch.setattr(runtime, "select_research_provider", select_provider)

    outcome = harness.run()

    assert outcome.finalization_required
    assert len(observed_models) == 2
    first_runtime = cast(Any, observed_models[0])._retry_runtime
    second_runtime = cast(Any, observed_models[1])._retry_runtime
    assert first_runtime is second_runtime
    deadline = harness.engine.inspect_deadline(harness.request.run_dir)
    expected = max(
        0.0,
        deadline.remaining_seconds - harness.request.config.runner.finalization_reserve_seconds,
    )
    assert first_runtime.remaining_research_seconds() == expected


def test_initial_two_provider_exhaustion_records_both_slots_and_closes_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_key_env = "KUAIRAND_TEST_MAIN_KEY"
    fallback_key_env = "KUAIRAND_TEST_FALLBACK_KEY"
    monkeypatch.setenv(main_key_env, "sk-test-main-provider-unavailable")
    monkeypatch.setenv(fallback_key_env, "sk-test-fallback-provider-unavailable")
    pricing = TokenPricing(
        input_usd_per_million="1",
        cached_input_usd_per_million="0.1",
        output_usd_per_million="2",
    )

    def endpoint(model: str, key_env: str) -> OpenAIResponsesModel:
        return OpenAIResponsesModel(
            OpenAIResponsesConfig(
                model=model,
                base_url=f"https://{model}.example/v1",
                reasoning_effort="high",
                pricing=pricing,
                api_key_env=key_env,
                timeout_seconds=1.0,
                max_response_bytes=1_048_576,
                max_output_tokens=4096,
                max_malformed_retries=1,
                max_transport_retries=0,
            ),
            transport=_UnavailableTransport(),
        )

    model = OpenAIFailoverModel(
        endpoint("main-model", main_key_env),
        endpoint("fallback-model", fallback_key_env),
    )
    research_config = ResearchConfig(
        provider="openai",
        max_repairs_per_experiment=2,
        run_kind="autonomous",
        openai=OpenAIResearchConfig(
            model="main-model",
            base_url="https://main-model.example/v1",
            reasoning_effort="high",
            api_key_env=main_key_env,
            timeout_seconds=1.0,
            max_response_bytes=1_048_576,
            max_output_tokens=4096,
            max_malformed_retries=1,
            max_transport_retries=0,
            pricing=OpenAITokenPricingConfig(
                input_usd_per_million="1",
                cached_input_usd_per_million="0.1",
                output_usd_per_million="2",
            ),
        ),
    )
    harness = _install_harness(
        tmp_path,
        monkeypatch,
        research_config=research_config,
        research_provider=AvailableResearchProvider(
            provider="openai",
            model=model,
            live_provider_used=True,
        ),
    )

    outcome = harness.run()

    assert outcome.finalization_required
    assert outcome.fallback_preserved
    assert [item.outcome for item in model.transcripts] == ["failed", "failed"]
    lineage = next(
        item
        for item in harness.progress().checkpoints()
        if item.stage is FullCampaignStage.LINEAGE_READY
    )
    diagnostic = cast(Mapping[str, Any], lineage.evidence["provider_diagnostic"])
    assert diagnostic["credential_envs"] == [main_key_env, fallback_key_env]
    assert [item["slot"] for item in diagnostic["provider_failures"]] == [
        "main",
        "fallback",
    ]
    usage = cast(Mapping[str, Any], lineage.evidence["provider_usage"])
    assert usage["active_slot"] == "fallback"
    assert usage["failover_count"] == 1
    assert [item["slot"] for item in usage["provider_chain"]] == ["main", "fallback"]
    assert usage["transcript_count"] == 2
    assert usage["retry_wait_seconds"] == 0.0
    assert harness.science_configs == []


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

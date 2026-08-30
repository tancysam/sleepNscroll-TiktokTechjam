"""Restart-safe local composition for the bounded KuaiRand-Pure research campaign.

The module is intentionally controller-owned.  Generated source receives only immutable numeric
NPY capabilities; public labels remain closed over protected scorers and final outcomes have no
representable input anywhere in this call graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import load_predictions
from kuairand_agent.baselines.fold_control_runner import (
    FoldFMControlExecutionRequest,
    SupervisedFoldFMExecutionError,
    SupervisedFoldFMRun,
    SupervisedFoldFMRunner,
)
from kuairand_agent.baselines.fold_controls import (
    FoldScoringContext,
    build_fold_scoring_context,
)
from kuairand_agent.campaign.budgets import AdmissionReason, LaunchCategory, WorkPhase
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateAdmissionError,
    CandidateJournalPolicy,
)
from kuairand_agent.campaign.controller import (
    CAMPAIGN_DATABASE_NAME,
    CampaignCreateRequest,
    CampaignEngine,
    CampaignStatus,
)
from kuairand_agent.campaign.full_campaign import (
    CampaignDataPlane,
    FinalizationSelectionPlan,
    FullCampaignCancelled,
    FullCampaignError,
    FullCampaignOutcome,
    FullCampaignOutcomeRepository,
    FullCampaignProgressLedger,
    FullCampaignStage,
    InnerFoldSelectionEvidence,
    MatchedSeedSelectionEvidence,
    ProductionFeatureBundle,
    QualifiedFMMemberPlan,
    build_finalization_candidate_inputs,
    build_production_feature_bundle,
    encode_numeric_user_groups,
    load_full_campaign_outcome,
    prepare_campaign_data_plane,
)
from kuairand_agent.campaign.generated_scientific_runner import (
    DurableGeneratedScientificRunner,
    FileScientificRunEvidenceRepository,
    FoldBFusionSelector,
    GeneratedScientificRunnerCancelledError,
    GeneratedScientificRunRecord,
    ProtectedScoreCallback,
    ProtectedScoringCapability,
    ScientificTierCapabilities,
    TrustedFMRankFusion,
)
from kuairand_agent.campaign.models import CampaignState
from kuairand_agent.campaign.provenance import (
    capture_environment_identity,
    hash_source_tree,
)
from kuairand_agent.campaign.qualification_evidence import (
    OfficialFMQualificationEvidence,
    OfficialFMSeedEvidence,
    QualificationExpectations,
    load_official_fm_qualification,
)
from kuairand_agent.campaign.receipt_outer_evaluation import (
    ReceiptAwareOuterEvaluationLedger,
)
from kuairand_agent.campaign.scientific import (
    CampaignStopReason,
    CandidateCampaignResult,
    CandidateOutcome,
    ExecutableChangeEvidence,
    OuterPromotionRequest,
    ScientificCampaignCancelled,
    ScientificCampaignConfig,
    ScientificCampaignResult,
    ScientificCandidate,
    ScientificRunEvidence,
    ScientificRunRequest,
    ScientificTier,
    run_scientific_campaign,
)
from kuairand_agent.campaign.scientific_store import (
    DurableScientificLedgerAdapter,
    TrustedOuterSeedEvidence,
)
from kuairand_agent.campaign.scoring_receipts import ScoringReceiptBook
from kuairand_agent.campaign.selector import (
    MATERIAL_PRIMARY_DELTA,
    GateEvidence,
    IncumbentEvidence,
    OrganizerMetrics,
    SeedMetrics,
)
from kuairand_agent.campaign.store import (
    ArtifactSpec,
    CampaignStore,
    OuterQueryLedger,
)
from kuairand_agent.config import ResearchConfig
from kuairand_agent.contract import benchmark_digest, verify_starter_kit
from kuairand_agent.data.canonical import (
    CanonicalInputs,
    ProtectedTargets,
    load_canonical_dataset,
)
from kuairand_agent.data.capabilities import DataPhase, build_candidate_inputs
from kuairand_agent.data.fields import (
    STANDARD_LATE_MEMBER,
    STANDARD_TRAIN_MEMBER,
    VIDEO_BASIC_MEMBER,
    FieldKey,
)
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
)
from kuairand_agent.execution.candidate_executor import (
    CandidateAction,
    GeneratedCandidateExecutor,
    LocalCandidateLimits,
    put_numpy_capability,
)
from kuairand_agent.execution.policy import WorkspacePolicy
from kuairand_agent.execution.runner import ExecutionOutcome, active_python_interpreter
from kuairand_agent.execution.workspace import WorkspaceMaterializer
from kuairand_agent.research.context import (
    AggregateRecord,
    MetricSummary,
    ResearchBudgetContext,
    SafeResearchContext,
    build_safe_research_context,
)
from kuairand_agent.research.factory import (
    AvailableResearchProvider,
    ProviderUnavailableDiagnostic,
    select_research_provider,
)
from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.loop import CampaignStoreResearchLedger
from kuairand_agent.research.materialize import snapshot_materialized_candidate
from kuairand_agent.research.production import (
    SCRIPTED_CANDIDATE_ID,
    SCRIPTED_PARENT_ID,
    LiveResearchBranchRejected,
    LiveResearchLineage,
    ScriptedLambdaRankLineage,
    classify_proposal_family,
    load_parent_snapshot,
    prepare_or_rehydrate_live_lineage,
    prepare_or_rehydrate_scripted_lambdarank_lineage,
)
from kuairand_agent.research.provider import (
    OpenAIChatCompletionsConfig,
    OpenAIChatCompletionsModel,
    OpenAIFailoverModel,
    OpenAIProviderChainError,
    OpenAIProviderError,
    ProviderErrorCode,
    ProviderModelLimitResolver,
    ProviderTranscript,
    ResponsesTransport,
    RetryRuntime,
)
from kuairand_agent.research.schemas import (
    ExperimentResultSummary,
    ParentSnapshot,
    Proposal,
    Reflection,
    ReflectionRequest,
    ResearchOperation,
    canonical_json_bytes,
)
from kuairand_agent.research.scripted import ScriptedResponse
from kuairand_agent.scoring.protected import (
    Alignment,
    ProtectedScorer,
    ScoreResult,
    SplitIdentity,
)

_SCHEMA_VERSION = 1
_PRODUCTION_DIR = "production"
_PORTFOLIO_REASON = "bounded_high_value_lambdarank_branch_prioritized"
_REJECTION_JOURNAL_DIR = "controller-rejection-journal"
_REJECTION_JOURNAL_LIMIT_BYTES = 65_536
_PROVIDER_ATTEMPT_JOURNAL_DIR = "provider-attempt-journal"
_SAFE_FOLD_FAILURE_OUTCOMES = frozenset(
    {
        ExecutionOutcome.TIMED_OUT,
        ExecutionOutcome.MEMORY_LIMIT,
        ExecutionOutcome.DISK_LIMIT,
        ExecutionOutcome.PROCESS_LIMIT,
        ExecutionOutcome.SPAWN_FAILED,
    }
)
_SAFE_FOLD_ADMISSION_REASONS = frozenset(
    {
        AdmissionReason.FINALIZATION_RESERVE,
        AdmissionReason.HARD_DEADLINE,
        AdmissionReason.HARD_LAUNCH_CAP,
        AdmissionReason.CATEGORY_CAP,
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical_json(value)).hexdigest()


def _resolve_directory(root: Path, value: Path, name: str) -> Path:
    candidate = value if value.is_absolute() else root / value
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.lstat()
    except (OSError, RuntimeError) as exc:
        raise FullCampaignError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FullCampaignError(f"{name} must be a real directory")
    return resolved


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise FullCampaignCancelled(
            "campaign cancelled by request; durable state is resumable and no new launch "
            "was admitted"
        )


def _rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


class _MetricValue(Protocol):
    @property
    def gauc(self) -> float: ...

    @property
    def ndcg_at_5(self) -> float: ...


def _metrics(value: _MetricValue) -> OrganizerMetrics:
    try:
        gauc = float(value.gauc)
        ndcg = float(value.ndcg_at_5)
    except (TypeError, ValueError) as exc:
        raise FullCampaignError("trusted metric evidence is malformed") from exc
    return OrganizerMetrics(gauc, ndcg)


def _artifact_spec(reference: ArtifactRef) -> ArtifactSpec:
    return ArtifactSpec(
        digest=reference.sha256,
        kind=reference.kind.value,
        relative_path=reference.object_relative_path.as_posix(),
        size_bytes=reference.size_bytes,
    )


def _qualified_member(
    qualification: OfficialFMQualificationEvidence,
    seed: OfficialFMSeedEvidence,
) -> QualifiedFMMemberPlan:
    return QualifiedFMMemberPlan(
        seed=seed.seed,
        checkpoint_sha256=seed.checkpoint_file_sha256,
        checkpoint_digest=seed.checkpoint_digest,
        encoding_sha256=seed.encoding_file_sha256,
        encoding_digest=seed.encoding_digest,
        config_digest=seed.config_digest,
        starter_manifest_digest=qualification.starter_manifest_digest,
        validation_prediction_digest=seed.validation_prediction_digest,
    )


@dataclass(frozen=True, slots=True)
class _FeatureArtifacts:
    fold_a_train: ArtifactRef
    fold_a_targets: ArtifactRef
    fold_a_groups: ArtifactRef
    fold_a_valid: ArtifactRef
    fold_b_train: ArtifactRef
    fold_b_targets: ArtifactRef
    fold_b_groups: ArtifactRef
    fold_b_valid: ArtifactRef
    outer_train: ArtifactRef
    outer_targets: ArtifactRef
    outer_groups: ArtifactRef
    outer_valid: ArtifactRef
    final: ArtifactRef

    def manifest(self) -> dict[str, object]:
        return {
            name: cast(ArtifactRef, getattr(self, name)).manifest()
            for name in self.__dataclass_fields__
        }


def _put_feature_artifacts(
    store: ArtifactStore,
    bundle: ProductionFeatureBundle,
    *,
    fold_a_targets: tuple[int, ...],
    fold_a_users: Sequence[object],
    fold_b_targets: tuple[int, ...],
    fold_b_users: Sequence[object],
    outer_targets: tuple[int, ...],
    outer_users: Sequence[object],
) -> _FeatureArtifacts:
    return _FeatureArtifacts(
        fold_a_train=put_numpy_capability(store, bundle.fold_a.prefix.values),
        fold_a_targets=put_numpy_capability(store, np.asarray(fold_a_targets, dtype=np.int8)),
        fold_a_groups=put_numpy_capability(store, encode_numeric_user_groups(fold_a_users)),
        fold_a_valid=put_numpy_capability(store, bundle.fold_a.query.values),
        fold_b_train=put_numpy_capability(store, bundle.fold_b.prefix.values),
        fold_b_targets=put_numpy_capability(store, np.asarray(fold_b_targets, dtype=np.int8)),
        fold_b_groups=put_numpy_capability(store, encode_numeric_user_groups(fold_b_users)),
        fold_b_valid=put_numpy_capability(store, bundle.fold_b.query.values),
        outer_train=put_numpy_capability(store, bundle.outer_and_final.prefix.values),
        outer_targets=put_numpy_capability(store, np.asarray(outer_targets, dtype=np.int8)),
        outer_groups=put_numpy_capability(store, encode_numeric_user_groups(outer_users)),
        outer_valid=put_numpy_capability(store, bundle.outer_validation.values),
        final=put_numpy_capability(store, bundle.final.values),
    )


def _limits(request: CampaignCreateRequest) -> LocalCandidateLimits:
    runner = request.config.runner
    memory = int(runner.memory_mb) * 1024**2
    disk = int(runner.disk_mb) * 1024**2
    return LocalCandidateLimits(
        timeout_seconds=float(runner.default_timeout_seconds),
        memory_limit_bytes=memory,
        workspace_disk_limit_bytes=disk,
        output_limit_bytes=min(disk, 512 * 1024**2),
        temp_limit_bytes=min(disk, 512 * 1024**2),
        threads=int(runner.threads),
        device=str(runner.device),
    )


def _stage(
    progress: FullCampaignProgressLedger,
    *,
    request_digest: str,
    stage: FullCampaignStage,
    evidence: Mapping[str, object],
) -> Mapping[str, object]:
    retained = next((item for item in progress.checkpoints() if item.stage is stage), None)
    if retained is not None:
        return retained.evidence
    return progress.append(
        request_digest=request_digest,
        stage=stage,
        evidence=evidence,
    ).evidence


@dataclass(frozen=True, slots=True)
class _ProtectedTierScorer:
    split: SplitIdentity
    alignment: Alignment
    scorer: ProtectedScorer
    labels: tuple[int, ...]

    def score(self, scores: npt.NDArray[np.float64]) -> ScoreResult:
        return self.scorer.score_with_encoded_labels(
            alignment=self.alignment,
            split=self.split,
            labels=self.labels,
            scores=scores,
            expected_count=len(self.labels),
        )


def _outer_scorer(
    *,
    starter_dir: Path,
    inputs: CanonicalInputs,
    targets: ProtectedTargets,
    split_token: str,
) -> _ProtectedTierScorer:
    split = SplitIdentity("outer_valid", split_token, len(inputs))
    alignment = Alignment.from_ids(
        split=split,
        user_ids=inputs.user_id,
        video_ids=inputs.video_id,
    )
    return _ProtectedTierScorer(
        split=split,
        alignment=alignment,
        scorer=ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment),
        labels=targets.reveal_for_scorer(),
    )


def _journal(
    *,
    engine: CampaignEngine,
    run_dir: Path,
    store: CampaignStore,
    artifacts: ArtifactStore,
    family: str,
    phase: WorkPhase,
    category: LaunchCategory | None,
    p95_seconds: float,
    cleanup_seconds: float,
    experiment_id: str | None,
    scientific_iteration: int | None,
    cancel_event: Event | None,
) -> CampaignStoreCandidateJournal:
    _check_cancel(cancel_event)
    observation = engine.observe_deadline(run_dir)
    _check_cancel(cancel_event)
    return CampaignStoreCandidateJournal(
        store=store,
        artifact_store=artifacts,
        deadline=observation,
        policy=CandidateJournalPolicy(
            family=family,
            phase=phase,
            category=category,
            p95_runtime_seconds=p95_seconds,
            cleanup_seconds=cleanup_seconds,
            experiment_id=experiment_id,
            scientific_iteration=scientific_iteration,
        ),
    )


def _fold_control(
    *,
    fold_name: str,
    fold_token: str,
    prefix_inputs: CanonicalInputs,
    prefix_labels: tuple[int, ...],
    query_inputs: CanonicalInputs,
    query_labels: tuple[int, ...],
    engine: CampaignEngine,
    run_dir: Path,
    campaign_store: CampaignStore,
    artifacts: ArtifactStore,
    workspaces: WorkspaceMaterializer,
    starter_dir: Path,
    limits: LocalCandidateLimits,
    cancel_event: Event | None,
) -> SupervisedFoldFMRun:
    if fold_name not in {"A", "B"}:
        raise FullCampaignError("fold control must name A or B")
    category = (
        LaunchCategory.TEMPORAL_FOLD_CONFIRMATION
        if fold_name == "A"
        else LaunchCategory.DIVERSE_INNER_SCREEN
    )
    journal = _journal(
        engine=engine,
        run_dir=run_dir,
        store=campaign_store,
        artifacts=artifacts,
        family="official_fm_fold_control",
        phase=WorkPhase.RESEARCH,
        category=category,
        p95_seconds=float(limits.timeout_seconds),
        cleanup_seconds=5.0,
        experiment_id=None,
        scientific_iteration=None,
        cancel_event=cancel_event,
    )
    runner = SupervisedFoldFMRunner(
        artifact_store=artifacts,
        workspace_materializer=workspaces,
        control_root=run_dir / _PRODUCTION_DIR / "fold-controls",
        interpreter=active_python_interpreter(),
        starter_dir=starter_dir,
        limits=limits,
    )
    return runner.run(
        FoldFMControlExecutionRequest(
            execution_id=f"fold-fm-control-{fold_name}-seed-0",
            fold_name=cast(Literal["A", "B"], fold_name),
            fold_token=fold_token,
            seed=0,
            prefix_inputs=prefix_inputs,
            prefix_labels=prefix_labels,
            query_inputs=query_inputs,
            query_labels=query_labels,
        ),
        journal=journal,
        cancel_event=cancel_event,
    )


def _feature_cache_build(
    *,
    data: CampaignDataPlane,
    builder_source_digest: str,
    cache_dir: Path,
    prior_stage: Mapping[str, object] | None,
) -> tuple[ProductionFeatureBundle, Mapping[str, object]]:
    cache_was_empty = not cache_dir.exists() or not any(cache_dir.iterdir())
    before_rss = _rss_bytes()
    started = time.perf_counter()
    bundle = build_production_feature_bundle(
        data,
        builder_source_digest=builder_source_digest,
        cache_dir=cache_dir,
    )
    initial_wall = time.perf_counter() - started
    after_initial_rss = _rss_bytes()
    started = time.perf_counter()
    replay = build_production_feature_bundle(
        data,
        builder_source_digest=builder_source_digest,
        cache_dir=cache_dir,
    )
    warm_wall = time.perf_counter() - started
    after_warm_rss = _rss_bytes()
    if replay.digest != bundle.digest:
        raise FullCampaignError("warm causal-feature replay changed the feature identity")
    if prior_stage is not None:
        receipt = prior_stage.get("cache_performance")
        if not isinstance(receipt, Mapping):
            raise FullCampaignError("retained feature stage lacks cache performance evidence")
        return bundle, receipt
    return bundle, {
        "schema_version": _SCHEMA_VERSION,
        "cache_dir": str(cache_dir),
        "initial_cache_state": "cold" if cache_was_empty else "warm",
        "initial_wall_seconds": initial_wall,
        "verified_warm_wall_seconds": warm_wall,
        "rss_before_bytes": before_rss,
        "rss_after_initial_bytes": after_initial_rss,
        "rss_after_warm_bytes": after_warm_rss,
        "bundle_identity_equal": True,
    }


def _fallback_receipt(
    qualification: OfficialFMQualificationEvidence,
    fold_a: SupervisedFoldFMRun,
    fold_b: SupervisedFoldFMRun,
) -> str:
    return _digest(
        b"kuairand-production-fallback-receipt-v1",
        {
            "qualification_manifest_digest": qualification.manifest_digest,
            "fallback_manifest_digest": qualification.fallback.manifest_digest,
            "fold_a_evidence_digest": fold_a.evidence.digest,
            "fold_b_evidence_digest": fold_b.evidence.digest,
            "outer_seeds": [item.seed for item in qualification.outer_runs],
            "replayable": True,
        },
    )


def _fallback_incumbent(
    qualification: OfficialFMQualificationEvidence,
    fold_a: SupervisedFoldFMRun,
    fold_b: SupervisedFoldFMRun,
    receipt_digest: str,
) -> IncumbentEvidence:
    return IncumbentEvidence(
        candidate_id=SCRIPTED_PARENT_ID,
        inner_by_fold=(
            ("A", _metrics(fold_a.control.metrics)),
            ("B", _metrics(fold_b.control.metrics)),
        ),
        outer_by_seed=tuple(
            SeedMetrics(item.seed, _metrics(item.metrics)) for item in qualification.outer_runs
        ),
        evidence_receipt_digest=receipt_digest,
        replayable=True,
        eligible=True,
        official_fm=True,
    )


def _safe_context(
    *,
    request: CampaignCreateRequest,
    qualification: OfficialFMQualificationEvidence,
    fold_a: SupervisedFoldFMRun,
    fold_b: SupervisedFoldFMRun,
    status: CampaignStatus,
    campaign_records: Sequence[AggregateRecord] = (),
    evidence: _ResearchContextEvidence | None = None,
) -> SafeResearchContext:
    remaining_seconds = max(0, int(status.deadline_remaining_seconds))
    context_evidence = _ResearchContextEvidence() if evidence is None else evidence
    return build_safe_research_context(
        starter_manifest_sha256=qualification.starter_manifest_digest,
        dataset_manifest_sha256=qualification.canonical_digest,
        capability_manifests=context_evidence.capability_manifests,
        budgets=ResearchBudgetContext(
            remaining_attempts=status.launches_remaining,
            remaining_wall_seconds=min(21_600, remaining_seconds),
            remaining_outer_promotions=status.outer_queries_remaining,
            intervention_count=0,
        ),
        inner_metrics=(
            MetricSummary(
                "official_fm_fold_A",
                fold_a.control.metrics.gauc,
                fold_a.control.metrics.ndcg_at_5,
                fold_a.control.metrics.primary,
                exact=True,
            ),
            MetricSummary(
                "official_fm_fold_B",
                fold_b.control.metrics.gauc,
                fold_b.control.metrics.ndcg_at_5,
                fold_b.control.metrics.primary,
                exact=True,
            ),
        ),
        outer_metrics=tuple(
            MetricSummary(
                f"official_fm_seed_{item.seed}",
                item.metrics.gauc,
                item.metrics.ndcg_at_5,
                item.metrics.primary,
            )
            for item in qualification.outer_runs
        ),
        train_eda=context_evidence.train_eda,
        validation_input_eda=context_evidence.validation_input_eda,
        method_cards=context_evidence.method_cards,
        campaign_records=campaign_records,
    )


@dataclass(frozen=True, slots=True)
class _ResearchContextEvidence:
    capability_manifests: tuple[Mapping[str, object], ...] = ()
    train_eda: tuple[AggregateRecord, ...] = ()
    validation_input_eda: tuple[AggregateRecord, ...] = ()
    method_cards: tuple[AggregateRecord, ...] = ()


def _input_capability_manifest(
    phase: DataPhase,
    inputs: CanonicalInputs,
) -> Mapping[str, object]:
    member = (
        STANDARD_LATE_MEMBER
        if phase in {DataPhase.OUTER_VALID, DataPhase.FINAL}
        else STANDARD_TRAIN_MEMBER
    )
    return build_candidate_inputs(
        phase,
        {
            FieldKey(member, "user_id"): inputs.user_id,
            FieldKey(member, "video_id"): inputs.video_id,
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): inputs.author_id,
            FieldKey(member, "tab"): inputs.tab,
            FieldKey(member, "duration_ms"): inputs.duration_ms,
        },
    ).manifest()


def _input_aggregate(name: str, inputs: CanonicalInputs) -> AggregateRecord:
    durations = np.asarray(inputs.duration_ms, dtype=np.float64)
    return AggregateRecord(
        name=name,
        values={
            "row_count": len(inputs),
            "unique_user_count": len(set(inputs.user_id)),
            "unique_video_count": len(set(inputs.video_id)),
            "unique_author_count": len(set(inputs.author_id)),
            "date_min": min(inputs.date),
            "date_max": max(inputs.date),
            "tab_cardinality": len(set(inputs.tab)),
            "duration_ms_mean": float(np.mean(durations)),
            "duration_ms_p10": float(np.quantile(durations, 0.10)),
            "duration_ms_p50": float(np.quantile(durations, 0.50)),
            "duration_ms_p90": float(np.quantile(durations, 0.90)),
        },
    )


def _research_context_evidence(
    data: CampaignDataPlane,
    features: ProductionFeatureBundle,
) -> _ResearchContextEvidence:
    train_positive_count = sum(data.outer_train_labels)
    train_click_count = sum(data.outer_train_click_labels)
    train_long_view_and_click_count = sum(
        long_view & click
        for long_view, click in zip(
            data.outer_train_labels,
            data.outer_train_click_labels,
            strict=True,
        )
    )
    train_count = len(data.outer_train_labels)
    watch_progress = np.asarray(data.outer_train_watch_progress, dtype=np.float64)
    long_view_negative_count = train_count - train_positive_count
    click_negative_count = train_count - train_click_count
    phi_denominator = math.sqrt(
        train_positive_count * long_view_negative_count * train_click_count * click_negative_count
    )
    long_view_click_phi = (
        0.0
        if phi_denominator == 0.0
        else (
            train_long_view_and_click_count * train_count - train_positive_count * train_click_count
        )
        / phi_denominator
    )
    return _ResearchContextEvidence(
        capability_manifests=(
            _input_capability_manifest(DataPhase.TRAIN, data.outer_train_inputs),
            _input_capability_manifest(DataPhase.INNER_VALID, data.fold_b.query_inputs),
            _input_capability_manifest(DataPhase.OUTER_VALID, data.outer_validation_inputs),
            _input_capability_manifest(DataPhase.FINAL, data.final_inputs),
        ),
        train_eda=(
            _input_aggregate("official_train_inputs", data.outer_train_inputs),
            AggregateRecord(
                "official_train_targets",
                {
                    "row_count": train_count,
                    "long_view_positive_count": train_positive_count,
                    "long_view_positive_rate": train_positive_count / train_count,
                    "click_positive_count": train_click_count,
                    "click_positive_rate": train_click_count / train_count,
                    "long_view_and_click_positive_count": train_long_view_and_click_count,
                    "click_rate_given_long_view": (
                        train_long_view_and_click_count / train_positive_count
                    ),
                    "long_view_click_phi": long_view_click_phi,
                    "watch_progress_mean": float(np.mean(watch_progress)),
                    "watch_progress_p50": float(np.quantile(watch_progress, 0.50)),
                    "watch_progress_p90": float(np.quantile(watch_progress, 0.90)),
                },
            ),
        ),
        validation_input_eda=(
            _input_aggregate("public_validation_inputs_only", data.outer_validation_inputs),
            _input_aggregate("prediction_period_inputs_only", data.final_inputs),
        ),
        method_cards=(
            AggregateRecord(
                "controller_causal_feature_bundle",
                {
                    "feature_count": features.final.feature_count,
                    "feature_names_csv": ",".join(features.final.feature_names),
                    "feature_bundle_digest": features.digest,
                    "fold_b_selected_fusion_only": True,
                    "uses_public_labels_for_features": False,
                    "uses_prediction_period_outcomes": False,
                },
            ),
            AggregateRecord(
                "input_only_strict_past_exposure",
                {
                    "status": "enabled_in_feature_schema_v8",
                    "feature_positions_zero_based_csv": "83,84,85,86,87,88,89,90,91,92,93,94",
                    "feature_names_csv": (
                        "user__strict_earlier_exposure_count,user__first_seen,"
                        "user__log1p_time_since_last_exposure,"
                        "video__strict_earlier_exposure_count,video__first_seen,"
                        "video__log1p_time_since_last_exposure,"
                        "author__strict_earlier_exposure_count,author__first_seen,"
                        "author__log1p_time_since_last_exposure,"
                        "user_video__strict_earlier_exposure_count,user_video__first_seen,"
                        "user_video__log1p_time_since_last_exposure"
                    ),
                    "source_fields_csv": "date,time_ms,user_id,video_id,author_id",
                    "chronology": "date first, then time_ms; equal timestamp buckets are atomic",
                    "training_policy": "each row sees only strictly earlier input impressions",
                    "query_policy": (
                        "earlier query inputs update later query features; no query outcome is "
                        "accepted or consulted"
                    ),
                    "query_outcomes_accepted_by_interface": False,
                    "raw_identifiers_exposed_to_candidate": False,
                },
            ),
            AggregateRecord(
                "generated_model_runtime",
                {
                    "numpy_version": np.__version__,
                    "lightgbm_version": "4.7.0",
                    "lightgbm_import_allowed": True,
                    "cpu_threads": 4,
                    "checkpoint_array_limit": 64,
                    "checkpoint_numeric_arrays_only": True,
                },
            ),
            AggregateRecord(
                "causal_feature_semantics",
                {
                    "history_scopes_csv": (
                        "user,video,author,tab,duration_bucket,user_author,user_tab,"
                        "author_tab,user_duration_bucket"
                    ),
                    "columns_per_history_scope_csv": (
                        "strict_past_exposure,strict_past_positive,smoothed_long_view_rate"
                    ),
                    "history_smoothing": 20.0,
                    "history_initial_prior": 0.5,
                    "training_history_policy": (
                        "strictly earlier timestamp buckets only; simultaneous rows never "
                        "observe one another"
                    ),
                    "query_history_policy": (
                        "outcome-bearing histories are frozen at each temporal-fold training "
                        "cutoff; only the separate input-only exposure family may update from "
                        "strictly earlier query inputs"
                    ),
                    "recency_companion_scopes_csv": "user,video,user_author",
                    "recency_companion_columns_csv": ("decayed_exposure,smoothed_long_view_rate"),
                    "recency_half_life_days_csv": "1,3,7",
                    "recency_previous_single_half_life_days": 3.0,
                    "recency_query_policy": (
                        "decay frozen training state to each query timestamp without updating it"
                    ),
                    "recency_search_evidence": (
                        "single-horizon and 1/3/7-day recency transforms were directionally "
                        "positive but non-material; do not add more horizons"
                    ),
                    "duration_bucket_edges_seconds_csv": "5,10,18,30,60",
                    "duration_seconds_transform": "duration_ms divided by 1000",
                    "duration_log_transform": "log1p duration_seconds",
                    "duration_threshold_transform": "one when duration_seconds is at least 18",
                    "date_transform": "integer date minus 20220408",
                    "tab_transform": "numeric tab value as float64",
                },
            ),
            AggregateRecord(
                "lightgbm_lambdarank_pattern",
                {
                    "recommended_use": "first-choice nonlinear grouped ranker over causal features",
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "label_gain": "0,1",
                    "eval_at": 5,
                    "deterministic_cpu": True,
                    "training_group_contract": (
                        "stable-sort rows by user_groups and pass contiguous query lengths"
                    ),
                    "prediction_group_required": False,
                    "fixed_tree_count_required": True,
                    "checkpoint_pattern": (
                        "encode Booster.model_to_string UTF-8 bytes as a uint8 NumPy array"
                    ),
                    "uses_protected_outcomes": False,
                },
            ),
            AggregateRecord(
                "strict_past_click_history",
                {
                    "status": "newly_enabled_after_56_feature_plateau",
                    "source_target": "official_train_is_click_only",
                    "history_scopes_csv": "user,video,author,user_author",
                    "columns_per_scope_csv": (
                        "strict_past_exposure,strict_past_click_positive,smoothed_click_rate"
                    ),
                    "global_feature": "click_global__prior",
                    "history_smoothing": 20.0,
                    "training_history_policy": (
                        "strictly earlier timestamp buckets only; simultaneous rows never "
                        "observe one another"
                    ),
                    "query_history_policy": (
                        "frozen at each temporal-fold training cutoff; query rows never update "
                        "history"
                    ),
                    "same_row_click_exposed": False,
                    "uses_late_period_outcomes": False,
                    "raw_click_target_exposed_to_candidate": False,
                    "recommended_experiment": (
                        "measure whether strict-past click propensity and its disagreement with "
                        "strict-past long_view propensity improve grouped ranking; preserve the "
                        "official FM control and avoid repeating plateaued recency-only, "
                        "categorical-only, or unchanged LambdaRank configurations"
                    ),
                },
            ),
            AggregateRecord(
                "organizer_fm_categorical_codes",
                {
                    "status": "newly_enabled_after_dense_39_feature_plateau",
                    "feature_names_csv": (
                        "starter_fm_code__user_id,starter_fm_code__video_id,"
                        "starter_fm_code__author_id,starter_fm_code__tab,"
                        "starter_fm_code__dur_bucket"
                    ),
                    "field_order_csv": "user_id,video_id,author_id,tab,dur_bucket",
                    "encoding": (
                        "exact organizer StarterEncoding global integer IDs represented exactly "
                        "as float64"
                    ),
                    "fit_policy": "fit independently on each exact temporal-fold prefix",
                    "query_unknown_policy": "one prefix-fitted unknown slot per field",
                    "candidate_input_policy": (
                        "candidate receives numeric codes only; raw identifier strings and "
                        "vocabularies remain controller-private"
                    ),
                    "recommended_experiment": (
                        "test a pairwise or listwise factorization model on these five code "
                        "columns, optionally using the 39 dense causal companions; do not repeat "
                        "the official pointwise-logloss FM"
                    ),
                    "starter_evidence": (
                        "the organizer identifies ranking-loss alignment and user history as the "
                        "highest-priority untested directions after static features and larger "
                        "FM capacity showed no material gain"
                    ),
                },
            ),
            AggregateRecord(
                "trusted_pairwise_fm_reference",
                {
                    "status": "inner_fold_verified_not_promoted",
                    "feature_positions_zero_based_csv": "51,52,53,54,55",
                    "feature_names_csv": (
                        "starter_fm_code__user_id,starter_fm_code__video_id,"
                        "starter_fm_code__author_id,starter_fm_code__tab,"
                        "starter_fm_code__dur_bucket"
                    ),
                    "mechanism": (
                        "five-field float32 factorization machine trained with logged "
                        "same-user pairwise logistic comparisons"
                    ),
                    "sampler": (
                        "uniform positive row among eligible users, then one observed negative "
                        "row from the same user"
                    ),
                    "factor_dim": 16,
                    "learning_rate": 0.001,
                    "l2": 0.000001,
                    "batch_size": 8192,
                    "pairs_per_epoch": 250000,
                    "epochs": 5,
                    "seed": 20260830,
                    "fold_b_raw_primary": 0.574186444,
                    "fold_b_fused_primary": 0.5763888955116272,
                    "fold_b_control_primary": 0.5754240155220032,
                    "fold_b_primary_delta": 0.000964879989624,
                    "frozen_candidate_weight": 0.4,
                    "frozen_control_weight": 0.6,
                    "fold_a_raw_primary": 0.6072738766670227,
                    "fold_a_fused_primary": 0.6081787347793579,
                    "fold_a_control_primary": 0.6071290373802185,
                    "fold_a_primary_delta": 0.0010496973991394,
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "this is the strongest directionally consistent unpromoted mechanism. "
                        "If testing it, implement the exact organizer categorical-code FM and "
                        "GAUC-aligned logged same-user sampler, preserve the official FM control, "
                        "and seek a substantive complement rather than merely retuning the same "
                        "six fixed hyperparameters"
                    ),
                    "uses_public_labels_for_tuning": False,
                },
            ),
            AggregateRecord(
                "protected_candidate_pairwise_fm_primitive",
                {
                    "status": "available_video_type_expansion_requires_fresh_train_fold_ablation",
                    "protected_path": "reference_pairwise_fm.py",
                    "train_function": (
                        "train_reference_pairwise_fm(features, targets, user_groups, *, seed)"
                    ),
                    "predict_function": ("reference_pairwise_fm_scores(features, checkpoint)"),
                    "diagnostics_function": "reference_pairwise_fm_diagnostics(checkpoint)",
                    "pair_sampler_function": (
                        "sample_reference_logged_pairs(user_groups, targets, *, pair_count, seed)"
                    ),
                    "checkpoint_prefix": "reference_",
                    "feature_positions_zero_based_csv": "51,52,53,54,55",
                    "sampler": (
                        "positive-ticket sampling: each eligible logged positive has equal "
                        "probability, then one logged negative is drawn from the same user"
                    ),
                    "encoding": "unmodified prefix-fitted organizer global integer codes",
                    "numeric_recipe": (
                        "float32 five-field FM reductions and dense Adam with the exact frozen "
                        "trusted_pairwise_fm_reference configuration"
                    ),
                    "composition_policy": (
                        "generated model_impl.py may import the primitive and compose a new "
                        "residual, calibration, or ensemble around its score; it must retain all "
                        "reference_* checkpoint arrays, must use the protected sampler for every "
                        "new pairwise objective, and must not reimplement group offsets, row-index "
                        "maps, pair sampling, or the primitive"
                    ),
                    "protected_from_generated_overlay": True,
                    "metric_or_scorer_access": False,
                },
            ),
            AggregateRecord(
                "protected_duration_conditioned_pair_ablation",
                {
                    "status": (
                        "implemented_full_budget_train_only_positive_mean_but_not_promotion_ready"
                    ),
                    "protected_paths_csv": (
                        "reference_observed_pair_objectives.py,reference_observed_pair_fm.py"
                    ),
                    "control_train_function": (
                        "train_reference_uniform_pairwise_fm(features, targets, user_groups, "
                        "*, seed)"
                    ),
                    "treatment_train_function": (
                        "train_reference_duration_pairwise_fm(features, targets, user_groups, "
                        "*, seed)"
                    ),
                    "predict_function": (
                        "reference_observed_pair_fm_scores(features, checkpoint)"
                    ),
                    "duration_feature_position_zero_based": 46,
                    "duration_bucket_edges_seconds_csv": "5,10,18,30,60",
                    "treatment_pair_policy": (
                        "exactly half uniform logged same-user pairs and half logged same-user "
                        "positive-negative pairs constrained to the same duration bucket"
                    ),
                    "control_admission": (
                        "uniform arm model-state arrays are byte-exact to the protected "
                        "reference pairwise FM at equal seed and budget"
                    ),
                    "equal_compute_budget": True,
                    "prediction_accepts_targets_or_groups": False,
                    "full_budget_replicate_seeds_csv": "0,1,2",
                    "full_budget_replicate_folds_csv": "A,B",
                    "full_budget_mean_primary_delta_vs_uniform": 0.0007370909,
                    "full_budget_worst_primary_delta_vs_uniform": -0.0001828671,
                    "full_budget_decision": (
                        "retain as experimental specialist only: positive mean evidence did not "
                        "pass the worst-cell robustness gate"
                    ),
                    "evidence_report": (
                        "docs/research/observed_pair_duration_pilot-20260830.md"
                    ),
                    "protected_from_generated_overlay": True,
                },
            ),
            AggregateRecord(
                "strict_past_watch_progress_history",
                {
                    "status": "newly_enabled_after_69_feature_click_history_plateau",
                    "source_target": "official_train_play_time_ms_only",
                    "history_scopes_csv": "user,video,author,user_author",
                    "columns_per_scope_csv": (
                        "strict_past_exposure,strict_past_progress_sum,smoothed_progress_mean"
                    ),
                    "global_feature": "watch_global__mean",
                    "transform": ("clip(play_time_ms / max(min(duration_ms, 18000), 1), 0, 2)"),
                    "history_smoothing": 20.0,
                    "training_history_policy": (
                        "strictly earlier timestamp buckets only; simultaneous rows never "
                        "observe one another"
                    ),
                    "query_history_policy": (
                        "frozen at each temporal-fold training cutoff; query rows never update "
                        "history"
                    ),
                    "same_row_play_time_exposed": False,
                    "uses_late_period_outcomes": False,
                    "raw_play_time_target_exposed_to_candidate": False,
                    "recommended_experiment": (
                        "test whether graded prior threshold progress complements binary "
                        "long_view and click histories, especially through scope disagreement "
                        "and reliability; do not repeat the exact 69-feature models"
                    ),
                },
            ),
            AggregateRecord(
                "historical_33_feature_lambdarank_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "configuration": (
                        "300 trees, learning_rate 0.05, num_leaves 31, min_data_in_leaf 20, "
                        "lambdarank with deterministic CPU"
                    ),
                    "fold_b_fused_primary": 0.5756440162658691,
                    "fold_b_control_primary": 0.5754240304231644,
                    "fold_a_fused_primary": 0.6074154376983643,
                    "fold_a_control_primary": 0.6071290373802185,
                    "selected_generated_weight": 0.25,
                    "selected_control_weight": 0.75,
                    "matched_seed_mean_primary_delta": 0.000042249759038289384,
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "do not repeat unchanged 33-feature LambdaRank; test the new recency "
                        "companions or another substantively different representation"
                    ),
                },
            ),
            AggregateRecord(
                "historical_39_feature_campaign_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "representation": "39 dense causal aggregate, recency, and static features",
                    "fold_b_control_primary": 0.5754240304231644,
                    "recency_lambdarank_generated_primary": 0.5682356506586075,
                    "recency_lambdarank_fused_primary": 0.5756224244832993,
                    "pointwise_gbdt_generated_primary": 0.5702996104955673,
                    "pointwise_gbdt_fused_primary": 0.5757940411567688,
                    "pairwise_dense_mlp_selected_primary": 0.5754240304231644,
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "do not repeat listwise trees, pointwise trees, or dense-only MLPs on the "
                        "39-feature representation; use the newly enabled organizer FM categorical "
                        "codes to test ranking-loss alignment on the actual interaction backbone"
                    ),
                },
            ),
            AggregateRecord(
                "historical_44_feature_campaign_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "representation": (
                        "39 dense causal companions plus five prefix-fitted organizer FM codes"
                    ),
                    "fold_b_control_primary": 0.5754240304231644,
                    "pairwise_code_fm_generated_primary": 0.5073919147253036,
                    "pairwise_code_fm_selected_primary": 0.5754240304231644,
                    "categorical_lambdarank_generated_primary": 0.5730865746736526,
                    "categorical_lambdarank_fused_primary": 0.5755051374435425,
                    "categorical_lambdarank_fold_a_fused_primary": 0.6081618070602417,
                    "categorical_lambdarank_fold_a_control_primary": 0.6071290373802185,
                    "deep_categorical_fold_a_fused_primary": 0.6076992154121399,
                    "fieldaware_listwise_fold_a_fused_primary": 0.6077238321304321,
                    "native_categorical_fold_a_fused_primary": 0.6073779463768005,
                    "history_reliability_lambdarank_evaluated": False,
                    "history_reliability_skip_reason": (
                        "the former loss-only novelty key incorrectly blocked this distinct "
                        "representation; the mechanism-axis guard now permits it"
                    ),
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "the exact pairwise-code FM, categorical LambdaRank, and shallow "
                        "DeepFM-style, field-aware listwise, and native-categorical tree "
                        "configurations were non-material; categorical LambdaRank was "
                        "directionally positive on Fold A but nearly neutral on Fold B. Do not "
                        "repeat those exact configurations. Preserve the FM control and "
                        "prioritize the not-yet-evaluated exposure-reliability and "
                        "cross-scope-disagreement strict-past history representation."
                    ),
                },
            ),
            AggregateRecord(
                "historical_56_feature_campaign_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "representation": (
                        "44-feature causal/categorical bundle plus 1-day, 3-day, and 7-day "
                        "strict-past recency horizons"
                    ),
                    "fold_a_control_primary": 0.6071290373802185,
                    "history_reliability_fold_a_fused_primary": 0.6076365411281586,
                    "deepfm_pairwise_fold_a_fused_primary": 0.60727858543396,
                    "multihorizon_pointwise_gbdt_fold_a_fused_primary": 0.6073309183120728,
                    "best_primary_delta": 0.0005075037479400635,
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "multi-horizon long_view recency remained directionally positive but "
                        "non-material. Do not add more recency horizons or repeat those exact "
                        "models; test the newly available, causally isolated click-history "
                        "signal and long_view-versus-click reliability disagreement."
                    ),
                },
            ),
            AggregateRecord(
                "historical_69_feature_campaign_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "representation": (
                        "56-feature causal/recency/categorical bundle plus 13 strict-past "
                        "click-history columns"
                    ),
                    "fold_a_control_primary": 0.6071290373802185,
                    "click_lambdarank_fold_a_fused_primary": 0.607760101556778,
                    "click_lambdarank_primary_delta": 0.0006310641765595,
                    "click_anchored_gbdt_fold_a_fused_primary": 0.6078385412693024,
                    "click_anchored_gbdt_primary_delta": 0.0007095038890839,
                    "click_pairwise_mlp_fold_b_selected_primary": 0.5754240304231644,
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "strict-past click histories were consistently directionally positive "
                        "for two tree mechanisms but non-material, while the pairwise MLP added "
                        "zero selected contribution. Do not repeat those exact models. Test the "
                        "new graded watch-progress histories as a richer causal representation."
                    ),
                },
            ),
            AggregateRecord(
                "historical_82_feature_campaign_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "representation": (
                        "69-feature causal/click/categorical bundle plus 13 strict-past graded "
                        "watch-progress columns"
                    ),
                    "fold_a_control_primary": 0.6071290373802185,
                    "watch_lambdarank_fold_a_fused_primary": 0.6071290373802185,
                    "watch_reliability_candidate_status": (
                        "candidate-local invalid-derived-feature failure; campaign recovered"
                    ),
                    "watch_pairwise_fold_b_selected_primary": 0.5754240304231644,
                    "material_threshold": 0.002,
                    "scientific_conclusion": (
                        "the exact watch-progress LambdaRank and pairwise-bilinear candidates "
                        "added zero selected contribution. Do not repeat them. Use the verified "
                        "five-field pairwise FM reference as the next interaction backbone and "
                        "treat watch progress only as an optional complement."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_10_composition_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "pairwise_fm_composite_fold_b_fused_primary": 0.5755205452442169,
                    "pairwise_fm_composite_fold_b_control_primary": 0.5754240304231644,
                    "pairwise_fm_composite_selected_generated_weight": 0.2,
                    "pairwise_fm_composite_fold_a_fused_primary": 0.6077823042869568,
                    "user_balanced_dart_fold_b_fused_primary": 0.5754691958427429,
                    "user_balanced_dart_fold_a_fused_primary": 0.6071479320526123,
                    "query_balanced_xendcg_fold_b_fused_primary": 0.5754531174898148,
                    "query_balanced_xendcg_fold_a_fused_primary": 0.6071521937847137,
                    "material_threshold": 0.002,
                    "diagnosed_pairwise_reimplementation_drift": (
                        "the generated pairwise helper sampled eligible users uniformly, "
                        "remapped organizer codes locally, and used plain SGD; the verified "
                        "reference uses positive-ticket sampling, unmodified global codes, and "
                        "dense Adam"
                    ),
                    "scientific_conclusion": (
                        "do not repeat these three exact candidates. Their result does not "
                        "falsify the protected exact pairwise-FM primitive. Reuse that immutable "
                        "primitive directly, first as an exact standalone score and then only "
                        "with a genuinely complementary composition whose ablation preserves the "
                        "reference signal."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_11_exact_reference_result",
                {
                    "status": "completed_and_retained_as_specialist_before_this_campaign",
                    "fold_b_generated_only_primary": 0.5741864740848541,
                    "fold_b_selected_primary": 0.5763889104127884,
                    "fold_b_control_primary": 0.5754240304231644,
                    "fold_b_selected_generated_weight": 0.4,
                    "fold_b_selected_control_weight": 0.6,
                    "fold_a_selected_primary": 0.6081787645816803,
                    "fold_a_control_primary": 0.6071290373802185,
                    "best_primary_delta": 0.0010497272014618,
                    "material_threshold": 0.002,
                    "primitive_implementation_exact": True,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "the autonomous campaign has already reproduced the immutable exact "
                        "pairwise-FM specialist. Do not repeat it standalone. Preserve its "
                        "reference_* state and score while testing one genuinely complementary "
                        "listwise, neural, or pointwise composition."
                    ),
                },
            ),
            AggregateRecord(
                "protected_candidate_categorical_ranker_primitive",
                {
                    "status": "available_and_immutable_in_candidate_parent",
                    "protected_path": "reference_categorical_ranker.py",
                    "train_function": (
                        "train_reference_categorical_ranker(features, targets, user_groups, "
                        "*, seed)"
                    ),
                    "predict_function": (
                        "reference_categorical_ranker_scores(features, checkpoint)"
                    ),
                    "diagnostics_function": (
                        "reference_categorical_ranker_diagnostics(checkpoint)"
                    ),
                    "checkpoint_prefixes_csv": "reference_,categorical_rank_",
                    "mechanism": (
                        "exact protected pairwise FM plus native-categorical LightGBM "
                        "LambdaRank over positions 0 through 82"
                    ),
                    "feature_count": 83,
                    "categorical_feature_positions_zero_based_csv": "51,52,53,54,55,82",
                    "historical_metrics_apply_to": "legacy first-82 predecessor only",
                    "current_83_column_fold_evidence": "not yet measured",
                    "tree_count": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "min_data_in_leaf": 200,
                    "lambdarank_truncation_level": 8,
                    "tree_score_scale": 0.2,
                    "fold_b_generated_only_primary": 0.5747565031051636,
                    "fold_b_selected_primary": 0.5769727230072021,
                    "fold_b_control_primary": 0.5754240155220032,
                    "fold_b_selected_generated_weight": 0.45,
                    "fold_b_selected_control_weight": 0.55,
                    "fold_b_primary_delta": 0.0015487074851989,
                    "fold_a_generated_only_primary": 0.607783854007721,
                    "fold_a_frozen_fusion_primary": 0.608400821685791,
                    "fold_a_control_primary": 0.6071290373802185,
                    "fold_a_primary_delta": 0.0012717843055725,
                    "material_threshold": 0.002,
                    "uses_public_labels_for_tuning": False,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "the recorded scores describe the legacy first-82 predecessor, not the "
                        "current video_type-aware expansion. Evaluate the current 83-column arm "
                        "against the byte-preserved legacy or official-FM control on both "
                        "train-derived folds before using it in any deployable portfolio."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_12_composition_result",
                {
                    "status": "completed_and_plateaued_before_this_campaign",
                    "listwise_composition_fold_b_selected_primary": 0.5764372497797012,
                    "listwise_composition_fold_a_frozen_primary": 0.608370840549469,
                    "hardness_pointwise_fold_b_selected_primary": 0.5755130499601364,
                    "hardness_pointwise_fold_a_frozen_primary": 0.607568770647049,
                    "neural_ranknet_execution_failure": (
                        "Adam moment shape (64,1) did not match gradient shape (128,1)"
                    ),
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "do not repeat the exact conditional LambdaRank or hardness-weighted "
                        "pointwise compositions. The neural mechanism was not scientifically "
                        "falsified because its implementation failed before evaluation; any "
                        "neural successor must use shape-derived optimizer state and retain the "
                        "protected categorical ranker."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_14_composition_result",
                {
                    "status": "completed_with_three_valid_nonmaterial_trials",
                    "shape_safe_pairwise_neural_fold_b_selected_primary": 0.5759039670228958,
                    "shape_safe_pairwise_neural_fold_a_frozen_primary": 0.6082998812198639,
                    "user_balanced_listnet_fold_b_selected_primary": 0.5770719051361084,
                    "user_balanced_listnet_fold_b_control_primary": 0.5754240304231644,
                    "user_balanced_listnet_fold_b_primary_delta": 0.001647874712944,
                    "user_balanced_listnet_fold_a_frozen_primary": 0.6086478531360626,
                    "user_balanced_listnet_fold_a_control_primary": 0.6071290373802185,
                    "user_balanced_listnet_fold_a_primary_delta": 0.0015188157558441,
                    "pointwise_offset_fold_b_selected_primary": 0.5773457437753677,
                    "pointwise_offset_fold_b_primary_delta": 0.0019217133522033,
                    "pointwise_offset_fold_a_frozen_primary": 0.6081205904483795,
                    "pointwise_offset_fold_a_primary_delta": 0.000991553068161,
                    "material_threshold": 0.002,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "retain the user-balanced ListNet composition because it was strongest "
                        "across both temporal folds. Do not repeat these three exact candidates. "
                        "Compose a genuinely distinct mechanism around the protected ListNet "
                        "score, prioritizing a gain consistent on both folds rather than the "
                        "pointwise candidate's larger Fold-B-only result."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_15_listnet_residual_plateau",
                {
                    "status": "completed_with_three_valid_nonmaterial_trials",
                    "gated_pairwise_fold_b_primary_delta": 0.0008881092071533,
                    "gated_pairwise_fold_a_primary_delta": 0.0015944838523865,
                    "deep_cross_fold_b_primary_delta": 0.0016001611948013,
                    "deep_cross_fold_a_primary_delta": 0.0016562640666962,
                    "denoising_listmle_fold_b_primary_delta": 0.0016210377216339,
                    "denoising_listmle_fold_a_primary_delta": 0.0015794038772583,
                    "material_threshold": 0.002,
                    "execution_failures": 0,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "three additional neural heads around the protected ListNet backbone "
                        "converged to the same sub-material band. Do not spend another trial on "
                        "a renamed MLP, deeper cross network, denoising head, pairwise residual, "
                        "or listwise residual unless the proposal identifies a genuinely new "
                        "ranking signal and an ablation that isolates it."
                    ),
                },
            ),
            AggregateRecord(
                "train_only_static_metadata_probe_result",
                {
                    "status": "video_type_enabled_after_bounded_portfolio_confirmation",
                    "backbone": (
                        "protected user-balanced ListNet plus one fixed LambdaRank residual"
                    ),
                    "video_type_fold_b_primary_delta": 0.0017284750938416,
                    "video_type_fold_a_frozen_primary_delta": 0.0018436312675476,
                    "upload_type_fold_b_primary_delta": 0.001442015171051,
                    "upload_type_fold_a_frozen_primary_delta": 0.0019646286964417,
                    "music_type_fold_b_primary_delta": 0.0014432668685913,
                    "music_type_fold_a_frozen_primary_delta": 0.0017603039741516,
                    "tag_fold_b_primary_delta": 0.0012959837913513,
                    "tag_fold_a_frozen_primary_delta": 0.0017802119255066,
                    "upload_age_fold_b_primary_delta": 0.0016232132911682,
                    "upload_age_fold_a_frozen_primary_delta": 0.001731276512146,
                    "server_height_fold_b_primary_delta": 0.0015340447425842,
                    "server_height_fold_a_frozen_primary_delta": 0.0018971562385559,
                    "material_threshold": 0.002,
                    "public_validation_used": False,
                    "outer_query_used": False,
                    "production_schema_changed": True,
                    "enabled_feature_name": "video_type_code",
                    "enabled_feature_position_zero_based": 82,
                    "enabled_feature_encoding": (
                        "lexical categories fit on each exact temporal-fold prefix; zero is the "
                        "query unknown slot"
                    ),
                    "other_five_fields_enabled": False,
                    "scientific_conclusion": (
                        "no single field cleared materiality alone, but video_type was the most "
                        "cross-fold-stable field and was the only member that complemented the "
                        "strongest pointwise ranker to the material boundary. Only its encoded "
                        "column is now present; the other side fields, snapshots, and outcome "
                        "statistics remain disabled."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_16_distinct_signal_result",
                {
                    "status": "completed_with_three_valid_nonmaterial_trials",
                    "watch_reliability_fold_b_primary_delta": 0.0016452074050903,
                    "watch_reliability_fold_a_primary_delta": 0.001433253288269,
                    "objective_disagreement_fold_b_primary_delta": 0.0017493963241577,
                    "objective_disagreement_fold_a_primary_delta": 0.0016002058982849,
                    "lightgcn_fold_b_primary_delta": 0.0010381191968918,
                    "lightgcn_fold_a_primary_delta": 0.0018841922283173,
                    "three_way_fold_b_selected_disagreement_weight": 0.45,
                    "three_way_fold_b_selected_lightgcn_weight": 0.0,
                    "three_way_fold_b_selected_control_weight": 0.55,
                    "malformed_implementation_retries": 1,
                    "execution_failures": 0,
                    "material_threshold": 0.002,
                    "public_validation_used": False,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "watch-reliability and protected-specialist disagreement remained in "
                        "the same sub-material band as earlier residual heads. The LightGCN "
                        "branch was weaker on Fold B, and a bounded three-way ensemble assigned "
                        "it zero weight. Do not repeat these exact branches or another generic "
                        "residual around the protected ListNet backbone."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_17_listnet_composition_result",
                {
                    "status": "completed_with_three_valid_nonmaterial_trials",
                    "tab_context_fold_b_primary_delta": 0.0012983679771423,
                    "tab_context_fold_a_primary_delta": 0.0016146004199982,
                    "causal_manifold_fold_b_primary_delta": 0.0015360713005066,
                    "causal_manifold_fold_a_primary_delta": 0.0016940832138062,
                    "temporal_localization_fold_b_primary_delta": 0.001493752002716,
                    "temporal_localization_fold_a_primary_delta": 0.0013605654239655,
                    "execution_failures": 0,
                    "malformed_retries": 0,
                    "material_threshold": 0.002,
                    "public_validation_used": False,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "three more contextual, manifold, and temporally weighted residuals "
                        "around the protected ListNet score remained sub-material. The next "
                        "portfolio must prioritize a standalone independently trained ranker "
                        "rather than another additive or gated ListNet composition."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_18_standalone_result",
                {
                    "status": "completed_with_three_valid_nonmaterial_trials",
                    "query_set_attention_fold_b_primary_delta": 0.0,
                    "query_set_attention_fold_a_primary_delta": 0.0,
                    "pairwise_leaf_region_fold_b_primary_delta": 0.0000923871994019,
                    "pairwise_leaf_region_fold_a_primary_delta": 0.0003617405891418,
                    "dcnv2_fold_b_primary_delta": 0.0001070499420166,
                    "dcnv2_fold_a_primary_delta": 0.0000415444374084,
                    "execution_failures": 0,
                    "malformed_retries": 0,
                    "material_threshold": 0.002,
                    "public_validation_used": False,
                    "outer_query_used": False,
                    "scientific_conclusion": (
                        "the three standalone 82-column mechanisms were substantially weaker "
                        "than retained residual specialists. Do not repeat them now that the "
                        "new video_type_code signal is available at column 82."
                    ),
                },
            ),
            AggregateRecord(
                "train_only_candidate_portfolio_result",
                {
                    "status": "video_type_is_the_only_supported_new_portfolio_member",
                    "candidate_count": 16,
                    "pointwise_video_type_control_weights_csv": "0.15,0.30,0.55",
                    "pointwise_video_type_fold_b_primary_delta": 0.0020057559013367,
                    "pointwise_video_type_fold_a_frozen_primary_delta": 0.0019877552986145,
                    "best_other_pair_fold_b_primary_delta": 0.0018074512481689,
                    "best_other_pair_fold_a_primary_delta": 0.0016923546791077,
                    "six_field_metadata_bundle_fold_b_primary_delta": 0.0015469193458557,
                    "six_field_metadata_bundle_fold_a_primary_delta": 0.0017505288124084,
                    "public_validation_used": False,
                    "outer_query_used": False,
                    "recommended_experiment": (
                        "use video_type_code in one ablated grouped-ranking mechanism that "
                        "preserves the first 82 columns and tests complementarity with a strong "
                        "pointwise or protected-specialist score; do not add the other five "
                        "metadata fields and do not repeat an 82-column standalone model"
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_20_video_type_result",
                {
                    "status": "completed_with_one_valid_trial_and_two_candidate_local_failures",
                    "valid_grouped_calibrator_fold_b_primary_delta": 0.0014815777540207,
                    "valid_grouped_calibrator_fold_a_primary_delta": 0.0012162327766419,
                    "fieldaware_pairwise_failure": (
                        "prediction derived categorical table sizes from training maxima and "
                        "rejected a legal reserved query unknown code"
                    ),
                    "pointwise_reliability_failure": (
                        "helper projected 32 compact columns then reused absolute feature "
                        "positions 32, 35, and 38 against that local matrix"
                    ),
                    "outer_query_used": False,
                    "public_validation_used": False,
                    "scientific_conclusion": (
                        "the grouped categorical-specialist calibration was valid but weaker "
                        "than the prior pointwise-plus-video_type portfolio evidence. The two "
                        "candidate-local failures did not test their scientific mechanisms. "
                        "Repair reserved-unknown handling and projection-local indexing rather "
                        "than treating either mechanism as falsified; prioritize a pointwise "
                        "and video_type composition for the next valid trial."
                    ),
                },
            ),
            AggregateRecord(
                "historical_attempt_21_repaired_video_type_result",
                {
                    "status": "completed_with_three_valid_nonmaterial_trials",
                    "pointwise_reliability_fold_b_primary_delta": 0.0003320574760437,
                    "pointwise_reliability_fold_a_primary_delta": 0.0005849897861481,
                    "fieldaware_pairwise_fold_b_primary_delta": 0.0,
                    "fieldaware_pairwise_selected_generated_weight": 0.0,
                    "lambdarank_video_type_fold_b_primary_delta": 0.0001890808343887,
                    "lambdarank_video_type_fold_a_primary_delta": 0.0005322396755219,
                    "execution_failures": 0,
                    "malformed_retries": 0,
                    "outer_query_used": False,
                    "public_validation_used": False,
                    "scientific_conclusion": (
                        "the two Attempt-20 execution mechanisms were repaired and all three "
                        "trials completed reproducibly, but none approached materiality. The "
                        "reserved-unknown and projection-local fixes are now verified. Do not "
                        "repeat these exact reliability, field-aware pairwise, or standalone "
                        "LambdaRank mechanisms; use the protected Attempt-14 pointwise score "
                        "for the evidence-backed three-way video_type portfolio instead."
                    ),
                },
            ),
            AggregateRecord(
                "protected_candidate_pointwise_ranker_primitive",
                {
                    "status": "available_and_immutable_in_candidate_parent",
                    "protected_path": "reference_pointwise_ranker.py",
                    "train_function": (
                        "train_reference_pointwise_ranker(features, targets, user_groups, *, seed)"
                    ),
                    "predict_function": (
                        "reference_pointwise_ranker_scores(features, checkpoint)"
                    ),
                    "diagnostics_function": (
                        "reference_pointwise_ranker_diagnostics(checkpoint)"
                    ),
                    "checkpoint_prefixes_csv": "reference_,categorical_rank_,pointwise_",
                    "frozen_input_slice": "features[:, :82]",
                    "nested_categorical_video_type_policy": "neutral zero column",
                    "mechanism": (
                        "exact protected categorical ranker plus the frozen query-balanced "
                        "pointwise neural correction retained from Attempt 14"
                    ),
                    "hidden_widths_csv": "256,64",
                    "epochs": 3,
                    "batch_size": 4096,
                    "learning_rate": 0.001,
                    "weight_decay": 0.00001,
                    "score_scale": 0.5,
                    "fold_b_selected_primary": 0.5773457437753677,
                    "fold_b_control_primary": 0.5754240304231644,
                    "fold_b_primary_delta": 0.0019217133522033,
                    "fold_a_frozen_primary": 0.6081205904483795,
                    "fold_a_control_primary": 0.6071290373802185,
                    "fold_a_primary_delta": 0.000991553068161,
                    "reviewed_video_type_portfolio_weights_csv": "0.15,0.30,0.55",
                    "reviewed_video_type_portfolio_fold_b_delta": 0.0020057559013367,
                    "reviewed_video_type_portfolio_fold_a_delta": 0.0019877552986145,
                    "material_threshold": 0.002,
                    "uses_public_labels_for_tuning": False,
                    "outer_query_used": False,
                    "composition_policy": (
                        "generated model_impl.py should import this exact specialist and add one "
                        "isolated video_type_code mechanism; it must retain all protected state "
                        "and must not reimplement or retune the pointwise composition"
                    ),
                    "protected_from_generated_overlay": True,
                    "metric_or_scorer_access": False,
                },
            ),
            AggregateRecord(
                "protected_candidate_listnet_ranker_primitive",
                {
                    "status": "available_and_immutable_in_candidate_parent",
                    "protected_path": "reference_listnet_ranker.py",
                    "train_function": (
                        "train_reference_listnet_ranker(features, targets, user_groups, *, seed)"
                    ),
                    "predict_function": "reference_listnet_ranker_scores(features, checkpoint)",
                    "diagnostics_function": (
                        "reference_listnet_ranker_diagnostics(checkpoint)"
                    ),
                    "checkpoint_prefixes_csv": "reference_,categorical_rank_,listwise_",
                    "frozen_input_slice": "features[:, :82]",
                    "nested_categorical_video_type_policy": "neutral zero column",
                    "mechanism": (
                        "exact protected pairwise FM and native-categorical LambdaRank plus an "
                        "equally user-weighted complete-query ListNet neural composition"
                    ),
                    "hidden_widths_csv": "192,64",
                    "epochs": 8,
                    "learning_rate": 0.001,
                    "weight_decay": 0.00001,
                    "query_batch_size": 128,
                    "score_scale": 0.15,
                    "fold_b_selected_primary": 0.5770719051361084,
                    "fold_b_control_primary": 0.5754240304231644,
                    "fold_b_selected_generated_weight": 0.45,
                    "fold_b_selected_control_weight": 0.55,
                    "fold_b_primary_delta": 0.001647874712944,
                    "fold_a_frozen_primary": 0.6086478531360626,
                    "fold_a_control_primary": 0.6071290373802185,
                    "fold_a_primary_delta": 0.0015188157558441,
                    "material_threshold": 0.002,
                    "uses_public_labels_for_tuning": False,
                    "outer_query_used": False,
                    "composition_policy": (
                        "generated model_impl.py may import this specialist and compose a new "
                        "mechanism around its score; it must retain all reference_*, "
                        "categorical_rank_*, and listwise_* arrays and must not reimplement or "
                        "retune the protected composition"
                    ),
                    "protected_from_generated_overlay": True,
                    "metric_or_scorer_access": False,
                },
            ),
        ),
    )


def _generated_scientific_candidate(
    *,
    request: CampaignCreateRequest,
    data: CampaignDataPlane,
    features: ProductionFeatureBundle,
    lineage: ScriptedLambdaRankLineage | LiveResearchLineage,
    candidate_id: str,
    candidate_limits: LocalCandidateLimits,
    scientific_iteration: int,
) -> ScientificCandidate:
    executable_change = ExecutableChangeEvidence(
        parent_source_digest=lineage.parent.digest,
        candidate_source_digest=lineage.materialized.source_digest,
        executable_diff_digest=lineage.materialized.diff_digest,
        controller_attestation_digest=_digest(
            b"kuairand-production-controller-attestation-v1",
            {
                "campaign_id": request.campaign_id,
                "request_digest": request.digest,
                "scientific_iteration": scientific_iteration,
                "parent_source_digest": lineage.parent.digest,
                "candidate_source_digest": lineage.materialized.source_digest,
                "diff_digest": lineage.materialized.diff_digest,
                "changed_symbols": list(lineage.material_change.changed_symbols),
                "reachable_python_files": list(lineage.material_change.reachable_python_files),
            },
        ),
        changed_symbols=lineage.material_change.changed_symbols,
        reachable_python_files=lineage.material_change.reachable_python_files,
    )
    training_policy_digest = _digest(
        b"kuairand-production-training-policy-v1",
        {
            "feature_bundle_digest": features.digest,
            "config_digest": lineage.config_artifact.sha256,
            "fold_a_digest": data.fold_a.fold.digest,
            "fold_b_digest": data.fold_b.fold.digest,
            "outer_training_inputs_digest": data.outer_train_inputs.digest,
            "outer_training_uses_public_labels": False,
            "fusion_selected_on": "fold_B_only",
            "matched_seeds": [0, 1, 2],
        },
    )
    return ScientificCandidate(
        candidate_id=candidate_id,
        parent_id=lineage.parent.candidate_id,
        family=("lambdarank" if lineage.provider == "scripted" else "openai_generated"),
        source_digest=lineage.materialized.source_digest,
        parent_source_digest=lineage.parent.digest,
        executable_change=executable_change,
        config_digest=lineage.config_artifact.sha256,
        training_policy_digest=training_policy_digest,
        gates=GateEvidence(),
        diversity_root=scientific_iteration == 1,
        metric_specialist_for_blending=True,
        sufficient_finalization_time=True,
        p95_runtime_seconds=float(candidate_limits.timeout_seconds),
        cleanup_seconds=30.0,
    )


@dataclass(slots=True)
class _ScientificRuntime:
    engine: CampaignEngine
    run_dir: Path
    campaign_store: CampaignStore
    artifacts: ArtifactStore
    executor: GeneratedCandidateExecutor
    lineage: ScriptedLambdaRankLineage | LiveResearchLineage
    candidate: ScientificCandidate
    experiment_id: str
    scientific_iteration: int
    config: ScientificCampaignConfig
    feature_artifacts: _FeatureArtifacts
    features: ProductionFeatureBundle
    fold_a: SupervisedFoldFMRun
    fold_b: SupervisedFoldFMRun
    fold_a_query_inputs: CanonicalInputs
    fold_b_query_inputs: CanonicalInputs
    fold_a_scorer: FoldScoringContext
    fold_b_scorer: FoldScoringContext
    outer_scorer: _ProtectedTierScorer
    outer_admission: ReceiptAwareOuterEvaluationLedger | None
    qualification: OfficialFMQualificationEvidence
    repository: FileScientificRunEvidenceRepository
    evidence_registry: dict[tuple[str, int], TrustedOuterSeedEvidence]
    cancel_event: Event | None
    records: dict[tuple[ScientificTier, int], GeneratedScientificRunRecord]

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ScientificCampaignCancelled(
                "trusted controller requested scientific campaign cancellation"
            )

    def _journal_factory(
        self,
        *,
        request: ScientificRunRequest,
        action: CandidateAction,
        execution_id: str,
    ) -> CampaignStoreCandidateJournal:
        del execution_id
        category: LaunchCategory | None = None
        phase = (
            WorkPhase.REQUIRED_CONFIRMATION
            if request.tier is ScientificTier.OUTER_MATCHED_SEED
            else WorkPhase.RESEARCH
        )
        if action is CandidateAction.TRAIN:
            if request.tier is ScientificTier.FOLD_B_SCREEN:
                category = LaunchCategory.DIVERSE_INNER_SCREEN
            elif request.tier is ScientificTier.FOLD_A_CONFIRMATION:
                category = LaunchCategory.TEMPORAL_FOLD_CONFIRMATION
            elif request.seed == 0:
                category = LaunchCategory.DISTINCT_OUTER_PROMOTION
            else:
                category = LaunchCategory.MATCHED_SEED_CONFIRMATION
        return _journal(
            engine=self.engine,
            run_dir=self.run_dir,
            store=self.campaign_store,
            artifacts=self.artifacts,
            family=self.candidate.family,
            phase=phase,
            category=category,
            p95_seconds=self.candidate.p95_runtime_seconds,
            cleanup_seconds=self.candidate.cleanup_seconds,
            experiment_id=self.experiment_id,
            scientific_iteration=self.scientific_iteration,
            cancel_event=self.cancel_event,
        )

    def _frozen_weights(self) -> tuple[float, float]:
        record = self.records.get((ScientificTier.FOLD_B_SCREEN, 0))
        if record is None:
            raise FullCampaignError("Fold B record is absent before fixed fusion reuse")
        weights = record.frozen_fusion_weights
        if weights is None:
            raise FullCampaignError("Fold B record did not freeze a fusion policy")
        return weights

    def _capability(self, request: ScientificRunRequest) -> ScientificTierCapabilities:
        if request.tier is ScientificTier.FOLD_B_SCREEN:
            return ScientificTierCapabilities(
                tier=request.tier,
                fold_id="B",
                scientific_data_digest=request.data_digest,
                training_data_digest=self.features.fold_b.prefix.logical_digest,
                prediction_data_digest=self.features.fold_b.query.logical_digest,
                training_split_token=self.features.fold_b.prefix.logical_digest,
                prediction_split_token=self.features.fold_b.query.logical_digest,
                training_features=self.feature_artifacts.fold_b_train,
                training_targets=self.feature_artifacts.fold_b_targets,
                training_user_groups=self.feature_artifacts.fold_b_groups,
                prediction_features=self.feature_artifacts.fold_b_valid,
                prediction_row_count=self.features.fold_b.query.row_count,
                fusion=FoldBFusionSelector(
                    user_ids=self.fold_b_query_inputs.user_id,
                    video_ids=self.fold_b_query_inputs.video_id,
                    control_scores=self.fold_b.control.predictions.scores,
                ),
            )
        if request.tier is ScientificTier.FOLD_A_CONFIRMATION:
            return ScientificTierCapabilities(
                tier=request.tier,
                fold_id="A",
                scientific_data_digest=request.data_digest,
                training_data_digest=self.features.fold_a.prefix.logical_digest,
                prediction_data_digest=self.features.fold_a.query.logical_digest,
                training_split_token=self.features.fold_a.prefix.logical_digest,
                prediction_split_token=self.features.fold_a.query.logical_digest,
                training_features=self.feature_artifacts.fold_a_train,
                training_targets=self.feature_artifacts.fold_a_targets,
                training_user_groups=self.feature_artifacts.fold_a_groups,
                prediction_features=self.feature_artifacts.fold_a_valid,
                prediction_row_count=self.features.fold_a.query.row_count,
                fusion=TrustedFMRankFusion(
                    phase=DataPhase.INNER_VALID,
                    weights=self._frozen_weights(),
                    user_ids=self.fold_a_query_inputs.user_id,
                    video_ids=self.fold_a_query_inputs.video_id,
                    control_scores=self.fold_a.control.predictions.scores,
                ),
            )
        qualified = self.qualification.outer_seed(request.seed)
        control = load_predictions(
            qualified.validation_predictions_path,
            expected_file_sha256=qualified.validation_predictions_file_sha256,
            expected_prediction_digest=qualified.validation_prediction_digest,
            expected_row_count=self.qualification.validation_row_count,
        )
        return ScientificTierCapabilities(
            tier=request.tier,
            fold_id=None,
            scientific_data_digest=request.data_digest,
            training_data_digest=self.features.outer_and_final.prefix.logical_digest,
            prediction_data_digest=self.features.outer_validation.logical_digest,
            training_split_token=self.features.outer_and_final.prefix.logical_digest,
            prediction_split_token=self.features.outer_validation.logical_digest,
            training_features=self.feature_artifacts.outer_train,
            training_targets=self.feature_artifacts.outer_targets,
            training_user_groups=self.feature_artifacts.outer_groups,
            prediction_features=self.feature_artifacts.outer_valid,
            prediction_row_count=self.features.outer_validation.row_count,
            fusion=TrustedFMRankFusion(
                phase=DataPhase.OUTER_VALID,
                weights=self._frozen_weights(),
                user_ids=self.outer_scorer.alignment.user_ids,
                video_ids=self.outer_scorer.alignment.video_ids,
                control_scores=control.scores,
            ),
        )

    def _scoring(self, request: ScientificRunRequest) -> ProtectedScoringCapability:
        callback: ProtectedScoreCallback
        if request.tier is ScientificTier.FOLD_B_SCREEN:
            context = self.fold_b_scorer
            callback = cast(ProtectedScoreCallback, context.score_with_encoded_labels)
            alignment_digest = context.query_alignment_digest
            rows = context.row_count
        elif request.tier is ScientificTier.FOLD_A_CONFIRMATION:
            context = self.fold_a_scorer
            callback = cast(ProtectedScoreCallback, context.score_with_encoded_labels)
            alignment_digest = context.query_alignment_digest
            rows = context.row_count
        else:
            if self.outer_admission is None:
                raise FullCampaignError("outer scoring requires lazy receipt-aware admission")
            promotion = self._promotion_request()
            admission = self.outer_admission
            protected = self.outer_scorer.score
            users = len(set(self.outer_scorer.alignment.user_ids))

            def score_outer(scores: npt.NDArray[np.float64]) -> ScoreResult:
                return admission.score(
                    request=promotion,
                    seed=request.seed,
                    scores=scores,
                    users=users,
                    rows=len(self.outer_scorer.labels),
                    protected_callback=protected,
                )

            callback = cast(ProtectedScoreCallback, score_outer)
            alignment_digest = self.outer_scorer.alignment.digest
            rows = len(self.outer_scorer.labels)
        return ProtectedScoringCapability(
            tier=request.tier,
            fold_id=request.fold_id,
            scientific_data_digest=request.data_digest,
            scorer_digest=request.scorer_digest,
            alignment_digest=alignment_digest,
            row_count=rows,
            callback=callback,
        )

    def _promotion_request(self) -> OuterPromotionRequest:
        return OuterPromotionRequest(
            campaign_digest=self.config.campaign_digest,
            candidate_id=self.candidate.candidate_id,
            candidate_fingerprint=self.candidate.fingerprint,
            source_digest=self.candidate.source_digest,
            parent_source_digest=self.candidate.parent_source_digest,
            executable_diff_digest=self.candidate.executable_change.executable_diff_digest,
            material_change_digest=self.candidate.executable_change.digest,
            controller_attestation_digest=(
                self.candidate.executable_change.controller_attestation_digest
            ),
            benchmark_digest=self.config.benchmark_digest,
            dataset_digest=self.config.dataset_digest,
            scorer_digest=self.config.scorer_digest,
            training_policy_digest=self.candidate.training_policy_digest,
        )

    def __call__(self, request: ScientificRunRequest) -> ScientificRunEvidence:
        self._raise_if_cancelled()
        capability = self._capability(request)
        scoring = self._scoring(request)
        runner = DurableGeneratedScientificRunner(
            executor=self.executor,
            artifact_store=self.artifacts,
            identity=self.lineage.identity,
            capabilities={request.tier: capability},
            scoring_callbacks={request.tier: scoring},
            journal_factory=self._journal_factory,
            evidence_repository=self.repository,
            cancel_event=self.cancel_event,
        )
        try:
            evidence = runner(request)
        except GeneratedScientificRunnerCancelledError as exc:
            raise ScientificCampaignCancelled(
                "generated scientific execution was cancelled by the trusted controller"
            ) from exc
        except Exception as exc:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise ScientificCampaignCancelled(
                    "generated scientific execution stopped after controller cancellation"
                ) from exc
            raise
        self._raise_if_cancelled()
        record = runner.load_record(request.digest)
        if record is None:
            raise FullCampaignError("generated scientific record was not durably committed")
        self.records[(request.tier, request.seed)] = record
        if request.tier is ScientificTier.OUTER_MATCHED_SEED:
            promotion = self._promotion_request()
            if evidence.metrics is None:
                raise FullCampaignError("outer generated run lacks protected aggregate metrics")
            receipt = (
                None
                if self.outer_admission is None
                else self.outer_admission.receipt_for(promotion.digest, request.seed)
            )
            trusted = TrustedOuterSeedEvidence(
                request_digest=promotion.digest,
                seed=request.seed,
                metrics=evidence.metrics,
                prediction_digest=record.scored_prediction_digest,
                score_evidence_digest=(evidence.digest if receipt is None else receipt.digest),
            )
            key = (promotion.digest, request.seed)
            prior = self.evidence_registry.get(key)
            if prior is not None and prior != trusted:
                raise FullCampaignError("outer trusted evidence retry is contradictory")
            self.evidence_registry[key] = trusted
        return evidence


def _candidate_artifact_clears_deployment_gate(
    *,
    result: ScientificCampaignResult,
    candidate_result: CandidateCampaignResult,
    representative_record: GeneratedScientificRunRecord,
    qualification: OfficialFMQualificationEvidence,
) -> bool:
    """Require the exact seed-0 artifact to materially beat immutable seed 4.

    Scientific selection remains a matched-seed mean decision.  Finalization is deliberately
    more conservative because it deploys one concrete representative artifact, not that mean.
    Any disagreement among the scientific result, its retained run evidence, and the immutable
    fallback identity therefore rejects deployment without changing the research incumbent.
    """

    candidate_id = candidate_result.candidate.candidate_id
    selection = candidate_result.selection
    fallback = qualification.fallback
    if (
        candidate_result.outcome is not CandidateOutcome.PROMOTED_CONFIRMED
        or selection is None
        or selection.selected_candidate_id != candidate_id
        or selection.challenger_candidate_id != candidate_id
        or result.incumbent.candidate_id != candidate_id
        or result.incumbent.official_fm
        or not result.incumbent.replayable
        or not result.incumbent.eligible
        or result.fallback.candidate_id != SCRIPTED_PARENT_ID
        or not result.fallback.official_fm
        or not result.fallback.replayable
        or not result.fallback.eligible
        or type(fallback.seed) is not int
        or fallback.seed != 4
    ):
        return False

    representative_metrics = representative_record.evidence.metrics
    if not isinstance(representative_metrics, OrganizerMetrics):
        return False
    incumbent_seed_zero = tuple(
        item for item in result.incumbent.outer_by_seed if item.seed == 0
    )
    if (
        len(incumbent_seed_zero) != 1
        or incumbent_seed_zero[0].metrics != representative_metrics
        or sum(run == representative_record.evidence for run in candidate_result.runs) != 1
    ):
        return False
    try:
        fallback_metrics = _metrics(fallback.metrics)
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        representative_metrics.primary_decimal - fallback_metrics.primary_decimal
        > MATERIAL_PRIMARY_DELTA
    )


def _candidate_selection(
    *,
    experiment_id: str,
    result: ScientificCampaignResult,
    runtime: _ScientificRuntime,
    qualification: OfficialFMQualificationEvidence,
    features: ProductionFeatureBundle,
    feature_artifacts: _FeatureArtifacts,
    dataset_digest: str,
    validation_inputs: CanonicalInputs,
    final_inputs: CanonicalInputs,
    limits: LocalCandidateLimits,
) -> FinalizationSelectionPlan | None:
    if result.incumbent.candidate_id != runtime.candidate.candidate_id:
        return None
    candidate_result = _candidate_result_for(
        result,
        candidate_id=runtime.candidate.candidate_id,
    )
    if candidate_result is None or candidate_result.selection is None:
        return None
    records = runtime.records
    representative_record = records.get((ScientificTier.OUTER_MATCHED_SEED, 0))
    if representative_record is None or not _candidate_artifact_clears_deployment_gate(
        result=result,
        candidate_result=candidate_result,
        representative_record=representative_record,
        qualification=qualification,
    ):
        return None
    fold_a_record = records.get((ScientificTier.FOLD_A_CONFIRMATION, 0))
    fold_b_record = records.get((ScientificTier.FOLD_B_SCREEN, 0))
    if fold_a_record is None or fold_b_record is None:
        raise FullCampaignError("generated incumbent lacks both train-derived fold records")
    weights = fold_b_record.frozen_fusion_weights
    if weights is None:
        raise FullCampaignError("generated incumbent lacks a frozen Fold B fusion policy")
    fallback_folds = dict(result.fallback.inner_by_fold)
    incumbent_folds = dict(result.incumbent.inner_by_fold)

    def disk_bytes(record: GeneratedScientificRunRecord) -> int:
        refs = (
            record.train_artifacts.entries
            + record.prediction_artifacts.entries
            + record.replay_artifacts.entries
        )
        return sum(reference.size_bytes for _, reference in refs)

    def parent_disk(control: SupervisedFoldFMRun) -> int:
        return sum(
            reference.size_bytes for _, reference in control.evidence.journal_artifacts.entries
        )

    inner = (
        InnerFoldSelectionEvidence(
            fold_id="A",
            candidate=incumbent_folds["A"],
            parent=fallback_folds["A"],
            reference=fallback_folds["A"],
            candidate_wall_seconds=fold_a_record.evidence.resources.wall_seconds,
            candidate_peak_rss_bytes=fold_a_record.evidence.resources.peak_rss_bytes,
            candidate_disk_bytes=disk_bytes(fold_a_record),
            parent_wall_seconds=runtime.fold_a.control.resources.wall_seconds,
            parent_peak_rss_bytes=(runtime.fold_a.control.resources.max_observed_rss_bytes),
            parent_disk_bytes=parent_disk(runtime.fold_a),
        ),
        InnerFoldSelectionEvidence(
            fold_id="B",
            candidate=incumbent_folds["B"],
            parent=fallback_folds["B"],
            reference=fallback_folds["B"],
            candidate_wall_seconds=fold_b_record.evidence.resources.wall_seconds,
            candidate_peak_rss_bytes=fold_b_record.evidence.resources.peak_rss_bytes,
            candidate_disk_bytes=disk_bytes(fold_b_record),
            parent_wall_seconds=runtime.fold_b.control.resources.wall_seconds,
            parent_peak_rss_bytes=(runtime.fold_b.control.resources.max_observed_rss_bytes),
            parent_disk_bytes=parent_disk(runtime.fold_b),
        ),
    )
    matched: list[MatchedSeedSelectionEvidence] = []
    for seed in (0, 1, 2):
        record = records.get((ScientificTier.OUTER_MATCHED_SEED, seed))
        if record is None or record.evidence.metrics is None:
            raise FullCampaignError(f"generated incumbent lacks matched seed {seed} evidence")
        qualified = qualification.outer_seed(seed)
        fm_prediction = runtime.artifacts.put_file(
            qualified.validation_predictions_path,
            kind=ArtifactKind.PREDICTION,
        )
        matched.append(
            MatchedSeedSelectionEvidence(
                seed=seed,
                scientific_request_digest=record.request_digest,
                scientific_record_digest=record.digest,
                checkpoint=record.checkpoint,
                candidate_validation_prediction=record.scored_prediction,
                fm_validation_prediction=fm_prediction,
                candidate_metrics=record.evidence.metrics,
                fm_metrics=_metrics(qualified.metrics),
                candidate_wall_seconds=record.evidence.resources.wall_seconds,
                candidate_peak_rss_bytes=record.evidence.resources.peak_rss_bytes,
                candidate_disk_bytes=disk_bytes(record),
                fm_wall_seconds=qualified.resources.wall_seconds,
                fm_peak_rss_bytes=qualified.resources.peak_rss_bytes,
                fm_disk_bytes=qualified.resources.disk_bytes,
                fm_member=_qualified_member(qualification, qualified),
            )
        )
    representative = matched[0]
    validation_capability = build_finalization_candidate_inputs(
        DataPhase.OUTER_VALID,
        validation_inputs,
    )
    final_capability = build_finalization_candidate_inputs(
        DataPhase.FINAL,
        final_inputs,
    )
    return FinalizationSelectionPlan(
        experiment_id=experiment_id,
        candidate_id=runtime.candidate.candidate_id,
        candidate_fingerprint=runtime.candidate.fingerprint,
        source_digest=runtime.candidate.source_digest,
        parent_source_digest=runtime.candidate.parent_source_digest,
        executable_change_digest=runtime.candidate.executable_change.digest,
        config_digest=runtime.candidate.config_digest,
        training_policy_digest=runtime.candidate.training_policy_digest,
        evidence_receipt_digest=result.incumbent.evidence_receipt_digest,
        source_snapshot=runtime.lineage.source_snapshot,
        training_features=feature_artifacts.outer_train,
        training_targets=feature_artifacts.outer_targets,
        training_user_groups=feature_artifacts.outer_groups,
        validation_features=feature_artifacts.outer_valid,
        final_features=feature_artifacts.final,
        feature_bundle_digest=features.digest,
        feature_count=features.final.feature_count,
        dataset_digest=dataset_digest,
        validation_inputs_digest=validation_capability.digest,
        final_inputs_digest=final_capability.digest,
        frozen_fusion_weights=weights,
        representative_seed=0,
        selected_outer_request_digest=representative.scientific_request_digest,
        scientific_record_digest=representative.scientific_record_digest,
        tree_checkpoint=representative.checkpoint,
        validation_prediction=representative.candidate_validation_prediction,
        fm_member=representative.fm_member,
        inner_folds=inner,
        matched_seeds=tuple(matched),
        timeout_seconds=int(limits.timeout_seconds),
        memory_limit_bytes=limits.memory_limit_bytes,
        threads=limits.threads,
    )


def _ensure_lineage_ledger(
    *,
    campaign_store: CampaignStore,
    lineage: ScriptedLambdaRankLineage | LiveResearchLineage,
    candidate_id: str,
    scientific_iteration: int,
) -> tuple[CampaignStoreResearchLedger, str]:
    ledger = CampaignStoreResearchLedger(
        campaign_store,
        provider_name=lineage.provider,
        fallback_candidate_id=SCRIPTED_PARENT_ID,
    )
    experiment_id = f"iteration-{scientific_iteration:02d}"
    snapshot_id = f"{candidate_id}-source"
    experiment = campaign_store.experiment(experiment_id)
    if experiment is None:
        ledger.create_iteration(experiment_id, scientific_iteration, lineage.proposal)
        experiment = campaign_store.experiment(experiment_id)
    if experiment is None:
        raise FullCampaignError("research experiment was not durably created")
    if campaign_store.proposal(lineage.proposal.proposal_id) is None:
        ledger.record_proposal(
            experiment_id,
            lineage.proposal_request.digest,
            lineage.proposal,
            _artifact_spec(lineage.transcript_artifact),
        )
    experiment = campaign_store.experiment(experiment_id)
    if experiment is None:
        raise FullCampaignError("research experiment disappeared")
    if experiment.status == "PLANNED":
        ledger.transition(
            experiment_id,
            "PLANNED",
            "PROPOSED",
            operation=ResearchOperation.PROPOSE.value,
            reason="persist scripted typed proposal and complete transcript",
        )
    if campaign_store.source_snapshot(snapshot_id) is None:
        ledger.record_source(
            snapshot_id=snapshot_id,
            experiment_id=experiment_id,
            candidate=lineage.materialized,
            parent_digest=lineage.parent.digest,
            source_manifest=_artifact_spec(lineage.source_snapshot.manifest_artifact),
            diff=_artifact_spec(lineage.diff_artifact),
            transcript_role="implementation_transcript",
            transcript=_artifact_spec(lineage.transcript_artifact),
        )
    experiment = campaign_store.experiment(experiment_id)
    if experiment is None:
        raise FullCampaignError("research experiment disappeared after source persistence")
    if experiment.status == "PROPOSED":
        ledger.transition(
            experiment_id,
            "PROPOSED",
            "MATERIALIZED",
            operation=ResearchOperation.IMPLEMENT.value,
            reason="persist material generated source and executable-change evidence",
        )
    experiment = campaign_store.experiment(experiment_id)
    if experiment is not None and experiment.status == "MATERIALIZED":
        ledger.transition(
            experiment_id,
            "MATERIALIZED",
            "RUNNING",
            operation="run",
            reason="start protected scientific evaluation of generated source",
        )
    experiment = campaign_store.experiment(experiment_id)
    if experiment is None:
        raise FullCampaignError("research experiment disappeared after lineage admission")
    return ledger, experiment.experiment_id


def _reflect(
    *,
    campaign_store: CampaignStore,
    research_ledger: CampaignStoreResearchLedger,
    lineage: ScriptedLambdaRankLineage | LiveResearchLineage,
    candidate_id: str,
    scientific_iteration: int,
    result: ScientificCampaignResult,
    safe_context: SafeResearchContext,
    research_config: ResearchConfig,
    artifacts: ArtifactStore,
    research_model: ResearchModel | None = None,
) -> tuple[str, str, ArtifactRef, Reflection]:
    summary = _scientific_reflection_summary(result, candidate_id=candidate_id)
    request = ReflectionRequest.create(
        request_id=f"iteration-{scientific_iteration:02d}-reflect",
        proposal_id=lineage.proposal.proposal_id,
        candidate_id=candidate_id,
        source_digest=lineage.materialized.source_digest,
        diff_digest=lineage.materialized.diff_digest,
        result=summary,
        safe_context=safe_context.to_wire(),
    )
    completed_scientific_evaluation = summary.status in {"promoted", "rejected"}
    scripted = Reflection(
        response_id="generated-causal-lambdarank-v1-reflection",
        summary=(
            "The bounded generated branch completed scientific evaluation; the typed result "
            "contains only metrics actually produced by that branch."
            if completed_scientific_evaluation
            else "The bounded generated branch did not complete a valid scientific evaluation; "
            "no incumbent metrics were substituted for the missing challenger result."
        ),
        recommendation="close_branch",
        lessons=(
            "Use only train-derived folds for fusion selection and keep matched-seed evidence.",
            "Retain the immutable FM fallback until clean final replay succeeds.",
        ),
    )
    if research_model is None:
        selected = select_research_provider(
            research_config,
            scripted_responses=(ScriptedResponse(ResearchOperation.REFLECT, scripted),),
        )
        if isinstance(selected, ProviderUnavailableDiagnostic):
            raise FullCampaignError(
                f"research provider unavailable during reflection: {selected.code.value}"
            )
        if not isinstance(selected, AvailableResearchProvider):
            raise FullCampaignError("research provider factory returned an unsupported result")
        reflection_model = selected.model
        reflection_provider = selected.provider
        reflection_live = selected.live_provider_used
    else:
        reflection_model = research_model
        reflection_provider = lineage.provider
        reflection_live = lineage.live_provider_used
    reflection = reflection_model.reflect(request)
    transcript = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                "operation": ResearchOperation.REFLECT.value,
                "provider": reflection_provider,
                "live_provider_used": reflection_live,
                "request": request.to_wire(),
                "response": reflection.to_wire(),
                "manual_source_edits": 0,
                "manual_interventions": 0,
            }
        ),
        kind=ArtifactKind.LOG,
    )
    experiment_id = f"iteration-{scientific_iteration:02d}"
    experiment = campaign_store.experiment(experiment_id)
    if experiment is None:
        raise FullCampaignError("research experiment is absent during reflection")
    if experiment.status == "RUNNING":
        research_ledger.transition(
            experiment_id,
            "RUNNING",
            "EVALUATED",
            operation="evaluate",
            reason="persist complete protected scientific campaign result",
            metadata={"scientific_result_digest": result.digest},
        )
    experiment = campaign_store.experiment(experiment_id)
    if experiment is not None and experiment.status == "EVALUATED":
        research_ledger.transition(
            experiment_id,
            "EVALUATED",
            "REFLECTED",
            operation=ResearchOperation.REFLECT.value,
            reason="persist exact typed reflection transcript",
            artifacts=(("reflection_transcript", _artifact_spec(transcript)),),
            metadata={"reflection_digest": reflection.digest},
        )
    experiment = campaign_store.experiment(experiment_id)
    if experiment is not None and experiment.status == "REFLECTED":
        research_ledger.transition(
            experiment_id,
            "REFLECTED",
            "CLOSED",
            operation="selection",
            reason="complete the bounded scientific iteration after reflection",
            metadata={"scientific_iteration": scientific_iteration},
        )
    return request.digest, reflection.digest, transcript, reflection


def _candidate_result_for(
    result: ScientificCampaignResult,
    *,
    candidate_id: str,
) -> CandidateCampaignResult | None:
    matches = tuple(
        item for item in result.candidates if item.candidate.candidate_id == candidate_id
    )
    return matches[0] if len(matches) == 1 else None


def _run_is_valid_scientific_evidence(run: ScientificRunEvidence) -> bool:
    return (
        run.metrics is not None
        and run.gates.failures == ()
        and run.replay_verified
    )


def _is_completed_scientific_rejection(candidate_result: CandidateCampaignResult) -> bool:
    if candidate_result.outcome not in {
        CandidateOutcome.SCREEN_REJECTED,
        CandidateOutcome.INNER_REJECTED,
        CandidateOutcome.RETAINED,
    }:
        return False
    return (
        candidate_result.candidate.gates.failures == ()
        and bool(candidate_result.runs)
        and all(_run_is_valid_scientific_evidence(run) for run in candidate_result.runs)
    )


def _scientific_reflection_summary(
    result: ScientificCampaignResult,
    *,
    candidate_id: str,
) -> ExperimentResultSummary:
    """Return only challenger evidence that the scientific loop actually produced."""

    candidate_result = _candidate_result_for(result, candidate_id=candidate_id)
    resource_run = (
        None
        if candidate_result is None or not candidate_result.runs
        else candidate_result.runs[-1]
    )
    runtime_seconds = None if resource_run is None else resource_run.resources.wall_seconds
    peak_memory_mb = (
        None
        if resource_run is None
        else resource_run.resources.peak_rss_bytes / float(1024**2)
    )
    outcome = None if candidate_result is None else candidate_result.outcome
    outer_outcomes = {
        CandidateOutcome.OUTER_FAILED,
        CandidateOutcome.RETAINED,
        CandidateOutcome.PROMOTED_CONFIRMED,
        CandidateOutcome.PROMOTED_UNCONFIRMED,
    }
    tier = "outer" if outcome in outer_outcomes else "inner"
    if outcome is CandidateOutcome.BUDGET_REJECTED:
        return ExperimentResultSummary(
            tier=tier,
            status="budget_blocked",
            gauc=None,
            ndcg_at_5=None,
            primary=None,
            runtime_seconds=runtime_seconds,
            peak_memory_mb=peak_memory_mb,
        )

    promotion_outcomes = {
        CandidateOutcome.PROMOTED_CONFIRMED,
        CandidateOutcome.PROMOTED_UNCONFIRMED,
    }
    promoted_by_outcome = outcome in promotion_outcomes
    promoted_by_incumbent = result.incumbent.candidate_id == candidate_id
    valid_runs = (
        candidate_result is not None
        and candidate_result.candidate.gates.failures == ()
        and bool(candidate_result.runs)
        and all(_run_is_valid_scientific_evidence(run) for run in candidate_result.runs)
    )
    completed_outcomes = {
        CandidateOutcome.SCREEN_REJECTED,
        CandidateOutcome.INNER_REJECTED,
        CandidateOutcome.RETAINED,
        *promotion_outcomes,
    }
    consistent_promotion = promoted_by_outcome == promoted_by_incumbent
    if outcome not in completed_outcomes or not valid_runs or not consistent_promotion:
        return ExperimentResultSummary(
            tier=tier,
            status="execution_failed",
            gauc=None,
            ndcg_at_5=None,
            primary=None,
            runtime_seconds=runtime_seconds,
            peak_memory_mb=peak_memory_mb,
        )

    assert resource_run is not None
    assert resource_run.metrics is not None
    metrics = resource_run.metrics
    return ExperimentResultSummary(
        tier=tier,
        status="promoted" if promoted_by_outcome else "rejected",
        gauc=metrics.gauc,
        ndcg_at_5=metrics.ndcg_at_5,
        primary=metrics.primary,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=peak_memory_mb,
    )


@dataclass(frozen=True, slots=True)
class _AutonomousFollowupResult:
    result: ScientificCampaignResult
    selection: FinalizationSelectionPlan | None
    reflection_request_digest: str
    reflection_response_digest: str
    reflection_transcript: ArtifactRef
    iterations_completed: int
    rejected_records: tuple[AggregateRecord, ...]


def _iteration_record(
    result: ScientificCampaignResult,
    reflection: Reflection,
    *,
    scientific_iteration: int,
    lineage: ScriptedLambdaRankLineage | LiveResearchLineage,
    run_records: Mapping[tuple[ScientificTier, int], GeneratedScientificRunRecord],
) -> AggregateRecord:
    candidate_result = result.candidates[-1] if result.candidates else None
    fold_b = run_records.get((ScientificTier.FOLD_B_SCREEN, 0))
    fold_a = run_records.get((ScientificTier.FOLD_A_CONFIRMATION, 0))
    fold_b_generated = None
    fold_b_selected_weights = None
    if fold_b is not None and fold_b.fusion_selection is not None:
        fold_b_selected_weights = fold_b.fusion_selection.selected_weights
        fold_b_generated = next(
            (
                point.metrics
                for point in fold_b.fusion_selection.points
                if point.weights == (1.0, 0.0)
            ),
            None,
        )
    promoted = result.incumbent.candidate_id == (
        None if candidate_result is None else candidate_result.candidate.candidate_id
    )
    proposal = getattr(lineage, "proposal", None)
    if isinstance(proposal, Proposal):
        family = classify_proposal_family(proposal)
        proposal_id: str | None = proposal.proposal_id
        proposal_objective: str | None = proposal.objective
        proposal_principal_change: str | None = proposal.principal_change
        proposal_mechanism: str | None = proposal.mechanism
    else:
        # Compatibility with provider-free typed test seams. Production lineages always carry
        # the admitted Proposal, so an absent proposal must never create a blocking family.
        family = "unknown"
        proposal_id = None
        proposal_objective = None
        proposal_principal_change = None
        proposal_mechanism = None
    completed_scientific_rejection = (
        candidate_result is not None
        and _is_completed_scientific_rejection(candidate_result)
        and not promoted
    )
    return AggregateRecord(
        name=f"scientific_iteration_{scientific_iteration:02d}",
        values={
            "scientific_iteration": scientific_iteration,
            "candidate_id": (
                None if candidate_result is None else candidate_result.candidate.candidate_id
            ),
            "candidate_outcome": (
                None if candidate_result is None else candidate_result.outcome.value
            ),
            "candidate_reason": None if candidate_result is None else candidate_result.reason,
            "proposal_id": proposal_id,
            "proposal_family": family,
            "proposal_family_blocked": (
                proposal is not None and completed_scientific_rejection
            ),
            "proposal_objective": proposal_objective,
            "proposal_principal_change": proposal_principal_change,
            "proposal_mechanism": proposal_mechanism,
            "candidate_promoted": promoted,
            "fold_b_generated_only_gauc": (
                None if fold_b_generated is None else fold_b_generated.gauc
            ),
            "fold_b_generated_only_ndcg_at_5": (
                None if fold_b_generated is None else fold_b_generated.ndcg_at_5
            ),
            "fold_b_generated_only_primary": (
                None if fold_b_generated is None else fold_b_generated.primary
            ),
            "fold_b_selected_generated_weight": (
                None if fold_b_selected_weights is None else fold_b_selected_weights[0]
            ),
            "fold_b_selected_control_weight": (
                None if fold_b_selected_weights is None else fold_b_selected_weights[1]
            ),
            "fold_b_selected_primary": (
                None
                if fold_b is None or fold_b.evidence.metrics is None
                else fold_b.evidence.metrics.primary
            ),
            "fold_a_selected_primary": (
                None
                if fold_a is None or fold_a.evidence.metrics is None
                else fold_a.evidence.metrics.primary
            ),
            "incumbent_candidate_id": result.incumbent.candidate_id,
            "launches_used": result.launches_used,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "stop_reason": result.stop_reason.value,
            "reflection_recommendation": reflection.recommendation,
            "reflection_summary": reflection.summary,
        },
    )


def _rejected_lineage_record(
    rejection: LiveResearchBranchRejected,
    *,
    scientific_iteration: int,
    root_failure_total_count: int = 1,
    root_failure_consecutive_count: int = 1,
    proposal_family_attempt_count: int = 1,
    proposal_family_blocked: bool = False,
) -> AggregateRecord:
    root = rejection.root_failure
    terminal = rejection.terminal_failure
    return AggregateRecord(
        name=f"rejected_lineage_{scientific_iteration:02d}",
        values={
            "scientific_iteration": scientific_iteration,
            "candidate_id": rejection.failed_candidate_id,
            "branch_outcome": "rejected_before_execution",
            "repairs_attempted": rejection.repairs_attempted,
            "diagnostic": rejection.diagnostic[:4096],
            "root_failure_stage": root.stage,
            "root_failure_category": root.category,
            "root_failure_code": root.code,
            "root_failure_subject": root.subject,
            "root_failure_fingerprint": root.fingerprint,
            "root_failure_diagnostic": root.diagnostic,
            "root_failure_total_count": root_failure_total_count,
            "root_failure_consecutive_count": root_failure_consecutive_count,
            "terminal_failure_stage": terminal.stage,
            "terminal_failure_category": terminal.category,
            "terminal_failure_code": terminal.code,
            "terminal_failure_subject": terminal.subject,
            "terminal_failure_fingerprint": terminal.fingerprint,
            "terminal_failure_diagnostic": terminal.diagnostic,
            "proposal_family": rejection.proposal_family,
            "proposal_signature": rejection.proposal_signature or None,
            "proposal_family_attempt_count": proposal_family_attempt_count,
            "proposal_family_blocked": proposal_family_blocked,
        },
    )


def _research_stage_counts(
    *,
    model: ResearchModel | None,
    branches_attempted: int,
    rejected_records: Sequence[AggregateRecord],
    candidates_admitted: int,
    training_started: int = 0,
    inner_evaluations_completed: int = 0,
    outer_evaluations_completed: int = 0,
) -> dict[str, int]:
    transcripts = (
        model.transcripts
        if isinstance(model, (OpenAIChatCompletionsModel, OpenAIFailoverModel))
        else ()
    )

    def accepted(operation: ResearchOperation) -> int:
        return sum(
            item.operation is operation and item.outcome == "accepted" for item in transcripts
        )

    return {
        "branches_attempted": branches_attempted,
        "proposal_responses_accepted": accepted(ResearchOperation.PROPOSE),
        "implementation_responses_accepted": accepted(ResearchOperation.IMPLEMENT),
        "repair_responses_accepted": accepted(ResearchOperation.REPAIR),
        "branches_rejected_pre_execution": len(rejected_records),
        "candidates_admitted": candidates_admitted,
        "training_started": training_started,
        "inner_evaluations_completed": inner_evaluations_completed,
        "outer_evaluations_completed": outer_evaluations_completed,
    }


def _failure_count_rows(
    records: Sequence[AggregateRecord],
    *,
    role: Literal["root", "terminal"],
) -> tuple[list[dict[str, object]], bool]:
    observed: dict[str, dict[str, object]] = {}
    prefix = f"{role}_failure_"
    for record in records:
        fingerprint = record.values.get(f"{prefix}fingerprint")
        if type(fingerprint) is not str:
            continue
        row = observed.setdefault(
            fingerprint,
            {
                "fingerprint": fingerprint,
                "stage": record.values.get(f"{prefix}stage"),
                "category": record.values.get(f"{prefix}category"),
                "code": record.values.get(f"{prefix}code"),
                "subject": record.values.get(f"{prefix}subject"),
                "count": 0,
            },
        )
        row["count"] = cast(int, row["count"]) + 1
    ordered = sorted(
        observed.values(),
        key=lambda item: (-cast(int, item["count"]), cast(str, item["fingerprint"])),
    )
    return ordered[:8], len(ordered) > 8


def _research_rejection_summary(
    records: Sequence[AggregateRecord],
) -> dict[str, object]:
    root_counts, root_truncated = _failure_count_rows(records, role="root")
    terminal_counts, terminal_truncated = _failure_count_rows(records, role="terminal")
    examples: list[dict[str, object]] = []
    for record in reversed(records):
        for role in ("terminal", "root"):
            prefix = f"{role}_failure_"
            fingerprint = record.values.get(f"{prefix}fingerprint")
            diagnostic = record.values.get(f"{prefix}diagnostic")
            if type(fingerprint) is not str or type(diagnostic) is not str:
                continue
            examples.append(
                {
                    "scientific_iteration": record.values.get("scientific_iteration"),
                    "candidate_id": record.values.get("candidate_id"),
                    "proposal_family": record.values.get("proposal_family"),
                    "proposal_signature": record.values.get("proposal_signature"),
                    "role": role,
                    "fingerprint": fingerprint,
                    "diagnostic": diagnostic,
                }
            )
            if len(examples) == 6:
                break
        if len(examples) == 6:
            break
    return {
        "branches_rejected_pre_execution": len(records),
        "root_counts": root_counts,
        "terminal_counts": terminal_counts,
        "examples": examples,
        "counts_truncated": root_truncated or terminal_truncated,
        "examples_truncated": len(records) * 2 > len(examples),
    }


def _research_rejection_evidence(
    records: Sequence[AggregateRecord],
    *,
    artifacts: ArtifactStore,
) -> dict[str, object]:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "records": [record.to_wire() for record in records],
    }
    ledger = artifacts.put_bytes(canonical_json_bytes(payload), kind=ArtifactKind.LOG)
    return {
        "research_rejection_summary": _research_rejection_summary(records),
        "research_rejection_ledger": ledger.manifest(),
    }


def _rejection_journal_directory(generated_root: Path) -> Path:
    try:
        generated_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_metadata = generated_root.lstat()
    except OSError as exc:
        raise FullCampaignError("generated-source root is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise FullCampaignError("generated-source root must be a real directory")
    journal = generated_root / _REJECTION_JOURNAL_DIR
    try:
        journal.mkdir(exist_ok=True, mode=0o700)
        journal_metadata = journal.lstat()
    except OSError as exc:
        raise FullCampaignError("research rejection journal is unavailable") from exc
    if stat.S_ISLNK(journal_metadata.st_mode) or not stat.S_ISDIR(journal_metadata.st_mode):
        raise FullCampaignError("research rejection journal must be a real directory")
    return journal


def _rejection_journal_payload(
    *,
    campaign_id: str,
    scientific_iteration: int,
    record: AggregateRecord,
    previous_journal_digest: str | None,
) -> dict[str, object]:
    record_wire = record.to_wire()
    record_digest = _digest(b"kuairand-rejected-lineage-record-v1", record_wire)
    unsigned = {
        "schema_version": _SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "scientific_iteration": scientific_iteration,
        "previous_journal_digest": previous_journal_digest,
        "record_digest": record_digest,
        "record": record_wire,
    }
    return unsigned | {"journal_digest": _digest(b"kuairand-rejection-journal-entry-v1", unsigned)}


def _decode_rejection_journal_entry(
    payload: bytes,
    *,
    campaign_id: str,
    scientific_iteration: int,
    previous_journal_digest: str | None,
) -> tuple[AggregateRecord, str]:
    if len(payload) > _REJECTION_JOURNAL_LIMIT_BYTES:
        raise FullCampaignError("research rejection journal entry exceeds its byte limit")
    try:
        decoded = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FullCampaignError("research rejection journal entry is malformed") from exc
    if not isinstance(decoded, dict) or payload != canonical_json_bytes(decoded):
        raise FullCampaignError("research rejection journal entry is not exact canonical JSON")
    expected_keys = {
        "schema_version",
        "campaign_id",
        "scientific_iteration",
        "previous_journal_digest",
        "record_digest",
        "record",
        "journal_digest",
    }
    if set(decoded) != expected_keys:
        raise FullCampaignError("research rejection journal entry has unexpected fields")
    if (
        decoded["schema_version"] != _SCHEMA_VERSION
        or decoded["campaign_id"] != campaign_id
        or decoded["scientific_iteration"] != scientific_iteration
        or decoded["previous_journal_digest"] != previous_journal_digest
    ):
        raise FullCampaignError("research rejection journal identity is inconsistent")
    record_wire = decoded["record"]
    if (
        not isinstance(record_wire, dict)
        or set(record_wire) != {"name", "values"}
        or type(record_wire["name"]) is not str
        or not isinstance(record_wire["values"], dict)
    ):
        raise FullCampaignError("research rejection journal record is malformed")
    record = AggregateRecord(record_wire["name"], record_wire["values"])
    if (
        record.name != f"rejected_lineage_{scientific_iteration:02d}"
        or record.values.get("scientific_iteration") != scientific_iteration
    ):
        raise FullCampaignError("research rejection journal iteration is inconsistent")
    expected_record_digest = _digest(b"kuairand-rejected-lineage-record-v1", record.to_wire())
    unsigned = {key: decoded[key] for key in expected_keys - {"journal_digest"}}
    expected_journal_digest = _digest(b"kuairand-rejection-journal-entry-v1", unsigned)
    if (
        decoded["record_digest"] != expected_record_digest
        or decoded["journal_digest"] != expected_journal_digest
    ):
        raise FullCampaignError("research rejection journal digest is inconsistent")
    return record, cast(str, decoded["journal_digest"])


def _load_rejection_journal(
    generated_root: Path,
    *,
    campaign_id: str,
    maximum_iterations: int,
) -> tuple[list[AggregateRecord], str | None]:
    journal = _rejection_journal_directory(generated_root)
    try:
        entries = tuple(sorted(journal.iterdir()))
    except OSError as exc:
        raise FullCampaignError("research rejection journal cannot be listed") from exc
    if len(entries) > maximum_iterations:
        raise FullCampaignError("research rejection journal exceeds the campaign iteration cap")
    records: list[AggregateRecord] = []
    previous_digest: str | None = None
    for iteration, path in enumerate(entries, start=1):
        expected_name = f"rejection-{iteration:02d}.json"
        try:
            metadata = path.lstat()
            payload = path.read_bytes()
        except OSError as exc:
            raise FullCampaignError("research rejection journal entry is unavailable") from exc
        if (
            path.name != expected_name
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise FullCampaignError("research rejection journal is not a contiguous file ledger")
        record, previous_digest = _decode_rejection_journal_entry(
            payload,
            campaign_id=campaign_id,
            scientific_iteration=iteration,
            previous_journal_digest=previous_digest,
        )
        records.append(record)
    return records, previous_digest


def _persist_rejection_journal_entry(
    generated_root: Path,
    *,
    campaign_id: str,
    scientific_iteration: int,
    record: AggregateRecord,
    previous_journal_digest: str | None,
) -> str:
    journal = _rejection_journal_directory(generated_root)
    payload = canonical_json_bytes(
        _rejection_journal_payload(
            campaign_id=campaign_id,
            scientific_iteration=scientific_iteration,
            record=record,
            previous_journal_digest=previous_journal_digest,
        )
    )
    if len(payload) > _REJECTION_JOURNAL_LIMIT_BYTES:
        raise FullCampaignError("research rejection journal entry exceeds its byte limit")
    path = journal / f"rejection-{scientific_iteration:02d}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            retained = path.read_bytes()
        except OSError as exc:
            raise FullCampaignError("retained research rejection entry is unavailable") from exc
        if retained != payload:
            raise FullCampaignError(
                "retained research rejection entry differs from replay"
            ) from None
    except OSError as exc:
        raise FullCampaignError("research rejection entry cannot be persisted") from exc
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        try:
            directory_descriptor = os.open(journal, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            raise FullCampaignError("research rejection journal cannot be synchronized") from exc
    _, journal_digest = _decode_rejection_journal_entry(
        payload,
        campaign_id=campaign_id,
        scientific_iteration=scientific_iteration,
        previous_journal_digest=previous_journal_digest,
    )
    return journal_digest


def _provider_transcript_sink(production: Path) -> Callable[[ProviderTranscript], None]:
    journal = production / _PROVIDER_ATTEMPT_JOURNAL_DIR
    journal.mkdir(parents=True, exist_ok=True, mode=0o700)
    if journal.is_symlink() or not journal.is_dir():
        raise FullCampaignError("provider-attempt journal must be a real directory")

    def persist(transcript: ProviderTranscript) -> None:
        payload = transcript.json_bytes
        if hashlib.sha256(payload).hexdigest() != transcript.digest:
            raise FullCampaignError("provider transcript digest does not identify its bytes")
        path = journal / f"{transcript.digest}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            try:
                retained = path.read_bytes()
            except OSError as exc:
                raise FullCampaignError("retained provider transcript is unavailable") from exc
            if retained != payload:
                raise FullCampaignError(
                    "retained provider transcript differs from replay"
                ) from None
            return
        except OSError as exc:
            raise FullCampaignError("provider transcript cannot be persisted") from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        try:
            directory_descriptor = os.open(journal, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            raise FullCampaignError("provider-attempt journal cannot be synchronized") from exc

    return persist


@dataclass(frozen=True, slots=True)
class _LiveLineagePreparation:
    status: Literal[
        "accepted",
        "admission_closed",
        "portfolio_exhausted",
        "repeated_pre_admission_failure",
        "provider_unavailable",
    ]
    lineage: LiveResearchLineage | None
    safe_context: SafeResearchContext | None
    scientific_iteration: int | None
    rejected_records: tuple[AggregateRecord, ...]
    branches_attempted: int
    provider_error: OpenAIProviderError | None = None


def _prepare_live_lineage_portfolio(
    *,
    campaign_id: str,
    parent: ParentSnapshot,
    generated_root: Path,
    artifact_store: ArtifactStore,
    model: ResearchModel,
    provider: str,
    maximum_iterations: int,
    safe_context_factory: Callable[[tuple[AggregateRecord, ...]], SafeResearchContext],
    continue_check: Callable[[], bool],
) -> _LiveLineagePreparation:
    """Prepare one valid lineage or return an exact, evidence-preserving closure reason."""

    rejected, previous_journal_digest = _load_rejection_journal(
        generated_root,
        campaign_id=campaign_id,
        maximum_iterations=maximum_iterations,
    )
    root_failure_totals: dict[str, int] = {}
    proposal_family_attempts: dict[str, int] = {}
    previous_root_fingerprint: str | None = None
    consecutive_root_failures = 0
    for retained_iteration, retained in enumerate(rejected, start=1):
        root_fingerprint = retained.values.get("root_failure_fingerprint")
        family = retained.values.get("proposal_family")
        if type(root_fingerprint) is not str or type(family) is not str:
            raise FullCampaignError("retained research rejection lacks typed identity")
        root_total = root_failure_totals.get(root_fingerprint, 0) + 1
        root_failure_totals[root_fingerprint] = root_total
        if root_fingerprint == previous_root_fingerprint:
            consecutive_root_failures += 1
        else:
            previous_root_fingerprint = root_fingerprint
            consecutive_root_failures = 1
        family_total = proposal_family_attempts.get(family, 0) + 1
        proposal_family_attempts[family] = family_total
        if (
            retained.values.get("root_failure_total_count") != root_total
            or retained.values.get("root_failure_consecutive_count") != consecutive_root_failures
            or retained.values.get("proposal_family_attempt_count") != family_total
            or retained.values.get("proposal_family_blocked") != (family_total >= 2)
        ):
            raise FullCampaignError("retained research rejection counters are inconsistent")
        if root_total >= 3:
            return _LiveLineagePreparation(
                status="repeated_pre_admission_failure",
                lineage=None,
                safe_context=None,
                scientific_iteration=None,
                rejected_records=tuple(rejected),
                branches_attempted=retained_iteration,
            )
    for iteration in range(len(rejected) + 1, maximum_iterations + 1):
        if not continue_check():
            return _LiveLineagePreparation(
                status="admission_closed",
                lineage=None,
                safe_context=None,
                scientific_iteration=None,
                rejected_records=tuple(rejected),
                branches_attempted=iteration - 1,
            )
        safe_context = safe_context_factory(tuple(rejected))
        try:
            lineage = prepare_or_rehydrate_live_lineage(
                campaign_id=campaign_id,
                scientific_iteration=iteration,
                parent=parent,
                generated_root=generated_root,
                artifact_store=artifact_store,
                safe_context=safe_context,
                model=model,
                provider=provider,
            )
        except OpenAIProviderError as error:
            return _LiveLineagePreparation(
                status="provider_unavailable",
                lineage=None,
                safe_context=None,
                scientific_iteration=None,
                rejected_records=tuple(rejected),
                branches_attempted=iteration,
                provider_error=error,
            )
        except LiveResearchBranchRejected as rejection:
            root_fingerprint = rejection.root_failure.fingerprint
            root_failure_total = root_failure_totals.get(root_fingerprint, 0) + 1
            root_failure_totals[root_fingerprint] = root_failure_total
            if root_fingerprint == previous_root_fingerprint:
                consecutive_root_failures += 1
            else:
                previous_root_fingerprint = root_fingerprint
                consecutive_root_failures = 1
            family = rejection.proposal_family
            family_attempt_count = proposal_family_attempts.get(family, 0) + 1
            proposal_family_attempts[family] = family_attempt_count
            record = _rejected_lineage_record(
                rejection,
                scientific_iteration=iteration,
                root_failure_total_count=root_failure_total,
                root_failure_consecutive_count=consecutive_root_failures,
                proposal_family_attempt_count=family_attempt_count,
                proposal_family_blocked=family_attempt_count >= 2,
            )
            previous_journal_digest = _persist_rejection_journal_entry(
                generated_root,
                campaign_id=campaign_id,
                scientific_iteration=iteration,
                record=record,
                previous_journal_digest=previous_journal_digest,
            )
            rejected.append(record)
            if root_failure_total >= 3:
                return _LiveLineagePreparation(
                    status="repeated_pre_admission_failure",
                    lineage=None,
                    safe_context=None,
                    scientific_iteration=None,
                    rejected_records=tuple(rejected),
                    branches_attempted=iteration,
                )
            continue
        return _LiveLineagePreparation(
            status="accepted",
            lineage=lineage,
            safe_context=safe_context,
            scientific_iteration=iteration,
            rejected_records=tuple(rejected),
            branches_attempted=iteration,
        )
    return _LiveLineagePreparation(
        status="portfolio_exhausted",
        lineage=None,
        safe_context=None,
        scientific_iteration=None,
        rejected_records=tuple(rejected),
        branches_attempted=maximum_iterations,
    )


def _provider_usage(model: ResearchModel | None) -> Mapping[str, object] | None:
    def context_limits(endpoint: OpenAIChatCompletionsModel) -> Mapping[str, object] | None:
        limits = endpoint.context_limits
        if limits is None:
            return None
        return {
            "context_length": limits.context_length,
            "max_completion_tokens": limits.max_completion_tokens,
            "source": limits.source,
        }

    provider_models: tuple[tuple[str, OpenAIChatCompletionsModel], ...]
    if isinstance(model, OpenAIChatCompletionsModel):
        provider_models = (("main", model),)
        active_slot = "main"
        transcripts = model.transcripts
        failovers: tuple[object, ...] = ()
        usage = model.total_usage
        active_model = model
    elif isinstance(model, OpenAIFailoverModel):
        provider_models = model.provider_models
        active_slot = model.active_slot
        transcripts = model.transcripts
        failovers = tuple(item.to_wire() for item in model.failover_events)
        usage = model.total_usage
        active_model = model.active_model
    else:
        return None
    return {
        "model": active_model.config.model,
        "base_url": active_model.config.base_url,
        "context_limits": context_limits(active_model),
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "unaccounted_attempts": usage.unaccounted_attempts,
        "retry_wait_seconds": round(usage.retry_wait_seconds, 6),
        "transcript_count": len(transcripts),
        "provider_wall_seconds": round(sum(item.latency_seconds for item in transcripts), 6),
        "active_slot": active_slot,
        "failover_count": len(failovers),
        "failover_events": list(failovers),
        "provider_chain": [
            {
                "slot": slot,
                "model": endpoint.config.model,
                "base_url": endpoint.config.base_url,
                "credential_env": endpoint.config.api_key_env,
                "context_limits": context_limits(endpoint),
                "input_tokens": endpoint.total_usage.input_tokens,
                "cached_input_tokens": endpoint.total_usage.cached_input_tokens,
                "output_tokens": endpoint.total_usage.output_tokens,
                "reasoning_tokens": endpoint.total_usage.reasoning_tokens,
                "total_tokens": endpoint.total_usage.total_tokens,
                "estimated_cost_usd": endpoint.total_usage.estimated_cost_usd,
                "unaccounted_attempts": endpoint.total_usage.unaccounted_attempts,
                "retry_wait_seconds": round(endpoint.total_usage.retry_wait_seconds, 6),
                "transcript_count": len(endpoint.transcripts),
            }
            for slot, endpoint in provider_models
        ],
    }


def _provider_transcripts(
    model: OpenAIChatCompletionsModel | OpenAIFailoverModel,
) -> tuple[object, ...]:
    return tuple(json.loads(item.json_bytes) for item in model.transcripts)


def _provider_credential_envs(
    model: OpenAIChatCompletionsModel | OpenAIFailoverModel,
) -> tuple[str, ...]:
    if isinstance(model, OpenAIChatCompletionsModel):
        return (model.config.api_key_env,)
    return tuple(endpoint.config.api_key_env for _, endpoint in model.provider_models)


def _run_autonomous_followups(
    *,
    request: CampaignCreateRequest,
    data: CampaignDataPlane,
    runtime_template: _ScientificRuntime,
    scientific_config: ScientificCampaignConfig,
    fallback: IncumbentEvidence,
    outer_ledger_path: Path,
    candidate_limits: LocalCandidateLimits,
    dataset_digest: str,
    context_evidence: _ResearchContextEvidence,
    validation_inputs: CanonicalInputs,
    final_inputs: CanonicalInputs,
    research_model: ResearchModel,
    first_lineage: LiveResearchLineage,
    first_result: ScientificCampaignResult,
    first_selection: FinalizationSelectionPlan | None,
    first_reflection: Reflection,
    first_reflection_evidence: tuple[str, str, ArtifactRef],
    prior_records: tuple[AggregateRecord, ...],
) -> _AutonomousFollowupResult:
    """Continue propose→implement→evaluate→reflect until an exact terminal condition."""

    result = first_result
    selection = first_selection
    lineage = first_lineage
    reflection = first_reflection
    reflection_request, reflection_response, reflection_transcript = first_reflection_evidence
    parent = (
        snapshot_materialized_candidate(
            lineage.materialized,
            candidate_id=runtime_template.candidate.candidate_id,
        )
        if result.incumbent.candidate_id == runtime_template.candidate.candidate_id
        else lineage.parent
    )
    candidates = list(result.candidates)
    feedback = list(result.public_feedback)
    records = [
        *prior_records,
        _iteration_record(
            result,
            reflection,
            scientific_iteration=runtime_template.scientific_iteration,
            lineage=lineage,
            run_records=runtime_template.records,
        ),
    ]
    rejected_records = list(prior_records)
    iteration = runtime_template.scientific_iteration
    terminal = result.stop_reason

    while True:
        if result.convergence.should_stop:
            terminal = CampaignStopReason.CONVERGED
            break
        if result.convergence.completed_iterations >= scientific_config.max_scientific_iterations:
            terminal = CampaignStopReason.ITERATION_CAP
            break
        if iteration >= scientific_config.max_scientific_iterations:
            terminal = CampaignStopReason.ITERATION_CAP
            break
        if result.launches_used >= scientific_config.max_launches:
            terminal = CampaignStopReason.LAUNCH_CAP
            break
        if result.stop_reason in {
            CampaignStopReason.LAUNCH_CAP,
            CampaignStopReason.FINALIZATION_RESERVE,
            CampaignStopReason.HARD_DEADLINE,
        }:
            terminal = result.stop_reason
            break
        status = runtime_template.engine.status(runtime_template.run_dir)
        reusable_receipt_count = (
            _reusable_score_receipt_count(
                outer_ledger_path,
                maximum=request.config.validation.outer_promotion_limit,
            )
            if status.outer_queries_remaining <= 0
            else 0
        )
        if not _may_attempt_outer_evaluation(
            outer_queries_remaining=status.outer_queries_remaining,
            reusable_receipt_count=reusable_receipt_count,
        ):
            terminal = CampaignStopReason.OUTER_PROMOTION_LIMIT
            break

        iteration += 1
        safe_context = _safe_context(
            request=request,
            qualification=runtime_template.qualification,
            fold_a=runtime_template.fold_a,
            fold_b=runtime_template.fold_b,
            status=status,
            campaign_records=tuple(records),
            evidence=context_evidence,
        )
        try:
            lineage = prepare_or_rehydrate_live_lineage(
                campaign_id=request.campaign_id,
                scientific_iteration=iteration,
                parent=parent,
                generated_root=runtime_template.run_dir / _PRODUCTION_DIR / "generated-source",
                artifact_store=runtime_template.artifacts,
                safe_context=safe_context,
                model=research_model,
                provider="openai",
            )
        except LiveResearchBranchRejected as rejection:
            rejected_record = _rejected_lineage_record(
                rejection,
                scientific_iteration=iteration,
            )
            records.append(rejected_record)
            rejected_records.append(rejected_record)
            terminal = CampaignStopReason.CANDIDATES_EXHAUSTED
            continue
        candidate_id = lineage.candidate_id
        research_ledger, experiment_id = _ensure_lineage_ledger(
            campaign_store=runtime_template.campaign_store,
            lineage=lineage,
            candidate_id=candidate_id,
            scientific_iteration=iteration,
        )
        candidate = _generated_scientific_candidate(
            request=request,
            data=data,
            features=runtime_template.features,
            lineage=lineage,
            candidate_id=candidate_id,
            candidate_limits=candidate_limits,
            scientific_iteration=iteration,
        )
        iteration_runtime = replace(
            runtime_template,
            lineage=lineage,
            candidate=candidate,
            experiment_id=experiment_id,
            scientific_iteration=iteration,
            records={},
        )
        with _open_outer_ledger(
            outer_ledger_path,
            maximum=request.config.validation.outer_promotion_limit,
        ) as project_ledger:
            adapter = DurableScientificLedgerAdapter(
                runtime_template.campaign_store,
                project_ledger,
                scorer_digest=runtime_template.outer_scorer.scorer.scorer_digest,
                evidence_registry=iteration_runtime.evidence_registry,
                representative_seed=0,
            )
            admission = ReceiptAwareOuterEvaluationLedger(
                adapter,
                ScoringReceiptBook.from_projection(project_ledger.projection()),
            )
            iteration_runtime.outer_admission = admission
            result = run_scientific_campaign(
                config=scientific_config,
                fallback=fallback,
                candidates=(candidate,),
                runner=iteration_runtime,
                outer_ledger=admission,
                initial_incumbent=result.incumbent,
                initial_convergence=result.convergence,
                initial_launches_used=result.launches_used,
                initial_elapsed_seconds=result.elapsed_seconds,
            )
        candidates.extend(result.candidates)
        feedback.extend(result.public_feedback)
        runtime_template.campaign_store.set_convergence_state(
            result.convergence.manifest(),
            expected_revision=runtime_template.campaign_store.snapshot().revision,
            reason=f"persist autonomous scientific convergence cursor after iteration {iteration}",
        )
        promoted_selection = _candidate_selection(
            experiment_id=experiment_id,
            result=result,
            runtime=iteration_runtime,
            qualification=runtime_template.qualification,
            features=runtime_template.features,
            feature_artifacts=runtime_template.feature_artifacts,
            dataset_digest=dataset_digest,
            validation_inputs=validation_inputs,
            final_inputs=final_inputs,
            limits=candidate_limits,
        )
        if promoted_selection is not None:
            selection = promoted_selection
        (
            reflection_request,
            reflection_response,
            reflection_transcript,
            reflection,
        ) = _reflect(
            campaign_store=runtime_template.campaign_store,
            research_ledger=research_ledger,
            lineage=lineage,
            candidate_id=candidate_id,
            scientific_iteration=iteration,
            result=result,
            safe_context=safe_context,
            research_config=request.config.research,
            artifacts=runtime_template.artifacts,
            research_model=research_model,
        )
        records.append(
            _iteration_record(
                result,
                reflection,
                scientific_iteration=iteration,
                lineage=lineage,
                run_records=iteration_runtime.records,
            )
        )
        if result.incumbent.candidate_id == candidate_id:
            parent = snapshot_materialized_candidate(
                lineage.materialized,
                candidate_id=candidate_id,
            )
        terminal = result.stop_reason

    aggregate = ScientificCampaignResult(
        config_digest=scientific_config.digest,
        fallback=fallback,
        incumbent=result.incumbent,
        candidates=tuple(candidates),
        public_feedback=tuple(feedback),
        convergence=result.convergence,
        launches_used=result.launches_used,
        elapsed_seconds=result.elapsed_seconds,
        stop_reason=terminal,
    )
    return _AutonomousFollowupResult(
        result=aggregate,
        selection=selection,
        reflection_request_digest=reflection_request,
        reflection_response_digest=reflection_response,
        reflection_transcript=reflection_transcript,
        iterations_completed=iteration,
        rejected_records=tuple(rejected_records),
    )


def _ensure_finalization_required(store: CampaignStore) -> None:
    snapshot = store.snapshot()
    if snapshot.status == CampaignState.FINALIZATION_REQUIRED.value:
        return
    if snapshot.status not in {
        CampaignState.CREATED.value,
        CampaignState.RUNNING.value,
        CampaignState.PAUSED.value,
    }:
        raise FullCampaignError(
            f"research campaign cannot enter finalization from {snapshot.status!r}"
        )
    store.set_campaign_phase(
        phase="finalization_required",
        status=CampaignState.FINALIZATION_REQUIRED.value,
        expected_revision=snapshot.revision,
        reason="bounded provider-free research portfolio is closed and evidence is retained",
    )


def _commit_outcome(
    *,
    run_dir: Path,
    request: CampaignCreateRequest,
    engine: CampaignEngine,
    campaign_store: CampaignStore,
    artifacts: ArtifactStore,
    progress: FullCampaignProgressLedger,
    qualification: OfficialFMQualificationEvidence,
    scorer_digest: str,
    fallback_receipt_digest: str,
    scientific_result_digest: str | None,
    reflection_request_digest: str | None,
    reflection_response_digest: str | None,
    reflection_transcript: ArtifactRef | None,
    selection: FinalizationSelectionPlan | None,
) -> FullCampaignOutcome:
    checkpoints = progress.checkpoints()
    if not checkpoints or checkpoints[-1].stage is not FullCampaignStage.REFLECTED:
        raise FullCampaignError("outcome requires a durable reflected research checkpoint")
    _ensure_finalization_required(campaign_store)
    snapshot = campaign_store.snapshot()
    if (
        snapshot.incumbent is None
        or snapshot.incumbent.incumbent_id != SCRIPTED_PARENT_ID
        or not snapshot.incumbent.is_fallback
        or not snapshot.incumbent.replay_verified
    ):
        raise FullCampaignError("qualified official-FM fallback was not preserved")
    outcome = FullCampaignOutcome(
        run_dir=run_dir,
        campaign_id=request.campaign_id,
        request_digest=request.digest,
        progress_predecessor_digest=checkpoints[-1].digest,
        fallback_candidate_id=SCRIPTED_PARENT_ID,
        fallback_receipt_digest=fallback_receipt_digest,
        qualification_manifest_digest=qualification.manifest_digest,
        dataset_digest=qualification.canonical_digest,
        scorer_digest=scorer_digest,
        validation_row_count=qualification.validation_row_count,
        final_row_count=qualification.final_row_count,
        scientific_result_digest=scientific_result_digest,
        reflection_request_digest=reflection_request_digest,
        reflection_response_digest=reflection_response_digest,
        reflection_transcript=reflection_transcript,
        selection=selection,
        launches_used=snapshot.launches_used,
        outer_queries_used=snapshot.outer_queries_used,
        manual_interventions=0,
    )
    committed = FullCampaignOutcomeRepository(
        run_dir=run_dir,
        artifact_store=artifacts,
        progress=progress,
    ).commit(outcome)
    reloaded = load_full_campaign_outcome(run_dir, engine=engine)
    if reloaded != committed:
        raise FullCampaignError("strict outcome reload differs from the committed outcome")
    return reloaded


def _provider_unavailable_outcome(
    *,
    diagnostic: ProviderUnavailableDiagnostic,
    run_dir: Path,
    request: CampaignCreateRequest,
    engine: CampaignEngine,
    campaign_store: CampaignStore,
    artifacts: ArtifactStore,
    progress: FullCampaignProgressLedger,
    qualification: OfficialFMQualificationEvidence,
    scorer_digest: str,
    fallback_receipt_digest: str,
) -> FullCampaignOutcome:
    rejection_evidence = _research_rejection_evidence((), artifacts=artifacts)
    stage_counts = _research_stage_counts(
        model=None,
        branches_attempted=0,
        rejected_records=(),
        candidates_admitted=0,
    )
    receipt = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                "diagnostic": diagnostic.to_wire(),
                "fallback_preserved": True,
                "manual_interventions": 0,
            }
        ),
        kind=ArtifactKind.LOG,
    )
    evidence = {
        "provider_diagnostic": diagnostic.to_wire(),
        "diagnostic_artifact": receipt.manifest(),
        "fallback_receipt_digest": fallback_receipt_digest,
        "fallback_preserved": True,
        "provider_usage": None,
        "research_stage_counts": stage_counts,
        **rejection_evidence,
    }
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.LINEAGE_READY,
        evidence=evidence,
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.SCIENCE_COMPLETE,
        evidence=evidence | {"scientific_result_digest": None},
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.REFLECTED,
        evidence=evidence
        | {
            "reflection_request_digest": None,
            "reflection_response_digest": None,
            "portfolio_count": 0,
            "portfolio_cap_reason": "configured_provider_unavailable",
        },
    )
    return _commit_outcome(
        run_dir=run_dir,
        request=request,
        engine=engine,
        campaign_store=campaign_store,
        artifacts=artifacts,
        progress=progress,
        qualification=qualification,
        scorer_digest=scorer_digest,
        fallback_receipt_digest=fallback_receipt_digest,
        scientific_result_digest=None,
        reflection_request_digest=None,
        reflection_response_digest=None,
        reflection_transcript=None,
        selection=None,
    )


def _runtime_provider_unavailable_outcome(
    *,
    error: OpenAIProviderError,
    model: OpenAIChatCompletionsModel | OpenAIFailoverModel,
    run_dir: Path,
    request: CampaignCreateRequest,
    engine: CampaignEngine,
    campaign_store: CampaignStore,
    artifacts: ArtifactStore,
    progress: FullCampaignProgressLedger,
    qualification: OfficialFMQualificationEvidence,
    scorer_digest: str,
    fallback_receipt_digest: str,
    rejected_records: Sequence[AggregateRecord] = (),
    branches_attempted: int = 1,
) -> FullCampaignOutcome:
    """Close initial research safely after the live provider exhausts bounded retries."""

    retryable = error.code in {
        ProviderErrorCode.TRANSPORT,
        ProviderErrorCode.HTTP,
        ProviderErrorCode.INCOMPLETE,
    }
    credential_envs = _provider_credential_envs(model)
    failures = (
        [item.to_wire() for item in error.failures]
        if isinstance(error, OpenAIProviderChainError)
        else []
    )
    diagnostic = {
        "category": "provider_unavailable",
        "provider": "openai",
        "code": error.code.value,
        "message": "The live research provider failed after its bounded retry policy.",
        "retryable": retryable,
        "credential_env": credential_envs[0],
        "credential_envs": list(credential_envs),
        "operation": None if error.operation is None else error.operation.value,
        "attempts": error.attempts,
        "status_code": error.status_code,
        "provider_failures": failures,
    }
    transcript_payloads = list(_provider_transcripts(model))
    rejection_evidence = _research_rejection_evidence(
        rejected_records,
        artifacts=artifacts,
    )
    stage_counts = _research_stage_counts(
        model=model,
        branches_attempted=branches_attempted,
        rejected_records=rejected_records,
        candidates_admitted=0,
    )
    receipt = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                "diagnostic": diagnostic,
                "provider_attempt_transcripts": transcript_payloads,
                "provider_failover_events": (
                    [item.to_wire() for item in model.failover_events]
                    if isinstance(model, OpenAIFailoverModel)
                    else []
                ),
                "rejected_lineage_records": [record.to_wire() for record in rejected_records],
                "research_stage_counts": stage_counts,
                "fallback_preserved": True,
                "manual_interventions": 0,
            }
        ),
        kind=ArtifactKind.LOG,
    )
    evidence = {
        "provider_diagnostic": diagnostic,
        "diagnostic_artifact": receipt.manifest(),
        "provider_usage": _provider_usage(model),
        "research_stage_counts": stage_counts,
        **rejection_evidence,
        "fallback_receipt_digest": fallback_receipt_digest,
        "fallback_preserved": True,
    }
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.LINEAGE_READY,
        evidence=evidence
        | {
            "generated_lineage_status": "provider_retry_exhausted",
            "manual_source_edits": 0,
        },
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.SCIENCE_COMPLETE,
        evidence=evidence
        | {
            "scientific_result_digest": None,
            "portfolio_count": branches_attempted,
            "portfolio_cap_reason": "runtime_provider_unavailable",
        },
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.REFLECTED,
        evidence=evidence
        | {
            "reflection_request_digest": None,
            "reflection_response_digest": None,
            "portfolio_count": branches_attempted,
            "portfolio_cap_reason": "runtime_provider_unavailable",
            "manual_interventions": 0,
        },
    )
    return _commit_outcome(
        run_dir=run_dir,
        request=request,
        engine=engine,
        campaign_store=campaign_store,
        artifacts=artifacts,
        progress=progress,
        qualification=qualification,
        scorer_digest=scorer_digest,
        fallback_receipt_digest=fallback_receipt_digest,
        scientific_result_digest=None,
        reflection_request_digest=None,
        reflection_response_digest=None,
        reflection_transcript=None,
        selection=None,
    )


def _qualification_fallback_receipt(
    qualification: OfficialFMQualificationEvidence,
    *,
    reason: str,
) -> str:
    return _digest(
        b"kuairand-production-qualification-only-fallback-receipt-v1",
        {
            "qualification_manifest_digest": qualification.manifest_digest,
            "fallback_manifest_digest": qualification.fallback.manifest_digest,
            "reason": reason,
            "replayable": True,
            "final_outcomes_scored": 0,
        },
    )


def _fold_branch_failure_details(
    error: CandidateAdmissionError | SupervisedFoldFMExecutionError,
    *,
    artifacts: ArtifactStore,
) -> Mapping[str, object]:
    """Classify one narrowly recoverable fold failure from trusted immutable evidence."""

    if isinstance(error, CandidateAdmissionError):
        if error.reason not in _SAFE_FOLD_ADMISSION_REASONS:
            raise error
        return {
            "error_type": type(error).__name__,
            "admission_reason": error.reason.value,
            "execution_outcome": None,
            "journal_closure_digest": None,
            "candidate_released": False,
            "cleanup_verified": True,
            "launch_charged": False,
        }
    closure = error.artifacts
    if closure is None:
        raise error
    manifest_ref = closure.artifact("execution_manifest")
    payload = artifacts.read_bytes(manifest_ref, max_bytes=manifest_ref.size_bytes)
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FullCampaignError("fold failure execution manifest is not JSON") from exc
    expected_fields = {
        "schema_version",
        "execution_id",
        "outcome",
        "process",
        "candidate_released",
        "exit_code",
        "terminating_signal",
        "started_at_utc",
        "ended_at_utc",
        "wall_seconds",
        "peak_tree_rss_bytes",
        "peak_workspace_bytes",
        "peak_process_count",
        "stdout",
        "stderr",
        "cleanup_verified",
        "device",
        "threads",
        "detail",
        "candidate_metrics_accepted",
    }
    if (
        not isinstance(decoded, dict)
        or set(decoded) != expected_fields
        or payload != _canonical_json(decoded)
    ):
        raise FullCampaignError("fold failure execution manifest is not exact canonical JSON")
    result = error.result
    if result is not None and payload != _canonical_json(result.manifest()):
        raise FullCampaignError("fold failure result differs from its immutable manifest")
    try:
        outcome = ExecutionOutcome(decoded["outcome"])
    except (TypeError, ValueError) as exc:
        raise FullCampaignError("fold failure execution outcome is invalid") from exc
    released = decoded["candidate_released"]
    cleanup_verified = decoded["cleanup_verified"]
    if type(released) is not bool or type(cleanup_verified) is not bool:
        raise FullCampaignError("fold failure release or cleanup evidence is malformed")
    if decoded["candidate_metrics_accepted"] is not False:
        raise FullCampaignError("failed fold execution accepted candidate metrics")
    if outcome is ExecutionOutcome.CANCELLED:
        raise FullCampaignCancelled(
            "campaign cancelled by supervised fold runner; durable state is resumable"
        ) from error
    if (
        outcome not in _SAFE_FOLD_FAILURE_OUTCOMES
        or cleanup_verified is not True
        or (result is not None and result.succeeded)
    ):
        raise error
    return {
        "error_type": type(error).__name__,
        "admission_reason": None,
        "execution_outcome": outcome.value,
        "journal_closure_digest": closure.closure_digest,
        "candidate_released": released,
        "cleanup_verified": cleanup_verified,
        "launch_charged": True,
    }


def _research_admission_closed_outcome(
    *,
    reason: str,
    run_dir: Path,
    request: CampaignCreateRequest,
    engine: CampaignEngine,
    campaign_store: CampaignStore,
    artifacts: ArtifactStore,
    progress: FullCampaignProgressLedger,
    qualification: OfficialFMQualificationEvidence,
    scorer_digest: str,
    fold_b: SupervisedFoldFMRun | None = None,
    fallback_receipt_digest: str | None = None,
    diagnostic_category: str = "research_admission_closed",
    fold_a_status: str = "not_started",
    fold_b_status: str | None = None,
    diagnostic_details: Mapping[str, object] | None = None,
    research_model: ResearchModel | None = None,
    rejected_records: Sequence[AggregateRecord] = (),
    branches_attempted: int = 0,
) -> FullCampaignOutcome:
    fallback_receipt = (
        _qualification_fallback_receipt(qualification, reason=reason)
        if fallback_receipt_digest is None
        else fallback_receipt_digest
    )
    diagnostic = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                "category": diagnostic_category,
                "reason": reason,
                "fallback_receipt_digest": fallback_receipt,
                "fallback_preserved": True,
                "final_outcomes_scored": 0,
                "details": {} if diagnostic_details is None else dict(diagnostic_details),
            }
        ),
        kind=ArtifactKind.LOG,
    )
    stage_counts = _research_stage_counts(
        model=research_model,
        branches_attempted=branches_attempted,
        rejected_records=rejected_records,
        candidates_admitted=0,
    )
    rejection_evidence = _research_rejection_evidence(
        rejected_records,
        artifacts=artifacts,
    )
    common = {
        "admission_closed": True,
        "reason": reason,
        "diagnostic_artifact": diagnostic.manifest(),
        "fallback_receipt_digest": fallback_receipt,
        "fallback_preserved": True,
        "provider_usage": _provider_usage(research_model),
        "research_stage_counts": stage_counts,
        **rejection_evidence,
    }
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.FOLD_CONTROLS_READY,
        evidence=common
        | {
            "fold_a_status": fold_a_status,
            "fold_b_status": ("not_started" if fold_b is None else "completed")
            if fold_b_status is None
            else fold_b_status,
            "fold_b_evidence_digest": None if fold_b is None else fold_b.evidence.digest,
            "new_research_launches_after_closure": 0,
        },
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.FEATURES_READY,
        evidence=common
        | {
            "feature_build_status": "not_required_for_qualified_fallback",
            "final_target_capability": None,
        },
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.LINEAGE_READY,
        evidence=common
        | {
            "generated_lineage_status": "not_started",
            "manual_source_edits": 0,
        },
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.SCIENCE_COMPLETE,
        evidence=common
        | {
            "scientific_result_digest": None,
            "portfolio_count": branches_attempted,
            "portfolio_cap_reason": reason,
        },
    )
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.REFLECTED,
        evidence=common
        | {
            "reflection_request_digest": None,
            "reflection_response_digest": None,
            "manual_interventions": 0,
        },
    )
    return _commit_outcome(
        run_dir=run_dir,
        request=request,
        engine=engine,
        campaign_store=campaign_store,
        artifacts=artifacts,
        progress=progress,
        qualification=qualification,
        scorer_digest=scorer_digest,
        fallback_receipt_digest=fallback_receipt,
        scientific_result_digest=None,
        reflection_request_digest=None,
        reflection_response_digest=None,
        reflection_transcript=None,
        selection=None,
    )


def _open_outer_ledger(path: Path, *, maximum: int) -> OuterQueryLedger:
    if maximum <= 0:
        raise FullCampaignError("a scientific campaign requires at least one outer slot")
    try:
        if path.exists() or path.is_symlink():
            return OuterQueryLedger.open(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent = path.parent.lstat()
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise FullCampaignError("outer-query ledger parent must be a real directory")
        return OuterQueryLedger.create(path, max_queries=maximum)
    except OSError as exc:
        raise FullCampaignError("outer-query ledger is unavailable") from exc


def _may_attempt_outer_evaluation(
    *,
    outer_queries_remaining: int,
    reusable_receipt_count: int,
) -> bool:
    """Allow an exhausted campaign to reach exact receipt lookup, never a new score call."""

    if type(outer_queries_remaining) is not int or outer_queries_remaining < 0:
        raise FullCampaignError("remaining outer-query count must be a non-negative integer")
    if type(reusable_receipt_count) is not int or reusable_receipt_count < 0:
        raise FullCampaignError("reusable receipt count must be a non-negative integer")
    return outer_queries_remaining > 0 or reusable_receipt_count > 0


def _reusable_score_receipt_count(path: Path, *, maximum: int) -> int:
    with _open_outer_ledger(path, maximum=maximum) as ledger:
        return len(ScoringReceiptBook.from_projection(ledger.projection()))


def _outer_ledger_location(
    *,
    run_dir: Path,
    project_root: Path,
    configured_path: Path | None,
) -> Path:
    return (
        run_dir.parent / "outer-query-ledger.sqlite3"
        if configured_path is None
        else configured_path
        if configured_path.is_absolute()
        else project_root / configured_path
    ).absolute()


@dataclass(frozen=True, slots=True)
class _OuterResumeAuthorization:
    request_digest: str
    missing_seeds: tuple[int, ...]


def _outer_resume_authorization(
    *,
    ledger_path: Path,
    request: CampaignCreateRequest,
    scorer_digest: str,
    campaign_store: CampaignStore,
) -> _OuterResumeAuthorization | None:
    """Verify whether a durable public reservation authorizes matched-seed completion."""

    local_used = campaign_store.snapshot().outer_queries_used
    if not ledger_path.exists() and not ledger_path.is_symlink():
        if local_used != 0:
            raise FullCampaignError(
                "campaign records an outer query but its project ledger is absent"
            )
        return None
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise FullCampaignError("outer-query ledger must be a real regular file")
    with OuterQueryLedger.open(ledger_path, read_only=True) as ledger:
        projection = ledger.projection()
    if projection.max_queries != request.config.validation.outer_promotion_limit:
        raise FullCampaignError("outer-query ledger limit differs from the campaign request")
    matches = tuple(item for item in projection.queries if item.campaign_id == request.campaign_id)
    if len(matches) > 1 or local_used > 1:
        raise FullCampaignError("provider-free campaign has multiple public outer reservations")
    if not matches:
        if local_used != 0:
            raise FullCampaignError(
                "campaign-local outer reservation disappeared from project ledger"
            )
        return None
    item = matches[0]
    metadata = item.reservation_metadata
    expected_metadata_fields = {
        "schema_version",
        "kind",
        "request_digest",
        "request",
        "candidate_id",
        "candidate_fingerprint",
        "source_digest",
        "parent_source_digest",
        "executable_diff_digest",
        "material_change_digest",
        "controller_attestation_digest",
        "consumes_slot",
    }
    request_manifest = metadata.get("request")
    if (
        set(metadata) != expected_metadata_fields
        or metadata.get("schema_version") != 1
        or metadata.get("kind") != "scientific_outer_promotion_reservation"
        or not isinstance(request_manifest, Mapping)
    ):
        raise FullCampaignError("outer reservation metadata is not the scientific contract")
    manifest_fields = set(OuterPromotionRequest.__dataclass_fields__) - {"digest"}
    if set(request_manifest) != manifest_fields | {"schema_version"}:
        raise FullCampaignError("outer promotion request manifest fields are not exact")
    if request_manifest.get("schema_version") != 1:
        raise FullCampaignError("outer promotion request schema is unsupported")
    try:
        promotion = OuterPromotionRequest(
            **{name: request_manifest[name] for name in manifest_fields}
        )
    except (TypeError, ValueError) as exc:
        raise FullCampaignError("outer promotion request manifest is invalid") from exc
    identity = campaign_store.identity()
    if (
        metadata.get("request_digest") != promotion.digest
        or promotion.campaign_digest != identity.config_digest
        or promotion.candidate_id != metadata.get("candidate_id")
        or promotion.candidate_fingerprint != item.candidate_fingerprint
        or promotion.benchmark_digest != request.benchmark_digest
        or promotion.dataset_digest != request.dataset_manifest_digest
        or promotion.scorer_digest != scorer_digest
        or item.benchmark_digest != request.benchmark_digest
        or item.dataset_digest != request.dataset_manifest_digest
        or item.scorer_digest != scorer_digest
        or item.state not in {"RESERVED", "COMPLETED"}
    ):
        raise FullCampaignError("outer reservation identity differs from the campaign")
    for name in (
        "candidate_fingerprint",
        "source_digest",
        "parent_source_digest",
        "executable_diff_digest",
        "material_change_digest",
        "controller_attestation_digest",
    ):
        if metadata.get(name) != getattr(promotion, name):
            raise FullCampaignError(f"outer reservation {name} differs from its request")
    outer_train = tuple(
        execution
        for execution in campaign_store.executions()
        if execution.kind == "generated_candidate_train"
        and execution.tier == "train"
        and execution.source_digest == promotion.source_digest
    )
    if any(execution.seed not in {0, 1, 2} for execution in outer_train):
        raise FullCampaignError("outer matched-seed execution has an unexpected seed")
    if len({execution.seed for execution in outer_train}) != len(outer_train):
        raise FullCampaignError("outer matched-seed execution identity is duplicated")
    observed_seeds = {cast(int, execution.seed) for execution in outer_train}
    return _OuterResumeAuthorization(
        request_digest=promotion.digest,
        missing_seeds=tuple(seed for seed in (0, 1, 2) if seed not in observed_seeds),
    )


def _validate_locked_runtime(
    project_root: Path,
    request: CampaignCreateRequest,
) -> None:
    if benchmark_digest() != request.benchmark_digest:
        raise FullCampaignError("frozen benchmark contract differs from the campaign request")
    environment = capture_environment_identity(project_root)
    if environment.digest != request.environment_digest:
        raise FullCampaignError("locked runtime environment differs from campaign creation")
    source = hash_source_tree(project_root)
    if source.digest != request.source_digest:
        raise FullCampaignError("trusted source tree differs from campaign creation")


def run_provider_free_campaign(
    run_dir: Path,
    *,
    project_root: Path,
    engine: CampaignEngine | None = None,
    outer_ledger_path: Path | None = None,
    cancel_event: Event | None = None,
) -> FullCampaignOutcome:
    """Run or restart the bounded local campaign through a strict finalization handoff.

    The function never constructs a final-outcome capability.  Its only final-period value is the
    canonical label-free input table, which is transformed alongside public validation by one
    train-frozen causal feature build.  Every train/evaluate child crosses a journal immediately
    after a fresh chained deadline observation.
    """

    if not isinstance(run_dir, Path) or not isinstance(project_root, Path):
        raise FullCampaignError("run_dir and project_root must be pathlib.Path values")
    if cancel_event is not None and not isinstance(cancel_event, Event):
        raise FullCampaignError("cancel_event must be threading.Event or None")
    _check_cancel(cancel_event)
    selected_engine = CampaignEngine() if engine is None else engine
    if not isinstance(selected_engine, CampaignEngine):
        raise FullCampaignError("engine must be a CampaignEngine")
    try:
        root = project_root.resolve(strict=True)
        run = run_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FullCampaignError("project or campaign run directory is unavailable") from exc
    if root.is_symlink() or not root.is_dir() or run.is_symlink() or not run.is_dir():
        raise FullCampaignError("project and campaign run roots must be real directories")

    request = selected_engine.load_request(run)
    if request.run_dir != run:
        raise FullCampaignError("campaign request run directory differs from the selected run")
    progress = FullCampaignProgressLedger(run / _PRODUCTION_DIR / "progress")
    checkpoints = progress.checkpoints()
    if checkpoints and checkpoints[-1].stage is FullCampaignStage.FINALIZATION_REQUIRED:
        return load_full_campaign_outcome(run, engine=selected_engine)

    _check_cancel(cancel_event)
    _validate_locked_runtime(root, request)
    status = selected_engine.resume(run)
    if status.status in {CampaignState.COMPLETED.value, CampaignState.INCOMPLETE.value}:
        raise FullCampaignError(f"campaign is terminal and cannot run research: {status.status}")
    _check_cancel(cancel_event)

    data_dir = _resolve_directory(root, request.config.benchmark.data_dir, "official data")
    starter_dir = _resolve_directory(
        root,
        request.config.benchmark.starter_dir,
        "organizer starter kit",
    )
    starter = verify_starter_kit(starter_dir)
    if starter.manifest_sha256 != request.starter_manifest_digest:
        raise FullCampaignError("organizer starter identity differs from campaign creation")

    dataset = load_canonical_dataset(data_dir)
    data = prepare_campaign_data_plane(
        dataset,
        expected_dataset_digest=request.dataset_manifest_digest,
    )
    valid_targets = dataset.valid.targets
    if not isinstance(valid_targets, ProtectedTargets):
        raise FullCampaignError("public validation targets are not protected")
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.DATA_PREPARED,
        evidence={
            "data_plane_digest": data.digest,
            "dataset_digest": dataset.digest,
            "train_rows": len(data.outer_train_inputs),
            "validation_rows": len(data.outer_validation_inputs),
            "final_rows": len(data.final_inputs),
            "final_target_capability": None,
            "final_outcome_parsed_fields": list(dataset.final.outcome_trace.parsed_fields),
            "final_outcome_parsed_cell_count": (dataset.final.outcome_trace.parsed_cell_count),
        },
    )

    outer_scorer = _outer_scorer(
        starter_dir=starter_dir,
        inputs=data.outer_validation_inputs,
        targets=valid_targets,
        split_token=data.outer_validation_inputs.digest,
    )
    scorer_digest = outer_scorer.scorer.scorer_digest
    qualification = load_official_fm_qualification(
        request.qualification_run_dir,
        expectations=QualificationExpectations(
            canonical_digest=dataset.digest,
            starter_manifest_digest=starter.manifest_sha256,
            scorer_digest=scorer_digest,
            validation_row_count=len(data.outer_validation_inputs),
            final_row_count=len(data.final_inputs),
        ),
    )
    if (
        qualification.manifest_digest != request.qualification_manifest_digest
        or qualification.benchmark_digest != request.benchmark_digest
    ):
        raise FullCampaignError("official FM qualification differs from campaign creation")
    _stage(
        progress,
        request_digest=request.digest,
        stage=FullCampaignStage.QUALIFICATION_VERIFIED,
        evidence={
            "qualification_manifest_digest": qualification.manifest_digest,
            "qualification_input_digest": qualification.qualification_input_digest,
            "audit_digest": qualification.audit_digest,
            "scorer_digest": scorer_digest,
            "outer_seeds": [item.seed for item in qualification.outer_runs],
            "fallback_manifest_digest": qualification.fallback.manifest_digest,
            "final_outcomes_scored": 0,
        },
    )

    artifacts = ArtifactStore(run / "artifacts")
    production = run / _PRODUCTION_DIR
    production.mkdir(parents=True, exist_ok=True, mode=0o700)
    fold_control_root = production / "fold-controls"
    candidate_control_root = production / "candidate-control"
    fold_control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate_control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    workspaces = WorkspaceMaterializer(
        production / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(),
    )
    candidate_limits = _limits(request)
    fold_limits = replace(candidate_limits, device="cpu")
    ledger_path = _outer_ledger_location(
        run_dir=run,
        project_root=root,
        configured_path=outer_ledger_path,
    )

    with CampaignStore.open(
        run / CAMPAIGN_DATABASE_NAME,
        campaign_id=request.campaign_id,
    ) as campaign_store:
        identity = campaign_store.identity()
        if (
            identity.config_digest != request.config.digest
            or identity.benchmark_digest != request.benchmark_digest
            or identity.dataset_manifest_digest != dataset.digest
            or identity.environment_digest != request.environment_digest
        ):
            raise FullCampaignError("campaign database identity differs from the request")

        retained_checkpoints = progress.checkpoints()
        retained_lineage = next(
            (
                checkpoint
                for checkpoint in retained_checkpoints
                if checkpoint.stage is FullCampaignStage.LINEAGE_READY
            ),
            None,
        )
        retained_fold = next(
            (
                checkpoint
                for checkpoint in retained_checkpoints
                if checkpoint.stage is FullCampaignStage.FOLD_CONTROLS_READY
            ),
            None,
        )
        closure_receipt = None
        if retained_fold is not None:
            value = retained_fold.evidence.get("fallback_receipt_digest")
            if type(value) is not str or len(value) != 64:
                raise FullCampaignError("retained fold stage lacks its fallback receipt")
            closure_receipt = value
        outer_resume = _outer_resume_authorization(
            ledger_path=ledger_path,
            request=request,
            scorer_digest=scorer_digest,
            campaign_store=campaign_store,
        )
        if retained_lineage is not None:
            retained_start = retained_lineage.evidence.get("scientific_start")
            if not isinstance(retained_start, Mapping):
                raise FullCampaignError("retained lineage lacks its scientific start cursor")
            start_launches = retained_start.get("launches_already_used")
            current_launches = campaign_store.snapshot().launches_used
            if type(start_launches) is not int or not 0 <= start_launches <= current_launches:
                raise FullCampaignError("retained scientific launch cursor moved backwards")
        current_deadline = selected_engine.inspect_deadline(run)
        if (
            status.finalization_required or current_deadline.finalization_reserve_active
        ) and outer_resume is None:
            return _research_admission_closed_outcome(
                reason="finalization_reserve_active_before_fold_B",
                run_dir=run,
                request=request,
                engine=selected_engine,
                campaign_store=campaign_store,
                artifacts=artifacts,
                progress=progress,
                qualification=qualification,
                scorer_digest=scorer_digest,
                fallback_receipt_digest=closure_receipt,
            )

        _check_cancel(cancel_event)
        try:
            fold_b = _fold_control(
                fold_name="B",
                fold_token=data.fold_b.fold.digest,
                prefix_inputs=data.fold_b.prefix_inputs,
                prefix_labels=data.fold_b.prefix_labels,
                query_inputs=data.fold_b.query_inputs,
                query_labels=data.fold_b.query_labels,
                engine=selected_engine,
                run_dir=run,
                campaign_store=campaign_store,
                artifacts=artifacts,
                workspaces=workspaces,
                starter_dir=starter_dir,
                limits=fold_limits,
                cancel_event=cancel_event,
            )
        except (CandidateAdmissionError, SupervisedFoldFMExecutionError) as exc:
            _check_cancel(cancel_event)
            return _research_admission_closed_outcome(
                reason="fold_B_control_branch_failed",
                run_dir=run,
                request=request,
                engine=selected_engine,
                campaign_store=campaign_store,
                artifacts=artifacts,
                progress=progress,
                qualification=qualification,
                scorer_digest=scorer_digest,
                fallback_receipt_digest=closure_receipt,
                diagnostic_category="fold_control_failed",
                fold_b_status="failed",
                diagnostic_details=_fold_branch_failure_details(exc, artifacts=artifacts),
            )
        if (
            selected_engine.inspect_deadline(run).finalization_reserve_active
            and outer_resume is None
        ):
            return _research_admission_closed_outcome(
                reason="finalization_reserve_active_between_fold_B_and_fold_A",
                run_dir=run,
                request=request,
                engine=selected_engine,
                campaign_store=campaign_store,
                artifacts=artifacts,
                progress=progress,
                qualification=qualification,
                scorer_digest=scorer_digest,
                fold_b=fold_b,
                fallback_receipt_digest=closure_receipt,
            )
        _check_cancel(cancel_event)
        try:
            fold_a = _fold_control(
                fold_name="A",
                fold_token=data.fold_a.fold.digest,
                prefix_inputs=data.fold_a.prefix_inputs,
                prefix_labels=data.fold_a.prefix_labels,
                query_inputs=data.fold_a.query_inputs,
                query_labels=data.fold_a.query_labels,
                engine=selected_engine,
                run_dir=run,
                campaign_store=campaign_store,
                artifacts=artifacts,
                workspaces=workspaces,
                starter_dir=starter_dir,
                limits=fold_limits,
                cancel_event=cancel_event,
            )
        except (CandidateAdmissionError, SupervisedFoldFMExecutionError) as exc:
            _check_cancel(cancel_event)
            return _research_admission_closed_outcome(
                reason="fold_A_control_branch_failed",
                run_dir=run,
                request=request,
                engine=selected_engine,
                campaign_store=campaign_store,
                artifacts=artifacts,
                progress=progress,
                qualification=qualification,
                scorer_digest=scorer_digest,
                fold_b=fold_b,
                fallback_receipt_digest=closure_receipt,
                diagnostic_category="fold_control_failed",
                fold_a_status="failed",
                diagnostic_details=_fold_branch_failure_details(exc, artifacts=artifacts),
            )
        fallback_receipt = _fallback_receipt(qualification, fold_a, fold_b)
        _stage(
            progress,
            request_digest=request.digest,
            stage=FullCampaignStage.FOLD_CONTROLS_READY,
            evidence={
                "fold_a_evidence_digest": fold_a.evidence.digest,
                "fold_b_evidence_digest": fold_b.evidence.digest,
                "fold_a_metrics": {
                    "GAUC": fold_a.control.metrics.gauc,
                    "nDCG@5": fold_a.control.metrics.ndcg_at_5,
                    "primary": fold_a.control.metrics.primary,
                },
                "fold_b_metrics": {
                    "GAUC": fold_b.control.metrics.gauc,
                    "nDCG@5": fold_b.control.metrics.ndcg_at_5,
                    "primary": fold_b.control.metrics.primary,
                },
                "fallback_receipt_digest": fallback_receipt,
                "official_fm_cpu_controls": True,
            },
        )
        if (
            selected_engine.inspect_deadline(run).finalization_reserve_active
            and outer_resume is None
        ):
            return _research_admission_closed_outcome(
                reason="finalization_reserve_active_before_generated_research",
                run_dir=run,
                request=request,
                engine=selected_engine,
                campaign_store=campaign_store,
                artifacts=artifacts,
                progress=progress,
                qualification=qualification,
                scorer_digest=scorer_digest,
                fold_b=fold_b,
                fallback_receipt_digest=fallback_receipt,
            )

        prior_feature_stage = next(
            (
                checkpoint.evidence
                for checkpoint in progress.checkpoints()
                if checkpoint.stage is FullCampaignStage.FEATURES_READY
            ),
            None,
        )
        cache_key = _digest(
            b"kuairand-production-feature-cache-key-v1",
            {
                "data_plane_digest": data.digest,
                "builder_source_digest": request.source_digest,
            },
        )
        cache_dir = production / "feature-cache" / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if cache_dir.is_symlink() or not cache_dir.is_dir():
            raise FullCampaignError("causal feature cache must be a real run-local directory")
        features, cache_receipt = _feature_cache_build(
            data=data,
            builder_source_digest=request.source_digest,
            cache_dir=cache_dir,
            prior_stage=prior_feature_stage,
        )
        feature_artifacts = _put_feature_artifacts(
            artifacts,
            features,
            fold_a_targets=data.fold_a.prefix_labels,
            fold_a_users=data.fold_a.prefix_inputs.user_id,
            fold_b_targets=data.fold_b.prefix_labels,
            fold_b_users=data.fold_b.prefix_inputs.user_id,
            outer_targets=data.outer_train_labels,
            outer_users=data.outer_train_inputs.user_id,
        )
        _stage(
            progress,
            request_digest=request.digest,
            stage=FullCampaignStage.FEATURES_READY,
            evidence={
                "feature_bundle_digest": features.digest,
                "feature_artifacts": feature_artifacts.manifest(),
                "feature_count": features.final.feature_count,
                "cache_key": cache_key,
                "cache_performance": dict(cache_receipt),
                "final_target_capability": None,
            },
        )

        status = selected_engine.status(run)
        context_evidence = _research_context_evidence(data, features)
        if retained_lineage is None:
            safe_context = _safe_context(
                request=request,
                qualification=qualification,
                fold_a=fold_a,
                fold_b=fold_b,
                status=status,
                evidence=context_evidence,
            )
        else:
            retained_context = retained_lineage.evidence.get("safe_context")
            retained_context_digest = retained_lineage.evidence.get("safe_context_digest")
            if (
                not isinstance(retained_context, Mapping)
                or type(retained_context_digest) is not str
            ):
                raise FullCampaignError("retained lineage lacks its frozen safe research context")
            context_copy = json.loads(_canonical_json(dict(retained_context)))
            if not isinstance(context_copy, dict):
                raise FullCampaignError("retained safe research context is malformed")
            safe_context = SafeResearchContext(
                MappingProxyType(context_copy),
                retained_context_digest,
            )

        _check_cancel(cancel_event)
        research_model: ResearchModel | None = None
        accepted_scientific_iteration = 1
        rejected_lineage_records: tuple[AggregateRecord, ...] = ()
        if request.config.research.provider == "scripted":
            template_dir = _resolve_directory(
                root,
                Path("candidate_templates/lambdarank"),
                "generated LambdaRank template",
            )
            lineage: ScriptedLambdaRankLineage | LiveResearchLineage = (
                prepare_or_rehydrate_scripted_lambdarank_lineage(
                    campaign_id=request.campaign_id,
                    scientific_iteration=1,
                    template_dir=template_dir,
                    generated_root=production / "generated-source",
                    artifact_store=artifacts,
                    safe_context=safe_context,
                )
            )
            candidate_id = SCRIPTED_CANDIDATE_ID
        else:

            def remaining_research_seconds() -> float:
                _check_cancel(cancel_event)
                observed = selected_engine.inspect_deadline(run)
                return max(
                    0.0,
                    observed.remaining_seconds - request.config.runner.finalization_reserve_seconds,
                )

            retry_runtime = RetryRuntime(remaining_research_seconds=remaining_research_seconds)
            persist_provider_transcript = _provider_transcript_sink(production)

            def build_openai_model(
                config: OpenAIChatCompletionsConfig,
                transport: ResponsesTransport | None,
            ) -> object:
                return OpenAIChatCompletionsModel(
                    config,
                    transport=transport,
                    retry_runtime=retry_runtime,
                    model_limit_resolver=ProviderModelLimitResolver(),
                    transcript_sink=persist_provider_transcript,
                )

            selected_provider = select_research_provider(
                request.config.research,
                openai_model_factory=build_openai_model,
            )
            if isinstance(selected_provider, ProviderUnavailableDiagnostic):
                return _provider_unavailable_outcome(
                    diagnostic=selected_provider,
                    run_dir=run,
                    request=request,
                    engine=selected_engine,
                    campaign_store=campaign_store,
                    artifacts=artifacts,
                    progress=progress,
                    qualification=qualification,
                    scorer_digest=scorer_digest,
                    fallback_receipt_digest=fallback_receipt,
                )
            if not isinstance(selected_provider, AvailableResearchProvider):
                raise FullCampaignError("research provider selection is malformed")
            if not selected_provider.live_provider_used:
                raise FullCampaignError("autonomous campaign requires a live research provider")
            research_model = selected_provider.model
            parent = load_parent_snapshot(
                _resolve_directory(root, Path("candidate_seed"), "candidate seed"),
                candidate_id=SCRIPTED_PARENT_ID,
            )

            def continue_live_lineage() -> bool:
                _check_cancel(cancel_event)
                return not selected_engine.inspect_deadline(run).finalization_reserve_active

            preparation = _prepare_live_lineage_portfolio(
                campaign_id=request.campaign_id,
                parent=parent,
                generated_root=production / "generated-source",
                artifact_store=artifacts,
                model=research_model,
                provider=selected_provider.provider,
                maximum_iterations=request.config.benchmark.max_iterations,
                safe_context_factory=(
                    (lambda _records: safe_context)
                    if retained_lineage is not None
                    else lambda records: _safe_context(
                        request=request,
                        qualification=qualification,
                        fold_a=fold_a,
                        fold_b=fold_b,
                        status=selected_engine.status(run),
                        campaign_records=records,
                        evidence=context_evidence,
                    )
                ),
                continue_check=continue_live_lineage,
            )
            if preparation.status == "provider_unavailable":
                if not isinstance(
                    research_model, (OpenAIChatCompletionsModel, OpenAIFailoverModel)
                ):
                    raise FullCampaignError(
                        "non-OpenAI research model returned an OpenAI provider failure"
                    )
                if preparation.provider_error is None:
                    raise FullCampaignError("provider-unavailable preparation lost its error")
                return _runtime_provider_unavailable_outcome(
                    error=preparation.provider_error,
                    model=research_model,
                    run_dir=run,
                    request=request,
                    engine=selected_engine,
                    campaign_store=campaign_store,
                    artifacts=artifacts,
                    progress=progress,
                    qualification=qualification,
                    scorer_digest=scorer_digest,
                    fallback_receipt_digest=fallback_receipt,
                    rejected_records=preparation.rejected_records,
                    branches_attempted=preparation.branches_attempted,
                )
            if preparation.status != "accepted":
                reason = (
                    "repeated_pre_admission_failure"
                    if preparation.status == "repeated_pre_admission_failure"
                    else "live_lineage_portfolio_exhausted_or_reserve_active"
                )
                return _research_admission_closed_outcome(
                    reason=reason,
                    run_dir=run,
                    request=request,
                    engine=selected_engine,
                    campaign_store=campaign_store,
                    artifacts=artifacts,
                    progress=progress,
                    qualification=qualification,
                    scorer_digest=scorer_digest,
                    fold_b=fold_b,
                    fallback_receipt_digest=fallback_receipt,
                    diagnostic_category="live_lineage_portfolio_closed",
                    fold_a_status="completed",
                    diagnostic_details={
                        "maximum_iterations": request.config.benchmark.max_iterations,
                        "preparation_status": preparation.status,
                    },
                    research_model=research_model,
                    rejected_records=preparation.rejected_records,
                    branches_attempted=preparation.branches_attempted,
                )
            if (
                preparation.lineage is None
                or preparation.safe_context is None
                or preparation.scientific_iteration is None
            ):
                raise FullCampaignError("accepted live-lineage preparation is incomplete")
            lineage = preparation.lineage
            safe_context = preparation.safe_context
            accepted_scientific_iteration = preparation.scientific_iteration
            rejected_lineage_records = preparation.rejected_records
            candidate_id = lineage.candidate_id
        research_ledger, experiment_id = _ensure_lineage_ledger(
            campaign_store=campaign_store,
            lineage=lineage,
            candidate_id=candidate_id,
            scientific_iteration=accepted_scientific_iteration,
        )
        observed_start = selected_engine.inspect_deadline(run)
        lineage_stage_counts = _research_stage_counts(
            model=research_model,
            branches_attempted=accepted_scientific_iteration,
            rejected_records=rejected_lineage_records,
            candidates_admitted=1,
        )
        lineage_rejection_evidence = _research_rejection_evidence(
            rejected_lineage_records,
            artifacts=artifacts,
        )
        lineage_stage = _stage(
            progress,
            request_digest=request.digest,
            stage=FullCampaignStage.LINEAGE_READY,
            evidence={
                "lineage": lineage.manifest(),
                "source_snapshot": lineage.source_snapshot.manifest(),
                "material_generated_source": True,
                "manual_source_edits": 0,
                "provider": lineage.provider,
                "live_provider_used": lineage.live_provider_used,
                "provider_usage": _provider_usage(research_model),
                "research_stage_counts": lineage_stage_counts,
                **lineage_rejection_evidence,
                "safe_context": safe_context.to_wire(),
                "safe_context_digest": safe_context.digest,
                "scientific_start": {
                    "launches_already_used": campaign_store.snapshot().launches_used,
                    "elapsed_seconds_at_start": observed_start.elapsed_seconds,
                    "wall_clock_seconds": request.config.benchmark.wall_clock_seconds,
                    "finalization_reserve_seconds": (
                        request.config.runner.finalization_reserve_seconds
                    ),
                },
            },
        )
        scientific_start = lineage_stage.get("scientific_start")
        if not isinstance(scientific_start, Mapping) or set(scientific_start) != {
            "launches_already_used",
            "elapsed_seconds_at_start",
            "wall_clock_seconds",
            "finalization_reserve_seconds",
        }:
            raise FullCampaignError("retained lineage lacks an exact scientific start cursor")
        launches_at_start = scientific_start["launches_already_used"]
        elapsed_at_start = scientific_start["elapsed_seconds_at_start"]
        if type(launches_at_start) is not int or not 0 <= launches_at_start <= 50:
            raise FullCampaignError("retained scientific launch cursor is invalid")
        if (
            isinstance(elapsed_at_start, bool)
            or not isinstance(elapsed_at_start, (int, float))
            or not np.isfinite(float(elapsed_at_start))
            or not 0.0 <= float(elapsed_at_start) <= request.config.benchmark.wall_clock_seconds
        ):
            raise FullCampaignError("retained scientific deadline cursor is invalid")
        if (
            scientific_start["wall_clock_seconds"] != request.config.benchmark.wall_clock_seconds
            or scientific_start["finalization_reserve_seconds"]
            != request.config.runner.finalization_reserve_seconds
        ):
            raise FullCampaignError("retained scientific budget differs from the request")

        candidate = _generated_scientific_candidate(
            request=request,
            data=data,
            features=features,
            lineage=lineage,
            candidate_id=candidate_id,
            candidate_limits=candidate_limits,
            scientific_iteration=accepted_scientific_iteration,
        )
        fold_a_data_digest = _digest(
            b"kuairand-production-fold-A-science-v1",
            {
                "fold_digest": data.fold_a.fold.digest,
                "feature_pair_digest": features.fold_a.digest,
                "fm_control_digest": fold_a.evidence.digest,
            },
        )
        fold_b_data_digest = _digest(
            b"kuairand-production-fold-B-science-v1",
            {
                "fold_digest": data.fold_b.fold.digest,
                "feature_pair_digest": features.fold_b.digest,
                "fm_control_digest": fold_b.evidence.digest,
            },
        )
        outer_data_digest = _digest(
            b"kuairand-production-outer-science-v1",
            {
                "training_inputs_digest": data.outer_train_inputs.digest,
                "validation_inputs_digest": data.outer_validation_inputs.digest,
                "feature_pair_digest": features.outer_and_final.digest,
                "scorer_digest": scorer_digest,
            },
        )
        scientific_config = ScientificCampaignConfig(
            benchmark_digest=request.benchmark_digest,
            dataset_digest=dataset.digest,
            scorer_digest=scorer_digest,
            fold_a_data_digest=fold_a_data_digest,
            fold_b_data_digest=fold_b_data_digest,
            outer_data_digest=outer_data_digest,
            environment_digest=request.environment_digest,
            campaign_digest=identity.config_digest,
            qualified_fallback_receipt_digest=fallback_receipt,
            max_scientific_iterations=request.config.benchmark.max_iterations,
            launches_already_used=launches_at_start,
            screen_margin=0.0,
            elapsed_seconds_at_start=float(elapsed_at_start),
            wall_clock_seconds=request.config.benchmark.wall_clock_seconds,
            finalization_reserve_seconds=(request.config.runner.finalization_reserve_seconds),
        )
        executor = GeneratedCandidateExecutor(
            artifact_store=artifacts,
            workspace_materializer=workspaces,
            control_root=candidate_control_root,
            interpreter=active_python_interpreter(),
            limits=candidate_limits,
        )
        runtime = _ScientificRuntime(
            engine=selected_engine,
            run_dir=run,
            campaign_store=campaign_store,
            artifacts=artifacts,
            executor=executor,
            lineage=lineage,
            candidate=candidate,
            experiment_id=experiment_id,
            scientific_iteration=accepted_scientific_iteration,
            config=scientific_config,
            feature_artifacts=feature_artifacts,
            features=features,
            fold_a=fold_a,
            fold_b=fold_b,
            fold_a_query_inputs=data.fold_a.query_inputs,
            fold_b_query_inputs=data.fold_b.query_inputs,
            fold_a_scorer=build_fold_scoring_context(
                starter_dir,
                "A",
                data.fold_a.fold.digest,
                data.fold_a.query_inputs,
                data.fold_a.query_labels,
            ),
            fold_b_scorer=build_fold_scoring_context(
                starter_dir,
                "B",
                data.fold_b.fold.digest,
                data.fold_b.query_inputs,
                data.fold_b.query_labels,
            ),
            outer_scorer=outer_scorer,
            outer_admission=None,
            qualification=qualification,
            repository=FileScientificRunEvidenceRepository(production / "scientific-records"),
            evidence_registry={},
            cancel_event=cancel_event,
            records={},
        )
        fallback_incumbent = _fallback_incumbent(
            qualification,
            fold_a,
            fold_b,
            fallback_receipt,
        )
        with _open_outer_ledger(
            ledger_path,
            maximum=request.config.validation.outer_promotion_limit,
        ) as project_ledger:
            adapter = DurableScientificLedgerAdapter(
                campaign_store,
                project_ledger,
                scorer_digest=scorer_digest,
                evidence_registry=runtime.evidence_registry,
                representative_seed=0,
            )
            admission = ReceiptAwareOuterEvaluationLedger(
                adapter,
                ScoringReceiptBook.from_projection(project_ledger.projection()),
            )
            runtime.outer_admission = admission
            try:
                result = run_scientific_campaign(
                    config=scientific_config,
                    fallback=fallback_incumbent,
                    candidates=(candidate,),
                    runner=runtime,
                    outer_ledger=admission,
                )
            except ScientificCampaignCancelled as exc:
                raise FullCampaignCancelled(
                    "campaign cancelled during scientific execution; durable child evidence "
                    "is resumable and no scientific success was published"
                ) from exc

        # Candidate callbacks deliberately contain ordinary branch failures inside the
        # scientific loop.  Cancellation is controller intent, however, and the shared Event
        # remains the authoritative cross-process signal even if a callback boundary reports a
        # contained failure.  Check it before publishing any scientific-success closure.
        _check_cancel(cancel_event)
        observed_launches = campaign_store.snapshot().launches_used
        if observed_launches != result.launches_used:
            raise FullCampaignError(
                "scientific launch cursor differs from authoritative campaign accounting"
            )
        campaign_store.set_convergence_state(
            result.convergence.manifest(),
            expected_revision=campaign_store.snapshot().revision,
            reason="persist provider-free scientific convergence cursor",
        )
        _check_cancel(cancel_event)
        selection = _candidate_selection(
            experiment_id=experiment_id,
            result=result,
            runtime=runtime,
            qualification=qualification,
            features=features,
            feature_artifacts=feature_artifacts,
            dataset_digest=dataset.digest,
            validation_inputs=data.outer_validation_inputs,
            final_inputs=data.final_inputs,
            limits=candidate_limits,
        )
        _check_cancel(cancel_event)
        reflection_request, reflection_response, reflection_transcript, reflection = _reflect(
            campaign_store=campaign_store,
            research_ledger=research_ledger,
            lineage=lineage,
            result=result,
            safe_context=safe_context,
            research_config=request.config.research,
            artifacts=artifacts,
            research_model=research_model,
            candidate_id=candidate_id,
            scientific_iteration=accepted_scientific_iteration,
        )
        _check_cancel(cancel_event)
        iterations_completed = accepted_scientific_iteration
        if isinstance(lineage, LiveResearchLineage):
            if research_model is None:
                raise FullCampaignError("live lineage lost its research model")
            followup = _run_autonomous_followups(
                request=request,
                data=data,
                runtime_template=runtime,
                scientific_config=scientific_config,
                fallback=fallback_incumbent,
                outer_ledger_path=ledger_path,
                candidate_limits=candidate_limits,
                dataset_digest=dataset.digest,
                context_evidence=context_evidence,
                validation_inputs=data.outer_validation_inputs,
                final_inputs=data.final_inputs,
                research_model=research_model,
                first_lineage=lineage,
                first_result=result,
                first_selection=selection,
                first_reflection=reflection,
                first_reflection_evidence=(
                    reflection_request,
                    reflection_response,
                    reflection_transcript,
                ),
                prior_records=rejected_lineage_records,
            )
            result = followup.result
            selection = followup.selection
            reflection_request = followup.reflection_request_digest
            reflection_response = followup.reflection_response_digest
            reflection_transcript = followup.reflection_transcript
            iterations_completed = followup.iterations_completed
            rejected_lineage_records = followup.rejected_records
        observed_launches = campaign_store.snapshot().launches_used
        if observed_launches != result.launches_used:
            raise FullCampaignError(
                "final scientific launch cursor differs from authoritative campaign accounting"
            )
        result_artifact = artifacts.put_bytes(
            _canonical_json(result.manifest() | {"digest": result.digest}),
            kind=ArtifactKind.MANIFEST,
        )
        inner_evaluations_completed = sum(
            run.metrics is not None
            for candidate_result in result.candidates
            for run in candidate_result.runs[:2]
        )
        outer_evaluations_completed = sum(
            run.metrics is not None
            for candidate_result in result.candidates
            for run in candidate_result.runs[2:]
        )
        final_stage_counts = _research_stage_counts(
            model=research_model,
            branches_attempted=iterations_completed,
            rejected_records=rejected_lineage_records,
            candidates_admitted=max(0, iterations_completed - len(rejected_lineage_records)),
            training_started=max(
                0,
                result.launches_used - scientific_config.launches_already_used,
            ),
            inner_evaluations_completed=inner_evaluations_completed,
            outer_evaluations_completed=outer_evaluations_completed,
        )
        final_rejection_evidence = _research_rejection_evidence(
            rejected_lineage_records,
            artifacts=artifacts,
        )
        _check_cancel(cancel_event)
        _stage(
            progress,
            request_digest=request.digest,
            stage=FullCampaignStage.SCIENCE_COMPLETE,
            evidence={
                "scientific_result_digest": result.digest,
                "scientific_result_artifact": result_artifact.manifest(),
                "incumbent_candidate_id": result.incumbent.candidate_id,
                "fallback_candidate_id": result.fallback.candidate_id,
                "fallback_preserved": True,
                "launches_used": result.launches_used,
                "stop_reason": result.stop_reason.value,
                "portfolio_count": iterations_completed,
                "portfolio_cap": request.config.benchmark.max_iterations,
                "iterations_completed": iterations_completed,
                "autonomous_loop_completed": isinstance(lineage, LiveResearchLineage),
                "provider_usage": _provider_usage(research_model),
                "research_stage_counts": final_stage_counts,
                **final_rejection_evidence,
            },
        )
        _check_cancel(cancel_event)
        _stage(
            progress,
            request_digest=request.digest,
            stage=FullCampaignStage.REFLECTED,
            evidence={
                "reflection_request_digest": reflection_request,
                "reflection_response_digest": reflection_response,
                "reflection_transcript": reflection_transcript.manifest(),
                "scientific_result_digest": result.digest,
                "selected_candidate_id": (None if selection is None else selection.candidate_id),
                "selection_digest": None if selection is None else selection.digest,
                "portfolio_count": iterations_completed,
                "portfolio_cap_reason": "exact_terminal_condition_reached",
                "manual_source_edits": 0,
                "manual_interventions": 0,
                "provider_usage": _provider_usage(research_model),
                "research_stage_counts": final_stage_counts,
                **final_rejection_evidence,
            },
        )
        _check_cancel(cancel_event)
        return _commit_outcome(
            run_dir=run,
            request=request,
            engine=selected_engine,
            campaign_store=campaign_store,
            artifacts=artifacts,
            progress=progress,
            qualification=qualification,
            scorer_digest=scorer_digest,
            fallback_receipt_digest=fallback_receipt,
            scientific_result_digest=result.digest,
            reflection_request_digest=reflection_request,
            reflection_response_digest=reflection_response,
            reflection_transcript=reflection_transcript,
            selection=selection,
        )


__all__ = ["run_provider_free_campaign"]

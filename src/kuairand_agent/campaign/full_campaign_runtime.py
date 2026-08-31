"""Restart-safe local composition for the bounded KuaiRand-Pure research campaign.

The module is intentionally controller-owned.  Generated source receives only immutable numeric
NPY capabilities; public labels remain closed over protected scorers and final outcomes have no
representable input anywhere in this call graph.
"""

from __future__ import annotations

import hashlib
import json
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
from typing import Final, Literal, Protocol, cast

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
from kuairand_agent.campaign.analysis import (
    AnalysisError,
    AnalysisKind,
    AnalysisQuery,
    TrainAnalysisInputs,
    run_requested_analyses,
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
from kuairand_agent.campaign.eda import (
    within_user_feature_diagnostics,
    within_user_label_structure,
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
from kuairand_agent.campaign.pure_features import ID_CODE_FEATURE_NAMES
from kuairand_agent.campaign.qualification_evidence import (
    OfficialFMQualificationEvidence,
    OfficialFMSeedEvidence,
    QualificationExpectations,
    load_official_fm_qualification,
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
from kuairand_agent.campaign.selector import (
    GateEvidence,
    IncumbentEvidence,
    OrganizerMetrics,
    SeedMetrics,
)
from kuairand_agent.campaign.store import (
    ArtifactSpec,
    CampaignStore,
    LineageFoldMetrics,
    OuterQueryLedger,
    ResearchLineageLedger,
)
from kuairand_agent.candidates.fusion import FUSION_WEIGHT_GRID
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
    AggregateScalar,
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
from kuairand_agent.research.interface import ResearchModel, ResearchModelError
from kuairand_agent.research.loop import CampaignStoreResearchLedger
from kuairand_agent.research.materialize import snapshot_materialized_candidate
from kuairand_agent.research.production import (
    SCRIPTED_CANDIDATE_ID,
    SCRIPTED_PARENT_ID,
    LiveResearchBranchRejected,
    LiveResearchLineage,
    ProductionResearchError,
    ScriptedLambdaRankLineage,
    load_parent_snapshot,
    prepare_or_rehydrate_live_lineage,
    prepare_or_rehydrate_scripted_lambdarank_lineage,
    proposal_family_of,
)
from kuairand_agent.research.production import _proposal_signature as _lineage_proposal_signature
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
    train_count = len(data.outer_train_labels)
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
                },
            ),
            # Train-only within-user diagnostics. Without these the agent proposes modelling
            # changes knowing only row counts and duration percentiles, which motivates no
            # hypothesis the briefing does not already name.
            *within_user_label_structure(
                labels=data.outer_train_labels,
                user_ids=data.outer_train_inputs.user_id,
            ),
            *within_user_feature_diagnostics(
                feature_names=features.outer_and_final.prefix.feature_names,
                feature_values=features.outer_and_final.prefix.values,
                labels=data.outer_train_labels,
                user_ids=data.outer_train_inputs.user_id,
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
                    "categorical_code_columns_csv": ",".join(ID_CODE_FEATURE_NAMES),
                    # Each fold fits its own vocabulary on its own training rows, so these sizes
                    # describe the outer build only.  A candidate must size any embedding table
                    # from the training matrix it is handed, never from these numbers.
                    "outer_code_cardinalities_csv": ",".join(
                        str(value) for value in features.outer_and_final.code_cardinalities
                    ),
                    "code_cardinalities_vary_by_fold": True,
                    # This was previously a bare boolean, which disclosed the mechanism without
                    # explaining it.  Candidate predictions are rank-blended with the official FM
                    # control, so the primary a candidate sees is the blend's, never its own.
                    "fold_b_selected_fusion_only": True,
                    "fusion_weight_grid_model_then_control_csv": ";".join(
                        f"{model}/{control}" for model, control in FUSION_WEIGHT_GRID
                    ),
                    "fusion_scored_metric_is_the_blend_not_the_model": True,
                    "fusion_weight_frozen_on_fold_b_and_reused_below": True,
                    "uses_public_labels_for_features": False,
                    "uses_prediction_period_outcomes": False,
                },
            ),
            # Without the cardinalities a candidate cannot size an embedding table, so the
            # identity columns would be unusable in practice even though they are present.
            AggregateRecord(
                "controller_identity_columns",
                {
                    "columns_csv": ",".join(ID_CODE_FEATURE_NAMES),
                    "outer_cardinalities_csv": ",".join(
                        str(value) for value in features.outer_and_final.code_cardinalities
                    ),
                    "encoding": (
                        "integer codes stored as float64 in the feature matrix; vocabularies are "
                        "fitted on training rows only and every value unseen in training resolves "
                        "to that field's final code, which is the UNK slot"
                    ),
                    # Each fold refits its vocabulary on its own training rows, so the cardinality
                    # above describes the outer build alone and is not a constant of the campaign.
                    "size_embeddings_from_the_training_matrix_not_this_number": True,
                    "intended_use": (
                        "learn an embedding per identity and cross it with the causal aggregate "
                        "columns; a feature constant within a user cannot reorder that user's own "
                        "impressions on its own, but becomes able to once crossed with an item "
                        "identity"
                    ),
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
            callback = cast(ProtectedScoreCallback, self.outer_scorer.score)
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
            promotion = OuterPromotionRequest(
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
            if evidence.metrics is None:
                raise FullCampaignError("outer generated run lacks protected aggregate metrics")
            trusted = TrustedOuterSeedEvidence(
                request_digest=promotion.digest,
                seed=request.seed,
                metrics=evidence.metrics,
                prediction_digest=record.scored_prediction_digest,
                score_evidence_digest=evidence.digest,
            )
            key = (promotion.digest, request.seed)
            prior = self.evidence_registry.get(key)
            if prior is not None and prior != trusted:
                raise FullCampaignError("outer trusted evidence retry is contradictory")
            self.evidence_registry[key] = trusted
        return evidence


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
    candidate_result = next(
        (
            item
            for item in result.candidates
            if item.candidate.candidate_id == runtime.candidate.candidate_id
        ),
        None,
    )
    if candidate_result is None or candidate_result.selection is None:
        raise FullCampaignError("generated incumbent lacks trusted selector evidence")
    records = runtime.records
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
    candidate_result = result.candidates[0] if result.candidates else None
    run = None
    if candidate_result is not None and candidate_result.runs:
        outer = [item for item in candidate_result.runs if item.metrics is not None]
        run = outer[-1] if outer else None
    # A branch that produced no run has no metrics.  Substituting the fallback's numbers here
    # told the model its crashed candidate had tied the baseline, and overnight-11's reflection
    # duly concluded a candidate that never executed was "indistinguishable from the official FM
    # reference".  Report zeros with the failure flag instead, so the summary cannot be read as a
    # measurement.
    execution_failed = run is None or run.metrics is None
    metrics = (
        _metrics(result.fallback.outer_by_seed[0].metrics)
        if run is None or run.metrics is None
        else run.metrics
    )
    promoted = result.incumbent.candidate_id == candidate_id
    summary = ExperimentResultSummary(
        tier="outer" if promoted else "inner",
        status="promoted" if promoted else "rejected",
        gauc=0.0 if execution_failed else metrics.gauc,
        ndcg_at_5=0.0 if execution_failed else metrics.ndcg_at_5,
        primary=0.0 if execution_failed else metrics.primary,
        execution_failed=execution_failed,
        runtime_seconds=0.0 if run is None else run.resources.wall_seconds,
        peak_memory_mb=(0.0 if run is None else run.resources.peak_rss_bytes / float(1024**2)),
    )
    request = ReflectionRequest.create(
        request_id=f"iteration-{scientific_iteration:02d}-reflect",
        proposal_id=lineage.proposal.proposal_id,
        candidate_id=candidate_id,
        source_digest=lineage.materialized.source_digest,
        diff_digest=lineage.materialized.diff_digest,
        result=summary,
        safe_context=safe_context.to_wire(),
    )
    scripted = Reflection(
        response_id="generated-causal-lambdarank-v1-reflection",
        summary=(
            "The bounded generated LambdaRank branch completed protected fold and matched-seed "
            "evaluation while preserving the qualified official-FM fallback."
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
    try:
        reflection = reflection_model.reflect(request)
    except ResearchModelError as exc:
        reflection = _unavailable_reflection(exc, scientific_iteration=scientific_iteration)
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


@dataclass(frozen=True, slots=True)
class _AutonomousFollowupResult:
    result: ScientificCampaignResult
    selection: FinalizationSelectionPlan | None
    reflection_request_digest: str
    reflection_response_digest: str
    reflection_transcript: ArtifactRef
    iterations_completed: int
    rejected_records: tuple[AggregateRecord, ...]


_MEASURED_TIER_RUN_COUNT: Final = 3


def _measured_primary(
    candidate_result: CandidateCampaignResult | None,
    incumbent: IncumbentEvidence,
) -> tuple[float | None, float | None, str | None]:
    """Return the candidate's measured primary, its delta, and the tier that produced it.

    The proposer sees only the records built here, so without this a direction that ran and
    scored is indistinguishable from one that was never tried.  Runs arrive in tier order:
    the Fold B screen, then the Fold A confirmation, then the matched outer seeds.
    """

    if candidate_result is None or not candidate_result.runs:
        return None, None, None
    scored = [run for run in candidate_result.runs if run.metrics is not None]
    if not scored:
        return None, None, None
    if len(scored) >= _MEASURED_TIER_RUN_COUNT:
        outer = scored[_MEASURED_TIER_RUN_COUNT - 1 :]
        primary = sum(float(run.metrics.primary) for run in outer if run.metrics) / len(outer)
        reference = sum(float(item.metrics.primary) for item in incumbent.outer_by_seed) / len(
            incumbent.outer_by_seed
        )
        return primary, primary - reference, "outer_matched_seed"
    metrics = scored[-1].metrics
    assert metrics is not None  # narrowed by the scored filter above
    folds = dict(incumbent.inner_by_fold)
    fold_b = folds.get("B")
    primary = float(metrics.primary)
    delta = None if fold_b is None else primary - float(fold_b.primary)
    return primary, delta, "inner_fold"


def _train_analysis_inputs(
    data: CampaignDataPlane,
    features: ProductionFeatureBundle,
) -> TrainAnalysisInputs:
    """Bind the training arrays an analysis may read, and only those.

    Built once at the call site where the feature bundle is in scope, so the train-only property is
    visible at the boundary rather than relying on the executor to be handed the right thing. The
    prefix matrix is the training half of the outer build; no validation or final-period array is
    reachable from the value this returns.
    """

    prefix = features.outer_and_final.prefix
    return TrainAnalysisInputs(
        feature_names=tuple(prefix.feature_names),
        feature_values=np.asarray(prefix.values, dtype=np.float64),
        labels=np.asarray(data.outer_train_labels, dtype=np.float64),
        user_ids=tuple(data.outer_train_inputs.user_id),
    )


def _answered_analyses(
    inputs: TrainAnalysisInputs | None,
    reflection: Reflection,
) -> tuple[AggregateRecord, ...]:
    """Answer the questions a reflection asked, for the next iteration's context.

    A malformed question is reported back as a record rather than raised: the model should learn
    to ask a better one, not lose the campaign. Requests are re-validated here through
    ``AnalysisQuery`` even though the response parser already checked their shape, because the
    vocabulary and the train-only guarantee belong to the analysis module rather than to the
    response contract.
    """

    if inputs is None or not reflection.analysis_requests:
        return ()
    queries: list[AnalysisQuery] = []
    rejected: list[AggregateRecord] = []
    for index, raw in enumerate(reflection.analysis_requests, start=1):
        try:
            queries.append(
                AnalysisQuery(
                    kind=AnalysisKind(str(raw.get("kind"))),
                    feature=str(raw.get("feature")),
                    second_feature=(
                        None if raw.get("second_feature") is None else str(raw["second_feature"])
                    ),
                    buckets=int(cast(int, raw.get("buckets", 5))),
                )
            )
        except (AnalysisError, ValueError) as error:
            rejected.append(
                AggregateRecord(
                    f"requested_analysis_{index:02d}_rejected",
                    {"reason": str(error)[:200]},
                )
            )
    return (*rejected, *run_requested_analyses(inputs, queries))


def _candidate_config(materialized: object) -> Mapping[str, object] | None:
    """Read a materialized candidate's ``config.json`` as a mapping, or ``None``.

    The configuration is the tuning the candidate actually performed -- decay half-lives, round
    counts, regularisation, the grid it searched. Recording only the family and the score told a
    later campaign that a direction lost without telling it at what setting, so it could not take
    a configuration that worked and push it further. Missing or malformed config is absent
    evidence, never a campaign failure.
    """

    reader = getattr(materialized, "file", None)
    if not callable(reader):
        return None
    try:
        content = reader("config.json").content
    except Exception:  # A candidate need not ship config.json, and a bad read is not fatal here.
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


# The rendered configuration reaches a later prompt, so it is bounded rather than trusted.
_RECORD_CONFIG_MAX_CHARS: Final = 600

#: How far a candidate must beat the incumbent on the Fold B screen before it is worth a Fold A
#: launch and an incumbent re-base.  One measured seed sigma (docs/RESULTS.md 3.4, observed 0.000316
#: across the five qualified official-FM seeds; the organizers publish 0.0008 on their own
#: platform).  At the previous 0.0 a candidate promoted on a +0.00004 Fold B delta, which is an
#: eighth of a sigma, so the incumbent chain was a random walk that spent Fold A launches on noise.
#:
#: This does NOT address Fold B overfitting: the two largest Fold B gains in the ledger (+0.00059,
#: +0.00034) clear this margin and still lost Fold A by more than they won.  It only stops the
#: churn below the noise floor.
FOLD_B_SCREEN_MARGIN: Final = 0.000316


def _bounded_config_json(config: Mapping[str, object] | None) -> str | None:
    """Render a candidate configuration as compact JSON, or ``None`` when unusable."""

    if not config:
        return None
    try:
        rendered = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return rendered if len(rendered) <= _RECORD_CONFIG_MAX_CHARS else None


def _decomposed_metrics(
    candidate_result: CandidateCampaignResult | None,
    incumbent: IncumbentEvidence,
) -> dict[str, AggregateScalar]:
    """Report GAUC and nDCG@5 per fold, with each delta against the incumbent's same fold.

    The record previously carried only the mean of the two, collapsed across folds. A candidate
    that gained 0.009 GAUC and lost 0.020 nDCG@5 was therefore remembered as one negative scalar,
    which reads as "that direction failed" rather than "the objective works and the truncated
    metric does not" -- the diagnosis an engineer would make from the same run. Reporting the
    folds separately also lets a replicated effect be told apart from one fold's noise.

    Runs arrive in the tier order ``_measured_primary`` documents: the Fold B screen first, then
    the Fold A confirmation, then the matched outer seeds. ``ScientificRunEvidence`` carries no
    fold label of its own, so that order is the only available mapping and it is applied
    positionally, never guessed from an attribute.
    """

    if candidate_result is None:
        return {}
    scored = [run for run in candidate_result.runs if run.metrics is not None]
    if not scored:
        return {}
    folds = dict(incumbent.inner_by_fold)
    values: dict[str, AggregateScalar] = {}
    for index, fold in enumerate(("B", "A")):
        if index >= len(scored):
            break
        metrics = scored[index].metrics
        if metrics is None:
            continue
        values[f"fold_{fold}_candidate_gauc"] = round(float(metrics.gauc), 6)
        values[f"fold_{fold}_candidate_ndcg_at_5"] = round(float(metrics.ndcg_at_5), 6)
        values[f"fold_{fold}_candidate_primary"] = round(float(metrics.primary), 6)
        reference = folds.get(fold)
        if reference is None:
            continue
        values[f"fold_{fold}_delta_gauc"] = round(float(metrics.gauc) - float(reference.gauc), 6)
        values[f"fold_{fold}_delta_ndcg_at_5"] = round(
            float(metrics.ndcg_at_5) - float(reference.ndcg_at_5), 6
        )
        values[f"fold_{fold}_delta_primary"] = round(
            float(metrics.primary) - float(reference.primary), 6
        )
    if values:
        values["metric_decomposition_note"] = (
            "The primary score is the MEAN of GAUC and nDCG@5, and a change can move them in "
            "opposite directions. A positive GAUC delta with a larger negative nDCG@5 delta means "
            "the ranking objective worked and the top-5 behaviour did not: that is a reason to "
            "fix the cutoff behaviour, not to abandon the direction. Deltas are against the "
            "incumbent on the same fold. An effect present on one fold only is not yet replicated."
        )
    return values


def _fusion_disclosure(
    records: Mapping[tuple[ScientificTier, int], GeneratedScientificRunRecord],
) -> dict[str, str | float | None]:
    """Describe what rank fusion did to this iteration's predictions.

    Every candidate prediction is rank-fused with the official FM control on a fixed five-point
    weight grid chosen on the Fold B screen and frozen for every tier below it.  The primary that
    reaches selection and reflection is the *blend's*, so a model far weaker than the control still
    reports a score near it, and a model the selector rejects outright reports the control's score
    exactly.  Without these fields the proposer reads "scored the same as the baseline" when the
    truth is "was discarded", and has no reason to change anything.
    """

    record = records.get((ScientificTier.FOLD_B_SCREEN, 0))
    selection = None if record is None else record.fusion_selection
    if selection is None:
        return {
            "candidate_standalone_primary": None,
            "fold_b_control_primary": None,
            "fusion_weights_selected": None,
            "fusion_model_discarded": None,
            "fusion_note": None,
        }
    standalone = selection.standalone_primary
    control = selection.control_primary
    weights = selection.selected_weights
    if selection.model_was_discarded:
        note = (
            "Rank fusion selected weight 0.0 for this model on the Fold B screen, so the reported "
            "primary is the official FM control's score with none of this model's ordering in it. "
            "The model itself scored candidate_standalone_primary. This is a rejection, NOT a tie."
        )
    elif standalone is not None and control is not None and standalone < control:
        note = (
            "The reported primary is a blend of this model with the official FM control at "
            "weights fusion_weights_selected. Scored alone the model was WEAKER than the control "
            "(candidate_standalone_primary below fold_b_control_primary), so most of the reported "
            "score is the control's, not this model's."
        )
    else:
        note = (
            "The reported primary is a blend of this model with the official FM control at "
            "weights fusion_weights_selected. Scored alone this model was at least as strong as "
            "the control."
        )
    return {
        "candidate_standalone_primary": None if standalone is None else round(standalone, 7),
        "fold_b_control_primary": None if control is None else round(control, 7),
        "fusion_weights_selected": f"{weights[0]} model, {weights[1]} official FM control",
        # The note above says this in prose; the flag says it to the family breaker, which
        # otherwise never closes the harshest verdict of all -- the selector threw the model out
        # and the record then reads as a tie with the baseline.
        "fusion_model_discarded": selection.model_was_discarded,
        "fusion_note": note,
    }


def _iteration_record(
    result: ScientificCampaignResult,
    reflection: Reflection,
    *,
    scientific_iteration: int,
    proposal: Proposal | None = None,
    materialized: object | None = None,
    fusion_records: Mapping[tuple[ScientificTier, int], GeneratedScientificRunRecord] | None = None,
) -> AggregateRecord:
    candidate_result = result.candidates[-1] if result.candidates else None
    primary, delta, tier = _measured_primary(candidate_result, result.incumbent)
    fusion = _fusion_disclosure({} if fusion_records is None else fusion_records)
    execution_failed = (
        candidate_result is not None
        and candidate_result.outcome is CandidateOutcome.CALLBACK_FAILED
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
            # A crashed candidate produced no score at all.  The reflection summary cannot say so,
            # because its schema requires three finite metrics and substitutes the fallback's, so
            # without this flag the next proposer reads a failed branch as one that tied the
            # baseline and has no reason to avoid repeating the defect.
            "execution_failed": execution_failed,
            "execution_failure_note": (
                None
                if not execution_failed
                else (
                    "This iteration produced NO measured score. Its training or prediction code "
                    "raised an exception before any evaluation. Treat it as a code defect to "
                    "avoid, not as evidence about the scientific direction."
                )
            ),
            # Without this the note above is all the proposer gets, and a candidate that broke a
            # stated rule cannot tell which one: four consecutive iterations were lost to the same
            # rejected diagnostics key because the reason never reached the model.
            "execution_failure_remedy": (
                None if candidate_result is None else candidate_result.remedy
            ),
            # The direction this iteration actually tested.  A proposer that cannot see this
            # cannot avoid restating it, and the convergence rule allows very few attempts.
            "proposal_family": (None if proposal is None else proposal_family_of(proposal)),
            "proposal_objective": None if proposal is None else proposal.objective,
            "proposal_principal_change": None if proposal is None else proposal.principal_change,
            "candidate_primary": None if primary is None else round(primary, 6),
            "candidate_primary_tier": tier,
            "delta_vs_incumbent": None if delta is None else round(delta, 6),
            # candidate_primary above is the FUSED score. These four say what the model
            # itself did, including whether the selector discarded it outright.
            **fusion,
            # GAUC and nDCG@5 per fold, with deltas against the incumbent's same fold. The
            # primary mean hides opposite movements in its two halves.
            **_decomposed_metrics(candidate_result, result.incumbent),
            # The hyperparameters that produced those numbers, so a later iteration can push
            # a setting that worked rather than re-searching for it.
            "candidate_config_json": (
                None
                if materialized is None
                else _bounded_config_json(_candidate_config(materialized))
            ),
            "incumbent_candidate_id": result.incumbent.candidate_id,
            "launches_used": result.launches_used,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "stop_reason": result.stop_reason.value,
            "reflection_recommendation": reflection.recommendation,
            "reflection_summary": reflection.summary,
            # Lessons are the richest thing reflection produces and were previously dropped here.
            "reflection_lessons": " | ".join(reflection.lessons),
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


@dataclass(frozen=True, slots=True)
class _LineageAdmissionMetrics:
    """This attempt's own inner-fold outcome, extracted for durable cross-run recording."""

    inner_fold_a: LineageFoldMetrics | None
    inner_fold_b: LineageFoldMetrics | None
    parent_fold_a_primary: float | None
    parent_fold_b_primary: float | None
    promoted: bool | None


def _lineage_admission_metrics(
    *,
    parent_incumbent: IncumbentEvidence,
    result: ScientificCampaignResult,
    candidate_id: str,
) -> _LineageAdmissionMetrics:
    """Extract one attempt's fold evidence against its actual parent, not the fixed FM baseline.

    ``parent_incumbent`` must be the incumbent as it stood *before* this attempt's own
    ``run_scientific_campaign`` call, since a promotion inside that call replaces
    ``result.incumbent`` with the just-processed candidate itself.
    """

    # Each fold's own (inner metrics, parent reference) pair travels together, but Fold A and
    # Fold B are independent of each other: a candidate that fails the Fold B screen never
    # reaches Fold A, and its real Fold B result is still worth a future campaign knowing rather
    # than discarding. `promoted` is only ever meaningful (and only ever stored) alongside full
    # evidence from both folds, since promotion is impossible without a Fold A confirmation.
    empty = _LineageAdmissionMetrics(None, None, None, None, None)
    own_result = next(
        (item for item in result.candidates if item.candidate.candidate_id == candidate_id),
        None,
    )
    if own_result is None or not own_result.runs:
        return empty
    parent_folds = dict(parent_incumbent.inner_by_fold)
    fold_b_raw = own_result.runs[0].metrics
    fold_a_raw = own_result.runs[1].metrics if len(own_result.runs) >= 2 else None

    inner_fold_b: LineageFoldMetrics | None = None
    parent_fold_b_primary: float | None = None
    parent_fold_b = parent_folds.get("B")
    if fold_b_raw is not None and parent_fold_b is not None:
        inner_fold_b = LineageFoldMetrics(
            gauc=fold_b_raw.gauc, ndcg_at_5=fold_b_raw.ndcg_at_5, primary=fold_b_raw.primary
        )
        parent_fold_b_primary = parent_fold_b.primary

    inner_fold_a: LineageFoldMetrics | None = None
    parent_fold_a_primary: float | None = None
    parent_fold_a = parent_folds.get("A")
    if fold_a_raw is not None and parent_fold_a is not None:
        inner_fold_a = LineageFoldMetrics(
            gauc=fold_a_raw.gauc, ndcg_at_5=fold_a_raw.ndcg_at_5, primary=fold_a_raw.primary
        )
        parent_fold_a_primary = parent_fold_a.primary

    promoted = (
        result.incumbent.candidate_id == candidate_id
        if inner_fold_a is not None and inner_fold_b is not None
        else None
    )
    return _LineageAdmissionMetrics(
        inner_fold_a=inner_fold_a,
        inner_fold_b=inner_fold_b,
        parent_fold_a_primary=parent_fold_a_primary,
        parent_fold_b_primary=parent_fold_b_primary,
        promoted=promoted,
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
    benchmark_digest: str = "",
    starter_digest: str = "",
    source_digest: str = "",
    lineage_ledger_path: Path | None = None,
    evaluation_digest: str = "",
    prior_root_failure_totals: Mapping[str, int] = MappingProxyType({}),
    prior_family_attempts: Mapping[str, int] = MappingProxyType({}),
    prior_advisory_records: tuple[AggregateRecord, ...] = (),
) -> _LiveLineagePreparation:
    """Prepare one valid lineage or return an exact, evidence-preserving closure reason.

    ``prior_*`` seeds the circuit breaker and the LLM-visible evidence with cross-run history from
    the project-wide :class:`~kuairand_agent.campaign.store.ResearchLineageLedger` (see
    ``_lineage_ledger_location``), so a fresh campaign against the exact same benchmark, starter
    kit, and trusted source identity does not repeat an already-characterized failure from
    scratch.  When ``lineage_ledger_path`` is given, every rejection and acceptance this call
    observes is also durably recorded there for the *next* campaign to inherit.

    ``prior_advisory_records`` only ever widens what the LLM is shown; it is deliberately kept out
    of the returned ``rejected_records``, which stays scoped to this campaign's own branches so
    downstream stage counts (``candidates_admitted`` and friends) remain accurate reporting of
    *this* run rather than a run's-worth of history borrowed from an earlier campaign.
    """

    if lineage_ledger_path is not None and not (
        benchmark_digest and starter_digest and source_digest
    ):
        raise FullCampaignError(
            "a research-lineage ledger requires benchmark, starter, and source digests"
        )

    def record_rejection_evidence(rejection: LiveResearchBranchRejected) -> None:
        if lineage_ledger_path is None:
            return
        with _open_lineage_ledger(lineage_ledger_path) as ledger:
            ledger.record_rejection(
                campaign_id=campaign_id,
                benchmark_digest=benchmark_digest,
                starter_digest=starter_digest,
                source_digest=source_digest,
                evaluation_digest=evaluation_digest,
                candidate_id=rejection.failed_candidate_id,
                proposal_family=rejection.proposal_family,
                proposal_signature=rejection.proposal_signature or None,
                repairs_attempted=rejection.repairs_attempted,
                root_failure_fingerprint=rejection.root_failure.fingerprint,
                root_failure_category=rejection.root_failure.category,
                root_failure_code=rejection.root_failure.code,
                root_failure_subject=rejection.root_failure.subject,
                terminal_failure_fingerprint=rejection.terminal_failure.fingerprint,
                terminal_failure_category=rejection.terminal_failure.category,
                terminal_failure_code=rejection.terminal_failure.code,
                terminal_failure_subject=rejection.terminal_failure.subject,
                diagnostic=rejection.diagnostic,
            )

    # Admission is recorded with real fold metrics by the caller, once `run_scientific_campaign`
    # has actually trained this candidate — this function only prepares the lineage.

    rejected, previous_journal_digest = _load_rejection_journal(
        generated_root,
        campaign_id=campaign_id,
        maximum_iterations=maximum_iterations,
    )
    root_failure_totals: dict[str, int] = dict(prior_root_failure_totals)
    proposal_family_attempts: dict[str, int] = dict(prior_family_attempts)
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
        safe_context = safe_context_factory(prior_advisory_records + tuple(rejected))
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
            record_rejection_evidence(rejection)
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


def _unavailable_reflection(
    exc: BaseException,
    *,
    scientific_iteration: int,
) -> Reflection:
    """Stand in for a reflection the provider did not return.

    Reflection is commentary on a result the controller already owns, so a provider failure here
    must degrade rather than propagate out of the research loop.  The diagnostic is collapsed to
    one line because ``report._text`` rejects newlines that ``schemas._text`` permits, and this
    text is model- and transport-authored.
    """

    detail = " ".join(str(exc).split())[:512] or type(exc).__name__
    return Reflection(
        response_id=f"reflection-unavailable-{scientific_iteration:02d}",
        summary=("The research provider did not return a reflection for this iteration: " + detail),
        recommendation="close_branch",
        lessons=("Provider reflection was unavailable; the measured result stands.",),
    )


def _research_closure_record(
    exc: BaseException,
    *,
    scientific_iteration: int,
    operation: str,
) -> AggregateRecord:
    """Record a research-plane failure that closes research without ending the campaign."""

    detail = " ".join(str(exc).split())[:2048] or exc.__class__.__name__
    code = getattr(exc, "code", None)
    return AggregateRecord(
        name=f"research_closure_{scientific_iteration:02d}",
        values={
            "scientific_iteration": scientific_iteration,
            "branch_outcome": "research_closed_by_provider_failure",
            "operation": operation,
            "failure_type": type(exc).__name__,
            "failure_code": str(getattr(code, "value", code)) if code is not None else "",
            "diagnostic": detail,
        },
    )


def _research_closure_reason(exc: BaseException) -> CampaignStopReason:
    """Map a research-plane failure onto the terminal condition it actually represents.

    The provider raises a DEADLINE error by design once the finalization reserve opens; that is a
    scheduled stop, not a fault.
    """

    code = getattr(exc, "code", None)
    if getattr(code, "value", None) == "deadline":
        return CampaignStopReason.FINALIZATION_RESERVE
    return CampaignStopReason.CANDIDATES_EXHAUSTED


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
    lineage_ledger_path: Path | None = None,
    evaluation_digest: str = "",
    prior_advisory_records: tuple[AggregateRecord, ...] = (),
    analysis_inputs: TrainAnalysisInputs | None = None,
) -> _AutonomousFollowupResult:
    """Continue propose→implement→evaluate→reflect until an exact terminal condition.

    ``prior_advisory_records`` carries cross-run lineage-ledger evidence.  It only ever widens the
    ``campaign_records`` shown to the LLM each iteration; it is deliberately kept out of
    ``rejected_records`` so this campaign's own stage counts (``candidates_admitted`` and friends)
    stay accurate reporting of *this* run.
    """

    def record_rejection_evidence(rejection: LiveResearchBranchRejected) -> None:
        if lineage_ledger_path is None:
            return
        with _open_lineage_ledger(lineage_ledger_path) as ledger:
            ledger.record_rejection(
                campaign_id=request.campaign_id,
                benchmark_digest=request.benchmark_digest,
                starter_digest=request.starter_manifest_digest,
                source_digest=request.source_digest,
                evaluation_digest=evaluation_digest,
                candidate_id=rejection.failed_candidate_id,
                proposal_family=rejection.proposal_family,
                proposal_signature=rejection.proposal_signature or None,
                repairs_attempted=rejection.repairs_attempted,
                root_failure_fingerprint=rejection.root_failure.fingerprint,
                root_failure_category=rejection.root_failure.category,
                root_failure_code=rejection.root_failure.code,
                root_failure_subject=rejection.root_failure.subject,
                terminal_failure_fingerprint=rejection.terminal_failure.fingerprint,
                terminal_failure_category=rejection.terminal_failure.category,
                terminal_failure_code=rejection.terminal_failure.code,
                terminal_failure_subject=rejection.terminal_failure.subject,
                diagnostic=rejection.diagnostic,
            )

    def record_admission_evidence(
        lineage: LiveResearchLineage,
        *,
        parent_incumbent: IncumbentEvidence,
        outcome: ScientificCampaignResult,
    ) -> None:
        if lineage_ledger_path is None:
            return
        metrics = _lineage_admission_metrics(
            parent_incumbent=parent_incumbent,
            result=outcome,
            candidate_id=lineage.candidate_id,
        )
        with _open_lineage_ledger(lineage_ledger_path) as ledger:
            ledger.record_admission(
                campaign_id=request.campaign_id,
                benchmark_digest=request.benchmark_digest,
                starter_digest=request.starter_manifest_digest,
                source_digest=request.source_digest,
                evaluation_digest=evaluation_digest,
                candidate_id=lineage.candidate_id,
                proposal_family=proposal_family_of(lineage.proposal),
                proposal_signature=_lineage_proposal_signature(lineage.proposal),
                inner_fold_a=metrics.inner_fold_a,
                inner_fold_b=metrics.inner_fold_b,
                parent_fold_a_primary=metrics.parent_fold_a_primary,
                parent_fold_b_primary=metrics.parent_fold_b_primary,
                promoted=metrics.promoted,
                candidate_config=_candidate_config(lineage.materialized),
            )

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
        *prior_advisory_records,
        *prior_records,
        _iteration_record(
            result,
            reflection,
            scientific_iteration=runtime_template.scientific_iteration,
            proposal=lineage.proposal,
            materialized=lineage.materialized,
            # The first iteration ran on runtime_template itself, so its fusion records are
            # here. Later iterations each get a fresh records dict from the replace() below.
            fusion_records=runtime_template.records,
        ),
    ]
    rejected_records = list(prior_records)
    iteration = runtime_template.scientific_iteration
    terminal = result.stop_reason
    proposal_breadth = max(1, request.config.research.proposal_breadth)
    round_attempt_count = 0
    pending_admitted_records: list[AggregateRecord] = []
    # Follow-up rejections accumulate the same counters the initial-admission loop keeps. Without
    # them every rejection record reported `attempt_count=1, proposal_family_blocked=False`, so the
    # model was told each repeat was its first attempt, `_proposal_family_is_blocked` could never
    # see an in-campaign block, and nothing ever stopped the loop: one observed campaign spent
    # 349,550 tokens re-proposing a family the controller refused fourteen times.
    followup_root_failures: dict[str, int] = {}
    followup_family_attempts: dict[str, int] = {}
    previous_root_fingerprint: str | None = None
    consecutive_root_failures = 0

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
        if status.outer_queries_remaining <= 0:
            terminal = CampaignStopReason.OUTER_PROMOTION_LIMIT
            break
        # ``result.elapsed_seconds`` accumulates only candidate subprocess wall time, so
        # provider latency, materialization and reflection are invisible to it.  Measured on
        # this project's own runs the unaccounted provider time reached 4011s in a single
        # partial campaign, larger than the whole finalization reserve.  Phase 1 consults the
        # real clock; this loop must too, or research runs straight through the reserve.
        observed_deadline = runtime_template.engine.inspect_deadline(runtime_template.run_dir)
        if observed_deadline.hard_expired:
            terminal = CampaignStopReason.HARD_DEADLINE
            break
        if observed_deadline.finalization_reserve_active:
            terminal = CampaignStopReason.FINALIZATION_RESERVE
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
            record_rejection_evidence(rejection)
            root_fingerprint = rejection.root_failure.fingerprint
            root_failure_total = followup_root_failures.get(root_fingerprint, 0) + 1
            followup_root_failures[root_fingerprint] = root_failure_total
            if root_fingerprint == previous_root_fingerprint:
                consecutive_root_failures += 1
            else:
                previous_root_fingerprint = root_fingerprint
                consecutive_root_failures = 1
            family = rejection.proposal_family
            family_attempt_count = followup_family_attempts.get(family, 0) + 1
            followup_family_attempts[family] = family_attempt_count
            rejected_record = _rejected_lineage_record(
                rejection,
                scientific_iteration=iteration,
                root_failure_total_count=root_failure_total,
                root_failure_consecutive_count=consecutive_root_failures,
                proposal_family_attempt_count=family_attempt_count,
                proposal_family_blocked=family_attempt_count >= 2,
            )
            records.append(rejected_record)
            rejected_records.append(rejected_record)
            terminal = CampaignStopReason.CANDIDATES_EXHAUSTED
            if root_failure_total >= 3:
                break
            continue
        except (ResearchModelError, ProductionResearchError) as exc:
            # A provider or lineage failure closes research; it must never end the campaign
            # before finalization.  Breaking here returns normally, so the caller finalizes
            # the retained incumbent exactly as it would on any other terminal condition.
            #
            # ``records`` is local and dies with this frame, so the closure also goes into
            # ``rejected_records``, which is what reaches the durable rejection ledger.  Without
            # that the report would show research simply stopping with no stated cause.  The
            # branch genuinely was rejected before execution, so the count stays honest.
            closure_record = _research_closure_record(
                exc,
                scientific_iteration=iteration,
                operation="lineage",
            )
            records.append(closure_record)
            rejected_records.append(closure_record)
            terminal = _research_closure_reason(exc)
            break
        candidate_id = lineage.candidate_id
        parent_incumbent = result.incumbent
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
            result = run_scientific_campaign(
                config=scientific_config,
                fallback=fallback,
                candidates=(candidate,),
                runner=iteration_runtime,
                outer_ledger=adapter,
                initial_incumbent=result.incumbent,
                initial_convergence=result.convergence,
                initial_launches_used=result.launches_used,
                initial_elapsed_seconds=result.elapsed_seconds,
            )
        record_admission_evidence(lineage, parent_incumbent=parent_incumbent, outcome=result)
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
        pending_admitted_records.extend(_answered_analyses(analysis_inputs, reflection))
        pending_admitted_records.append(
            _iteration_record(
                result,
                reflection,
                scientific_iteration=iteration,
                proposal=lineage.proposal,
                materialized=lineage.materialized,
                fusion_records=iteration_runtime.records,
            )
        )
        round_attempt_count += 1
        promoted = result.incumbent.candidate_id == candidate_id
        if promoted:
            parent = snapshot_materialized_candidate(
                lineage.materialized,
                candidate_id=candidate_id,
            )
        # A round is up to `proposal_breadth` independently proposed attempts against the same
        # parent. Each attempt's own reflection is held back from `records` (and therefore from
        # the *next* attempt's `campaign_records`) until the round concludes, so a same-round
        # follow-up attempt is proposed from the same starting evidence rather than anchored to a
        # critique of the attempt immediately before it. Once the round ends — by promotion or by
        # exhausting its breadth — every attempt's reflection is flushed for later rounds to see.
        if promoted or round_attempt_count >= proposal_breadth:
            records.extend(pending_admitted_records)
            pending_admitted_records = []
            round_attempt_count = 0
        terminal = result.stop_reason

    records.extend(pending_admitted_records)
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


def _open_lineage_ledger(path: Path) -> ResearchLineageLedger:
    """Open the project-wide research-lineage ledger, creating it on first use.

    Unlike the outer-query ledger this carries no hard limit, so opening never fails on a
    populated ledger: it exists purely to give a fresh campaign typed cross-run evidence about
    which proposal families and root failures were already characterized against this exact
    benchmark, starter kit, and trusted source identity.
    """

    try:
        if path.exists() or path.is_symlink():
            return ResearchLineageLedger.open(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent = path.parent.lstat()
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise FullCampaignError("research-lineage ledger parent must be a real directory")
        return ResearchLineageLedger.create(path)
    except OSError as exc:
        raise FullCampaignError("research-lineage ledger is unavailable") from exc


def _evaluation_scope_digest(
    *,
    benchmark_digest: str,
    dataset_digest: str,
    scorer_digest: str,
    feature_bundle_digest: str,
) -> str:
    """Identity of everything that determines what a recorded metric means.

    Cross-run metric evidence is scoped by this rather than by the trusted-source digest. A fold
    score is a fact about the benchmark, the data, the scorer and the controller-engineered
    feature bundle; editing a prompt, a selector or a circuit breaker cannot make it false, so
    that evidence should outlive our own development. Changing the features or the scorer does
    change what the number means, and both are inputs here, so the scope correctly resets then.
    """

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "benchmark_digest": benchmark_digest,
                "dataset_digest": dataset_digest,
                "scorer_digest": scorer_digest,
                "feature_bundle_digest": feature_bundle_digest,
            }
        )
    ).hexdigest()


def _lineage_ledger_location(
    *,
    run_dir: Path,
    project_root: Path,
    configured_path: Path | None,
) -> Path:
    return (
        run_dir.parent / "research-lineage-ledger.sqlite3"
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
    lineage_ledger_path: Path | None = None,
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
    lineage_path = _lineage_ledger_location(
        run_dir=run,
        project_root=root,
        configured_path=lineage_ledger_path,
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
        # Metric evidence is scoped by what determines a metric's meaning, not by our source tree,
        # so cross-run knowledge survives controller edits. See _evaluation_scope_digest.
        evaluation_scope = _evaluation_scope_digest(
            benchmark_digest=request.benchmark_digest,
            dataset_digest=qualification.canonical_digest,
            scorer_digest=scorer_digest,
            # The bundle's own digest embeds `builder_source_digest=request.source_digest`, i.e.
            # the whole trusted source tree, so it moves on every commit. Using it here made the
            # evaluation scope inherit that source dependence transitively and reset cross-run
            # memory on each code change -- the exact failure this scope exists to prevent. The
            # matrix digest hashes the feature names, shape and float bytes, so it changes when
            # the features genuinely change and not when a prompt does.
            feature_bundle_digest=features.outer_and_final.prefix.digest,
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
            with _open_lineage_ledger(lineage_path) as prior_lineage_ledger:
                # Controller-attributable failures stay scoped to this exact source tree, so a
                # corrective fix starts them clean.
                prior_lineage = prior_lineage_ledger.summary(
                    benchmark_digest=request.benchmark_digest,
                    starter_digest=request.starter_manifest_digest,
                    source_digest=request.source_digest,
                )
                # Measured outcomes are scoped by evaluation identity instead, so what a proposal
                # family actually scored survives our own commits. Reading these through the
                # source scope was what made cross-run memory reset on every code change.
                durable_admissions = prior_lineage_ledger.admissions_for_evaluation(
                    benchmark_digest=request.benchmark_digest,
                    evaluation_digest=evaluation_scope,
                )
            prior_events = {event.event_id: event for event in prior_lineage.recent_events}
            prior_events.update({event.event_id: event for event in durable_admissions})
            prior_lineage_advisory_records = tuple(
                AggregateRecord(
                    name=f"prior_campaign_lesson_{event.event_id:04d}",
                    values={
                        key: value
                        for key, value in {
                            "campaign_id": event.campaign_id,
                            "branch_outcome": event.outcome,
                            "candidate_id": event.candidate_id,
                            "proposal_family": event.proposal_family,
                            "proposal_signature": event.proposal_signature,
                            "repairs_attempted": event.repairs_attempted,
                            "root_failure_fingerprint": event.root_failure_fingerprint,
                            "root_failure_category": event.root_failure_category,
                            "root_failure_code": event.root_failure_code,
                            "root_failure_subject": event.root_failure_subject,
                            "inner_fold_a_gauc": (
                                None if event.inner_fold_a is None else event.inner_fold_a.gauc
                            ),
                            "inner_fold_a_ndcg_at_5": (
                                None if event.inner_fold_a is None else event.inner_fold_a.ndcg_at_5
                            ),
                            "inner_fold_a_primary": (
                                None if event.inner_fold_a is None else event.inner_fold_a.primary
                            ),
                            "inner_fold_b_gauc": (
                                None if event.inner_fold_b is None else event.inner_fold_b.gauc
                            ),
                            "inner_fold_b_ndcg_at_5": (
                                None if event.inner_fold_b is None else event.inner_fold_b.ndcg_at_5
                            ),
                            "inner_fold_b_primary": (
                                None if event.inner_fold_b is None else event.inner_fold_b.primary
                            ),
                            "parent_fold_a_primary": event.parent_fold_a_primary,
                            "parent_fold_b_primary": event.parent_fold_b_primary,
                            "promoted": event.promoted,
                            # The setting that produced those metrics. A family that lost at one
                            # configuration has not been shown to be a dead direction, and a
                            # family that gained is worth pushing further from where it landed.
                            "candidate_config_json": _bounded_config_json(event.candidate_config),
                        }.items()
                        if value is not None
                    },
                )
                for event in sorted(prior_events.values(), key=lambda item: item.event_id)
            )

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
                benchmark_digest=request.benchmark_digest,
                starter_digest=request.starter_manifest_digest,
                source_digest=request.source_digest,
                lineage_ledger_path=lineage_path,
                evaluation_digest=evaluation_scope,
                prior_root_failure_totals=prior_lineage.root_failure_totals,
                prior_family_attempts=prior_lineage.proposal_family_rejection_totals,
                prior_advisory_records=prior_lineage_advisory_records,
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
            screen_margin=FOLD_B_SCREEN_MARGIN,
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
            try:
                result = run_scientific_campaign(
                    config=scientific_config,
                    fallback=fallback_incumbent,
                    candidates=(candidate,),
                    runner=runtime,
                    outer_ledger=adapter,
                )
            except ScientificCampaignCancelled as exc:
                raise FullCampaignCancelled(
                    "campaign cancelled during scientific execution; durable child evidence "
                    "is resumable and no scientific success was published"
                ) from exc
        if isinstance(lineage, LiveResearchLineage):
            metrics = _lineage_admission_metrics(
                parent_incumbent=fallback_incumbent,
                result=result,
                candidate_id=candidate.candidate_id,
            )
            with _open_lineage_ledger(lineage_path) as project_lineage_ledger:
                project_lineage_ledger.record_admission(
                    campaign_id=request.campaign_id,
                    benchmark_digest=request.benchmark_digest,
                    starter_digest=request.starter_manifest_digest,
                    source_digest=request.source_digest,
                    evaluation_digest=evaluation_scope,
                    candidate_id=lineage.candidate_id,
                    proposal_family=proposal_family_of(lineage.proposal),
                    proposal_signature=_lineage_proposal_signature(lineage.proposal),
                    inner_fold_a=metrics.inner_fold_a,
                    inner_fold_b=metrics.inner_fold_b,
                    parent_fold_a_primary=metrics.parent_fold_a_primary,
                    parent_fold_b_primary=metrics.parent_fold_b_primary,
                    promoted=metrics.promoted,
                )

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
                analysis_inputs=_train_analysis_inputs(data, features),
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
                lineage_ledger_path=lineage_path,
                evaluation_digest=evaluation_scope,
                prior_advisory_records=prior_lineage_advisory_records,
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

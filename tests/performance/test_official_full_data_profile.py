from __future__ import annotations

import hashlib
import json
import os
import time
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kuairand_agent.campaign.budgets import (
    BudgetLedger,
    LaunchCategory,
    LaunchRequest,
    WorkKind,
    WorkPhase,
)
from kuairand_agent.campaign.full_campaign import (
    build_production_feature_bundle,
    prepare_campaign_data_plane,
)
from kuairand_agent.campaign.qualification_evidence import (
    QualificationExpectations,
    load_official_fm_qualification,
)
from kuairand_agent.candidates.grouping import build_user_grouping
from kuairand_agent.candidates.pairwise import GAUCPairSampler
from kuairand_agent.contract import STARTER_FILE_SHA256, verify_starter_kit
from kuairand_agent.data.audit import DataAuditReport, audit_dataset
from kuairand_agent.data.canonical import CanonicalDataset, TrainingTargets
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.performance import (
    PerformanceProfile,
    TimingReceipt,
    measure_operation,
    write_performance_profile,
)

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
QUALIFICATION_ENV = "KUAIRAND_QUALIFICATION_RUN_DIR"
OUTPUT_ENV = "KUAIRAND_PERFORMANCE_OUTPUT"
FINALIZATION_RESERVE_SECONDS = 3_600.0
CAMPAIGN_LIMIT_SECONDS = 21_600.0
EXPECTED_ELIGIBLE_TRAIN_ROWS = 1_130_240

pytestmark = pytest.mark.skipif(
    QUALIFICATION_ENV not in os.environ,
    reason=f"set {QUALIFICATION_ENV} to the verified official qualification run",
)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AssertionError(f"{name} must be a real number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AssertionError(f"{name} must be an integer")
    return int(value)


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _builder_source_digest() -> str:
    path = ROOT / "src" / "kuairand_agent" / "campaign" / "pure_features.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification_dir() -> Path:
    requested = os.environ[QUALIFICATION_ENV]
    try:
        root = Path(requested).resolve(strict=True)
    except OSError as exc:
        pytest.skip(f"{QUALIFICATION_ENV} does not resolve to an existing run: {exc}")
    if not root.is_dir():
        pytest.skip(f"{QUALIFICATION_ENV} is not a directory: {root}")
    return root


def _clean_replay_receipt(root: Path) -> TimingReceipt:
    manifest = cast(
        dict[str, object],
        json.loads((root / "manifest.json").read_text(encoding="ascii")),
    )
    fm = cast(dict[str, object], manifest["fm"])
    clean = cast(dict[str, object], fm["clean_seed_zero"])
    resources = cast(dict[str, object], clean["resources"])
    trace = cast(dict[str, object], clean["training_trace"])
    child = cast(dict[str, object], trace["clean_subprocess"])
    return TimingReceipt(
        family="final_replay",
        wall_seconds=_real(resources["wall_seconds"], "clean replay wall_seconds"),
        cpu_seconds=_real(resources["cpu_seconds"], "clean replay cpu_seconds"),
        peak_rss_bytes=_integer(
            resources["peak_rss_bytes"],
            "clean replay peak_rss_bytes",
        ),
        rows=1_141_112 + 124_909,
        evidence_digest=_digest(
            child["evidence_digest"],
            "clean replay evidence_digest",
        ),
    )


def _controller_overhead(model_p95_seconds: float) -> tuple[TimingReceipt, float]:
    ledger = BudgetLedger.after_qualification()
    request = LaunchRequest(
        execution_id="performance-controller-probe",
        family="causal_lambdarank",
        kind=WorkKind.FULL_TRAIN_EVALUATE,
        phase=WorkPhase.RESEARCH,
        p95_runtime_seconds=model_p95_seconds,
        cleanup_seconds=60.0,
        category=LaunchCategory.DIVERSE_INNER_SCREEN,
    )
    iterations = 10_000
    started = time.perf_counter()
    last = None
    for _ in range(iterations):
        last = ledger.admit(
            request,
            remaining_seconds=CAMPAIGN_LIMIT_SECONDS,
            finalization_reserve_seconds=FINALIZATION_RESERVE_SECONDS,
        )
    elapsed = time.perf_counter() - started
    assert last is not None and last.allowed
    digest = hashlib.sha256(
        (
            f"{last.reason.value}:{last.required_seconds}:{last.remaining_seconds}:{iterations}"
        ).encode("ascii")
    ).hexdigest()
    return (
        TimingReceipt(
            family="controller_admission",
            wall_seconds=elapsed,
            cpu_seconds=elapsed,
            peak_rss_bytes=1,
            rows=iterations,
            evidence_digest=digest,
        ),
        elapsed,
    )


def test_official_profile_is_linear_cached_and_fits_campaign_reserves(
    tmp_path: Path,
    official_data_dir: Path,
    official_dataset: CanonicalDataset,
    official_audit: DataAuditReport,
) -> None:
    qualification_dir = _qualification_dir()
    starter = verify_starter_kit(STARTER)
    qualification = load_official_fm_qualification(
        qualification_dir,
        expectations=QualificationExpectations(
            canonical_digest=official_dataset.digest,
            starter_manifest_digest=starter.manifest_sha256,
            scorer_digest=STARTER_FILE_SHA256["evaluate.py"],
            validation_row_count=official_dataset.valid.row_count,
            final_row_count=official_dataset.final.row_count,
        ),
    )
    assert qualification.audit_digest == official_audit.digest

    audit_receipt, audited = measure_operation(
        family="data_audit",
        rows=1_141_112 + 124_909 + 170_588,
        operation=lambda: (
            (report := audit_dataset(official_data_dir)),
            report.digest,
        ),
    )
    assert audited.digest == official_audit.digest
    assert audited.final_outcome_trace.manifest()["outcome_cells_scored"] == 0

    data_plane = prepare_campaign_data_plane(
        official_dataset,
        expected_dataset_digest=official_dataset.digest,
    )
    cache = tmp_path / "causal-cache"
    cold_receipt, cold = measure_operation(
        family="causal_feature_cold",
        rows=1_141_112 + 124_909 + 170_588,
        operation=lambda: (
            (
                bundle := build_production_feature_bundle(
                    data_plane,
                    builder_source_digest=_builder_source_digest(),
                    cache_dir=cache,
                )
            ),
            bundle.digest,
        ),
    )
    warm_receipt, warm = measure_operation(
        family="causal_feature_warm",
        rows=1_141_112 + 124_909 + 170_588,
        operation=lambda: (
            (
                bundle := build_production_feature_bundle(
                    data_plane,
                    builder_source_digest=_builder_source_digest(),
                    cache_dir=cache,
                )
            ),
            bundle.digest,
        ),
    )
    assert cold.digest == warm.digest
    assert warm_receipt.wall_seconds < cold_receipt.wall_seconds

    grouping_receipt, grouping = measure_operation(
        family="full_data_grouping",
        rows=official_dataset.train.row_count,
        operation=lambda: (
            (
                grouped := build_user_grouping(
                    official_dataset.train.inputs.user_id,
                    official_dataset.train.inputs.video_id,
                    phase=DataPhase.TRAIN,
                )
            ),
            grouped.digest,
        ),
    )
    assert grouping.row_count == official_dataset.train.row_count
    assert sum(grouping.group_sizes) == grouping.row_count
    assert grouping_receipt.wall_seconds < 180.0

    targets = official_dataset.train.targets
    assert isinstance(targets, TrainingTargets)
    sampler_receipt, sampler = measure_operation(
        family="pairwise_sampler",
        rows=official_dataset.train.row_count,
        operation=lambda: (
            (
                built := GAUCPairSampler(
                    official_dataset.train.inputs.user_id,
                    np.asarray(targets.long_view, dtype=np.int8),
                    phase=DataPhase.TRAIN,
                )
            ),
            hashlib.sha256(
                (
                    f"{built.eligible_user_count}:{built.eligible_positive_count}:"
                    f"{built.stored_row_index_count}:{built.pair_space_size}"
                ).encode("ascii")
            ).hexdigest(),
        ),
    )
    batch = sampler.sample(100_000, seed=20260828)
    assert batch.pair_count == 100_000
    assert sampler.stored_row_index_count == EXPECTED_ELIGIBLE_TRAIN_ROWS
    assert sampler.stored_row_index_count < official_dataset.train.row_count
    assert sampler.pair_space_size > sampler.stored_row_index_count
    assert sampler_receipt.wall_seconds < 60.0

    fm_receipts = tuple(
        TimingReceipt(
            family="official_fm",
            wall_seconds=run.resources.wall_seconds,
            cpu_seconds=run.resources.cpu_seconds,
            peak_rss_bytes=max(run.resources.peak_rss_bytes, 1),
            rows=official_dataset.train.row_count,
            evidence_digest=run.checkpoint_digest,
        )
        for run in qualification.outer_runs
    )
    fm_p95 = max(item.wall_seconds for item in fm_receipts)
    controller_receipt, controller_seconds = _controller_overhead(fm_p95)
    final_replay = _clean_replay_receipt(qualification_dir)
    receipts = (
        audit_receipt,
        cold_receipt,
        warm_receipt,
        grouping_receipt,
        sampler_receipt,
        *fm_receipts,
        final_replay,
        controller_receipt,
    )
    model_runtime = sum(
        item.wall_seconds for item in receipts if item.family != "controller_admission"
    )
    worst_observed_model_p95 = max(
        cold_receipt.wall_seconds,
        grouping_receipt.wall_seconds,
        fm_p95,
        final_replay.wall_seconds,
    )
    projected = (
        audit_receipt.wall_seconds
        + cold_receipt.wall_seconds
        + 50 * worst_observed_model_p95
        + FINALIZATION_RESERVE_SECONDS
        + controller_seconds
    )
    profile = PerformanceProfile.create(
        receipts=receipts,
        controller_overhead_seconds=controller_seconds,
        model_runtime_seconds=model_runtime,
        finalization_reserve_seconds=FINALIZATION_RESERVE_SECONDS,
        finalization_families=(
            "data_audit",
            "causal_feature_warm",
            "final_replay",
        ),
        projected_campaign_seconds=projected,
        campaign_limit_seconds=CAMPAIGN_LIMIT_SECONDS,
    )

    assert profile.family("official_fm").sample_count == 3
    assert profile.family("causal_feature_cold").p95_seconds < 900.0
    assert profile.family("causal_feature_warm").p95_seconds < 180.0
    assert profile.family("full_data_grouping").p95_seconds < 180.0
    assert profile.family("pairwise_sampler").p95_seconds < 60.0
    assert profile.controller_overhead_ratio < 0.01
    assert profile.finalization_reserve_sufficient
    assert profile.campaign_time_sufficient
    assert max(item.peak_rss_bytes for item in receipts) < 8 * 1024**3

    requested_output = os.environ.get(OUTPUT_ENV)
    if requested_output is not None:
        write_performance_profile(profile, Path(requested_output))

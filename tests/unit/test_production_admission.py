from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kuairand_agent.contract import SplitName, StarterVerification
from kuairand_agent.data.audit import DataAuditReport, FinalOutcomeSkipTrace
from kuairand_agent.data.canonical import (
    OUTCOME_FIELDS,
    CanonicalAlignment,
    CanonicalDataset,
    CanonicalFinalSplit,
    CanonicalInputs,
    CanonicalTrainingSplit,
    CanonicalValidationSplit,
    OutcomeAccessTrace,
    ProtectedTargets,
    TrainingTargets,
)
from kuairand_agent.observability.receipts import StartupReceipt
from kuairand_agent.performance import LoadedPerformanceProfile, PerformanceProfile, TimingReceipt
from kuairand_agent.production import admission
from kuairand_agent.production.admission import (
    ControllerCapabilityReceipt,
    ProductionAdmissionError,
    admit_cpu_fallback,
)
from kuairand_agent.resource_profiles import ResourceProfile, load_resource_profile

ROOT = Path(__file__).parents[2]
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _split(
    name: SplitName,
    *,
    rows: int,
) -> CanonicalTrainingSplit | CanonicalValidationSplit | CanonicalFinalSplit:
    inputs = CanonicalInputs(
        user_id=tuple(f"u-{index}" for index in range(rows)),
        video_id=tuple(f"v-{index}" for index in range(rows)),
        date=tuple(20220410 + index for index in range(rows)),
        duration_ms=tuple(1_000.0 for _ in range(rows)),
        tab=tuple("0" for _ in range(rows)),
        author_id=tuple(f"a-{index}" for index in range(rows)),
        time_ms=tuple(index for index in range(rows)),
        video_type=tuple("NORMAL" for _ in range(rows)),
    )
    alignment = CanonicalAlignment(
        split=name,
        row_id=tuple(range(rows)),
        user_id=inputs.user_id,
        video_id=inputs.video_id,
    )
    parsed: tuple[str, ...] = OUTCOME_FIELDS if name is SplitName.TRAIN else ()
    if name is SplitName.VALID:
        parsed = ("long_view",)
    trace = OutcomeAccessTrace(
        split=name,
        row_count=rows,
        parsed_fields=parsed,
        skipped_fields=tuple(field for field in OUTCOME_FIELDS if field not in parsed),
    )
    if name is SplitName.TRAIN:
        return CanonicalTrainingSplit(
            name=name,
            inputs=inputs,
            alignment=alignment,
            outcome_trace=trace,
            targets=TrainingTargets(
                {field: tuple(0 for _ in range(rows)) for field in OUTCOME_FIELDS}
            ),
        )
    if name is SplitName.VALID:
        return CanonicalValidationSplit(
            name=name,
            inputs=inputs,
            alignment=alignment,
            outcome_trace=trace,
            targets=ProtectedTargets(tuple(index % 2 for index in range(rows))),
        )
    return CanonicalFinalSplit(
        name=name,
        inputs=inputs,
        alignment=alignment,
        outcome_trace=trace,
    )


def _dataset() -> CanonicalDataset:
    return CanonicalDataset(
        train=cast(CanonicalTrainingSplit, _split(SplitName.TRAIN, rows=3)),
        valid=cast(CanonicalValidationSplit, _split(SplitName.VALID, rows=2)),
        final=cast(CanonicalFinalSplit, _split(SplitName.TEST, rows=2)),
        author_map_digest=HASH_A,
    )


def _audit(root: Path) -> DataAuditReport:
    return DataAuditReport(
        data_dir=root,
        sources=(),
        splits=(),
        train_associations={},
        final_outcome_trace=FinalOutcomeSkipTrace(
            member="data/log_standard_4_22_to_5_08_pure.csv",
            split=SplitName.TEST,
            row_count=2,
            skipped_fields=OUTCOME_FIELDS,
            selected_fields=("user_id", "video_id"),
        ),
        field_policy={},
    )


def _performance(
    *,
    audit_digest: str,
    official_digests: tuple[str, str, str],
) -> PerformanceProfile:
    receipt_specs = (
        ("data_audit", 7, audit_digest),
        ("causal_feature_cold", 7, HASH_A),
        ("causal_feature_warm", 7, HASH_A),
        ("full_data_grouping", 3, HASH_A),
        ("pairwise_sampler", 3, HASH_A),
        *(("official_fm", 3, digest) for digest in official_digests),
        ("final_replay", 5, HASH_B),
        ("controller_admission", 1, HASH_C),
    )
    receipts = tuple(
        TimingReceipt(
            family=family,
            wall_seconds=1.0,
            cpu_seconds=1.0,
            peak_rss_bytes=1024,
            rows=rows,
            evidence_digest=digest,
        )
        for family, rows, digest in receipt_specs
    )
    return PerformanceProfile.create(
        receipts=receipts,
        controller_overhead_seconds=0.01,
        model_runtime_seconds=100.0,
        finalization_reserve_seconds=3600.0,
        finalization_families=("data_audit", "causal_feature_warm", "final_replay"),
        projected_campaign_seconds=10_000.0,
        campaign_limit_seconds=21_600.0,
    )


def _qualification(audit_digest: str) -> Any:
    outer = tuple(
        SimpleNamespace(seed=seed, checkpoint_digest=digest)
        for seed, digest in enumerate((HASH_A, HASH_B, HASH_C))
    )
    return SimpleNamespace(
        manifest_digest=HASH_A,
        qualification_input_digest=HASH_B,
        audit_digest=audit_digest,
        outer_runs=outer,
        fallback=SimpleNamespace(seed=4, manifest_digest=HASH_C),
    )


def _startup(*, profile: str = "competition-cpu") -> StartupReceipt:
    return StartupReceipt(
        contract_id=HASH_A,
        expected_contract_id=HASH_A,
        profile=profile,
        verified=True,
    )


def _profile() -> ResourceProfile:
    return load_resource_profile(ROOT / "configs/competition-cpu.toml")


def test_admit_cpu_fallback_returns_typed_path_free_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    starter_root = tmp_path / "starter"
    qualification_root = tmp_path / "qualification"
    for path in (data, starter_root, qualification_root):
        path.mkdir()
    performance_path = tmp_path / "performance.json"
    performance_path.write_bytes(b"{}\n")

    dataset = _dataset()
    audited = _audit(data)
    qualification = _qualification(audited.digest)
    profile = _performance(
        audit_digest=audited.digest,
        official_digests=(HASH_A, HASH_B, HASH_C),
    )
    physical_sha256 = hashlib.sha256(performance_path.read_bytes()).hexdigest()
    monkeypatch.setattr(admission, "audit_dataset", lambda _path: audited)
    monkeypatch.setattr(admission, "load_canonical_dataset", lambda _path: dataset)
    monkeypatch.setattr(
        admission,
        "verify_starter_kit",
        lambda _path: StarterVerification(
            root=starter_root,
            files={"evaluate.py": HASH_A},
            manifest_sha256=HASH_B,
        ),
    )
    monkeypatch.setattr(
        admission,
        "load_official_fm_qualification",
        lambda _path, *, expectations: qualification,
    )
    monkeypatch.setattr(
        admission,
        "load_performance_profile",
        lambda _path: LoadedPerformanceProfile(
            profile=profile,
            physical_sha256=physical_sha256,
            size_bytes=3,
        ),
    )

    profile_path = ROOT / "configs/competition-cpu.toml"
    admitted = admit_cpu_fallback(
        repository_root=ROOT,
        startup_receipt=_startup(),
        resource_profile=_profile(),
        resource_profile_file_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        data_root=data,
        starter_root=starter_root,
        qualification_run_dir=qualification_root,
        performance_profile_path=performance_path,
    )

    assert admitted.runtime.dataset is dataset
    assert admitted.runtime.qualification is qualification
    assert admitted.runtime.fallback.seed == 4
    assert admitted.runtime.provider_request_limit == 0
    assert admitted.runtime.protected_query_limit == 0
    assert admitted.runtime.controller.state_repository_only is True
    assert admitted.runtime.controller.legacy_writer_imports == ()
    assert tuple(item.family for item in admitted.measured_families) == (
        "data_audit",
        "causal_feature_cold",
        "causal_feature_warm",
        "full_data_grouping",
        "pairwise_sampler",
        "official_fm",
        "final_replay",
        "controller_admission",
    )
    encoded = json.dumps(admitted.manifest(), sort_keys=True)
    assert str(tmp_path) not in encoded
    assert (
        admitted.receipt_id
        == admit_cpu_fallback(
            repository_root=ROOT,
            startup_receipt=_startup(),
            resource_profile=_profile(),
            resource_profile_file_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            data_root=data,
            starter_root=starter_root,
            qualification_run_dir=qualification_root,
            performance_profile_path=performance_path,
        ).receipt_id
    )


def test_cpu_profile_rejection_happens_before_data_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_audit(_path: Path) -> None:
        raise AssertionError("data audit must not run for a rejected startup profile")

    monkeypatch.setattr(admission, "audit_dataset", must_not_audit)
    with pytest.raises(ProductionAdmissionError, match="competition-cpu startup"):
        admit_cpu_fallback(
            repository_root=ROOT,
            startup_receipt=_startup(profile="competition-gpu"),
            resource_profile=_profile(),
            resource_profile_file_sha256=HASH_A,
            data_root=ROOT,
            starter_root=ROOT,
            qualification_run_dir=ROOT,
            performance_profile_path=ROOT / "uv.lock",
        )


def test_controller_receipt_cannot_claim_calls_or_legacy_writers() -> None:
    with pytest.raises(ProductionAdmissionError, match="zero provider and protected"):
        ControllerCapabilityReceipt(
            controller_module="kuairand_agent.production.controller",
            controller_source_sha256=HASH_A,
            provider_request_limit=1,
        )
    with pytest.raises(ProductionAdmissionError, match="sole authority"):
        ControllerCapabilityReceipt(
            controller_module="kuairand_agent.production.controller",
            controller_source_sha256=HASH_A,
            legacy_writer_imports=("kuairand_agent.campaign.store",),
        )


def test_controller_source_audit_rejects_a_legacy_writer_import(tmp_path: Path) -> None:
    source = tmp_path / "src/kuairand_agent/production/controller.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from kuairand_agent.state.repository import StateRepository\n"
        "from kuairand_agent.campaign.store import CampaignStore\n",
        encoding="utf-8",
    )

    with pytest.raises(ProductionAdmissionError, match="legacy mutable authority"):
        admission._controller_receipt(tmp_path)

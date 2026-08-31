from __future__ import annotations

import importlib
import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kuairand_agent.resource_profiles import (
    SCIENTIFIC_POLICY,
    MeasuredDuration,
    ResourceProfileError,
    TrainingAdmissionReason,
    load_resource_profile,
    qualify_lightgbm_gpu,
)

ROOT = Path(__file__).parents[2]
CPU_CONFIG = ROOT / "configs" / "competition-cpu.toml"
GPU_CONFIG = ROOT / "configs" / "competition-gpu.toml"


def _duration(receipt_id: str, seconds: float) -> MeasuredDuration:
    return MeasuredDuration(receipt_id=receipt_id, sample_count=5, p95_seconds=seconds)


def test_cpu_and_gpu_profiles_share_one_frozen_scientific_policy() -> None:
    cpu = load_resource_profile(CPU_CONFIG)
    gpu = load_resource_profile(GPU_CONFIG)

    assert cpu.scientific_policy is SCIENTIFIC_POLICY
    assert gpu.scientific_policy is SCIENTIFIC_POLICY
    assert cpu.scientific_policy.digest == gpu.scientific_policy.digest
    assert cpu.scientific_policy.confirmation_seeds == (0, 1, 2)
    assert cpu.scientific_policy.strict_past_features
    assert not cpu.scientific_policy.final_period_outcomes_accessible
    assert cpu.scientific_policy.required_replay_grades == ("SCORING_EXACT", "BUNDLE_EXACT")
    with pytest.raises(FrozenInstanceError):
        cpu.scientific_policy.convergence_patience = 2  # type: ignore[misc]


def test_resource_profiles_have_exact_six_hour_limits_and_distinct_throughput() -> None:
    cpu = load_resource_profile(CPU_CONFIG)
    gpu = load_resource_profile(GPU_CONFIG)

    for profile in (cpu, gpu):
        assert profile.wall_clock_seconds == 21_600
        assert profile.finalization_reserve_seconds == 3_600
        assert profile.latest_training_launch_seconds == 18_000
        assert profile.training_window_seconds == 18_000
        assert profile.max_candidate_processes == 1
        assert profile.threads_per_candidate == 4
        assert profile.candidate_rss_target_bytes == 12 * 1024**3
        assert profile.process_tree_rss_hard_cap_bytes == 14 * 1024**3
        assert profile.candidate_disk_hard_cap_bytes == 20 * 1024**3
        assert profile.requires_measured_p95
        assert not profile.in_attempt_device_fallback
        assert not profile.stock_lightgbm_gpu_qualified

    assert cpu.name == "competition-cpu"
    assert cpu.device == "cpu"
    assert cpu.preferred_backend == "lightgbm-cpu"
    assert cpu.dependency_group == "tree-cpu"
    assert cpu.planned_full_scientific_screens == 4
    assert cpu.planned_frozen_finalists == 1
    assert cpu.planned_protected_evaluations == 1
    assert not cpu.gpu_probe_allowed
    assert not cpu.verified_gpu_build_required

    assert gpu.name == "competition-gpu"
    assert gpu.device == "gpu"
    assert gpu.preferred_backend == "lightgbm-gpu"
    assert gpu.dependency_group == "tree-gpu"
    assert gpu.planned_full_scientific_screens == 8
    assert gpu.planned_frozen_finalists == 2
    assert gpu.planned_protected_evaluations == 2
    assert gpu.gpu_probe_allowed
    assert gpu.verified_gpu_build_required


def test_cpu_profile_loading_does_not_import_or_probe_gpu_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def unexpected_import(name: str) -> object:
        imported.append(name)
        raise AssertionError(f"unexpected dependency import: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)

    profile = load_resource_profile(CPU_CONFIG)

    assert profile.device == "cpu"
    assert imported == []


def test_training_launch_requires_measured_p95_and_fit_before_minute_300() -> None:
    profile = load_resource_profile(CPU_CONFIG)

    missing = profile.admit_training_launch(elapsed_seconds=100, duration=None)
    assert not missing.allowed
    assert missing.reason is TrainingAdmissionReason.MEASUREMENT_REQUIRED

    exact = profile.admit_training_launch(
        elapsed_seconds=17_850,
        duration=_duration("runtime-1", 100),
        cleanup_seconds=50,
    )
    assert exact.allowed
    assert exact.reason is TrainingAdmissionReason.ALLOWED
    assert exact.projected_completion_seconds == 18_000
    assert exact.receipt_ids == ("runtime-1",)

    too_slow = profile.admit_training_launch(
        elapsed_seconds=17_850,
        duration=_duration("runtime-2", 100.001),
        cleanup_seconds=50,
    )
    assert not too_slow.allowed
    assert too_slow.reason is TrainingAdmissionReason.INSUFFICIENT_MEASURED_TIME

    closed = profile.admit_training_launch(
        elapsed_seconds=18_000.001,
        duration=_duration("runtime-3", 1),
    )
    assert not closed.allowed
    assert closed.reason is TrainingAdmissionReason.LAUNCH_WINDOW_CLOSED


def test_protected_eligible_confirmation_bundle_requires_all_frozen_seeds() -> None:
    profile = load_resource_profile(GPU_CONFIG)
    incomplete = profile.admit_confirmation_bundle(
        elapsed_seconds=17_000,
        durations_by_seed={
            0: _duration("seed-0", 100),
            1: _duration("seed-1", 100),
        },
    )
    assert not incomplete.allowed
    assert incomplete.reason is TrainingAdmissionReason.CONFIRMATION_SEEDS_INCOMPLETE

    durations = {
        0: _duration("seed-0", 100),
        1: _duration("seed-1", 100),
        2: _duration("seed-2", 100),
    }
    admitted = profile.admit_confirmation_bundle(
        elapsed_seconds=17_670,
        durations_by_seed=durations,
        cleanup_seconds_per_seed=10,
    )
    assert admitted.allowed
    assert admitted.projected_completion_seconds == 18_000
    assert admitted.receipt_ids == ("seed-0", "seed-1", "seed-2")

    rejected = profile.admit_confirmation_bundle(
        elapsed_seconds=17_671,
        durations_by_seed=durations,
        cleanup_seconds_per_seed=10,
    )
    assert not rejected.allowed
    assert rejected.reason is TrainingAdmissionReason.INSUFFICIENT_MEASURED_TIME


def test_resource_profile_loader_fails_closed_on_policy_or_resource_drift(
    tmp_path: Path,
) -> None:
    raw = tomllib.loads(CPU_CONFIG.read_text(encoding="utf-8"))
    raw["scientific_policy"]["practical_improvement_margin"] = 0.0
    changed_policy = CPU_CONFIG.read_text(encoding="utf-8").replace(
        "practical_improvement_margin = 0.002",
        "practical_improvement_margin = 0.0",
    )
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(changed_policy, encoding="utf-8")
    with pytest.raises(ResourceProfileError, match="practical_improvement_margin"):
        load_resource_profile(policy_path)

    changed_resource = CPU_CONFIG.read_text(encoding="utf-8").replace(
        "threads_per_candidate = 4",
        "threads_per_candidate = 8",
    )
    resource_path = tmp_path / "resource.toml"
    resource_path.write_text(changed_resource, encoding="utf-8")
    with pytest.raises(ResourceProfileError, match="threads_per_candidate"):
        load_resource_profile(resource_path)


def test_gpu_qualification_requires_a_runtime_probe_not_a_version_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)

    result = qualify_lightgbm_gpu()

    assert not result.qualified
    assert result.backend_identity == "unavailable"
    assert "dependency unavailable" in result.reason


@pytest.mark.parametrize(
    ("receipt_id", "sample_count", "p95_seconds"),
    [
        ("", 1, 1.0),
        ("receipt", 0, 1.0),
        ("receipt", 1, 0.0),
        ("receipt", 1, float("inf")),
    ],
)
def test_measured_duration_receipts_are_strict(
    receipt_id: str,
    sample_count: int,
    p95_seconds: float,
) -> None:
    with pytest.raises(ResourceProfileError):
        MeasuredDuration(
            receipt_id=receipt_id,
            sample_count=sample_count,
            p95_seconds=p95_seconds,
        )

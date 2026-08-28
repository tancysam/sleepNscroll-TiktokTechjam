from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kuairand_agent.performance import (
    PerformanceAcceptanceError,
    PerformanceProfile,
    TimingReceipt,
    measure_operation,
    write_performance_profile,
)


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def _receipt(family: str, wall_seconds: float, *, rows: int = 100) -> TimingReceipt:
    return TimingReceipt(
        family=family,
        wall_seconds=wall_seconds,
        cpu_seconds=wall_seconds / 2,
        peak_rss_bytes=4096,
        rows=rows,
        evidence_digest=_digest(f"{family}-{wall_seconds}"),
    )


def test_profile_uses_median_and_nearest_rank_p95_and_checks_reserve() -> None:
    profile = PerformanceProfile.create(
        receipts=(
            _receipt("tree", 1.0),
            _receipt("tree", 2.0),
            _receipt("tree", 4.0),
            _receipt("tree", 8.0),
            _receipt("feature_cache_warm", 0.2),
            _receipt("final_replay", 10.0),
            _receipt("submission_check", 5.0),
        ),
        controller_overhead_seconds=0.25,
        model_runtime_seconds=25.0,
        finalization_reserve_seconds=20.0,
        finalization_families=("final_replay", "submission_check"),
        projected_campaign_seconds=120.0,
        campaign_limit_seconds=360.0,
    )

    tree = profile.family("tree")
    assert tree.sample_count == 4
    assert tree.p50_seconds == 3.0
    assert tree.p95_seconds == 8.0
    assert tree.peak_rss_bytes == 4096
    assert profile.controller_overhead_ratio == 0.01
    assert profile.finalization_p95_seconds == 15.0
    assert profile.finalization_reserve_sufficient is True
    assert profile.campaign_time_sufficient is True


def test_profile_fails_closed_for_unknown_finalization_family_and_bad_receipt() -> None:
    with pytest.raises(PerformanceAcceptanceError, match="finalization family"):
        PerformanceProfile.create(
            receipts=(_receipt("tree", 1.0),),
            controller_overhead_seconds=0.1,
            model_runtime_seconds=1.0,
            finalization_reserve_seconds=10.0,
            finalization_families=("missing",),
            projected_campaign_seconds=2.0,
            campaign_limit_seconds=20.0,
        )
    with pytest.raises(PerformanceAcceptanceError, match="evidence_digest"):
        TimingReceipt("tree", 1.0, 0.5, 1, 1, "not-a-digest")


def test_measure_operation_binds_result_receipt_and_profile_writes_canonical_json(
    tmp_path: Path,
) -> None:
    receipt, result = measure_operation(
        family="causal_feature_cold",
        rows=3,
        operation=lambda: ("feature-result", _digest("feature-result")),
    )
    assert result == "feature-result"
    assert receipt.evidence_digest == _digest("feature-result")
    assert receipt.wall_seconds >= 0.0
    assert receipt.cpu_seconds >= 0.0
    assert receipt.peak_rss_bytes > 0

    profile = PerformanceProfile.create(
        receipts=(receipt, _receipt("final_replay", 1.0)),
        controller_overhead_seconds=0.001,
        model_runtime_seconds=1.0,
        finalization_reserve_seconds=10.0,
        finalization_families=("final_replay",),
        projected_campaign_seconds=2.0,
        campaign_limit_seconds=20.0,
    )
    output = write_performance_profile(profile, tmp_path / "performance.json")
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload) == profile.manifest()

    with pytest.raises(PerformanceAcceptanceError, match="refusing to overwrite"):
        write_performance_profile(profile, output)

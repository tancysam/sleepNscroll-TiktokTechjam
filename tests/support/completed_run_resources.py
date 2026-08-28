"""Fail-closed acceptance helpers for retained production resource evidence.

This module deliberately parses only measurements emitted by production.  It never fills a
missing phase with a guessed duration, zero CPU, or sentinel memory value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

from kuairand_agent.performance import TimingReceipt

CAMPAIGN_LIMIT_SECONDS = 21_600.0
FINALIZATION_RESERVE_SECONDS = 3_600.0
MAXIMUM_PEAK_RSS_BYTES = 8 * 1024**3
FINALIZATION_COVERAGE = (
    "canonical_context_reopen",
    "generated_final_training_if_selected",
    "validation_replay",
    "final_inference",
    "clean_replay_evidence",
    "untouched_organizer_check",
    "judge_report_and_reproduce_script",
    "bundle_construction_fsync_publication_and_closure",
)


class CompletedRunResourceError(AssertionError):
    """Retained production evidence cannot prove the performance acceptance contract."""


def _mapping(value: object, name: str, expected: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CompletedRunResourceError(f"{name} must be a string-keyed mapping")
    if set(value) != expected:
        raise CompletedRunResourceError(
            f"{name} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}"
        )
    return value


def _real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CompletedRunResourceError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise CompletedRunResourceError(f"{name} must be finite and {qualifier}")
    return result


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CompletedRunResourceError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise CompletedRunResourceError(f"{name} must be {qualifier}")
    return result


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompletedRunResourceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _truth(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise CompletedRunResourceError(f"{name} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class FinalTrainingResourceReceipt:
    wall_seconds: float
    peak_rss_bytes: int
    disk_bytes: int
    device: str
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class CompletedFinalizationResources:
    campaign_elapsed_seconds: float
    finalization_started_elapsed_seconds: float
    finalization_elapsed_seconds: float
    local_monotonic_wall_seconds: float
    aggregate: TimingReceipt
    final_training: FinalTrainingResourceReceipt | None


def validate_completed_finalization_resources(
    raw: object,
    *,
    generated_selection: bool,
    expected_bundle_manifest_sha256: str,
    training_replay: object,
) -> CompletedFinalizationResources:
    """Validate actual aggregate finalization evidence and its generated-training child.

    The aggregate is intentionally authoritative for reserve accounting because it covers the
    inclusive finalization interval.  Detailed generated training is checked for identity and
    resource retention, but is not added again to the interval and therefore cannot be double
    counted.
    """

    expected_bundle = _digest(
        expected_bundle_manifest_sha256,
        "expected bundle manifest SHA-256",
    )
    common_fields = {
        "schema_version",
        "clock_basis",
        "campaign_elapsed_seconds",
        "hard_wall_seconds",
        "finalization_reserve_seconds",
        "finalization_started_elapsed_seconds",
        "finalization_elapsed_seconds",
        "coverage",
        "within_reserve",
        "within_hard_wall",
        "aggregate",
    }
    fields = common_fields | ({"final_training"} if generated_selection else set())
    evidence = _mapping(raw, "resource_evidence", fields)
    if evidence["schema_version"] != 1:
        raise CompletedRunResourceError("resource_evidence schema_version must be 1")
    if evidence["clock_basis"] != "durable_max_of_monotonic_and_utc_elapsed":
        raise CompletedRunResourceError("resource_evidence clock basis is not durable")
    coverage = evidence["coverage"]
    if not isinstance(coverage, list) or tuple(coverage) != FINALIZATION_COVERAGE:
        raise CompletedRunResourceError("resource_evidence coverage is incomplete or reordered")

    hard_wall = _real(evidence["hard_wall_seconds"], "hard_wall_seconds", positive=True)
    reserve = _real(
        evidence["finalization_reserve_seconds"],
        "finalization_reserve_seconds",
        positive=True,
    )
    if hard_wall != CAMPAIGN_LIMIT_SECONDS:
        raise CompletedRunResourceError("production hard wall differs from 21,600 seconds")
    if reserve != FINALIZATION_RESERVE_SECONDS:
        raise CompletedRunResourceError("finalization reserve differs from 3,600 seconds")
    campaign_elapsed = _real(
        evidence["campaign_elapsed_seconds"],
        "campaign_elapsed_seconds",
        positive=True,
    )
    finalization_started = _real(
        evidence["finalization_started_elapsed_seconds"],
        "finalization_started_elapsed_seconds",
    )
    finalization_elapsed = _real(
        evidence["finalization_elapsed_seconds"],
        "finalization_elapsed_seconds",
        positive=True,
    )
    if not math.isclose(
        campaign_elapsed - finalization_started,
        finalization_elapsed,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise CompletedRunResourceError("finalization elapsed differs from durable campaign delta")
    if finalization_started > hard_wall - reserve:
        raise CompletedRunResourceError("finalization started after the protected reserve began")
    expected_within_reserve = finalization_elapsed <= reserve
    expected_within_wall = campaign_elapsed <= hard_wall
    if _truth(evidence["within_reserve"], "within_reserve") != expected_within_reserve:
        raise CompletedRunResourceError("within_reserve differs from measured elapsed time")
    if _truth(evidence["within_hard_wall"], "within_hard_wall") != expected_within_wall:
        raise CompletedRunResourceError("within_hard_wall differs from measured elapsed time")
    if not expected_within_wall:
        raise CompletedRunResourceError("production campaign exceeded the 21,600-second hard wall")
    if not expected_within_reserve:
        raise CompletedRunResourceError("production finalization exceeded the 3,600-second reserve")

    aggregate = _mapping(
        evidence["aggregate"],
        "resource_evidence.aggregate",
        {
            "family",
            "wall_seconds",
            "local_monotonic_wall_seconds",
            "cpu_seconds",
            "peak_rss_bytes",
            "rows",
            "evidence_digest",
            "rss_accounting",
        },
    )
    if aggregate["family"] != "production_finalization_total":
        raise CompletedRunResourceError("aggregate family is not production_finalization_total")
    if aggregate["rss_accounting"] != (
        "conservative_sum_of_process_lifetime_self_and_child_high_water_marks"
    ):
        raise CompletedRunResourceError("aggregate RSS accounting disclosure changed")
    local_monotonic = _real(
        aggregate["local_monotonic_wall_seconds"],
        "aggregate local_monotonic_wall_seconds",
        positive=True,
    )
    wall_seconds = _real(
        aggregate["wall_seconds"],
        "aggregate wall_seconds",
        positive=True,
    )
    if wall_seconds != finalization_elapsed:
        raise CompletedRunResourceError(
            "aggregate wall time differs from durable finalization time"
        )
    if local_monotonic > wall_seconds + 1e-9:
        raise CompletedRunResourceError("local monotonic time exceeds conservative durable time")
    aggregate_receipt = TimingReceipt(
        family="production_finalization_total",
        wall_seconds=wall_seconds,
        cpu_seconds=_real(aggregate["cpu_seconds"], "aggregate cpu_seconds"),
        peak_rss_bytes=_integer(
            aggregate["peak_rss_bytes"],
            "aggregate peak_rss_bytes",
            positive=True,
        ),
        rows=_integer(aggregate["rows"], "aggregate rows", positive=True),
        evidence_digest=_digest(
            aggregate["evidence_digest"],
            "aggregate evidence_digest",
        ),
    )
    if aggregate_receipt.evidence_digest != expected_bundle:
        raise CompletedRunResourceError("aggregate receipt is not bound to the closed bundle")
    if aggregate_receipt.peak_rss_bytes >= MAXIMUM_PEAK_RSS_BYTES:
        raise CompletedRunResourceError("production finalization exceeded the 8-GiB RSS ceiling")

    replay = _mapping(
        training_replay,
        "training_replay",
        set(training_replay) if isinstance(training_replay, Mapping) else set(),
    )
    final_training: FinalTrainingResourceReceipt | None = None
    if generated_selection:
        if replay.get("required") is not True or replay.get("completed") is not True:
            raise CompletedRunResourceError("generated selection lacks completed final training")
        if replay.get("exact_checkpoint_bytes") is not True:
            raise CompletedRunResourceError("generated final training did not replay exact bytes")
        if replay.get("charged_launch") is not True:
            raise CompletedRunResourceError("generated final training was not durably charged")
        checkpoint = _digest(
            replay.get("checkpoint_sha256"),
            "training_replay checkpoint_sha256",
        )
        if checkpoint != _digest(
            replay.get("selected_checkpoint_sha256"),
            "training_replay selected_checkpoint_sha256",
        ):
            raise CompletedRunResourceError("final-training checkpoint differs from selection")
        replay_resources = _mapping(
            replay.get("resources"),
            "training_replay.resources",
            {"wall_seconds", "peak_rss_bytes", "disk_bytes"},
        )
        retained = _mapping(
            evidence["final_training"],
            "resource_evidence.final_training",
            {
                "family",
                "wall_seconds",
                "peak_rss_bytes",
                "disk_bytes",
                "device",
                "evidence_digest",
            },
        )
        if retained["family"] != "generated_final_training_replay":
            raise CompletedRunResourceError("final-training family changed")
        if retained["device"] != replay.get("device") or retained["device"] != "cpu":
            raise CompletedRunResourceError("final-training device differs from retained execution")
        final_training = FinalTrainingResourceReceipt(
            wall_seconds=_real(
                retained["wall_seconds"],
                "final_training wall_seconds",
                positive=True,
            ),
            peak_rss_bytes=_integer(
                retained["peak_rss_bytes"],
                "final_training peak_rss_bytes",
                positive=True,
            ),
            disk_bytes=_integer(
                retained["disk_bytes"],
                "final_training disk_bytes",
                positive=True,
            ),
            device="cpu",
            evidence_digest=_digest(
                retained["evidence_digest"],
                "final_training evidence_digest",
            ),
        )
        if final_training.evidence_digest != checkpoint:
            raise CompletedRunResourceError("final-training receipt is not checkpoint-bound")
        if final_training.wall_seconds != _real(
            replay_resources["wall_seconds"],
            "training_replay resource wall_seconds",
            positive=True,
        ):
            raise CompletedRunResourceError("final-training wall receipts differ")
        if final_training.peak_rss_bytes != _integer(
            replay_resources["peak_rss_bytes"],
            "training_replay resource peak_rss_bytes",
            positive=True,
        ):
            raise CompletedRunResourceError("final-training RSS receipts differ")
        if final_training.disk_bytes != _integer(
            replay_resources["disk_bytes"],
            "training_replay resource disk_bytes",
            positive=True,
        ):
            raise CompletedRunResourceError("final-training disk receipts differ")
        if final_training.wall_seconds > finalization_elapsed:
            raise CompletedRunResourceError("final-training time exceeds total finalization time")
        if final_training.peak_rss_bytes >= MAXIMUM_PEAK_RSS_BYTES:
            raise CompletedRunResourceError("final training exceeded the 8-GiB RSS ceiling")
    elif "final_training" in evidence:
        raise CompletedRunResourceError("fallback resource evidence contains generated training")

    return CompletedFinalizationResources(
        campaign_elapsed_seconds=campaign_elapsed,
        finalization_started_elapsed_seconds=finalization_started,
        finalization_elapsed_seconds=finalization_elapsed,
        local_monotonic_wall_seconds=local_monotonic,
        aggregate=aggregate_receipt,
        final_training=final_training,
    )


__all__ = [
    "CAMPAIGN_LIMIT_SECONDS",
    "FINALIZATION_COVERAGE",
    "FINALIZATION_RESERVE_SECONDS",
    "CompletedFinalizationResources",
    "CompletedRunResourceError",
    "FinalTrainingResourceReceipt",
    "validate_completed_finalization_resources",
]

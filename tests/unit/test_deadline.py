from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from kuairand_agent.campaign.clock import (
    NANOSECONDS_PER_SECOND,
    DeadlineAdmissionReason,
    DeadlineError,
    DeadlineState,
    SystemClock,
)


@dataclass
class FakeClock:
    monotonic_value_ns: int
    utc_value: datetime
    boot_value: str = "boot-a"

    def monotonic_ns(self) -> int:
        return self.monotonic_value_ns

    def utc_now(self) -> datetime:
        return self.utc_value

    def boot_identity(self) -> str:
        return self.boot_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value_ns += int(seconds * NANOSECONDS_PER_SECOND)
        self.utc_value += timedelta(seconds=seconds)


def _clock() -> FakeClock:
    return FakeClock(10 * NANOSECONDS_PER_SECOND, datetime(2026, 8, 28, tzinfo=UTC))


def test_original_monotonic_deadline_survives_manifest_restart() -> None:
    clock = _clock()
    original = DeadlineState.start(clock)
    original_deadline = original.monotonic_deadline_ns
    clock.advance(1_234)
    observed = original.observe(clock)
    persisted = json.loads(json.dumps(observed.state.manifest(), sort_keys=True))
    resumed = DeadlineState.from_manifest(persisted)
    assert resumed.monotonic_deadline_ns == original_deadline
    assert resumed.started_utc == original.started_utc
    assert resumed.observe(clock).remaining_seconds == 21_600 - 1_234


def test_boot_change_falls_back_to_original_utc_deadline_and_counts_downtime() -> None:
    clock = _clock()
    deadline = DeadlineState.start(clock)
    clock.advance(600)
    first = deadline.observe(clock)

    clock.boot_value = "boot-b"
    clock.monotonic_value_ns = 5 * NANOSECONDS_PER_SECOND
    clock.utc_value += timedelta(seconds=1_400)
    rebooted = first.state.observe(clock)
    assert not rebooted.same_boot
    assert rebooted.elapsed_seconds == 2_000
    assert rebooted.remaining_seconds == 19_600
    assert rebooted.state.monotonic_deadline_ns == deadline.monotonic_deadline_ns


def test_suspend_or_process_downtime_cannot_grant_new_time() -> None:
    clock = _clock()
    deadline = DeadlineState.start(clock)
    clock.advance(8_000)
    checkpoint = deadline.observe(clock).state
    clock.advance(2_000)  # time passes while no controller process is running
    resumed = checkpoint.observe(clock)
    assert resumed.elapsed_seconds == 10_000
    assert resumed.remaining_seconds == 11_600


def test_wall_clock_and_monotonic_rollback_fail_closed() -> None:
    clock = _clock()
    state = DeadlineState.start(clock)
    clock.advance(10)
    state = state.observe(clock).state

    clock.utc_value -= timedelta(seconds=1)
    with pytest.raises(DeadlineError, match="UTC wall clock moved backwards"):
        state.observe(clock)

    clock.utc_value += timedelta(seconds=1)
    clock.monotonic_value_ns -= NANOSECONDS_PER_SECOND
    with pytest.raises(DeadlineError, match="monotonic clock moved backwards"):
        state.observe(clock)


def test_research_stops_when_only_finalization_reserve_remains() -> None:
    clock = _clock()
    state = DeadlineState.start(clock)
    clock.advance(18_000)
    observation = state.observe(clock)
    assert observation.remaining_seconds == 3_600
    assert observation.finalization_reserve_active
    research = observation.admit_research(p95_runtime_seconds=0)
    assert not research.allowed
    assert research.reason is DeadlineAdmissionReason.FINALIZATION_RESERVE

    finalization = observation.admit_required_completion(p95_runtime_seconds=3_600)
    assert finalization.allowed


def test_exact_p95_cleanup_and_reserve_edge_is_deterministic() -> None:
    clock = _clock()
    state = DeadlineState.start(clock)
    clock.advance(17_000)
    observation = state.observe(clock)
    assert observation.remaining_seconds == 4_600
    exact = observation.admit_research(p95_runtime_seconds=900, cleanup_seconds=100)
    assert exact.allowed
    assert exact.required_seconds == 4_600
    too_slow = observation.admit_research(p95_runtime_seconds=900.000_001, cleanup_seconds=100)
    assert not too_slow.allowed
    assert too_slow.reason is DeadlineAdmissionReason.FINALIZATION_RESERVE


def test_hard_six_hour_ceiling_marks_work_expired() -> None:
    clock = _clock()
    state = DeadlineState.start(clock)
    clock.advance(21_600)
    observation = state.observe(clock)
    assert observation.elapsed_seconds == 21_600
    assert observation.remaining_seconds == 0
    assert observation.hard_expired
    assert (
        observation.admit_required_completion(p95_runtime_seconds=0).reason
        is DeadlineAdmissionReason.HARD_DEADLINE
    )


def test_deadline_manifest_is_strict_and_reserve_is_at_least_one_hour() -> None:
    clock = _clock()
    state = DeadlineState.start(clock)
    invalid = state.manifest()
    invalid["unexpected"] = 1
    with pytest.raises(DeadlineError, match="unknown"):
        DeadlineState.from_manifest(invalid)
    with pytest.raises(DeadlineError, match="at least 3600"):
        DeadlineState.start(clock, wall_clock_seconds=10_000, finalization_reserve_seconds=3_599)


def test_system_clock_has_stable_unprivileged_boot_identity_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied() -> float:
        raise PermissionError("sandbox denied kern.boottime")

    monkeypatch.setattr("kuairand_agent.campaign.clock.psutil.boot_time", denied)
    monkeypatch.setattr("kuairand_agent.campaign.clock.time.time", lambda: 10_000.2)
    monkeypatch.setattr("kuairand_agent.campaign.clock.time.monotonic", lambda: 100.1)

    first = SystemClock().boot_identity()
    second = SystemClock().boot_identity()

    assert first == second
    assert len(first) == 64

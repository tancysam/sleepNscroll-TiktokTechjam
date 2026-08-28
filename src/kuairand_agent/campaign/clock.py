"""Restart-safe monotonic six-hour campaign deadline policy."""

from __future__ import annotations

import hashlib
import math
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final, Protocol, Self

import psutil  # type: ignore[import-untyped]

DEADLINE_SCHEMA_VERSION: Final = 1
MAX_WALL_CLOCK_SECONDS: Final = 21_600
MIN_FINALIZATION_RESERVE_SECONDS: Final = 3_600
NANOSECONDS_PER_SECOND: Final = 1_000_000_000


class DeadlineError(ValueError):
    """Raised when deadline evidence is invalid or time moves backwards."""


class Clock(Protocol):
    """The narrow time seam used by the campaign and deterministic tests."""

    def monotonic_ns(self) -> int: ...

    def utc_now(self) -> datetime: ...

    def boot_identity(self) -> str: ...


class SystemClock:
    """Production clock using the host monotonic counter and UTC wall clock."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def utc_now(self) -> datetime:
        return datetime.now(UTC)

    def boot_identity(self) -> str:
        try:
            boot_epoch = round(float(psutil.boot_time()))
            source = "psutil"
        except (OSError, PermissionError, psutil.Error):
            # Sandboxed macOS processes can be denied the kern.boottime sysctl used by psutil.
            # The wall-clock epoch corresponding to monotonic zero is stable across processes on
            # one boot.  Rounding to a whole second removes sampling jitter; if the wall clock is
            # adjusted enough to change it, resume treats the identity as a boot change and falls
            # back to the original UTC deadline rather than granting more time.
            boot_epoch = round(time.time() - time.monotonic())
            source = "wall-minus-monotonic"
        evidence = f"{platform.node()}\0{source}\0{boot_epoch}".encode()
        return hashlib.sha256(evidence).hexdigest()


class DeadlineAdmissionReason(StrEnum):
    ALLOWED = "allowed"
    HARD_DEADLINE = "hard_deadline"
    FINALIZATION_RESERVE = "finalization_reserve"


def _seconds(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeadlineError(f"{location} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise DeadlineError(f"{location} must be a non-negative finite number")
    return result


def _aware_utc(value: object, location: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DeadlineError(f"{location} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _boot_identity(value: object) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise DeadlineError("boot_identity must be a non-empty string without NUL bytes")
    return value


@dataclass(frozen=True, slots=True)
class DeadlineAdmission:
    allowed: bool
    reason: DeadlineAdmissionReason
    remaining_seconds: float
    required_seconds: float

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise DeadlineError("allowed must be boolean")
        if not isinstance(self.reason, DeadlineAdmissionReason):
            raise DeadlineError("reason must be DeadlineAdmissionReason")
        if self.allowed != (self.reason is DeadlineAdmissionReason.ALLOWED):
            raise DeadlineError("allowed and admission reason disagree")
        _seconds(self.remaining_seconds, "remaining_seconds")
        _seconds(self.required_seconds, "required_seconds")


@dataclass(frozen=True, slots=True)
class DeadlineObservation:
    """One conservative observation plus the state that must be persisted."""

    state: DeadlineState
    elapsed_seconds: float
    remaining_seconds: float
    same_boot: bool

    @property
    def hard_expired(self) -> bool:
        return self.remaining_seconds <= 0.0

    @property
    def finalization_reserve_active(self) -> bool:
        return self.remaining_seconds <= self.state.finalization_reserve_seconds

    def admit_research(
        self, *, p95_runtime_seconds: float, cleanup_seconds: float = 0.0
    ) -> DeadlineAdmission:
        p95 = _seconds(p95_runtime_seconds, "p95_runtime_seconds")
        cleanup = _seconds(cleanup_seconds, "cleanup_seconds")
        base_required = p95 + cleanup
        required = base_required + self.state.finalization_reserve_seconds
        if self.hard_expired or base_required > self.remaining_seconds:
            reason = DeadlineAdmissionReason.HARD_DEADLINE
        elif self.finalization_reserve_active or required > self.remaining_seconds:
            reason = DeadlineAdmissionReason.FINALIZATION_RESERVE
        else:
            reason = DeadlineAdmissionReason.ALLOWED
        return DeadlineAdmission(
            allowed=reason is DeadlineAdmissionReason.ALLOWED,
            reason=reason,
            remaining_seconds=self.remaining_seconds,
            required_seconds=required,
        )

    def admit_required_completion(
        self, *, p95_runtime_seconds: float, cleanup_seconds: float = 0.0
    ) -> DeadlineAdmission:
        required = _seconds(p95_runtime_seconds, "p95_runtime_seconds") + _seconds(
            cleanup_seconds, "cleanup_seconds"
        )
        reason = (
            DeadlineAdmissionReason.ALLOWED
            if not self.hard_expired and required <= self.remaining_seconds
            else DeadlineAdmissionReason.HARD_DEADLINE
        )
        return DeadlineAdmission(
            allowed=reason is DeadlineAdmissionReason.ALLOWED,
            reason=reason,
            remaining_seconds=self.remaining_seconds,
            required_seconds=required,
        )


@dataclass(frozen=True, slots=True)
class DeadlineState:
    """Persisted evidence for one deadline that can never reset on resume."""

    wall_clock_seconds: int
    finalization_reserve_seconds: int
    started_utc: datetime
    utc_deadline: datetime
    original_boot_identity: str
    monotonic_started_ns: int
    monotonic_deadline_ns: int
    last_observed_utc: datetime
    last_observed_monotonic_ns: int
    last_elapsed_seconds: float = 0.0
    schema_version: int = DEADLINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.wall_clock_seconds) is not int or not (
            1 <= self.wall_clock_seconds <= MAX_WALL_CLOCK_SECONDS
        ):
            raise DeadlineError("wall_clock_seconds must be an integer in [1, 21600]")
        if type(self.finalization_reserve_seconds) is not int or not (
            MIN_FINALIZATION_RESERVE_SECONDS
            <= self.finalization_reserve_seconds
            < self.wall_clock_seconds
        ):
            raise DeadlineError(
                "finalization_reserve_seconds must be at least 3600 and below wall clock"
            )
        started = _aware_utc(self.started_utc, "started_utc")
        deadline = _aware_utc(self.utc_deadline, "utc_deadline")
        last_utc = _aware_utc(self.last_observed_utc, "last_observed_utc")
        if (
            started != self.started_utc
            or deadline != self.utc_deadline
            or last_utc != self.last_observed_utc
        ):
            raise DeadlineError("deadline datetimes must be normalized to UTC")
        if deadline != started + timedelta(seconds=self.wall_clock_seconds):
            raise DeadlineError("utc_deadline does not match the original wall-clock budget")
        if last_utc < started:
            raise DeadlineError("last_observed_utc cannot precede campaign start")
        _boot_identity(self.original_boot_identity)
        if type(self.monotonic_started_ns) is not int or self.monotonic_started_ns < 0:
            raise DeadlineError("monotonic_started_ns must be a non-negative integer")
        if type(self.monotonic_deadline_ns) is not int or self.monotonic_deadline_ns != (
            self.monotonic_started_ns + self.wall_clock_seconds * NANOSECONDS_PER_SECOND
        ):
            raise DeadlineError("monotonic_deadline_ns does not match the original budget")
        if (
            type(self.last_observed_monotonic_ns) is not int
            or self.last_observed_monotonic_ns < self.monotonic_started_ns
        ):
            raise DeadlineError("last observed monotonic time cannot precede campaign start")
        elapsed = _seconds(self.last_elapsed_seconds, "last_elapsed_seconds")
        if elapsed > self.wall_clock_seconds:
            raise DeadlineError("last_elapsed_seconds cannot exceed wall-clock budget")
        if self.schema_version != DEADLINE_SCHEMA_VERSION:
            raise DeadlineError(f"unsupported deadline schema_version {self.schema_version}")

    @classmethod
    def start(
        cls,
        clock: Clock,
        *,
        wall_clock_seconds: int = MAX_WALL_CLOCK_SECONDS,
        finalization_reserve_seconds: int = MIN_FINALIZATION_RESERVE_SECONDS,
    ) -> Self:
        now_mono = clock.monotonic_ns()
        if type(now_mono) is not int or now_mono < 0:
            raise DeadlineError("clock monotonic_ns must return a non-negative integer")
        now_utc = _aware_utc(clock.utc_now(), "clock utc_now")
        boot = _boot_identity(clock.boot_identity())
        return cls(
            wall_clock_seconds=wall_clock_seconds,
            finalization_reserve_seconds=finalization_reserve_seconds,
            started_utc=now_utc,
            utc_deadline=now_utc + timedelta(seconds=wall_clock_seconds),
            original_boot_identity=boot,
            monotonic_started_ns=now_mono,
            monotonic_deadline_ns=now_mono + wall_clock_seconds * NANOSECONDS_PER_SECOND,
            last_observed_utc=now_utc,
            last_observed_monotonic_ns=now_mono,
        )

    def observe(self, clock: Clock) -> DeadlineObservation:
        """Observe conservatively and return the advanced state for durable persistence.

        When the boot identity still matches, the larger of monotonic and UTC elapsed time is
        used.  After a boot change, only the original UTC deadline is comparable.  A backwards
        wall or monotonic clock fails closed instead of granting extra campaign time.
        """

        now_utc = _aware_utc(clock.utc_now(), "clock utc_now")
        if now_utc < self.last_observed_utc:
            raise DeadlineError("UTC wall clock moved backwards; deadline fails closed")
        now_mono = clock.monotonic_ns()
        if type(now_mono) is not int or now_mono < 0:
            raise DeadlineError("clock monotonic_ns must return a non-negative integer")
        current_boot = _boot_identity(clock.boot_identity())
        same_boot = current_boot == self.original_boot_identity
        utc_elapsed = max(0.0, (now_utc - self.started_utc).total_seconds())
        if same_boot:
            if now_mono < self.last_observed_monotonic_ns:
                raise DeadlineError("monotonic clock moved backwards; deadline fails closed")
            monotonic_elapsed = (now_mono - self.monotonic_started_ns) / NANOSECONDS_PER_SECOND
            elapsed = max(self.last_elapsed_seconds, utc_elapsed, monotonic_elapsed)
            persisted_mono = now_mono
        else:
            elapsed = max(self.last_elapsed_seconds, utc_elapsed)
            # The new boot's monotonic epoch is incomparable; retain the last trusted value.
            persisted_mono = self.last_observed_monotonic_ns
        elapsed = min(float(self.wall_clock_seconds), elapsed)
        remaining = max(0.0, self.wall_clock_seconds - elapsed)
        advanced = replace(
            self,
            last_observed_utc=now_utc,
            last_observed_monotonic_ns=persisted_mono,
            last_elapsed_seconds=elapsed,
        )
        return DeadlineObservation(
            state=advanced,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            same_boot=same_boot,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "wall_clock_seconds": self.wall_clock_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "started_utc": self.started_utc.isoformat().replace("+00:00", "Z"),
            "utc_deadline": self.utc_deadline.isoformat().replace("+00:00", "Z"),
            "original_boot_identity": self.original_boot_identity,
            "monotonic_started_ns": self.monotonic_started_ns,
            "monotonic_deadline_ns": self.monotonic_deadline_ns,
            "last_observed_utc": self.last_observed_utc.isoformat().replace("+00:00", "Z"),
            "last_observed_monotonic_ns": self.last_observed_monotonic_ns,
            "last_elapsed_seconds": self.last_elapsed_seconds,
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        expected = {
            "schema_version",
            "wall_clock_seconds",
            "finalization_reserve_seconds",
            "started_utc",
            "utc_deadline",
            "original_boot_identity",
            "monotonic_started_ns",
            "monotonic_deadline_ns",
            "last_observed_utc",
            "last_observed_monotonic_ns",
            "last_elapsed_seconds",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            issue = "unknown" if unknown else "missing"
            fields = unknown or missing
            raise DeadlineError(f"{issue} deadline field(s): {', '.join(sorted(fields))}")

        def integer(name: str) -> int:
            value = raw[name]
            if type(value) is not int:
                raise DeadlineError(f"{name} must be an integer")
            return value

        def timestamp(name: str) -> datetime:
            value = raw[name]
            if type(value) is not str:
                raise DeadlineError(f"{name} must be an ISO-8601 string")
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise DeadlineError(f"{name} must be a valid ISO-8601 string") from exc
            return _aware_utc(parsed, name)

        return cls(
            schema_version=integer("schema_version"),
            wall_clock_seconds=integer("wall_clock_seconds"),
            finalization_reserve_seconds=integer("finalization_reserve_seconds"),
            started_utc=timestamp("started_utc"),
            utc_deadline=timestamp("utc_deadline"),
            original_boot_identity=_boot_identity(raw["original_boot_identity"]),
            monotonic_started_ns=integer("monotonic_started_ns"),
            monotonic_deadline_ns=integer("monotonic_deadline_ns"),
            last_observed_utc=timestamp("last_observed_utc"),
            last_observed_monotonic_ns=integer("last_observed_monotonic_ns"),
            last_elapsed_seconds=_seconds(raw["last_elapsed_seconds"], "last_elapsed_seconds"),
        )

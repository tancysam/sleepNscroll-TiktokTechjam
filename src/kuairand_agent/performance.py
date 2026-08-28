"""Deterministic performance acceptance receipts for full-data campaign evidence.

Wall-clock and memory observations are deliberately excluded from model, data, and prediction
identity.  Each observation instead binds to an already trusted evidence digest, allowing a
judge-readable profile to state what was measured without turning timing noise into scientific
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

PERFORMANCE_PROFILE_SCHEMA_VERSION: Final = 1
_FAMILY_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_DOMAIN: Final = b"kuairand-performance-profile-v1\0"


class PerformanceAcceptanceError(ValueError):
    """A performance observation or acceptance profile is malformed."""


def _seconds(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerformanceAcceptanceError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise PerformanceAcceptanceError(f"{name} must be a finite {qualifier} number")
    return result


def _family(value: object, name: str = "family") -> str:
    if type(value) is not str or _FAMILY_RE.fullmatch(value) is None:
        raise PerformanceAcceptanceError(
            f"{name} must be a lowercase snake-case identifier of at most 80 characters"
        )
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise PerformanceAcceptanceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PerformanceAcceptanceError(
            "performance profile must be finite canonical JSON"
        ) from exc


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSD compatibility layer report KiB.
    if sys.platform.startswith("linux"):
        observed *= 1024
    return max(observed, 1)


@dataclass(frozen=True, slots=True)
class TimingReceipt:
    """One local operation observation bound to a trusted output/evidence digest."""

    family: str
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    rows: int
    evidence_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _family(self.family))
        object.__setattr__(
            self,
            "wall_seconds",
            _seconds(self.wall_seconds, "wall_seconds"),
        )
        object.__setattr__(
            self,
            "cpu_seconds",
            _seconds(self.cpu_seconds, "cpu_seconds"),
        )
        if type(self.peak_rss_bytes) is not int or self.peak_rss_bytes <= 0:
            raise PerformanceAcceptanceError("peak_rss_bytes must be a positive integer")
        if type(self.rows) is not int or self.rows <= 0:
            raise PerformanceAcceptanceError("rows must be a positive integer")
        _digest(self.evidence_digest, "evidence_digest")

    def manifest(self) -> dict[str, object]:
        return {
            "family": self.family,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "rows": self.rows,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class FamilyTimingSummary:
    """Median and nearest-rank p95 for one operation family."""

    family: str
    sample_count: int
    p50_seconds: float
    p95_seconds: float
    peak_rss_bytes: int
    maximum_rows: int
    evidence_digests: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "maximum_rows": self.maximum_rows,
            "evidence_digests": list(self.evidence_digests),
        }


def _summarize(receipts: Sequence[TimingReceipt]) -> FamilyTimingSummary:
    if not receipts:
        raise PerformanceAcceptanceError("cannot summarize an empty timing family")
    family = receipts[0].family
    if any(item.family != family for item in receipts):
        raise PerformanceAcceptanceError("timing summary received mixed families")
    ordered = sorted(item.wall_seconds for item in receipts)
    p95_rank = max(1, math.ceil(0.95 * len(ordered)))
    return FamilyTimingSummary(
        family=family,
        sample_count=len(receipts),
        p50_seconds=float(statistics.median(ordered)),
        p95_seconds=ordered[p95_rank - 1],
        peak_rss_bytes=max(item.peak_rss_bytes for item in receipts),
        maximum_rows=max(item.rows for item in receipts),
        evidence_digests=tuple(item.evidence_digest for item in receipts),
    )


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    """Hash-bound acceptance summary for runtime, overhead, and finalization reserve."""

    receipts: tuple[TimingReceipt, ...]
    families: Mapping[str, FamilyTimingSummary]
    controller_overhead_seconds: float
    model_runtime_seconds: float
    finalization_reserve_seconds: float
    finalization_families: tuple[str, ...]
    projected_campaign_seconds: float
    campaign_limit_seconds: float
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.receipts or any(not isinstance(item, TimingReceipt) for item in self.receipts):
            raise PerformanceAcceptanceError("receipts must contain TimingReceipt values")
        normalized_families = dict(self.families)
        expected_names = tuple(sorted({item.family for item in self.receipts}))
        if tuple(sorted(normalized_families)) != expected_names:
            raise PerformanceAcceptanceError("family summaries differ from timing receipts")
        for name, summary in normalized_families.items():
            if not isinstance(summary, FamilyTimingSummary) or summary.family != name:
                raise PerformanceAcceptanceError("family summary identity mismatch")
        object.__setattr__(self, "families", MappingProxyType(normalized_families))
        for name in (
            "controller_overhead_seconds",
            "finalization_reserve_seconds",
            "projected_campaign_seconds",
            "campaign_limit_seconds",
        ):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))
        object.__setattr__(
            self,
            "model_runtime_seconds",
            _seconds(self.model_runtime_seconds, "model_runtime_seconds", positive=True),
        )
        if self.projected_campaign_seconds > self.campaign_limit_seconds:
            # Retain the receipt and expose the failed acceptance bit; malformed negative time is
            # rejected above, while an infeasible measured plan is valid evidence.
            pass
        finalization = tuple(
            _family(value, "finalization family") for value in self.finalization_families
        )
        if not finalization or len(finalization) != len(set(finalization)):
            raise PerformanceAcceptanceError(
                "finalization families must be a non-empty unique sequence"
            )
        unknown = tuple(name for name in finalization if name not in normalized_families)
        if unknown:
            raise PerformanceAcceptanceError(
                f"finalization family is absent from receipts: {unknown!r}"
            )
        object.__setattr__(self, "finalization_families", finalization)
        object.__setattr__(
            self,
            "digest",
            hashlib.sha256(_PROFILE_DOMAIN + _canonical_json(self.body())).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        *,
        receipts: Sequence[TimingReceipt],
        controller_overhead_seconds: float,
        model_runtime_seconds: float,
        finalization_reserve_seconds: float,
        finalization_families: Sequence[str],
        projected_campaign_seconds: float,
        campaign_limit_seconds: float,
    ) -> PerformanceProfile:
        retained = tuple(receipts)
        grouped: dict[str, list[TimingReceipt]] = {}
        for receipt in retained:
            if not isinstance(receipt, TimingReceipt):
                raise PerformanceAcceptanceError("receipts must contain TimingReceipt values")
            grouped.setdefault(receipt.family, []).append(receipt)
        summaries = {name: _summarize(values) for name, values in sorted(grouped.items())}
        return cls(
            receipts=retained,
            families=summaries,
            controller_overhead_seconds=controller_overhead_seconds,
            model_runtime_seconds=model_runtime_seconds,
            finalization_reserve_seconds=finalization_reserve_seconds,
            finalization_families=tuple(finalization_families),
            projected_campaign_seconds=projected_campaign_seconds,
            campaign_limit_seconds=campaign_limit_seconds,
        )

    def family(self, name: str) -> FamilyTimingSummary:
        normalized = _family(name)
        try:
            return self.families[normalized]
        except KeyError as exc:
            raise PerformanceAcceptanceError(
                f"performance profile has no family {normalized!r}"
            ) from exc

    @property
    def controller_overhead_ratio(self) -> float:
        return self.controller_overhead_seconds / self.model_runtime_seconds

    @property
    def finalization_p95_seconds(self) -> float:
        return sum(self.families[name].p95_seconds for name in self.finalization_families)

    @property
    def finalization_reserve_sufficient(self) -> bool:
        return self.finalization_p95_seconds <= self.finalization_reserve_seconds

    @property
    def campaign_time_sufficient(self) -> bool:
        return self.projected_campaign_seconds <= self.campaign_limit_seconds

    def body(self) -> dict[str, object]:
        return {
            "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
            "receipts": [item.manifest() for item in self.receipts],
            "families": {
                name: summary.manifest() for name, summary in sorted(self.families.items())
            },
            "controller": {
                "overhead_seconds": self.controller_overhead_seconds,
                "model_runtime_seconds": self.model_runtime_seconds,
                "overhead_ratio": self.controller_overhead_ratio,
            },
            "finalization": {
                "families": list(self.finalization_families),
                "p95_seconds": self.finalization_p95_seconds,
                "reserve_seconds": self.finalization_reserve_seconds,
                "reserve_sufficient": self.finalization_reserve_sufficient,
            },
            "campaign": {
                "projected_seconds": self.projected_campaign_seconds,
                "limit_seconds": self.campaign_limit_seconds,
                "time_sufficient": self.campaign_time_sufficient,
            },
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}


def measure_operation[T](
    *,
    family: str,
    rows: int,
    operation: Callable[[], tuple[T, str]],
) -> tuple[TimingReceipt, T]:
    """Measure one operation whose return value includes its trusted evidence digest."""

    normalized_family = _family(family)
    if type(rows) is not int or rows <= 0:
        raise PerformanceAcceptanceError("rows must be a positive integer")
    if not callable(operation):
        raise PerformanceAcceptanceError("operation must be callable")
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result, evidence_digest = operation()
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    receipt = TimingReceipt(
        family=normalized_family,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=_peak_rss_bytes(),
        rows=rows,
        evidence_digest=_digest(evidence_digest, "operation evidence digest"),
    )
    return receipt, result


def write_performance_profile(profile: PerformanceProfile, path: Path) -> Path:
    """Atomically write one canonical, non-overwriting profile artifact."""

    if not isinstance(profile, PerformanceProfile):
        raise PerformanceAcceptanceError("profile must be a PerformanceProfile")
    if not isinstance(path, Path):
        raise PerformanceAcceptanceError("path must be a pathlib.Path")
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PerformanceAcceptanceError(
            f"refusing to overwrite performance profile: {destination}"
        )
    payload = _canonical_json(profile.manifest()) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise PerformanceAcceptanceError(
                f"refusing to overwrite performance profile: {destination}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "PERFORMANCE_PROFILE_SCHEMA_VERSION",
    "FamilyTimingSummary",
    "PerformanceAcceptanceError",
    "PerformanceProfile",
    "TimingReceipt",
    "measure_operation",
    "write_performance_profile",
]

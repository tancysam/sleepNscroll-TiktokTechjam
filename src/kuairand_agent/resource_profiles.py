"""Frozen CPU/GPU resource profiles for the six-hour competition facade.

The resource profile is deliberately separate from the legacy ``AgentConfig``.  A device
choice may change throughput and the number of *planned* protected evaluations, but it may
not change scientific policy.  Loading either checked-in profile therefore replaces the
TOML scientific-policy table with the same immutable canonical object after exact validation.

Nothing in the normal import or CPU-profile load path imports or probes a GPU runtime.  The
LightGBM GPU probe is an explicit CLI action used only by ``scripts/qualify_gpu.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

PROFILE_SCHEMA_VERSION: Final = 1
MIB: Final = 1024**2


class ResourceProfileError(ValueError):
    """Raised when a resource profile or measured-duration receipt is invalid."""


@dataclass(frozen=True, slots=True)
class ScientificPolicy:
    """Scientific policy shared by every hardware profile.

    The fields here are the policy elements which a resource downgrade is most likely to
    weaken accidentally.  Other scientific rules remain owned by the campaign controller.
    """

    policy_version: str
    benchmark: str
    target: str
    primary_metric: str
    strict_past_features: bool
    inner_folds: tuple[str, ...]
    confirmation_seeds: tuple[int, ...]
    practical_improvement_margin: float
    convergence_patience: int
    global_protected_evaluation_limit: int
    final_period_outcomes_accessible: bool
    required_replay_grades: tuple[str, ...]

    @property
    def digest(self) -> str:
        """Return a stable identity independent of the selected resource profile."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


SCIENTIFIC_POLICY: Final = ScientificPolicy(
    policy_version="competition-scientific-v1",
    benchmark="kuairand-pure",
    target="long_view",
    primary_metric="(GAUC+nDCG@5)/2",
    strict_past_features=True,
    inner_folds=("20220416:20220418", "20220419:20220421"),
    confirmation_seeds=(0, 1, 2),
    practical_improvement_margin=0.002,
    convergence_patience=3,
    global_protected_evaluation_limit=6,
    final_period_outcomes_accessible=False,
    required_replay_grades=("SCORING_EXACT", "BUNDLE_EXACT"),
)


@dataclass(frozen=True, slots=True)
class MeasuredDuration:
    """A measured p95 duration which is admissible as launch-budget evidence."""

    receipt_id: str
    sample_count: int
    p95_seconds: float

    def __post_init__(self) -> None:
        if not self.receipt_id or "\x00" in self.receipt_id:
            raise ResourceProfileError("duration receipt_id must be non-empty and contain no NUL")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ResourceProfileError("duration sample_count must be a positive integer")
        _positive_finite(self.p95_seconds, "duration p95_seconds")


class TrainingAdmissionReason(StrEnum):
    """Closed reason set for measured-duration launch admission."""

    ALLOWED = "allowed"
    MEASUREMENT_REQUIRED = "measurement_required"
    LAUNCH_WINDOW_CLOSED = "launch_window_closed"
    INSUFFICIENT_MEASURED_TIME = "insufficient_measured_time"
    CONFIRMATION_SEEDS_INCOMPLETE = "confirmation_seeds_incomplete"


@dataclass(frozen=True, slots=True)
class TrainingAdmission:
    """Resource-only admission decision; it never spends a launch or protected query."""

    allowed: bool
    reason: TrainingAdmissionReason
    elapsed_seconds: float
    launch_cutoff_seconds: int
    measured_runtime_seconds: float | None
    cleanup_seconds: float
    projected_completion_seconds: float | None
    receipt_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Immutable execution limits for one competition hardware profile."""

    name: str
    preferred_backend: str
    device: str
    dependency_group: str
    wall_clock_seconds: int
    finalization_reserve_seconds: int
    latest_training_launch_seconds: int
    max_candidate_processes: int
    threads_per_candidate: int
    candidate_rss_target_mb: int
    process_tree_rss_hard_cap_mb: int
    candidate_disk_hard_cap_mb: int
    planned_full_scientific_screens: int
    planned_frozen_finalists: int
    planned_protected_evaluations: int
    requires_measured_p95: bool
    in_attempt_device_fallback: bool
    gpu_probe_allowed: bool
    verified_gpu_build_required: bool
    stock_lightgbm_gpu_qualified: bool
    scientific_policy: ScientificPolicy = SCIENTIFIC_POLICY

    @property
    def candidate_rss_target_bytes(self) -> int:
        return self.candidate_rss_target_mb * MIB

    @property
    def process_tree_rss_hard_cap_bytes(self) -> int:
        return self.process_tree_rss_hard_cap_mb * MIB

    @property
    def candidate_disk_hard_cap_bytes(self) -> int:
        return self.candidate_disk_hard_cap_mb * MIB

    @property
    def training_window_seconds(self) -> int:
        return self.wall_clock_seconds - self.finalization_reserve_seconds

    @property
    def digest(self) -> str:
        """Return the profile identity, including the shared scientific-policy digest."""

        payload = self.manifest()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> dict[str, object]:
        """Return a stable JSON-compatible profile manifest."""

        result = asdict(self)
        result["scientific_policy"] = {
            "digest": self.scientific_policy.digest,
            **asdict(self.scientific_policy),
        }
        return cast(dict[str, object], result)

    def admit_training_launch(
        self,
        *,
        elapsed_seconds: float,
        duration: MeasuredDuration | None,
        cleanup_seconds: float = 0.0,
    ) -> TrainingAdmission:
        """Admit one launch only when measured p95 work fits before minute 300."""

        elapsed = _nonnegative_finite(elapsed_seconds, "elapsed_seconds")
        cleanup = _nonnegative_finite(cleanup_seconds, "cleanup_seconds")
        if elapsed > self.latest_training_launch_seconds:
            return TrainingAdmission(
                allowed=False,
                reason=TrainingAdmissionReason.LAUNCH_WINDOW_CLOSED,
                elapsed_seconds=elapsed,
                launch_cutoff_seconds=self.latest_training_launch_seconds,
                measured_runtime_seconds=None if duration is None else duration.p95_seconds,
                cleanup_seconds=cleanup,
                projected_completion_seconds=None,
                receipt_ids=() if duration is None else (duration.receipt_id,),
            )
        if duration is None:
            return TrainingAdmission(
                allowed=False,
                reason=TrainingAdmissionReason.MEASUREMENT_REQUIRED,
                elapsed_seconds=elapsed,
                launch_cutoff_seconds=self.latest_training_launch_seconds,
                measured_runtime_seconds=None,
                cleanup_seconds=cleanup,
                projected_completion_seconds=None,
                receipt_ids=(),
            )
        projected = elapsed + duration.p95_seconds + cleanup
        allowed = projected <= self.latest_training_launch_seconds
        return TrainingAdmission(
            allowed=allowed,
            reason=(
                TrainingAdmissionReason.ALLOWED
                if allowed
                else TrainingAdmissionReason.INSUFFICIENT_MEASURED_TIME
            ),
            elapsed_seconds=elapsed,
            launch_cutoff_seconds=self.latest_training_launch_seconds,
            measured_runtime_seconds=duration.p95_seconds,
            cleanup_seconds=cleanup,
            projected_completion_seconds=projected,
            receipt_ids=(duration.receipt_id,),
        )

    def admit_confirmation_bundle(
        self,
        *,
        elapsed_seconds: float,
        durations_by_seed: Mapping[int, MeasuredDuration],
        cleanup_seconds_per_seed: float = 0.0,
    ) -> TrainingAdmission:
        """Atomically admit the complete frozen seed bundle or reject the finalist."""

        elapsed = _nonnegative_finite(elapsed_seconds, "elapsed_seconds")
        cleanup = _nonnegative_finite(cleanup_seconds_per_seed, "cleanup_seconds_per_seed")
        expected_seeds = self.scientific_policy.confirmation_seeds
        if tuple(sorted(durations_by_seed)) != expected_seeds:
            return TrainingAdmission(
                allowed=False,
                reason=TrainingAdmissionReason.CONFIRMATION_SEEDS_INCOMPLETE,
                elapsed_seconds=elapsed,
                launch_cutoff_seconds=self.latest_training_launch_seconds,
                measured_runtime_seconds=None,
                cleanup_seconds=cleanup,
                projected_completion_seconds=None,
                receipt_ids=(),
            )
        receipts = tuple(durations_by_seed[seed] for seed in expected_seeds)
        measured = sum(receipt.p95_seconds for receipt in receipts)
        projected = elapsed + measured + cleanup * len(receipts)
        if elapsed > self.latest_training_launch_seconds:
            reason = TrainingAdmissionReason.LAUNCH_WINDOW_CLOSED
            allowed = False
        else:
            allowed = projected <= self.latest_training_launch_seconds
            reason = (
                TrainingAdmissionReason.ALLOWED
                if allowed
                else TrainingAdmissionReason.INSUFFICIENT_MEASURED_TIME
            )
        return TrainingAdmission(
            allowed=allowed,
            reason=reason,
            elapsed_seconds=elapsed,
            launch_cutoff_seconds=self.latest_training_launch_seconds,
            measured_runtime_seconds=measured,
            cleanup_seconds=cleanup * len(receipts),
            projected_completion_seconds=projected,
            receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
        )


_COMMON_RESOURCE: Final[dict[str, object]] = {
    "wall_clock_seconds": 21_600,
    "finalization_reserve_seconds": 3_600,
    "latest_training_launch_seconds": 18_000,
    "max_candidate_processes": 1,
    "threads_per_candidate": 4,
    "candidate_rss_target_mb": 12 * 1024,
    "process_tree_rss_hard_cap_mb": 14 * 1024,
    "candidate_disk_hard_cap_mb": 20 * 1024,
    "requires_measured_p95": True,
    "in_attempt_device_fallback": False,
    "stock_lightgbm_gpu_qualified": False,
}

_EXPECTED_PROFILES: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {
        "competition-cpu": MappingProxyType(
            {
                **_COMMON_RESOURCE,
                "name": "competition-cpu",
                "preferred_backend": "lightgbm-cpu",
                "device": "cpu",
                "dependency_group": "tree-cpu",
                "planned_full_scientific_screens": 4,
                "planned_frozen_finalists": 1,
                "planned_protected_evaluations": 1,
                "gpu_probe_allowed": False,
                "verified_gpu_build_required": False,
            }
        ),
        "competition-gpu": MappingProxyType(
            {
                **_COMMON_RESOURCE,
                "name": "competition-gpu",
                "preferred_backend": "lightgbm-gpu",
                "device": "gpu",
                "dependency_group": "tree-gpu",
                "planned_full_scientific_screens": 8,
                "planned_frozen_finalists": 2,
                "planned_protected_evaluations": 2,
                "gpu_probe_allowed": True,
                "verified_gpu_build_required": True,
            }
        ),
    }
)

_EXPECTED_SCIENTIFIC_POLICY: Final[Mapping[str, object]] = MappingProxyType(
    cast(dict[str, object], asdict(SCIENTIFIC_POLICY))
)


def load_resource_profile(path: str | Path) -> ResourceProfile:
    """Load one exact checked-in resource profile.

    Unknown fields fail closed.  Every scientific-policy field must match the canonical
    object exactly, preventing a CPU profile from silently weakening a scientific gate.
    """

    profile_path = Path(path)
    try:
        with profile_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ResourceProfileError(f"cannot load resource profile {profile_path}: {exc}") from exc
    if set(raw) != {"schema_version", "scientific_policy", "resource_profile"}:
        _raise_table_difference(
            "top level",
            set(raw),
            {"schema_version", "scientific_policy", "resource_profile"},
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ResourceProfileError(
            f"schema_version must equal {PROFILE_SCHEMA_VERSION}, got {raw['schema_version']!r}"
        )
    scientific = _mapping(raw["scientific_policy"], "scientific_policy")
    _validate_exact_table(scientific, _EXPECTED_SCIENTIFIC_POLICY, "scientific_policy")
    resource = _mapping(raw["resource_profile"], "resource_profile")
    name = resource.get("name")
    if type(name) is not str or name not in _EXPECTED_PROFILES:
        expected = ", ".join(sorted(_EXPECTED_PROFILES))
        raise ResourceProfileError(f"resource_profile.name must be one of: {expected}")
    expected_resource = _EXPECTED_PROFILES[name]
    _validate_exact_table(resource, expected_resource, "resource_profile")
    profile = ResourceProfile(
        **cast(dict[str, Any], dict(resource)),
        scientific_policy=SCIENTIFIC_POLICY,
    )
    if profile.training_window_seconds != profile.latest_training_launch_seconds:
        raise ResourceProfileError(
            "training launch cutoff must preserve the full finalization reserve"
        )
    return profile


@dataclass(frozen=True, slots=True)
class GPUQualificationResult:
    """Result of the explicit, local LightGBM GPU capability probe."""

    qualified: bool
    backend_identity: str
    reason: str
    replay_max_absolute_error: float | None
    replay_ranking_equal: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def qualify_lightgbm_gpu() -> GPUQualificationResult:
    """Train and replay a tiny local GPU LambdaRank fixture.

    This function is never called while loading the CPU profile.  A package version or a
    successful import is insufficient: qualification requires the GPU learner to train and
    reproduce the same ranking on an actual fixture.
    """

    try:
        lightgbm = importlib.import_module("lightgbm")
        numpy = importlib.import_module("numpy")
    except (ImportError, OSError) as exc:
        return GPUQualificationResult(
            qualified=False,
            backend_identity="unavailable",
            reason=f"GPU dependency unavailable: {type(exc).__name__}",
            replay_max_absolute_error=None,
            replay_ranking_equal=False,
        )
    version = getattr(lightgbm, "__version__", "unknown")
    identity = f"lightgbm:{version}:gpu-runtime-probe"
    try:
        values = numpy.arange(256 * 8, dtype=numpy.float64).reshape(256, 8)
        features = numpy.sin(values / 17.0) + numpy.cos(values / 29.0)
        labels = numpy.asarray([(index * 7) % 2 for index in range(256)], dtype=numpy.int32)
        groups = [8] * 32
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5],
            "label_gain": [0, 1],
            "device_type": "gpu",
            "num_threads": 4,
            "seed": 0,
            "data_random_seed": 0,
            "feature_fraction_seed": 0,
            "bagging_seed": 0,
            "extra_seed": 0,
            "verbosity": -1,
        }

        def fit_predict() -> Any:
            dataset = lightgbm.Dataset(
                features,
                label=labels,
                group=groups,
                free_raw_data=False,
            )
            booster = lightgbm.train(params, dataset, num_boost_round=4)
            return numpy.asarray(booster.predict(features, num_iteration=4), dtype=numpy.float64)

        first = fit_predict()
        second = fit_predict()
        if first.shape != (256,) or second.shape != (256,):
            raise RuntimeError("GPU probe returned an unexpected prediction shape")
        if not bool(numpy.isfinite(first).all()) or not bool(numpy.isfinite(second).all()):
            raise RuntimeError("GPU probe returned non-finite predictions")
        max_error = float(numpy.max(numpy.abs(first - second)))
        first_rank = numpy.lexsort((numpy.arange(first.size), -first))
        second_rank = numpy.lexsort((numpy.arange(second.size), -second))
        ranking_equal = bool(numpy.array_equal(first_rank, second_rank))
        qualified = ranking_equal and max_error <= 1e-12
        return GPUQualificationResult(
            qualified=qualified,
            backend_identity=identity,
            reason=(
                "qualified by GPU training and same-backend replay"
                if qualified
                else "same-backend GPU replay exceeded the frozen tolerance"
            ),
            replay_max_absolute_error=max_error,
            replay_ranking_equal=ranking_equal,
        )
    except Exception as exc:  # the native library exposes several backend-specific exception types
        return GPUQualificationResult(
            qualified=False,
            backend_identity=identity,
            reason=f"GPU training probe failed: {type(exc).__name__}: {exc}",
            replay_max_absolute_error=None,
            replay_ranking_equal=False,
        )


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ResourceProfileError(f"{location} must be a TOML table")
    if not all(type(key) is str for key in value):
        raise ResourceProfileError(f"{location} field names must be strings")
    return cast(Mapping[str, object], value)


def _raise_table_difference(location: str, actual: set[str], expected: set[str]) -> None:
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ResourceProfileError(f"unknown {location} field(s): {', '.join(sorted(unknown))}")
    raise ResourceProfileError(f"missing {location} field(s): {', '.join(sorted(missing))}")


def _validate_exact_table(
    actual: Mapping[str, object], expected: Mapping[str, object], location: str
) -> None:
    if set(actual) != set(expected):
        _raise_table_difference(location, set(actual), set(expected))
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, tuple):
            if not isinstance(actual_value, list) or tuple(actual_value) != expected_value:
                raise ResourceProfileError(
                    f"{location}.{key} must equal the frozen value {list(expected_value)!r}"
                )
        elif type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ResourceProfileError(
                f"{location}.{key} must equal the frozen value {expected_value!r}"
            )


def _nonnegative_finite(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceProfileError(f"{location} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ResourceProfileError(f"{location} must be a non-negative finite number")
    return result


def _positive_finite(value: object, location: str) -> float:
    result = _nonnegative_finite(value, location)
    if result == 0.0:
        raise ResourceProfileError(f"{location} must be positive")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate competition resource profiles")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("profile", type=Path)
    qualify = subparsers.add_parser("qualify-gpu")
    qualify.add_argument("profile", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a profile or run the explicitly requested GPU capability probe."""

    arguments = _build_parser().parse_args(argv)
    profile = load_resource_profile(arguments.profile)
    if arguments.command == "validate":
        print(json.dumps(profile.manifest(), sort_keys=True, separators=(",", ":")))
        return 0
    if profile.device != "gpu" or not profile.gpu_probe_allowed:
        raise ResourceProfileError("qualify-gpu requires the competition-gpu profile")
    result = qualify_lightgbm_gpu()
    print(result.to_json())
    return 0 if result.qualified else 1


if __name__ == "__main__":  # pragma: no cover - exercised by opt-in shell scripts
    raise SystemExit(main())


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "SCIENTIFIC_POLICY",
    "GPUQualificationResult",
    "MeasuredDuration",
    "ResourceProfile",
    "ResourceProfileError",
    "ScientificPolicy",
    "TrainingAdmission",
    "TrainingAdmissionReason",
    "load_resource_profile",
    "main",
    "qualify_lightgbm_gpu",
]

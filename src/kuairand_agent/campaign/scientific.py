"""Trusted, family-neutral orchestration for the bounded scientific campaign.

Candidate callbacks receive only immutable experiment identities and one of three development
tiers: Fold B screen, Fold A confirmation, or public-validation matched seed.  The callback
never receives final-period targets, row-level outcomes, or authority to choose an epoch from a
public score.  Public promotion is mediated by an injected append-only ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final, Protocol

from kuairand_agent.campaign.convergence import CompletedScientificIteration, ConvergenceState
from kuairand_agent.campaign.selector import (
    CandidateEvidence,
    FoldEvidence,
    GateEvidence,
    IncumbentEvidence,
    OrganizerMetrics,
    OuterEligibilityClass,
    SeedMetrics,
    SelectionContext,
    SelectionDecision,
    SelectionOutcome,
    SelectionReason,
    Selector,
)

SCIENTIFIC_SCHEMA_VERSION: Final = 1
HARD_LAUNCH_LIMIT: Final = 50
HARD_OUTER_PROMOTION_LIMIT: Final = 6
HARD_WALL_CLOCK_SECONDS: Final = 21_600
MIN_FINALIZATION_RESERVE_SECONDS: Final = 600
DEFAULT_FINALIZATION_RESERVE_SECONDS: Final = 3_600
MATCHED_SEEDS: Final = (0, 1, 2)
SAFE_REPORTING_PRECISION: Final = 4
FOLD_DATES: Final = {
    "A": (20220408, 20220415, 20220416, 20220418),
    "B": (20220408, 20220418, 20220419, 20220421),
}


class ScientificCampaignError(ValueError):
    """Raised when scientific evidence or orchestration authority is malformed."""


class ScientificCampaignCancelled(RuntimeError):
    """Trusted controller cancellation that must cross candidate-failure containment."""


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
        raise ScientificCampaignError("scientific identity must be finite canonical JSON") from exc


def _digest_manifest(domain: bytes, value: object) -> str:
    digest = hashlib.sha256(domain)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScientificCampaignError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 120
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ScientificCampaignError(f"{name} must be short non-empty single-line text")
    return value


def _nonnegative_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ScientificCampaignError(f"{name} must be an integer in [0, {maximum}]")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ScientificCampaignError(f"{name} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ScientificCampaignError(f"{name} must be finite and non-negative")
    return result


def _primary(metrics: OrganizerMetrics) -> Decimal:
    if not isinstance(metrics, OrganizerMetrics):
        raise ScientificCampaignError("metrics must be organizer aggregate metrics")
    return metrics.primary_decimal


class ScientificTier(StrEnum):
    """The complete set of scientific scoring tiers; no final tier exists."""

    FOLD_B_SCREEN = "fold_b_screen"
    FOLD_A_CONFIRMATION = "fold_a_confirmation"
    OUTER_MATCHED_SEED = "outer_matched_seed"


class CandidateOutcome(StrEnum):
    SCREEN_REJECTED = "screen_rejected"
    INNER_REJECTED = "inner_rejected"
    OUTER_FAILED = "outer_failed"
    RETAINED = "retained"
    PROMOTED_CONFIRMED = "promoted_confirmed"
    PROMOTED_UNCONFIRMED = "promoted_unconfirmed"
    CALLBACK_FAILED = "callback_failed"
    BUDGET_REJECTED = "budget_rejected"


class CampaignStopReason(StrEnum):
    CANDIDATES_EXHAUSTED = "candidates_exhausted"
    CONVERGED = "converged"
    ITERATION_CAP = "iteration_cap"
    LAUNCH_CAP = "launch_cap"
    FINALIZATION_RESERVE = "finalization_reserve"
    HARD_DEADLINE = "hard_deadline"
    OUTER_PROMOTION_LIMIT = "outer_promotion_limit"


@dataclass(frozen=True, slots=True)
class ScientificCampaignConfig:
    """Frozen identities and hard bounds for one verified Pure campaign."""

    benchmark_digest: str
    dataset_digest: str
    scorer_digest: str
    fold_a_data_digest: str
    fold_b_data_digest: str
    outer_data_digest: str
    environment_digest: str
    campaign_digest: str
    qualified_fallback_receipt_digest: str
    max_scientific_iterations: int
    launches_already_used: int
    screen_margin: float = 0.0
    elapsed_seconds_at_start: float = 0.0
    wall_clock_seconds: int = HARD_WALL_CLOCK_SECONDS
    finalization_reserve_seconds: int = DEFAULT_FINALIZATION_RESERVE_SECONDS
    reporting_precision: int = field(init=False, default=SAFE_REPORTING_PRECISION)
    max_launches: int = field(init=False, default=HARD_LAUNCH_LIMIT)
    outer_promotion_limit: int = field(init=False, default=HARD_OUTER_PROMOTION_LIMIT)
    matched_seeds: tuple[int, ...] = field(init=False, default=MATCHED_SEEDS)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "benchmark_digest",
            "dataset_digest",
            "scorer_digest",
            "fold_a_data_digest",
            "fold_b_data_digest",
            "outer_data_digest",
            "environment_digest",
            "campaign_digest",
            "qualified_fallback_receipt_digest",
        ):
            _digest(getattr(self, name), name)
        if type(self.max_scientific_iterations) is not int or not (
            1 <= self.max_scientific_iterations <= HARD_LAUNCH_LIMIT
        ):
            raise ScientificCampaignError("max_scientific_iterations must be in [1, 50]")
        _nonnegative_int(self.launches_already_used, "launches_already_used", HARD_LAUNCH_LIMIT)
        margin = _finite_nonnegative(self.screen_margin, "screen_margin")
        if margin > 1.0:
            raise ScientificCampaignError("screen_margin must be at most one")
        if type(self.wall_clock_seconds) is not int or not (
            60 <= self.wall_clock_seconds <= HARD_WALL_CLOCK_SECONDS
        ):
            raise ScientificCampaignError("wall_clock_seconds must be an integer in [60, 21600]")
        if type(self.finalization_reserve_seconds) is not int or not (
            MIN_FINALIZATION_RESERVE_SECONDS
            <= self.finalization_reserve_seconds
            < self.wall_clock_seconds
        ):
            raise ScientificCampaignError(
                "finalization_reserve_seconds must be at least 600 and below the wall clock"
            )
        elapsed = _finite_nonnegative(self.elapsed_seconds_at_start, "elapsed_seconds_at_start")
        if elapsed > self.wall_clock_seconds:
            raise ScientificCampaignError("elapsed_seconds_at_start exceeds the hard deadline")
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-scientific-config-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "benchmark_digest": self.benchmark_digest,
            "dataset_digest": self.dataset_digest,
            "scorer_digest": self.scorer_digest,
            "fold_data_digests": {
                "A": self.fold_a_data_digest,
                "B": self.fold_b_data_digest,
            },
            "outer_data_digest": self.outer_data_digest,
            "qualified_fallback_receipt_digest": self.qualified_fallback_receipt_digest,
            "environment_digest": self.environment_digest,
            "campaign_digest": self.campaign_digest,
            "fold_dates": {name: list(values) for name, values in FOLD_DATES.items()},
            "screen_margin": self.screen_margin,
            "matched_seeds": list(self.matched_seeds),
            "max_scientific_iterations": self.max_scientific_iterations,
            "max_launches": self.max_launches,
            "launches_already_used": self.launches_already_used,
            "outer_promotion_limit": self.outer_promotion_limit,
            "reporting_precision": self.reporting_precision,
            "wall_clock_seconds": self.wall_clock_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "elapsed_seconds_at_start": self.elapsed_seconds_at_start,
            "public_epoch_selection": "forbidden; training_policy_digest frozen before outer",
        }


def _executable_python_path(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 240
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ScientificCampaignError(f"{name} must name a reachable executable Python file")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or path.suffix != ".py"
        or any(part in {"", ".", ".."} or part.startswith(".") for part in path.parts)
    ):
        raise ScientificCampaignError(f"{name} must name a reachable executable Python file")
    return value


@dataclass(frozen=True, slots=True)
class ExecutableChangeEvidence:
    """Controller-attested material Python-AST change from one immutable parent tree."""

    parent_source_digest: str
    candidate_source_digest: str
    executable_diff_digest: str
    controller_attestation_digest: str
    changed_symbols: tuple[str, ...]
    reachable_python_files: tuple[str, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "parent_source_digest",
            "candidate_source_digest",
            "executable_diff_digest",
            "controller_attestation_digest",
        ):
            _digest(getattr(self, name), name)
        if self.parent_source_digest == self.candidate_source_digest:
            raise ScientificCampaignError(
                "parent and candidate source digests must differ for a material change"
            )
        if type(self.reachable_python_files) is not tuple or not self.reachable_python_files:
            raise ScientificCampaignError(
                "reachable_python_files must contain reachable executable Python files"
            )
        reachable = tuple(
            _executable_python_path(value, "reachable_python_files item")
            for value in self.reachable_python_files
        )
        if reachable != tuple(sorted(set(reachable))):
            raise ScientificCampaignError(
                "reachable_python_files must be sorted, unique, and canonical"
            )
        if "candidate.py" not in reachable:
            raise ScientificCampaignError(
                "reachable_python_files must include the candidate.py executable entry point"
            )
        if type(self.changed_symbols) is not tuple or not self.changed_symbols:
            raise ScientificCampaignError("changed_symbols cannot be empty")
        if self.changed_symbols != tuple(sorted(set(self.changed_symbols))):
            raise ScientificCampaignError("changed_symbols must be sorted and unique")
        for value in self.changed_symbols:
            if type(value) is not str or value.count(":") != 1:
                raise ScientificCampaignError(
                    "changed symbol must use canonical '<python-path>:<symbol>' syntax"
                )
            path, symbol = value.split(":", 1)
            _executable_python_path(path, "changed symbol path")
            if path not in reachable:
                raise ScientificCampaignError("changed symbol must name a reachable Python file")
            if not symbol.isidentifier():
                raise ScientificCampaignError("changed symbol must be a Python identifier")
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-executable-change-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "parent_source_digest": self.parent_source_digest,
            "candidate_source_digest": self.candidate_source_digest,
            "executable_diff_digest": self.executable_diff_digest,
            "controller_attestation_digest": self.controller_attestation_digest,
            "changed_symbols": list(self.changed_symbols),
            "reachable_python_files": list(self.reachable_python_files),
        }


@dataclass(frozen=True, slots=True)
class ScientificCandidate:
    """One family-neutral generated candidate with fixed training-policy identity."""

    candidate_id: str
    parent_id: str
    family: str
    source_digest: str
    parent_source_digest: str
    executable_change: ExecutableChangeEvidence
    config_digest: str
    training_policy_digest: str
    gates: GateEvidence
    diversity_root: bool = False
    metric_specialist_for_blending: bool = False
    sufficient_finalization_time: bool = True
    p95_runtime_seconds: float = 60.0
    cleanup_seconds: float = 5.0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        _identifier(self.parent_id, "parent_id")
        _identifier(self.family, "family")
        if self.candidate_id == self.parent_id:
            raise ScientificCampaignError("candidate_id must differ from parent_id")
        for name in (
            "source_digest",
            "parent_source_digest",
            "config_digest",
            "training_policy_digest",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.executable_change, ExecutableChangeEvidence):
            raise ScientificCampaignError(
                "candidate requires controller-attested executable-change evidence"
            )
        if (
            self.executable_change.candidate_source_digest != self.source_digest
            or self.executable_change.parent_source_digest != self.parent_source_digest
        ):
            raise ScientificCampaignError(
                "candidate source identities disagree with executable-change evidence"
            )
        if not isinstance(self.gates, GateEvidence):
            raise ScientificCampaignError("gates must be GateEvidence")
        for name in (
            "diversity_root",
            "metric_specialist_for_blending",
            "sufficient_finalization_time",
        ):
            if type(getattr(self, name)) is not bool:
                raise ScientificCampaignError(f"{name} must be boolean")
        _finite_nonnegative(self.p95_runtime_seconds, "p95_runtime_seconds")
        _finite_nonnegative(self.cleanup_seconds, "cleanup_seconds")
        manifest = {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "family": self.family,
            "source_digest": self.source_digest,
            "parent_source_digest": self.parent_source_digest,
            "executable_change_digest": self.executable_change.digest,
            "config_digest": self.config_digest,
            "training_policy_digest": self.training_policy_digest,
        }
        object.__setattr__(
            self,
            "fingerprint",
            _digest_manifest(b"kuairand-scientific-candidate-v1\0", manifest),
        )


@dataclass(frozen=True, slots=True)
class ScientificRunRequest:
    """Label-free controller request presented to one trusted family callback."""

    campaign_digest: str
    candidate_id: str
    reference_candidate_id: str
    family: str
    tier: ScientificTier
    fold_id: str | None
    seed: int
    source_digest: str
    parent_source_digest: str
    executable_diff_digest: str
    material_change_digest: str
    controller_attestation_digest: str
    config_digest: str
    training_policy_digest: str
    data_digest: str
    scorer_digest: str
    environment_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_digest",
            "source_digest",
            "parent_source_digest",
            "executable_diff_digest",
            "material_change_digest",
            "controller_attestation_digest",
            "config_digest",
        ):
            _digest(getattr(self, name), name)
        if self.parent_source_digest == self.source_digest:
            raise ScientificCampaignError(
                "run request parent and candidate source digests must differ"
            )
        for name in (
            "training_policy_digest",
            "data_digest",
            "scorer_digest",
            "environment_digest",
        ):
            _digest(getattr(self, name), name)
        for name in ("candidate_id", "reference_candidate_id", "family"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.tier, ScientificTier):
            raise ScientificCampaignError("tier must be ScientificTier")
        expected_fold = {
            ScientificTier.FOLD_B_SCREEN: "B",
            ScientificTier.FOLD_A_CONFIRMATION: "A",
            ScientificTier.OUTER_MATCHED_SEED: None,
        }[self.tier]
        if self.fold_id != expected_fold:
            raise ScientificCampaignError("fold_id does not match scientific tier")
        _nonnegative_int(self.seed, "seed", 2**32 - 1)
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-scientific-run-request-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "campaign_digest": self.campaign_digest,
            "candidate_id": self.candidate_id,
            "reference_candidate_id": self.reference_candidate_id,
            "family": self.family,
            "tier": self.tier.value,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "source_digest": self.source_digest,
            "parent_source_digest": self.parent_source_digest,
            "executable_diff_digest": self.executable_diff_digest,
            "material_change_digest": self.material_change_digest,
            "controller_attestation_digest": self.controller_attestation_digest,
            "config_digest": self.config_digest,
            "training_policy_digest": self.training_policy_digest,
            "data_digest": self.data_digest,
            "scorer_digest": self.scorer_digest,
            "environment_digest": self.environment_digest,
        }


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    wall_seconds: float
    peak_rss_bytes: int
    disk_bytes: int

    def __post_init__(self) -> None:
        _finite_nonnegative(self.wall_seconds, "wall_seconds")
        _nonnegative_int(self.peak_rss_bytes, "peak_rss_bytes", 2**63 - 1)
        _nonnegative_int(self.disk_bytes, "disk_bytes", 2**63 - 1)

    def manifest(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "disk_bytes": self.disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class RunIdentityEvidence:
    source_digest: str
    parent_source_digest: str
    executable_diff_digest: str
    material_change_digest: str
    controller_attestation_digest: str
    config_digest: str
    training_policy_digest: str
    data_digest: str
    environment_digest: str
    execution_digest: str
    checkpoint_digest: str
    prediction_digest: str
    scorer_digest: str
    artifact_closure_digest: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _digest(getattr(self, name), name)

    def manifest(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ScientificRunEvidence:
    """Aggregate score plus durable identities; row-level predictions never cross this seam."""

    request_digest: str
    metrics: OrganizerMetrics | None
    gates: GateEvidence
    identities: RunIdentityEvidence
    resources: ResourceEvidence
    replay_verified: bool
    failure_fingerprint: str | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.request_digest, "request_digest")
        if self.metrics is not None and not isinstance(self.metrics, OrganizerMetrics):
            raise ScientificCampaignError("metrics must be OrganizerMetrics or None")
        if not isinstance(self.gates, GateEvidence):
            raise ScientificCampaignError("run gates must be GateEvidence")
        if not isinstance(self.identities, RunIdentityEvidence):
            raise ScientificCampaignError("identities must be RunIdentityEvidence")
        if not isinstance(self.resources, ResourceEvidence):
            raise ScientificCampaignError("resources must be ResourceEvidence")
        if type(self.replay_verified) is not bool:
            raise ScientificCampaignError("replay_verified must be boolean")
        if self.failure_fingerprint is not None:
            _digest(self.failure_fingerprint, "failure_fingerprint")
        if self.metrics is None and self.failure_fingerprint is None:
            raise ScientificCampaignError("failed run evidence requires a failure fingerprint")
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-scientific-run-evidence-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "request_digest": self.request_digest,
            "metrics": None
            if self.metrics is None
            else {
                "GAUC": self.metrics.gauc,
                "nDCG@5": self.metrics.ndcg_at_5,
                "primary": self.metrics.primary,
            },
            "gates": {name: getattr(self.gates, name) for name in self.gates.__dataclass_fields__},
            "identities": self.identities.manifest(),
            "resources": self.resources.manifest(),
            "replay_verified": self.replay_verified,
            "failure_fingerprint": self.failure_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class OuterPromotionLedgerSnapshot:
    revision: int
    campaign_digest: str
    benchmark_digest: str
    dataset_digest: str
    scorer_digest: str
    max_distinct_candidates: int
    candidate_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.revision, "ledger revision", 2**63 - 1)
        for name in (
            "campaign_digest",
            "benchmark_digest",
            "dataset_digest",
            "scorer_digest",
        ):
            _digest(getattr(self, name), f"ledger {name}")
        if type(self.max_distinct_candidates) is not int or not (
            0 <= self.max_distinct_candidates <= HARD_OUTER_PROMOTION_LIMIT
        ):
            raise ScientificCampaignError("ledger maximum must be in [0, 6]")
        if len(self.candidate_fingerprints) != len(set(self.candidate_fingerprints)):
            raise ScientificCampaignError("ledger candidate fingerprints must be distinct")
        for value in self.candidate_fingerprints:
            _digest(value, "ledger candidate fingerprint")
        if len(self.candidate_fingerprints) > self.max_distinct_candidates:
            raise ScientificCampaignError("ledger exceeds its distinct-candidate maximum")


@dataclass(frozen=True, slots=True)
class OuterPromotionRequest:
    campaign_digest: str
    candidate_id: str
    candidate_fingerprint: str
    source_digest: str
    parent_source_digest: str
    executable_diff_digest: str
    material_change_digest: str
    controller_attestation_digest: str
    benchmark_digest: str
    dataset_digest: str
    scorer_digest: str
    training_policy_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "outer candidate_id")
        for name in (
            "campaign_digest",
            "candidate_fingerprint",
            "source_digest",
            "parent_source_digest",
            "executable_diff_digest",
            "material_change_digest",
            "controller_attestation_digest",
            "benchmark_digest",
            "dataset_digest",
            "scorer_digest",
            "training_policy_digest",
        ):
            _digest(getattr(self, name), name)
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-outer-promotion-request-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, str | int]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "campaign_digest": self.campaign_digest,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "source_digest": self.source_digest,
            "parent_source_digest": self.parent_source_digest,
            "executable_diff_digest": self.executable_diff_digest,
            "material_change_digest": self.material_change_digest,
            "controller_attestation_digest": self.controller_attestation_digest,
            "benchmark_digest": self.benchmark_digest,
            "dataset_digest": self.dataset_digest,
            "scorer_digest": self.scorer_digest,
            "training_policy_digest": self.training_policy_digest,
        }


@dataclass(frozen=True, slots=True)
class OuterPromotionReservation:
    reservation_id: str
    request_digest: str
    ledger_revision: int
    candidate_id: str
    candidate_fingerprint: str
    consumes_slot: bool

    def __post_init__(self) -> None:
        _identifier(self.reservation_id, "reservation_id")
        _digest(self.request_digest, "reservation request_digest")
        _nonnegative_int(self.ledger_revision, "reservation ledger_revision", 2**63 - 1)
        _identifier(self.candidate_id, "reservation candidate_id")
        _digest(self.candidate_fingerprint, "reservation candidate_fingerprint")
        if type(self.consumes_slot) is not bool:
            raise ScientificCampaignError("reservation consumes_slot must be boolean")


@dataclass(frozen=True, slots=True)
class OuterPromotionCompletion:
    reservation_id: str
    request_digest: str
    reservation_revision: int
    candidate_fingerprint: str
    successful: bool
    seed_metrics: tuple[SeedMetrics, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        _identifier(self.reservation_id, "completion reservation_id")
        _digest(self.request_digest, "completion request_digest")
        _nonnegative_int(
            self.reservation_revision,
            "completion reservation_revision",
            2**63 - 1,
        )
        _digest(self.candidate_fingerprint, "completion candidate_fingerprint")
        if type(self.successful) is not bool:
            raise ScientificCampaignError("completion successful must be boolean")
        if any(not isinstance(item, SeedMetrics) for item in self.seed_metrics):
            raise ScientificCampaignError("completion seed_metrics must contain SeedMetrics")
        _digest(self.evidence_digest, "completion evidence_digest")


class OuterPromotionLedger(Protocol):
    def snapshot(self) -> OuterPromotionLedgerSnapshot: ...

    def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation: ...

    def complete(
        self,
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
    ) -> None: ...


type ScientificRunCallback = Callable[[ScientificRunRequest], ScientificRunEvidence]


@dataclass(frozen=True, slots=True)
class CandidateCampaignResult:
    candidate: ScientificCandidate
    outcome: CandidateOutcome
    runs: tuple[ScientificRunEvidence, ...]
    reason: str
    selection: SelectionDecision | None = None


def _direction(candidate: float, incumbent: float) -> str:
    if candidate > incumbent:
        return "higher"
    if candidate < incumbent:
        return "lower"
    return "equal"


@dataclass(frozen=True, slots=True)
class PublicAggregateFeedback:
    """Rounded aggregate-only public feedback safe for the research model."""

    candidate_id: str
    seed: int
    gauc: float
    ndcg_at_5: float
    primary: float
    gauc_direction: str
    ndcg_at_5_direction: str
    primary_direction: str

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "feedback candidate_id")
        _nonnegative_int(self.seed, "feedback seed", 2**32 - 1)
        for name in ("gauc", "ndcg_at_5", "primary"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ScientificCampaignError(f"feedback {name} must be finite in [0, 1]")
        for name in ("gauc_direction", "ndcg_at_5_direction", "primary_direction"):
            if getattr(self, name) not in {"higher", "lower", "equal"}:
                raise ScientificCampaignError(f"feedback {name} is invalid")

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
            "GAUC_direction": self.gauc_direction,
            "nDCG@5_direction": self.ndcg_at_5_direction,
            "primary_direction": self.primary_direction,
        }


@dataclass(frozen=True, slots=True)
class ScientificCampaignResult:
    config_digest: str
    fallback: IncumbentEvidence
    incumbent: IncumbentEvidence
    candidates: tuple[CandidateCampaignResult, ...]
    public_feedback: tuple[PublicAggregateFeedback, ...]
    convergence: ConvergenceState
    launches_used: int
    elapsed_seconds: float
    stop_reason: CampaignStopReason
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.config_digest, "result config_digest")
        if not self.fallback.official_fm or not self.fallback.replayable:
            raise ScientificCampaignError("campaign result fallback must be replayable official FM")
        _nonnegative_int(self.launches_used, "result launches_used", HARD_LAUNCH_LIMIT)
        _finite_nonnegative(self.elapsed_seconds, "result elapsed_seconds")
        object.__setattr__(
            self,
            "digest",
            _digest_manifest(b"kuairand-scientific-result-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCIENTIFIC_SCHEMA_VERSION,
            "config_digest": self.config_digest,
            "fallback_candidate_id": self.fallback.candidate_id,
            "fallback_evidence_receipt_digest": self.fallback.evidence_receipt_digest,
            "incumbent_candidate_id": self.incumbent.candidate_id,
            "incumbent_evidence_receipt_digest": self.incumbent.evidence_receipt_digest,
            "candidate_outcomes": [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "outcome": item.outcome.value,
                    "run_digests": [run.digest for run in item.runs],
                    "reason": item.reason,
                }
                for item in self.candidates
            ],
            "public_feedback": [item.manifest() for item in self.public_feedback],
            "convergence": self.convergence.manifest(),
            "launches_used": self.launches_used,
            "elapsed_seconds": self.elapsed_seconds,
            "stop_reason": self.stop_reason.value,
        }


def _request(
    config: ScientificCampaignConfig,
    candidate: ScientificCandidate,
    *,
    reference_candidate_id: str,
    tier: ScientificTier,
    seed: int,
) -> ScientificRunRequest:
    fold_id = {
        ScientificTier.FOLD_B_SCREEN: "B",
        ScientificTier.FOLD_A_CONFIRMATION: "A",
        ScientificTier.OUTER_MATCHED_SEED: None,
    }[tier]
    data_digest = {
        ScientificTier.FOLD_B_SCREEN: config.fold_b_data_digest,
        ScientificTier.FOLD_A_CONFIRMATION: config.fold_a_data_digest,
        ScientificTier.OUTER_MATCHED_SEED: config.outer_data_digest,
    }[tier]
    return ScientificRunRequest(
        campaign_digest=config.campaign_digest,
        candidate_id=candidate.candidate_id,
        reference_candidate_id=reference_candidate_id,
        family=candidate.family,
        tier=tier,
        fold_id=fold_id,
        seed=seed,
        source_digest=candidate.source_digest,
        parent_source_digest=candidate.parent_source_digest,
        executable_diff_digest=candidate.executable_change.executable_diff_digest,
        material_change_digest=candidate.executable_change.digest,
        controller_attestation_digest=(candidate.executable_change.controller_attestation_digest),
        config_digest=candidate.config_digest,
        training_policy_digest=candidate.training_policy_digest,
        data_digest=data_digest,
        scorer_digest=config.scorer_digest,
        environment_digest=config.environment_digest,
    )


def _validate_evidence(
    evidence: ScientificRunEvidence,
    request: ScientificRunRequest,
) -> None:
    if not isinstance(evidence, ScientificRunEvidence):
        raise ScientificCampaignError("runner must return ScientificRunEvidence")
    if evidence.request_digest != request.digest:
        raise ScientificCampaignError("run evidence request identity mismatch")
    identities = evidence.identities
    for name in (
        "source_digest",
        "parent_source_digest",
        "executable_diff_digest",
        "material_change_digest",
        "controller_attestation_digest",
        "config_digest",
        "training_policy_digest",
        "data_digest",
        "environment_digest",
        "scorer_digest",
    ):
        if getattr(identities, name) != getattr(request, name):
            raise ScientificCampaignError(f"run evidence {name} mismatch")


def _invoke_runner(
    runner: ScientificRunCallback,
    request: ScientificRunRequest,
) -> tuple[ScientificRunEvidence | None, str | None]:
    """Contain candidate-controlled failures, including malformed returned evidence."""

    try:
        evidence = runner(request)
        _validate_evidence(evidence, request)
    except ScientificCampaignCancelled:
        raise
    except Exception as exc:
        return None, type(exc).__name__
    return evidence, None


def _effective_gates(evidence: ScientificRunEvidence) -> GateEvidence:
    """Bind the explicit replay fact into the selector's structural gate vector."""

    return replace(
        evidence.gates,
        replay=evidence.gates.replay and evidence.replay_verified,
    )


def _fallback_is_complete(
    fallback: IncumbentEvidence,
    config: ScientificCampaignConfig,
) -> None:
    if not isinstance(fallback, IncumbentEvidence):
        raise ScientificCampaignError("fallback must be IncumbentEvidence")
    if not fallback.official_fm or not fallback.replayable or not fallback.eligible:
        raise ScientificCampaignError("fallback must be eligible replayable official FM")
    if {name for name, _ in fallback.inner_by_fold} != {"A", "B"}:
        raise ScientificCampaignError("fallback requires Fold A and Fold B evidence")
    if {item.seed for item in fallback.outer_by_seed} != set(MATCHED_SEEDS):
        raise ScientificCampaignError("fallback requires matched seeds 0, 1, and 2")
    if fallback.evidence_receipt_digest != config.qualified_fallback_receipt_digest:
        raise ScientificCampaignError("fallback receipt identity mismatch")


def _combined_gates(*gates: GateEvidence) -> GateEvidence:
    values = {
        name: all(getattr(item, name) for item in gates)
        for name in GateEvidence.__dataclass_fields__
    }
    return GateEvidence(**values)


def _outer_candidate_ids(
    snapshot: OuterPromotionLedgerSnapshot,
    candidate: ScientificCandidate,
) -> frozenset[str]:
    values = list(snapshot.candidate_fingerprints)
    if candidate.fingerprint in values:
        values.remove(candidate.fingerprint)
        values.append(candidate.candidate_id)
    return frozenset(values)


def _evidence_bundle_digest(runs: Sequence[ScientificRunEvidence]) -> str:
    return _digest_manifest(
        b"kuairand-scientific-evidence-bundle-v1\0",
        {"run_digests": [item.digest for item in runs]},
    )


def _feedback(
    *,
    candidate_id: str,
    candidate: SeedMetrics,
    incumbent: SeedMetrics,
    precision: int,
) -> PublicAggregateFeedback:
    return PublicAggregateFeedback(
        candidate_id=candidate_id,
        seed=candidate.seed,
        gauc=round(candidate.metrics.gauc, precision),
        ndcg_at_5=round(candidate.metrics.ndcg_at_5, precision),
        primary=round(candidate.metrics.primary, precision),
        gauc_direction=_direction(candidate.metrics.gauc, incumbent.metrics.gauc),
        ndcg_at_5_direction=_direction(
            candidate.metrics.ndcg_at_5,
            incumbent.metrics.ndcg_at_5,
        ),
        primary_direction=_direction(candidate.metrics.primary, incumbent.metrics.primary),
    )


def _time_admission_reason(
    config: ScientificCampaignConfig,
    candidate: ScientificCandidate,
    *,
    elapsed_seconds: float,
    required_completion: bool,
) -> CampaignStopReason | None:
    remaining = config.wall_clock_seconds - elapsed_seconds
    base_required = candidate.p95_runtime_seconds + candidate.cleanup_seconds
    if remaining <= 0.0 or base_required > remaining:
        return CampaignStopReason.HARD_DEADLINE
    if not required_completion and base_required + config.finalization_reserve_seconds > remaining:
        return CampaignStopReason.FINALIZATION_RESERVE
    return None


def _outer_bundle_admission_reason(
    config: ScientificCampaignConfig,
    candidate: ScientificCandidate,
    *,
    launches_used: int,
    elapsed_seconds: float,
) -> CampaignStopReason | None:
    """Require the whole matched-seed bundle to fit before consuming an outer slot."""

    if launches_used + len(config.matched_seeds) > config.max_launches:
        return CampaignStopReason.LAUNCH_CAP
    remaining = config.wall_clock_seconds - elapsed_seconds
    per_seed = candidate.p95_runtime_seconds + candidate.cleanup_seconds
    bundle_required = len(config.matched_seeds) * per_seed
    if remaining <= 0.0 or bundle_required > remaining:
        return CampaignStopReason.HARD_DEADLINE
    if bundle_required + config.finalization_reserve_seconds > remaining:
        return CampaignStopReason.FINALIZATION_RESERVE
    return None


def _validate_ledger_snapshot(
    snapshot: OuterPromotionLedgerSnapshot,
    config: ScientificCampaignConfig,
    *,
    minimum_revision: int = 0,
) -> None:
    if not isinstance(snapshot, OuterPromotionLedgerSnapshot):
        raise ScientificCampaignError("outer ledger returned an invalid snapshot")
    if snapshot.revision < minimum_revision:
        raise ScientificCampaignError("outer ledger revision moved backwards")
    for name in (
        "campaign_digest",
        "benchmark_digest",
        "dataset_digest",
        "scorer_digest",
    ):
        if getattr(snapshot, name) != getattr(config, name):
            raise ScientificCampaignError(f"outer ledger {name} mismatch")
    if snapshot.max_distinct_candidates > config.outer_promotion_limit:
        raise ScientificCampaignError("outer ledger maximum exceeds campaign policy")


def run_scientific_campaign(
    *,
    config: ScientificCampaignConfig,
    fallback: IncumbentEvidence,
    candidates: Sequence[ScientificCandidate],
    runner: ScientificRunCallback,
    outer_ledger: OuterPromotionLedger,
    initial_incumbent: IncumbentEvidence | None = None,
    initial_convergence: ConvergenceState | None = None,
    initial_launches_used: int | None = None,
    initial_elapsed_seconds: float | None = None,
) -> ScientificCampaignResult:
    """Run candidates in order while preserving fallback and optional durable loop cursors."""

    if not isinstance(config, ScientificCampaignConfig):
        raise ScientificCampaignError("config must be ScientificCampaignConfig")
    _fallback_is_complete(fallback, config)
    if not callable(runner):
        raise ScientificCampaignError("runner must be callable")
    ledger_snapshot = outer_ledger.snapshot()
    _validate_ledger_snapshot(ledger_snapshot, config)
    ledger_revision = ledger_snapshot.revision

    incumbent = fallback if initial_incumbent is None else initial_incumbent
    if not isinstance(incumbent, IncumbentEvidence):
        raise ScientificCampaignError("initial_incumbent must be IncumbentEvidence")
    if not incumbent.replayable or not incumbent.eligible:
        raise ScientificCampaignError("initial_incumbent must be eligible and replayable")
    if {name for name, _ in incumbent.inner_by_fold} != {"A", "B"}:
        raise ScientificCampaignError("initial_incumbent requires Fold A and Fold B evidence")
    if {item.seed for item in incumbent.outer_by_seed} != set(MATCHED_SEEDS):
        raise ScientificCampaignError("initial_incumbent requires matched seeds 0, 1, and 2")
    fallback_outer_mean = sum(
        (item.metrics.primary_decimal for item in fallback.outer_by_seed),
        Decimal(0),
    ) / len(fallback.outer_by_seed)
    convergence = (
        ConvergenceState.initial(float(fallback_outer_mean))
        if initial_convergence is None
        else initial_convergence
    )
    if not isinstance(convergence, ConvergenceState):
        raise ScientificCampaignError("initial_convergence must be ConvergenceState")
    if convergence.completed_iterations > config.max_scientific_iterations:
        raise ScientificCampaignError("initial_convergence exceeds the scientific iteration cap")
    launches_used = (
        config.launches_already_used
        if initial_launches_used is None
        else _nonnegative_int(initial_launches_used, "initial_launches_used", config.max_launches)
    )
    if launches_used < config.launches_already_used:
        raise ScientificCampaignError("initial_launches_used moved before the frozen cursor")
    elapsed_seconds = (
        config.elapsed_seconds_at_start
        if initial_elapsed_seconds is None
        else _finite_nonnegative(initial_elapsed_seconds, "initial_elapsed_seconds")
    )
    if elapsed_seconds < config.elapsed_seconds_at_start:
        raise ScientificCampaignError("initial_elapsed_seconds moved before the frozen cursor")
    if elapsed_seconds > config.wall_clock_seconds:
        raise ScientificCampaignError("initial_elapsed_seconds exceeds the hard deadline")
    results: list[CandidateCampaignResult] = []
    public_feedback: list[PublicAggregateFeedback] = []
    stop_reason = CampaignStopReason.CANDIDATES_EXHAUSTED

    for candidate in candidates:
        if convergence.should_stop:
            stop_reason = CampaignStopReason.CONVERGED
            break
        if convergence.completed_iterations >= config.max_scientific_iterations:
            stop_reason = CampaignStopReason.ITERATION_CAP
            break
        if launches_used >= config.max_launches:
            stop_reason = CampaignStopReason.LAUNCH_CAP
            break
        if not isinstance(candidate, ScientificCandidate):
            raise ScientificCampaignError("candidates must contain ScientificCandidate values")
        if candidate.parent_id != incumbent.candidate_id:
            raise ScientificCampaignError("candidate parent must be the current incumbent")

        admission_stop = _time_admission_reason(
            config,
            candidate,
            elapsed_seconds=elapsed_seconds,
            required_completion=False,
        )
        if admission_stop is not None:
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.BUDGET_REJECTED,
                    runs=(),
                    reason=admission_stop.value,
                )
            )
            stop_reason = admission_stop
            break

        request = _request(
            config,
            candidate,
            reference_candidate_id=incumbent.candidate_id,
            tier=ScientificTier.FOLD_B_SCREEN,
            seed=0,
        )
        launches_used += 1
        evidence, callback_failure = _invoke_runner(runner, request)
        if evidence is None:
            elapsed_seconds += candidate.p95_runtime_seconds
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.CALLBACK_FAILED,
                    runs=(),
                    reason=f"callback_failed:{callback_failure}",
                )
            )
            continue
        elapsed_seconds += evidence.resources.wall_seconds
        parent_fold_b = dict(incumbent.inner_by_fold)["B"]
        scientifically_valid_screen = (
            evidence.metrics is not None
            and not _effective_gates(evidence).failures
            and candidate.gates.failures == ()
        )
        passed = False
        if scientifically_valid_screen:
            assert evidence.metrics is not None  # narrowed by the validity predicate above.
            passed = candidate.diversity_root or _primary(evidence.metrics) - _primary(
                parent_fold_b
            ) > Decimal(str(config.screen_margin))
        if not passed:
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.SCREEN_REJECTED,
                    runs=(evidence,),
                    reason="fold_b_screen_failed",
                )
            )
            # A valid below-threshold result is non-material science.  Missing metrics or failed
            # execution/policy gates are implementation attempts, not scientific observations;
            # they remain launch- and iteration-budgeted by the outer controller but must not
            # satisfy the convergence patience rule.
            if scientifically_valid_screen:
                convergence = convergence.observe(CompletedScientificIteration(None))
            continue
        assert evidence.metrics is not None  # narrowed by the trusted screen predicate

        if launches_used >= config.max_launches:
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.BUDGET_REJECTED,
                    runs=(evidence,),
                    reason="launch_cap_before_fold_a",
                )
            )
            stop_reason = CampaignStopReason.LAUNCH_CAP
            break
        admission_stop = _time_admission_reason(
            config,
            candidate,
            elapsed_seconds=elapsed_seconds,
            required_completion=False,
        )
        if admission_stop is not None:
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.BUDGET_REJECTED,
                    runs=(evidence,),
                    reason=admission_stop.value,
                )
            )
            stop_reason = admission_stop
            break
        fold_a_request = _request(
            config,
            candidate,
            reference_candidate_id=incumbent.candidate_id,
            tier=ScientificTier.FOLD_A_CONFIRMATION,
            seed=0,
        )
        launches_used += 1
        fold_a_evidence, callback_failure = _invoke_runner(runner, fold_a_request)
        if fold_a_evidence is None:
            elapsed_seconds += candidate.p95_runtime_seconds
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.CALLBACK_FAILED,
                    runs=(evidence,),
                    reason=f"fold_a_callback_failed:{callback_failure}",
                )
            )
            continue
        elapsed_seconds += fold_a_evidence.resources.wall_seconds
        runs: list[ScientificRunEvidence] = [evidence, fold_a_evidence]
        if fold_a_evidence.metrics is None:
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.CALLBACK_FAILED,
                    runs=tuple(runs),
                    reason="fold_a_failed",
                )
            )
            continue

        incumbent_folds = dict(incumbent.inner_by_fold)
        fallback_folds = dict(fallback.inner_by_fold)
        folds = (
            FoldEvidence(
                "A",
                fold_a_evidence.metrics,
                incumbent_folds["A"],
                fallback_folds["A"],
            ),
            FoldEvidence(
                "B",
                evidence.metrics,
                incumbent_folds["B"],
                fallback_folds["B"],
            ),
        )
        combined_gates = _combined_gates(
            candidate.gates,
            _effective_gates(evidence),
            _effective_gates(fold_a_evidence),
        )
        inner_candidate = CandidateEvidence(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
            gates=combined_gates,
            folds=folds,
            diversity_root=candidate.diversity_root,
            metric_specialist_for_blending=candidate.metric_specialist_for_blending,
        )
        ledger_snapshot = outer_ledger.snapshot()
        _validate_ledger_snapshot(
            ledger_snapshot,
            config,
            minimum_revision=ledger_revision,
        )
        ledger_revision = ledger_snapshot.revision
        retrying_reserved_candidate = (
            candidate.fingerprint in ledger_snapshot.candidate_fingerprints
        )
        context = SelectionContext(
            configured_seeds=config.matched_seeds,
            outer_candidate_ids=_outer_candidate_ids(ledger_snapshot, candidate),
            outer_promotion_limit=min(
                config.outer_promotion_limit,
                ledger_snapshot.max_distinct_candidates,
            ),
            screen_margin=config.screen_margin,
            sufficient_finalization_time=candidate.sufficient_finalization_time,
        )
        selector = Selector()
        eligibility = selector.assess_outer_eligibility(inner_candidate, context)
        if not eligibility.eligible:
            if eligibility.classification is OuterEligibilityClass.BUDGET_BLOCKED:
                if SelectionReason.OUTER_CANDIDATE_LIMIT in eligibility.reasons:
                    reason = SelectionReason.OUTER_CANDIDATE_LIMIT.value
                    stop_reason = CampaignStopReason.OUTER_PROMOTION_LIMIT
                else:
                    reason = SelectionReason.INSUFFICIENT_FINALIZATION_TIME.value
                    stop_reason = CampaignStopReason.FINALIZATION_RESERVE
                results.append(
                    CandidateCampaignResult(
                        candidate=candidate,
                        outcome=CandidateOutcome.BUDGET_REJECTED,
                        runs=tuple(runs),
                        reason=reason,
                    )
                )
                break
            if eligibility.classification is OuterEligibilityClass.INVALID_EVIDENCE:
                results.append(
                    CandidateCampaignResult(
                        candidate=candidate,
                        outcome=CandidateOutcome.CALLBACK_FAILED,
                        runs=tuple(runs),
                        reason=eligibility.reasons[0].value,
                    )
                )
                continue
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.INNER_REJECTED,
                    runs=tuple(runs),
                    reason=eligibility.reasons[0].value,
                )
            )
            convergence = convergence.observe(CompletedScientificIteration(None))
            continue

        admission_stop = _outer_bundle_admission_reason(
            config,
            candidate,
            launches_used=launches_used,
            elapsed_seconds=elapsed_seconds,
        )
        if admission_stop is not None:
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.BUDGET_REJECTED,
                    runs=tuple(runs),
                    reason=f"{admission_stop.value}_before_outer_reservation",
                )
            )
            stop_reason = admission_stop
            break

        promotion_request = OuterPromotionRequest(
            campaign_digest=config.campaign_digest,
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint,
            source_digest=candidate.source_digest,
            parent_source_digest=candidate.parent_source_digest,
            executable_diff_digest=candidate.executable_change.executable_diff_digest,
            material_change_digest=candidate.executable_change.digest,
            controller_attestation_digest=(
                candidate.executable_change.controller_attestation_digest
            ),
            benchmark_digest=config.benchmark_digest,
            dataset_digest=config.dataset_digest,
            scorer_digest=config.scorer_digest,
            training_policy_digest=candidate.training_policy_digest,
        )
        reservation = outer_ledger.reserve(promotion_request)
        if not isinstance(reservation, OuterPromotionReservation):
            raise ScientificCampaignError("outer ledger returned an invalid reservation")
        identity_mismatch = (
            reservation.request_digest != promotion_request.digest
            or reservation.candidate_id != candidate.candidate_id
            or reservation.candidate_fingerprint != candidate.fingerprint
        )
        if retrying_reserved_candidate:
            revision_or_slot_mismatch = reservation.ledger_revision > ledger_snapshot.revision
        else:
            revision_or_slot_mismatch = (
                reservation.ledger_revision <= ledger_snapshot.revision
                or reservation.consumes_slot != eligibility.consumes_outer_slot
            )
        if identity_mismatch or revision_or_slot_mismatch:
            raise ScientificCampaignError("outer reservation identity mismatch")
        ledger_revision = max(ledger_snapshot.revision, reservation.ledger_revision)

        outer_seed_metrics: list[SeedMetrics] = []
        outer_failed = False
        for seed in config.matched_seeds:
            if launches_used >= config.max_launches:
                outer_failed = True
                break
            admission_stop = _time_admission_reason(
                config,
                candidate,
                elapsed_seconds=elapsed_seconds,
                required_completion=seed != config.matched_seeds[0],
            )
            if admission_stop is not None:
                outer_failed = True
                stop_reason = admission_stop
                break
            outer_request = _request(
                config,
                candidate,
                reference_candidate_id=incumbent.candidate_id,
                tier=ScientificTier.OUTER_MATCHED_SEED,
                seed=seed,
            )
            launches_used += 1
            outer_evidence, _ = _invoke_runner(runner, outer_request)
            if outer_evidence is None:
                elapsed_seconds += candidate.p95_runtime_seconds
                outer_failed = True
                break
            elapsed_seconds += outer_evidence.resources.wall_seconds
            runs.append(outer_evidence)
            if outer_evidence.metrics is None:
                outer_failed = True
                break
            outer_seed_metrics.append(SeedMetrics(seed, outer_evidence.metrics))

        if outer_failed or tuple(item.seed for item in outer_seed_metrics) != config.matched_seeds:
            outer_ledger.complete(
                reservation,
                OuterPromotionCompletion(
                    reservation_id=reservation.reservation_id,
                    request_digest=promotion_request.digest,
                    reservation_revision=reservation.ledger_revision,
                    candidate_fingerprint=candidate.fingerprint,
                    successful=False,
                    seed_metrics=tuple(outer_seed_metrics),
                    evidence_digest=_evidence_bundle_digest(runs),
                ),
            )
            results.append(
                CandidateCampaignResult(
                    candidate=candidate,
                    outcome=CandidateOutcome.OUTER_FAILED,
                    runs=tuple(runs),
                    reason="matched_seed_confirmation_incomplete",
                )
            )
            if launches_used >= config.max_launches:
                stop_reason = CampaignStopReason.LAUNCH_CAP
                break
            if stop_reason in {
                CampaignStopReason.FINALIZATION_RESERVE,
                CampaignStopReason.HARD_DEADLINE,
            }:
                break
            continue

        complete_gates = _combined_gates(
            combined_gates,
            *(_effective_gates(item) for item in runs[2:]),
        )
        challenger = CandidateEvidence(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
            gates=complete_gates,
            folds=folds,
            outer_by_seed=tuple(outer_seed_metrics),
            diversity_root=candidate.diversity_root,
            metric_specialist_for_blending=candidate.metric_specialist_for_blending,
        )
        decision = selector.decide(incumbent, challenger, context)
        outer_ledger.complete(
            reservation,
            OuterPromotionCompletion(
                reservation_id=reservation.reservation_id,
                request_digest=promotion_request.digest,
                reservation_revision=reservation.ledger_revision,
                candidate_fingerprint=candidate.fingerprint,
                successful=True,
                seed_metrics=tuple(outer_seed_metrics),
                evidence_digest=_evidence_bundle_digest(runs),
            ),
        )
        incumbent_by_seed = {item.seed: item for item in incumbent.outer_by_seed}
        candidate_feedback = tuple(
            _feedback(
                candidate_id=candidate.candidate_id,
                candidate=item,
                incumbent=incumbent_by_seed[item.seed],
                precision=config.reporting_precision,
            )
            for item in outer_seed_metrics
        )
        public_feedback.extend(candidate_feedback)
        if decision.outcome in {
            SelectionOutcome.PROMOTE_CONFIRMED,
            SelectionOutcome.PROMOTE_UNCONFIRMED,
        }:
            incumbent = IncumbentEvidence(
                candidate_id=candidate.candidate_id,
                inner_by_fold=tuple(
                    (fold.fold_id, fold.candidate)
                    for fold in sorted(folds, key=lambda x: x.fold_id)
                ),
                outer_by_seed=tuple(outer_seed_metrics),
                evidence_receipt_digest=_evidence_bundle_digest(runs),
                replayable=True,
                eligible=True,
                official_fm=False,
            )
        outcome = {
            SelectionOutcome.PROMOTE_CONFIRMED: CandidateOutcome.PROMOTED_CONFIRMED,
            SelectionOutcome.PROMOTE_UNCONFIRMED: CandidateOutcome.PROMOTED_UNCONFIRMED,
        }.get(decision.outcome, CandidateOutcome.RETAINED)
        results.append(
            CandidateCampaignResult(
                candidate=candidate,
                outcome=outcome,
                runs=tuple(runs),
                reason=decision.reason.value,
                selection=decision,
            )
        )
        eligible_outer_primary_decimal = sum(
            (item.metrics.primary_decimal for item in outer_seed_metrics),
            Decimal(0),
        ) / len(outer_seed_metrics)
        convergence = convergence.observe(
            CompletedScientificIteration(float(eligible_outer_primary_decimal))
        )

    if stop_reason is CampaignStopReason.CANDIDATES_EXHAUSTED and convergence.should_stop:
        stop_reason = CampaignStopReason.CONVERGED

    return ScientificCampaignResult(
        config_digest=config.digest,
        fallback=fallback,
        incumbent=incumbent,
        candidates=tuple(results),
        public_feedback=tuple(public_feedback),
        convergence=convergence,
        launches_used=launches_used,
        elapsed_seconds=elapsed_seconds,
        stop_reason=stop_reason,
    )

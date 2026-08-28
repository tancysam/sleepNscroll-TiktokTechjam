"""Deterministic launch, scientific-iteration, and runtime admission policy."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from kuairand_agent.campaign.models import ScientificIterationCount, TrainingLaunchCount

MAX_TRAINING_LAUNCHES: Final = 50
QUALIFICATION_LAUNCHES: Final = 6
MAX_SCIENTIFIC_ITERATIONS: Final = 50
DEFAULT_RUNTIME_WINDOW: Final = 20


class BudgetPolicyError(ValueError):
    """Raised when launch accounting or admission state is invalid."""


class LaunchCategory(StrEnum):
    """Frozen launch allocation from the benchmark plan."""

    BASELINE_QUALIFICATION_REPLAY = "baseline_qualification_replay"
    DIVERSE_INNER_SCREEN = "diverse_inner_screen"
    TEMPORAL_FOLD_CONFIRMATION = "temporal_fold_confirmation"
    DISTINCT_OUTER_PROMOTION = "distinct_outer_promotion"
    MATCHED_SEED_CONFIRMATION = "matched_seed_confirmation"
    BLEND_FUSION = "blend_fusion"
    FINAL_TRAINING_REPLAY = "final_training_replay"
    RECOVERY_RESERVE = "recovery_reserve"


CATEGORY_CEILINGS: Final[Mapping[LaunchCategory, int]] = MappingProxyType(
    {
        LaunchCategory.BASELINE_QUALIFICATION_REPLAY: 6,
        LaunchCategory.DIVERSE_INNER_SCREEN: 20,
        LaunchCategory.TEMPORAL_FOLD_CONFIRMATION: 6,
        LaunchCategory.DISTINCT_OUTER_PROMOTION: 6,
        LaunchCategory.MATCHED_SEED_CONFIRMATION: 5,
        LaunchCategory.BLEND_FUSION: 3,
        LaunchCategory.FINAL_TRAINING_REPLAY: 2,
        LaunchCategory.RECOVERY_RESERVE: 2,
    }
)
if sum(CATEGORY_CEILINGS.values()) != MAX_TRAINING_LAUNCHES:  # pragma: no cover
    raise RuntimeError("launch allocation must total exactly 50")


class WorkKind(StrEnum):
    """External action kinds, only one of which consumes a training launch."""

    FULL_TRAIN_EVALUATE = "full_train_evaluate"
    CHECKPOINT_INFERENCE_REPLAY = "checkpoint_inference_replay"
    STATIC_CHECK = "static_check"
    SYNTHETIC_SMOKE = "synthetic_smoke"
    PROVIDER_ACTION = "provider_action"

    @property
    def consumes_training_launch(self) -> bool:
        return self is WorkKind.FULL_TRAIN_EVALUATE


class WorkPhase(StrEnum):
    RESEARCH = "research"
    REQUIRED_CONFIRMATION = "required_confirmation"
    FINALIZATION = "finalization"


class AdmissionReason(StrEnum):
    ALLOWED = "allowed"
    HARD_LAUNCH_CAP = "hard_launch_cap"
    CATEGORY_CAP = "category_cap"
    SCIENTIFIC_ITERATION_CAP = "scientific_iteration_cap"
    HARD_DEADLINE = "hard_deadline"
    FINALIZATION_RESERVE = "finalization_reserve"


def _identifier(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise BudgetPolicyError(f"{location} must be a non-empty string without NUL bytes")
    return value


def _seconds(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetPolicyError(f"{location} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BudgetPolicyError(f"{location} must be a non-negative finite number")
    return result


@dataclass(frozen=True, slots=True)
class BudgetReallocation:
    """Explicit transfer of unused category ceiling without increasing the total."""

    source: LaunchCategory
    target: LaunchCategory
    amount: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, LaunchCategory) or not isinstance(
            self.target, LaunchCategory
        ):
            raise BudgetPolicyError("reallocation categories must be LaunchCategory values")
        if self.source is self.target:
            raise BudgetPolicyError("reallocation source and target must differ")
        if type(self.amount) is not int or self.amount <= 0:
            raise BudgetPolicyError("reallocation amount must be a positive integer")
        _identifier(self.reason, "reallocation reason")


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    """A proposed external action before it is admitted or charged."""

    execution_id: str
    family: str
    kind: WorkKind
    phase: WorkPhase
    p95_runtime_seconds: float
    cleanup_seconds: float = 0.0
    category: LaunchCategory | None = None
    original_category: LaunchCategory | None = None
    repair_child: bool = False

    def __post_init__(self) -> None:
        _identifier(self.execution_id, "execution_id")
        _identifier(self.family, "family")
        if not isinstance(self.kind, WorkKind):
            raise BudgetPolicyError("kind must be a WorkKind")
        if not isinstance(self.phase, WorkPhase):
            raise BudgetPolicyError("phase must be a WorkPhase")
        _seconds(self.p95_runtime_seconds, "p95_runtime_seconds")
        _seconds(self.cleanup_seconds, "cleanup_seconds")
        if type(self.repair_child) is not bool:
            raise BudgetPolicyError("repair_child must be boolean")
        if self.kind.consumes_training_launch:
            if not isinstance(self.category, LaunchCategory):
                raise BudgetPolicyError("full train/evaluate work requires a launch category")
            if self.repair_child:
                if self.category is not LaunchCategory.RECOVERY_RESERVE:
                    raise BudgetPolicyError("a training repair child must use recovery_reserve")
                if not isinstance(self.original_category, LaunchCategory):
                    raise BudgetPolicyError(
                        "a training repair child must record its original category"
                    )
                if self.original_category is LaunchCategory.RECOVERY_RESERVE:
                    raise BudgetPolicyError(
                        "a training repair child's original category cannot be recovery_reserve"
                    )
            elif self.original_category not in (None, self.category):
                raise BudgetPolicyError(
                    "non-repair launch original_category must be absent or equal category"
                )
            if self.category is LaunchCategory.RECOVERY_RESERVE and not self.repair_child:
                raise BudgetPolicyError("recovery_reserve is only for training repair children")
        elif self.category is not None or self.original_category is not None or self.repair_child:
            raise BudgetPolicyError(
                "uncharged work cannot claim a launch category or repair launch"
            )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    allowed: bool
    reason: AdmissionReason
    request: LaunchRequest
    remaining_seconds: float
    required_seconds: float
    projected_launch_number: int | None

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise BudgetPolicyError("allowed must be boolean")
        if not isinstance(self.reason, AdmissionReason):
            raise BudgetPolicyError("reason must be an AdmissionReason")
        _seconds(self.remaining_seconds, "remaining_seconds")
        _seconds(self.required_seconds, "required_seconds")
        if self.allowed != (self.reason is AdmissionReason.ALLOWED):
            raise BudgetPolicyError("allowed and admission reason disagree")
        if self.projected_launch_number is not None and (
            type(self.projected_launch_number) is not int or self.projected_launch_number <= 0
        ):
            raise BudgetPolicyError("projected_launch_number must be positive when present")


@dataclass(frozen=True, slots=True)
class LaunchCharge:
    launch_number: int
    execution_id: str
    family: str
    category: LaunchCategory
    original_category: LaunchCategory
    repair_child: bool
    imported_from_qualification: bool = False

    def __post_init__(self) -> None:
        if type(self.launch_number) is not int or self.launch_number <= 0:
            raise BudgetPolicyError("launch_number must be a positive integer")
        _identifier(self.execution_id, "execution_id")
        _identifier(self.family, "family")
        if not isinstance(self.category, LaunchCategory) or not isinstance(
            self.original_category, LaunchCategory
        ):
            raise BudgetPolicyError("charge categories must be LaunchCategory values")
        if (
            type(self.repair_child) is not bool
            or type(self.imported_from_qualification) is not bool
        ):
            raise BudgetPolicyError("charge flags must be boolean")


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    """Immutable authoritative counters and launch records for one campaign."""

    charges: tuple[LaunchCharge, ...]
    scientific_iterations: ScientificIterationCount = field(
        default_factory=ScientificIterationCount
    )
    reallocations: tuple[BudgetReallocation, ...] = ()
    max_training_launches: int = MAX_TRAINING_LAUNCHES
    max_scientific_iterations: int = MAX_SCIENTIFIC_ITERATIONS

    def __post_init__(self) -> None:
        if type(self.max_training_launches) is not int or not (
            QUALIFICATION_LAUNCHES <= self.max_training_launches <= MAX_TRAINING_LAUNCHES
        ):
            raise BudgetPolicyError("max_training_launches must be an integer in [6, 50]")
        if type(self.max_scientific_iterations) is not int or not (
            1 <= self.max_scientific_iterations <= MAX_SCIENTIFIC_ITERATIONS
        ):
            raise BudgetPolicyError("max_scientific_iterations must be an integer in [1, 50]")
        if not isinstance(self.scientific_iterations, ScientificIterationCount):
            raise BudgetPolicyError("scientific_iterations must be ScientificIterationCount")
        if self.scientific_iterations.value > self.max_scientific_iterations:
            raise BudgetPolicyError("scientific iteration count exceeds configured maximum")
        if len(self.charges) > self.max_training_launches:
            raise BudgetPolicyError("training launch count exceeds configured maximum")
        expected_numbers = tuple(range(1, len(self.charges) + 1))
        if tuple(charge.launch_number for charge in self.charges) != expected_numbers:
            raise BudgetPolicyError("launch numbers must be contiguous and one-based")
        execution_ids = [charge.execution_id for charge in self.charges]
        if len(execution_ids) != len(set(execution_ids)):
            raise BudgetPolicyError("execution_id must be unique across charged launches")
        effective = self.effective_ceilings
        for category in LaunchCategory:
            if self.used(category) > effective[category]:
                raise BudgetPolicyError(f"usage exceeds effective ceiling for {category.value}")
        if sum(effective.values()) != MAX_TRAINING_LAUNCHES:
            raise BudgetPolicyError("effective category ceilings must total exactly 50")

    @classmethod
    def after_qualification(
        cls,
        *,
        max_training_launches: int = MAX_TRAINING_LAUNCHES,
        max_scientific_iterations: int = MAX_SCIENTIFIC_ITERATIONS,
    ) -> Self:
        """Create a campaign ledger with the exact six WP3 launches already charged."""

        charges = tuple(
            LaunchCharge(
                launch_number=index,
                execution_id=f"qualification-launch-{index}",
                family="official_starter_fm",
                category=LaunchCategory.BASELINE_QUALIFICATION_REPLAY,
                original_category=LaunchCategory.BASELINE_QUALIFICATION_REPLAY,
                repair_child=False,
                imported_from_qualification=True,
            )
            for index in range(1, QUALIFICATION_LAUNCHES + 1)
        )
        return cls(
            charges=charges,
            max_training_launches=max_training_launches,
            max_scientific_iterations=max_scientific_iterations,
        )

    @property
    def training_launches(self) -> TrainingLaunchCount:
        return TrainingLaunchCount(len(self.charges))

    @property
    def remaining_training_launches(self) -> int:
        return self.max_training_launches - self.training_launches.value

    @property
    def remaining_scientific_iterations(self) -> int:
        return self.max_scientific_iterations - self.scientific_iterations.value

    @property
    def effective_ceilings(self) -> Mapping[LaunchCategory, int]:
        limits = dict(CATEGORY_CEILINGS)
        for transfer in self.reallocations:
            limits[transfer.source] -= transfer.amount
            limits[transfer.target] += transfer.amount
            if limits[transfer.source] < 0:
                raise BudgetPolicyError("reallocation makes a category ceiling negative")
        return MappingProxyType(limits)

    def used(self, category: LaunchCategory) -> int:
        if not isinstance(category, LaunchCategory):
            raise BudgetPolicyError("category must be a LaunchCategory")
        return sum(charge.category is category for charge in self.charges)

    def remaining_in_category(self, category: LaunchCategory) -> int:
        return self.effective_ceilings[category] - self.used(category)

    def approve_reallocation(self, transfer: BudgetReallocation) -> Self:
        if not isinstance(transfer, BudgetReallocation):
            raise BudgetPolicyError("transfer must be a BudgetReallocation")
        source_remaining = self.remaining_in_category(transfer.source)
        if transfer.amount > source_remaining:
            raise BudgetPolicyError(
                f"cannot reallocate {transfer.amount} from {transfer.source.value}; "
                f"only {source_remaining} unused"
            )
        return replace(self, reallocations=(*self.reallocations, transfer))

    def may_begin_scientific_iteration(self) -> AdmissionReason:
        if self.scientific_iterations.value >= self.max_scientific_iterations:
            return AdmissionReason.SCIENTIFIC_ITERATION_CAP
        return AdmissionReason.ALLOWED

    def complete_scientific_iteration(self) -> Self:
        """Close one hypothesis; repairs and individual launches do not call this method."""

        if self.may_begin_scientific_iteration() is not AdmissionReason.ALLOWED:
            raise BudgetPolicyError("scientific iteration cap reached")
        return replace(self, scientific_iterations=self.scientific_iterations.increment())

    def admit(
        self,
        request: LaunchRequest,
        *,
        remaining_seconds: float,
        finalization_reserve_seconds: float,
    ) -> AdmissionDecision:
        """Evaluate caps and the conservative p95 deadline without charging anything."""

        if not isinstance(request, LaunchRequest):
            raise BudgetPolicyError("request must be a LaunchRequest")
        remaining = _seconds(remaining_seconds, "remaining_seconds")
        reserve = _seconds(finalization_reserve_seconds, "finalization_reserve_seconds")
        base_required = request.p95_runtime_seconds + request.cleanup_seconds
        required = base_required + (reserve if request.phase is WorkPhase.RESEARCH else 0.0)

        reason = AdmissionReason.ALLOWED
        if base_required > remaining or remaining <= 0.0:
            reason = AdmissionReason.HARD_DEADLINE
        elif request.phase is WorkPhase.RESEARCH and (remaining <= reserve or required > remaining):
            reason = AdmissionReason.FINALIZATION_RESERVE
        elif request.kind.consumes_training_launch:
            assert request.category is not None  # validated by LaunchRequest
            if self.training_launches.value >= self.max_training_launches:
                reason = AdmissionReason.HARD_LAUNCH_CAP
            elif self.remaining_in_category(request.category) <= 0:
                reason = AdmissionReason.CATEGORY_CAP

        projected = (
            self.training_launches.value + 1
            if reason is AdmissionReason.ALLOWED and request.kind.consumes_training_launch
            else None
        )
        return AdmissionDecision(
            allowed=reason is AdmissionReason.ALLOWED,
            reason=reason,
            request=request,
            remaining_seconds=remaining,
            required_seconds=required,
            projected_launch_number=projected,
        )

    def charge_started_launch(self, admission: AdmissionDecision) -> Self:
        """Charge an admitted full launch at process start; outcomes never refund it."""

        if not isinstance(admission, AdmissionDecision):
            raise BudgetPolicyError("admission must be an AdmissionDecision")
        if not admission.allowed:
            raise BudgetPolicyError("cannot charge a rejected launch admission")
        request = admission.request
        if not request.kind.consumes_training_launch:
            return self
        if admission.projected_launch_number != self.training_launches.value + 1:
            raise BudgetPolicyError("stale launch admission; ledger changed before charge")
        if any(charge.execution_id == request.execution_id for charge in self.charges):
            raise BudgetPolicyError(f"execution_id {request.execution_id!r} is already charged")
        assert request.category is not None
        original = request.original_category or request.category
        charge = LaunchCharge(
            launch_number=self.training_launches.value + 1,
            execution_id=request.execution_id,
            family=request.family,
            category=request.category,
            original_category=original,
            repair_child=request.repair_child,
        )
        return replace(self, charges=(*self.charges, charge))

    def manifest(self) -> dict[str, object]:
        return {
            "max_training_launches": self.max_training_launches,
            "max_scientific_iterations": self.max_scientific_iterations,
            "scientific_iterations": self.scientific_iterations.value,
            "charges": [
                {
                    "launch_number": charge.launch_number,
                    "execution_id": charge.execution_id,
                    "family": charge.family,
                    "category": charge.category.value,
                    "original_category": charge.original_category.value,
                    "repair_child": charge.repair_child,
                    "imported_from_qualification": charge.imported_from_qualification,
                }
                for charge in self.charges
            ],
            "reallocations": [
                {
                    "source": transfer.source.value,
                    "target": transfer.target.value,
                    "amount": transfer.amount,
                    "reason": transfer.reason,
                }
                for transfer in self.reallocations
            ],
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        expected = {
            "max_training_launches",
            "max_scientific_iterations",
            "scientific_iterations",
            "charges",
            "reallocations",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            issue = "unknown" if unknown else "missing"
            fields = unknown or missing
            raise BudgetPolicyError(f"{issue} budget field(s): {', '.join(sorted(fields))}")
        charges_raw = raw["charges"]
        reallocations_raw = raw["reallocations"]
        if not isinstance(charges_raw, list) or not isinstance(reallocations_raw, list):
            raise BudgetPolicyError("charges and reallocations must be lists")
        charges: list[LaunchCharge] = []
        for index, item in enumerate(charges_raw):
            if not isinstance(item, Mapping):
                raise BudgetPolicyError(f"charges[{index}] must be an object")
            expected_charge = {
                "launch_number",
                "execution_id",
                "family",
                "category",
                "original_category",
                "repair_child",
                "imported_from_qualification",
            }
            if set(item) != expected_charge:
                raise BudgetPolicyError(f"charges[{index}] has missing or unknown fields")
            category = item["category"]
            original = item["original_category"]
            if type(category) is not str or type(original) is not str:
                raise BudgetPolicyError(f"charges[{index}] categories must be strings")
            try:
                parsed_category = LaunchCategory(category)
                parsed_original = LaunchCategory(original)
            except ValueError as exc:
                raise BudgetPolicyError(f"charges[{index}] has an unknown category") from exc
            launch_number = item["launch_number"]
            repair_child = item["repair_child"]
            imported = item["imported_from_qualification"]
            if type(launch_number) is not int:
                raise BudgetPolicyError(f"charges[{index}].launch_number must be an integer")
            if type(repair_child) is not bool or type(imported) is not bool:
                raise BudgetPolicyError(f"charges[{index}] flags must be boolean")
            charges.append(
                LaunchCharge(
                    launch_number=launch_number,
                    execution_id=_identifier(item["execution_id"], "execution_id"),
                    family=_identifier(item["family"], "family"),
                    category=parsed_category,
                    original_category=parsed_original,
                    repair_child=repair_child,
                    imported_from_qualification=imported,
                )
            )
        transfers: list[BudgetReallocation] = []
        for index, item in enumerate(reallocations_raw):
            if not isinstance(item, Mapping) or set(item) != {
                "source",
                "target",
                "amount",
                "reason",
            }:
                raise BudgetPolicyError(f"reallocations[{index}] has missing or unknown fields")
            source = item["source"]
            target = item["target"]
            amount = item["amount"]
            if type(source) is not str or type(target) is not str or type(amount) is not int:
                raise BudgetPolicyError(f"reallocations[{index}] has invalid field types")
            try:
                parsed_source = LaunchCategory(source)
                parsed_target = LaunchCategory(target)
            except ValueError as exc:
                raise BudgetPolicyError(f"reallocations[{index}] has an unknown category") from exc
            transfers.append(
                BudgetReallocation(
                    source=parsed_source,
                    target=parsed_target,
                    amount=amount,
                    reason=_identifier(item["reason"], "reallocation reason"),
                )
            )
        max_training = raw["max_training_launches"]
        max_iterations = raw["max_scientific_iterations"]
        scientific_iterations = raw["scientific_iterations"]
        if type(max_training) is not int or type(max_iterations) is not int:
            raise BudgetPolicyError("budget maxima must be integers")
        if type(scientific_iterations) is not int:
            raise BudgetPolicyError("scientific_iterations must be an integer")
        ledger = cls(
            charges=tuple(charges),
            scientific_iterations=ScientificIterationCount(scientific_iterations),
            reallocations=tuple(transfers),
            max_training_launches=max_training,
            max_scientific_iterations=max_iterations,
        )
        imported = tuple(charge for charge in ledger.charges if charge.imported_from_qualification)
        if len(imported) != QUALIFICATION_LAUNCHES or any(
            charge.category is not LaunchCategory.BASELINE_QUALIFICATION_REPLAY
            for charge in imported
        ):
            raise BudgetPolicyError(
                "restored campaign must retain exactly six qualification launches"
            )
        if any(
            not charge.imported_from_qualification
            for charge in ledger.charges[:QUALIFICATION_LAUNCHES]
        ) or any(
            charge.imported_from_qualification for charge in ledger.charges[QUALIFICATION_LAUNCHES:]
        ):
            raise BudgetPolicyError(
                "the six qualification launches must precede every campaign launch"
            )
        return ledger


@dataclass(frozen=True, slots=True)
class RuntimeEstimate:
    family: str
    sample_count: int
    p50_seconds: float
    p95_seconds: float


@dataclass(frozen=True, slots=True)
class RuntimeSample:
    family: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        _identifier(self.family, "runtime family")
        _seconds(self.elapsed_seconds, "elapsed_seconds")


@dataclass(frozen=True, slots=True)
class RuntimeHistory:
    """Rolling per-family timings with deterministic median and nearest-rank p95."""

    samples: tuple[RuntimeSample, ...] = ()
    window_size: int = DEFAULT_RUNTIME_WINDOW

    def __post_init__(self) -> None:
        if type(self.window_size) is not int or self.window_size <= 0:
            raise BudgetPolicyError("runtime window_size must be a positive integer")
        counts: dict[str, int] = {}
        for sample in self.samples:
            if not isinstance(sample, RuntimeSample):
                raise BudgetPolicyError("samples must contain RuntimeSample values")
            counts[sample.family] = counts.get(sample.family, 0) + 1
        if any(count > self.window_size for count in counts.values()):
            raise BudgetPolicyError("runtime history exceeds its per-family rolling window")

    def record(self, family: str, elapsed_seconds: float) -> Self:
        sample = RuntimeSample(family=family, elapsed_seconds=elapsed_seconds)
        updated = [*self.samples, sample]
        family_indexes = [index for index, item in enumerate(updated) if item.family == family]
        excess = len(family_indexes) - self.window_size
        if excess > 0:
            remove = set(family_indexes[:excess])
            updated = [item for index, item in enumerate(updated) if index not in remove]
        return replace(self, samples=tuple(updated))

    def estimate(self, family: str, *, fallback_seconds: float) -> RuntimeEstimate:
        _identifier(family, "runtime family")
        fallback = _seconds(fallback_seconds, "fallback_seconds")
        values = sorted(
            sample.elapsed_seconds for sample in self.samples if sample.family == family
        )
        if not values:
            return RuntimeEstimate(
                family=family,
                sample_count=0,
                p50_seconds=fallback,
                p95_seconds=fallback,
            )
        rank = max(1, math.ceil(0.95 * len(values)))
        return RuntimeEstimate(
            family=family,
            sample_count=len(values),
            p50_seconds=float(statistics.median(values)),
            p95_seconds=values[rank - 1],
        )

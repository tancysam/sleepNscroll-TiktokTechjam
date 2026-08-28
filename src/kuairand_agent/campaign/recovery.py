"""Typed failure classification, redacted fingerprints, and bounded recovery policy."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MAX_DIAGNOSTIC_CHARACTERS: Final = 4_096
MAX_CHILD_REPAIRS: Final = 2
MAX_PROVIDER_RETRIES: Final = 1


class RecoveryPolicyError(ValueError):
    """Raised when failure or recovery evidence is malformed."""


class FailureSignal(StrEnum):
    """Typed observations from the provider, static gate, runner, or finalizer."""

    MALFORMED_PROVIDER_OUTPUT = "malformed_provider_output"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    STATIC_POLICY_VIOLATION = "static_policy_violation"
    DETERMINISTIC_DATA_BUG = "deterministic_data_bug"
    SHAPE_BUG = "shape_bug"
    NON_FINITE_OUTPUT = "non_finite_output"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    INTERRUPTED = "interrupted"
    CANDIDATE_INVALID_METRIC = "candidate_invalid_metric"
    PROTECTED_SCORER_ERROR = "protected_scorer_error"
    TRUSTED_CONTRACT_ERROR = "trusted_contract_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    FINAL_REPLAY_MISMATCH = "final_replay_mismatch"
    UNKNOWN_CANDIDATE_FAILURE = "unknown_candidate_failure"


class FailureCategory(StrEnum):
    MALFORMED_MODEL_RESPONSE = "malformed_model_response"
    STATIC_IMPORT_POLICY = "static_import_policy"
    DETERMINISTIC_DATA_SHAPE = "deterministic_data_shape"
    NON_FINITE_SCORES = "non_finite_scores"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    INTERRUPTED_PROCESS = "interrupted_process"
    CANDIDATE_INVALID_METRIC = "candidate_invalid_metric"
    PROTECTED_SCORER_CONTRACT = "protected_scorer_contract"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    FINAL_REPLAY_MISMATCH = "final_replay_mismatch"
    UNKNOWN_CANDIDATE_FAILURE = "unknown_candidate_failure"


_SIGNAL_CATEGORY: Final[dict[FailureSignal, FailureCategory]] = {
    FailureSignal.MALFORMED_PROVIDER_OUTPUT: FailureCategory.MALFORMED_MODEL_RESPONSE,
    FailureSignal.SYNTAX_ERROR: FailureCategory.STATIC_IMPORT_POLICY,
    FailureSignal.IMPORT_ERROR: FailureCategory.STATIC_IMPORT_POLICY,
    FailureSignal.STATIC_POLICY_VIOLATION: FailureCategory.STATIC_IMPORT_POLICY,
    FailureSignal.DETERMINISTIC_DATA_BUG: FailureCategory.DETERMINISTIC_DATA_SHAPE,
    FailureSignal.SHAPE_BUG: FailureCategory.DETERMINISTIC_DATA_SHAPE,
    FailureSignal.NON_FINITE_OUTPUT: FailureCategory.NON_FINITE_SCORES,
    FailureSignal.TIMEOUT: FailureCategory.TIMEOUT,
    FailureSignal.MEMORY_LIMIT: FailureCategory.OUT_OF_MEMORY,
    FailureSignal.INTERRUPTED: FailureCategory.INTERRUPTED_PROCESS,
    FailureSignal.CANDIDATE_INVALID_METRIC: FailureCategory.CANDIDATE_INVALID_METRIC,
    FailureSignal.PROTECTED_SCORER_ERROR: FailureCategory.PROTECTED_SCORER_CONTRACT,
    FailureSignal.TRUSTED_CONTRACT_ERROR: FailureCategory.PROTECTED_SCORER_CONTRACT,
    FailureSignal.PROVIDER_UNAVAILABLE: FailureCategory.PROVIDER_UNAVAILABLE,
    FailureSignal.FINAL_REPLAY_MISMATCH: FailureCategory.FINAL_REPLAY_MISMATCH,
    FailureSignal.UNKNOWN_CANDIDATE_FAILURE: FailureCategory.UNKNOWN_CANDIDATE_FAILURE,
}

_SECRET_PATTERNS: Final = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|bearer|token|password|secret)\b\s*[:=]\s*[^\s,;]+"
    ),
)
_POSIX_PATH: Final = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s:'\"]+/)*[^\s:'\"]*")
_WINDOWS_PATH: Final = re.compile(r"\b[A-Za-z]:\\(?:[^\s:'\"]+\\)*[^\s:'\"]*")
_VOLATILE_PROCESS: Final = re.compile(r"(?i)\b(pid|process|pgid)\s*[=: ]\s*\d+\b")
_MEMORY_ADDRESS: Final = re.compile(r"\b0x[0-9a-fA-F]+\b")


def _identifier(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise RecoveryPolicyError(f"{location} must be a non-empty string without NUL bytes")
    return value


def redact_diagnostic(value: str) -> str:
    """Remove credentials, absolute paths, and volatile process identities, then bound size."""

    if type(value) is not str:
        raise RecoveryPolicyError("diagnostic must be a string")
    result = value.replace("\x00", "<nul>")
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("<redacted-secret>", result)
    result = _WINDOWS_PATH.sub("<path>", result)
    result = _POSIX_PATH.sub("<path>", result)
    result = _VOLATILE_PROCESS.sub(lambda match: f"{match.group(1).lower()}=<id>", result)
    result = _MEMORY_ADDRESS.sub("<address>", result)
    return result[:MAX_DIAGNOSTIC_CHARACTERS]


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    signal: FailureSignal
    stage: str
    exception_type: str
    message: str
    traceback: str
    source_digest: str
    config_digest: str
    data_digest: str
    launched_training: bool

    def __post_init__(self) -> None:
        if not isinstance(self.signal, FailureSignal):
            raise RecoveryPolicyError("signal must be FailureSignal")
        for name in ("stage", "exception_type", "source_digest", "config_digest", "data_digest"):
            _identifier(getattr(self, name), name)
        if type(self.message) is not str or type(self.traceback) is not str:
            raise RecoveryPolicyError("failure diagnostics must be strings")
        if type(self.launched_training) is not bool:
            raise RecoveryPolicyError("launched_training must be boolean")


@dataclass(frozen=True, slots=True)
class ClassifiedFailure:
    category: FailureCategory
    fingerprint: str
    bounded_diagnostic: str
    launched_training: bool
    source_digest: str
    config_digest: str
    data_digest: str


class RecoveryAction(StrEnum):
    REPARSE_RESPONSE = "reparse_response"
    SCHEMA_PROVIDER_RETRY = "schema_provider_retry"
    REPAIR_CHILD = "repair_child"
    NUMERICAL_STABILIZATION_REPAIR = "numerical_stabilization_repair"
    REDUCE_COST_REPAIR = "reduce_cost_repair"
    SMALLER_MODEL_REPAIR = "smaller_model_repair"
    RESUME_MATCHING_CHECKPOINT = "resume_matching_checkpoint"
    IGNORE_CANDIDATE_METRIC = "ignore_candidate_metric"
    RETRY_PROVIDER = "retry_provider"
    CLOSE_BRANCH = "close_branch"
    STOP_RESEARCH_PRESERVE_STATE = "stop_research_preserve_state"
    FINALIZE_INCUMBENT = "finalize_incumbent"
    FALLBACK_REPLAYABLE_INCUMBENT = "fallback_replayable_incumbent"


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    category: FailureCategory
    fingerprint: str
    action: RecoveryAction
    source_digest: str
    config_digest: str
    consumes_repair_budget: bool

    def __post_init__(self) -> None:
        if not isinstance(self.category, FailureCategory) or not isinstance(
            self.action, RecoveryAction
        ):
            raise RecoveryPolicyError("recovery event category/action types are invalid")
        _identifier(self.fingerprint, "fingerprint")
        _identifier(self.source_digest, "source_digest")
        _identifier(self.config_digest, "config_digest")
        if type(self.consumes_repair_budget) is not bool:
            raise RecoveryPolicyError("consumes_repair_budget must be boolean")


@dataclass(frozen=True, slots=True)
class RecoveryHistory:
    events: tuple[RecoveryEvent, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(event, RecoveryEvent) for event in self.events):
            raise RecoveryPolicyError("history must contain RecoveryEvent values")

    @property
    def repairs_used(self) -> int:
        return sum(event.consumes_repair_budget for event in self.events)

    def action_count(self, action: RecoveryAction) -> int:
        return sum(event.action is action for event in self.events)

    def saw(self, failure: ClassifiedFailure, action: RecoveryAction | None = None) -> bool:
        return any(
            event.fingerprint == failure.fingerprint and (action is None or event.action is action)
            for event in self.events
        )

    def append(self, failure: ClassifiedFailure, decision: RecoveryDecision) -> RecoveryHistory:
        return RecoveryHistory(
            events=(
                *self.events,
                RecoveryEvent(
                    category=failure.category,
                    fingerprint=failure.fingerprint,
                    action=decision.action,
                    source_digest=failure.source_digest,
                    config_digest=failure.config_digest,
                    consumes_repair_budget=decision.consumes_repair_budget,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class RecoveryResources:
    remaining_repairs: int
    recovery_launches_remaining: int
    provider_retries_remaining: int = MAX_PROVIDER_RETRIES
    matching_checkpoint_identity: bool = False
    changed_configuration_available: bool = True
    smaller_repair_predicted_to_fit: bool = True
    eligible_incumbent_available: bool = True

    def __post_init__(self) -> None:
        for name, maximum in (
            ("remaining_repairs", MAX_CHILD_REPAIRS),
            ("recovery_launches_remaining", 2),
            ("provider_retries_remaining", MAX_PROVIDER_RETRIES),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise RecoveryPolicyError(f"{name} must be an integer in [0, {maximum}]")
        for name in (
            "matching_checkpoint_identity",
            "changed_configuration_available",
            "smaller_repair_predicted_to_fit",
            "eligible_incumbent_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise RecoveryPolicyError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    rationale: str
    consumes_repair_budget: bool = False
    may_launch_training: bool = False
    requires_changed_source_or_config: bool = False
    stop_research: bool = False
    preserve_incumbent: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecoveryAction):
            raise RecoveryPolicyError("action must be RecoveryAction")
        _identifier(self.rationale, "rationale")
        for name in (
            "consumes_repair_budget",
            "may_launch_training",
            "requires_changed_source_or_config",
            "stop_research",
            "preserve_incumbent",
        ):
            if type(getattr(self, name)) is not bool:
                raise RecoveryPolicyError(f"{name} must be boolean")
        if not self.preserve_incumbent:
            raise RecoveryPolicyError("recovery policy may never discard the incumbent")

    @property
    def scientific_iteration_delta(self) -> int:
        """Repairs are executions inside an iteration, never a second iteration."""

        return 0


class RecoveryPolicy:
    """Implement the bounded recovery matrix without touching trusted components."""

    def __init__(self, *, max_child_repairs: int = MAX_CHILD_REPAIRS) -> None:
        if type(max_child_repairs) is not int or not 0 <= max_child_repairs <= MAX_CHILD_REPAIRS:
            raise RecoveryPolicyError("max_child_repairs must be an integer in [0, 2]")
        self._max_child_repairs = max_child_repairs

    def classify(self, evidence: FailureEvidence) -> ClassifiedFailure:
        if not isinstance(evidence, FailureEvidence):
            raise RecoveryPolicyError("evidence must be FailureEvidence")
        category = _SIGNAL_CATEGORY[evidence.signal]
        diagnostic = redact_diagnostic(
            f"{evidence.exception_type}: {evidence.message}\n{evidence.traceback}"
        )
        fingerprint_payload = {
            "category": category.value,
            "stage": evidence.stage,
            "exception_type": evidence.exception_type,
            "diagnostic": diagnostic,
            "source_digest": evidence.source_digest,
            "config_digest": evidence.config_digest,
            "data_digest": evidence.data_digest,
        }
        encoded = json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return ClassifiedFailure(
            category=category,
            fingerprint=hashlib.sha256(encoded).hexdigest(),
            bounded_diagnostic=diagnostic,
            launched_training=evidence.launched_training,
            source_digest=evidence.source_digest,
            config_digest=evidence.config_digest,
            data_digest=evidence.data_digest,
        )

    def decide(
        self,
        failure: ClassifiedFailure,
        history: RecoveryHistory,
        resources: RecoveryResources,
    ) -> RecoveryDecision:
        if not isinstance(failure, ClassifiedFailure):
            raise RecoveryPolicyError("failure must be ClassifiedFailure")
        if not isinstance(history, RecoveryHistory) or not isinstance(resources, RecoveryResources):
            raise RecoveryPolicyError("invalid recovery history or resources")

        category = failure.category
        if category is FailureCategory.MALFORMED_MODEL_RESPONSE:
            if history.action_count(RecoveryAction.REPARSE_RESPONSE) < 1:
                return RecoveryDecision(
                    RecoveryAction.REPARSE_RESPONSE,
                    "reparse the same bounded provider response once",
                )
            if (
                resources.provider_retries_remaining > 0
                and history.action_count(RecoveryAction.SCHEMA_PROVIDER_RETRY) < 1
            ):
                return RecoveryDecision(
                    RecoveryAction.SCHEMA_PROVIDER_RETRY,
                    "make one schema-focused provider retry without training",
                )
            return self._close("malformed response recovery exhausted")

        if category in {
            FailureCategory.STATIC_IMPORT_POLICY,
            FailureCategory.DETERMINISTIC_DATA_SHAPE,
        }:
            if not self._repair_available(history, resources) or history.saw(
                failure, RecoveryAction.REPAIR_CHILD
            ):
                return self._close("repair budget exhausted or identical failure repeated")
            return RecoveryDecision(
                RecoveryAction.REPAIR_CHILD,
                "create one bounded child repair from the exact failed source",
                consumes_repair_budget=True,
                may_launch_training=category is FailureCategory.DETERMINISTIC_DATA_SHAPE,
                requires_changed_source_or_config=True,
            )

        if category is FailureCategory.NON_FINITE_SCORES:
            if (
                not self._training_repair_available(history, resources)
                or history.action_count(RecoveryAction.NUMERICAL_STABILIZATION_REPAIR) >= 1
            ):
                return self._close("the single numerical-stabilization repair is unavailable")
            return RecoveryDecision(
                RecoveryAction.NUMERICAL_STABILIZATION_REPAIR,
                "allow one repair that adds explicit numerical stabilization",
                consumes_repair_budget=True,
                may_launch_training=True,
                requires_changed_source_or_config=True,
            )

        if category is FailureCategory.TIMEOUT:
            if (
                not self._training_repair_available(history, resources)
                or not resources.changed_configuration_available
                or history.saw(failure, RecoveryAction.REDUCE_COST_REPAIR)
            ):
                return self._close("an identical timed-out configuration cannot be relaunched")
            return RecoveryDecision(
                RecoveryAction.REDUCE_COST_REPAIR,
                "repair must explicitly reduce predicted runtime before a new launch",
                consumes_repair_budget=True,
                may_launch_training=True,
                requires_changed_source_or_config=True,
            )

        if category is FailureCategory.OUT_OF_MEMORY:
            if (
                not self._training_repair_available(history, resources)
                or not resources.smaller_repair_predicted_to_fit
                or history.action_count(RecoveryAction.SMALLER_MODEL_REPAIR) >= 1
            ):
                return self._close("the single predicted-to-fit smaller repair is unavailable")
            return RecoveryDecision(
                RecoveryAction.SMALLER_MODEL_REPAIR,
                "allow one explicitly smaller batch or model repair predicted to fit",
                consumes_repair_budget=True,
                may_launch_training=True,
                requires_changed_source_or_config=True,
            )

        if category is FailureCategory.INTERRUPTED_PROCESS:
            if resources.matching_checkpoint_identity and resources.recovery_launches_remaining > 0:
                return RecoveryDecision(
                    RecoveryAction.RESUME_MATCHING_CHECKPOINT,
                    "resume only the checkpoint with matching source, config, and data identities",
                    may_launch_training=True,
                )
            return self._close("no identity-matching checkpoint can be resumed safely")

        if category is FailureCategory.CANDIDATE_INVALID_METRIC:
            return RecoveryDecision(
                RecoveryAction.IGNORE_CANDIDATE_METRIC,
                "ignore candidate-reported metrics and retain trusted scoring authority",
            )

        if category is FailureCategory.PROTECTED_SCORER_CONTRACT:
            return RecoveryDecision(
                RecoveryAction.STOP_RESEARCH_PRESERVE_STATE,
                "trusted scorer or contract failure is not repairable by generated code",
                stop_research=True,
            )

        if category is FailureCategory.PROVIDER_UNAVAILABLE:
            if (
                resources.provider_retries_remaining > 0
                and history.action_count(RecoveryAction.RETRY_PROVIDER) < 1
            ):
                return RecoveryDecision(
                    RecoveryAction.RETRY_PROVIDER,
                    "retry once within provider policy without launching training",
                )
            if resources.eligible_incumbent_available:
                return RecoveryDecision(
                    RecoveryAction.FINALIZE_INCUMBENT,
                    "provider retry exhausted; stop research and finalize the eligible incumbent",
                    stop_research=True,
                )
            return RecoveryDecision(
                RecoveryAction.STOP_RESEARCH_PRESERVE_STATE,
                "provider retry exhausted and no eligible final candidate is available",
                stop_research=True,
            )

        if category is FailureCategory.FINAL_REPLAY_MISMATCH:
            return RecoveryDecision(
                RecoveryAction.FALLBACK_REPLAYABLE_INCUMBENT,
                "fall back through replayable incumbent history, ultimately to official FM",
                stop_research=True,
            )

        return self._close("unknown candidate failure closes the branch")

    def _repair_available(self, history: RecoveryHistory, resources: RecoveryResources) -> bool:
        return history.repairs_used < self._max_child_repairs and resources.remaining_repairs > 0

    def _training_repair_available(
        self, history: RecoveryHistory, resources: RecoveryResources
    ) -> bool:
        return (
            self._repair_available(history, resources) and resources.recovery_launches_remaining > 0
        )

    @staticmethod
    def _close(rationale: str) -> RecoveryDecision:
        return RecoveryDecision(RecoveryAction.CLOSE_BRANCH, rationale)

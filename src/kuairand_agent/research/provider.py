"""Bounded OpenAI-compatible Chat Completions adapter for the typed research-model seam.

Credentials are resolved only while a call is being dispatched.  The adapter has no file, shell,
tool, or evaluator authority; its only optional side effect is one bounded HTTPS POST per attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Final, Protocol, cast
from urllib.parse import quote, urlsplit

from kuairand_agent.research.interface import ResearchModelError
from kuairand_agent.research.prompts import PROMPT_VERSION, instructions_for
from kuairand_agent.research.schemas import (
    GeneratedPackage,
    ImplementationRequest,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RepairRequest,
    ResearchOperation,
    SchemaValidationError,
    canonical_json_bytes,
    parse_json_object,
    response_json_schema,
)
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateManifestPolicyError,
)

_MODEL_RE: Final = re.compile(
    r"(?=.{1,128}\Z)[A-Za-z0-9][A-Za-z0-9._:-]*(?:/[A-Za-z0-9][A-Za-z0-9._:-]*)?\Z"
)
_ENV_RE: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_PRICE_RE: Final = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?\Z")
_REASONING_EFFORTS: Final = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_GATEWAY_REASONING_HOSTS: Final = frozenset({"openrouter.ai"})
# Hosts that ignore both ``reasoning_effort`` and the gateway ``reasoning`` object, and
# instead expose an explicit thinking budget. Measured against api.tokenrouter.com: the
# OpenAI-style controls left 92% of completion tokens as reasoning and drove implement calls
# past the configured timeout, while an explicit budget is honoured exactly.
_THINKING_BUDGET_HOSTS: Final = frozenset({"api.tokenrouter.com"})
_THINKING_BUDGET_TOKENS: Final = {
    "none": 0,
    "minimal": 0,
    "low": 4096,
    "medium": 16384,
    "high": 32768,
    "xhigh": 49152,
    "max": 65536,
}
_SECRET_PATTERN: Final = re.compile(r"(?i)(?:bearer\s+|sk-(?:proj-)?)[A-Za-z0-9_.-]{8,}")


class ProviderErrorCode(StrEnum):
    CREDENTIAL_MISSING = "credential_missing"
    DEADLINE = "deadline"
    TRANSPORT = "transport"
    RESPONSE_TOO_LARGE = "response_too_large"
    HTTP = "http"
    INCOMPLETE = "incomplete"
    REFUSAL = "refusal"
    MALFORMED_RESPONSE = "malformed_response"
    CONTEXT_LIMIT = "context_limit"


class OpenAIProviderError(ResearchModelError):
    """Typed, credential-safe provider failure."""

    def __init__(
        self,
        code: ProviderErrorCode,
        message: str,
        *,
        operation: ResearchOperation | None = None,
        attempts: int = 0,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.operation = operation
        self.attempts = attempts
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Credential-free evidence for one exhausted provider slot."""

    slot: str
    code: ProviderErrorCode
    operation: ResearchOperation | None
    attempts: int
    status_code: int | None

    def to_wire(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "code": self.code.value,
            "operation": None if self.operation is None else self.operation.value,
            "attempts": self.attempts,
            "status_code": self.status_code,
        }


class OpenAIProviderChainError(OpenAIProviderError):
    """Both the active provider path and every available fallback have failed."""

    def __init__(
        self,
        final_error: OpenAIProviderError,
        failures: tuple[ProviderFailure, ...],
    ) -> None:
        super().__init__(
            final_error.code,
            "all configured inference providers failed after bounded retries",
            operation=final_error.operation,
            attempts=sum(item.attempts for item in failures),
            status_code=final_error.status_code,
        )
        self.failures = failures


@dataclass(frozen=True, slots=True)
class ProviderFailoverEvent:
    """Safe evidence that the sticky provider slot changed."""

    operation: ResearchOperation
    from_slot: str
    to_slot: str
    failure: ProviderFailure

    def to_wire(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "from_slot": self.from_slot,
            "to_slot": self.to_slot,
            "failure": self.failure.to_wire(),
        }


class ResponsesTransportError(RuntimeError):
    """A transport failed without returning a bounded HTTP response."""


class ResponseTooLargeError(ResponsesTransportError):
    """The response exceeded its exact configured byte ceiling."""


def _price(value: object, name: str) -> str:
    if type(value) is not str or _PRICE_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be bounded non-negative decimal text")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex already excludes it
        raise ValueError(f"{name} is invalid") from exc
    if parsed > Decimal("10000"):
        raise ValueError(f"{name} exceeds 10000 USD per million tokens")
    return format(parsed.normalize(), "f")


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Explicit prices; no temporally unstable model price is inferred by the adapter."""

    input_usd_per_million: str
    cached_input_usd_per_million: str
    output_usd_per_million: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_usd_per_million",
            _price(self.input_usd_per_million, "input_usd_per_million"),
        )
        object.__setattr__(
            self,
            "cached_input_usd_per_million",
            _price(self.cached_input_usd_per_million, "cached_input_usd_per_million"),
        )
        object.__setattr__(
            self,
            "output_usd_per_million",
            _price(self.output_usd_per_million, "output_usd_per_million"),
        )


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsConfig:
    model: str
    base_url: str
    reasoning_effort: str
    pricing: TokenPricing
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 120.0
    max_response_bytes: int = 8 * 1024 * 1024
    max_output_tokens: int = 32_768
    max_malformed_retries: int = 1
    max_transport_retries: int = 0
    response_format: str = "json_schema"
    max_tokens_parameter: str = "max_completion_tokens"
    send_reasoning_effort: bool = True

    def __post_init__(self) -> None:
        if type(self.model) is not str or _MODEL_RE.fullmatch(self.model) is None:
            raise ValueError("model must be a portable explicit model identifier")
        if (
            type(self.reasoning_effort) is not str
            or self.reasoning_effort not in _REASONING_EFFORTS
        ):
            raise ValueError("reasoning_effort is unsupported")
        if not isinstance(self.pricing, TokenPricing):
            raise ValueError("pricing must be TokenPricing")
        if type(self.api_key_env) is not str or _ENV_RE.fullmatch(self.api_key_env) is None:
            raise ValueError("api_key_env must be a portable uppercase environment name")
        if type(self.timeout_seconds) is not float or not 0.1 <= self.timeout_seconds <= 600.0:
            raise ValueError("timeout_seconds must be a float in [0.1, 600.0]")
        if (
            type(self.max_response_bytes) is not int
            or not 1024 <= self.max_response_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("max_response_bytes must be an integer in [1024, 16777216]")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 200_000:
            raise ValueError("max_output_tokens must be an integer in [1, 200000]")
        if type(self.max_malformed_retries) is not int or not 0 <= self.max_malformed_retries <= 1:
            raise ValueError("max_malformed_retries must be 0 or 1")
        if type(self.max_transport_retries) is not int or not 0 <= self.max_transport_retries <= 3:
            raise ValueError("max_transport_retries must be an integer in [0, 3]")
        if self.response_format not in {"json_schema", "json_object"}:
            raise ValueError("response_format must be 'json_schema' or 'json_object'")
        if self.max_tokens_parameter not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError("max_tokens_parameter must be 'max_completion_tokens' or 'max_tokens'")
        if type(self.send_reasoning_effort) is not bool:
            raise ValueError("send_reasoning_effort must be boolean")
        if type(self.base_url) is not str:
            raise ValueError(
                "base_url must be a credential-free HTTPS URL without query or fragment"
            )
        try:
            parsed = urlsplit(self.base_url)
        except ValueError as exc:
            raise ValueError(
                "base_url must be a credential-free HTTPS URL without query or fragment"
            ) from exc
        if (
            len(self.base_url) > 2048
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or ".." in parsed.path.split("/")
        ):
            raise ValueError(
                "base_url must be a credential-free HTTPS URL without query or fragment"
            )
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def response_url(self) -> str:
        """Backward-compatible alias for callers that only need the dispatch URL."""

        return self.chat_completions_url


# Public compatibility alias for existing callers and persisted test fixtures.  New code should
# use the protocol-accurate Chat Completions name.
OpenAIResponsesConfig = OpenAIChatCompletionsConfig


@dataclass(frozen=True, slots=True)
class ModelContextLimits:
    """Provider-advertised model limits used to size requests without model-name tables."""

    context_length: int
    max_completion_tokens: int
    source: str

    def __post_init__(self) -> None:
        if type(self.context_length) is not int or not 1 <= self.context_length <= 10_000_000:
            raise ValueError("context_length must be a positive bounded integer")
        if (
            type(self.max_completion_tokens) is not int
            or not 1 <= self.max_completion_tokens <= self.context_length
        ):
            raise ValueError("max_completion_tokens must fit within context_length")
        if type(self.source) is not str or not self.source or len(self.source) > 128:
            raise ValueError("model-limit source must be bounded non-empty text")


ModelLimitResolver = Callable[[OpenAIChatCompletionsConfig, str], ModelContextLimits | None]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _no_remaining_research_limit() -> float | None:
    return None


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded provider retry timing independent of campaign policy."""

    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    max_retry_after_seconds: float = 60.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "initial_backoff_seconds",
            "max_backoff_seconds",
            "max_retry_after_seconds",
            "jitter_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial_backoff_seconds cannot exceed max_backoff_seconds")
        if self.max_backoff_seconds > 300.0 or self.max_retry_after_seconds > 300.0:
            raise ValueError("provider retry waits cannot exceed 300 seconds")
        if self.jitter_ratio > 1.0:
            raise ValueError("jitter_ratio cannot exceed 1.0")


@dataclass(frozen=True, slots=True)
class RetryRuntime:
    """Narrow injected boundary for time, sleep, jitter, and campaign time remaining."""

    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.perf_counter
    utc_now: Callable[[], datetime] = _utc_now
    jitter: Callable[[float, float], float] = random.uniform
    remaining_research_seconds: Callable[[], float | None] = _no_remaining_research_limit

    def __post_init__(self) -> None:
        for name in (
            "sleep",
            "monotonic",
            "utc_now",
            "jitter",
            "remaining_research_seconds",
        ):
            if not callable(getattr(self, name)):
                raise ValueError(f"{name} must be callable")


@dataclass(frozen=True, slots=True)
class TransportRequest:
    url: str
    body: bytes = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(repr=False)
    timeout_seconds: float
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: bytes = field(repr=False)
    retry_after: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("status_code must be an HTTP status")
        if type(self.body) is not bytes:
            raise ValueError("transport response body must be bytes")
        if self.retry_after is not None and (
            type(self.retry_after) is not str
            or not self.retry_after
            or len(self.retry_after) > 128
            or any(ord(char) < 32 or ord(char) > 126 for char in self.retry_after)
        ):
            raise ValueError("retry_after must be bounded visible ASCII text")


class ResponsesTransport(Protocol):
    def send(self, request: TransportRequest) -> TransportResponse: ...


def _bounded_read(stream: object, maximum: int) -> bytes:
    reader = getattr(stream, "read", None)
    if not callable(reader):
        raise ResponsesTransportError("HTTP stream has no read method")
    body = cast(bytes, reader(maximum + 1))
    if type(body) is not bytes:
        raise ResponsesTransportError("HTTP stream returned non-bytes")
    if len(body) > maximum:
        raise ResponseTooLargeError("provider response exceeded the configured byte ceiling")
    return body


class UrllibResponsesTransport:
    """Dependency-free bounded HTTPS transport."""

    @staticmethod
    def _retry_after(headers: object) -> str | None:
        getter = getattr(headers, "get", None)
        if not callable(getter):
            return None
        value = getter("Retry-After")
        if type(value) is not str:
            return None
        bounded = value.strip()
        if (
            not bounded
            or len(bounded) > 128
            or any(ord(char) < 32 or ord(char) > 126 for char in bounded)
        ):
            return None
        return bounded

    def send(self, request: TransportRequest) -> TransportResponse:
        raw_request = urllib.request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(raw_request, timeout=request.timeout_seconds) as response:
                return TransportResponse(
                    status_code=int(response.status),
                    body=_bounded_read(response, request.max_response_bytes),
                    retry_after=self._retry_after(getattr(response, "headers", None)),
                )
        except urllib.error.HTTPError as exc:
            return TransportResponse(
                status_code=exc.code,
                body=_bounded_read(exc, request.max_response_bytes),
                retry_after=self._retry_after(getattr(exc, "headers", None)),
            )
        except ResponseTooLargeError:
            raise
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            raise ResponsesTransportError(type(exc).__name__) from exc


class ProviderModelLimitResolver:
    """Discover limits from OpenRouter or an informative OpenAI-compatible models endpoint.

    OpenRouter has an authoritative per-model contract. Other gateways are queried through
    ``/models`` and used only when the selected record exposes exact positive ``context_length``
    and ``max_completion_tokens`` fields. Otherwise the configured conservative ceiling remains.
    """

    def __call__(
        self, config: OpenAIChatCompletionsConfig, secret: str
    ) -> ModelContextLimits | None:
        hostname = (urlsplit(config.base_url).hostname or "").lower()
        if hostname == "openrouter.ai":
            author, separator, slug = config.model.partition("/")
            if not separator or not author or not slug:
                return None
            safe_author = quote(author, safe="._-")
            safe_slug = quote(slug, safe="._-:")
            metadata_url = f"{config.base_url}/model/{safe_author}/{safe_slug}"
            maximum_bytes = 256 * 1024
        else:
            metadata_url = f"{config.base_url}/models"
            maximum_bytes = 2 * 1024 * 1024
        request = urllib.request.Request(
            metadata_url,
            headers={
                "Authorization": f"Bearer {secret}",
                "Accept": "application/json",
                "User-Agent": "kuairand-agent/0.1.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=min(config.timeout_seconds, 10.0)
            ) as response:
                if int(response.status) != 200:
                    return None
                body = _bounded_read(response, maximum_bytes)
            envelope = parse_json_object(body.decode("utf-8", errors="strict"))
        except (
            OSError,
            TimeoutError,
            UnicodeError,
            urllib.error.URLError,
            json.JSONDecodeError,
            SchemaValidationError,
            ResponseTooLargeError,
        ):
            return None
        raw_data = envelope.get("data")
        data: Mapping[str, object] | None = None
        if isinstance(raw_data, Mapping):
            data = cast(Mapping[str, object], raw_data)
        elif isinstance(raw_data, list):
            data = next(
                (
                    cast(Mapping[str, object], item)
                    for item in raw_data
                    if isinstance(item, Mapping) and item.get("id") == config.model
                ),
                None,
            )
        if data is None:
            return None
        top_provider = data.get("top_provider")
        provider_data = top_provider if isinstance(top_provider, Mapping) else {}
        context_values = tuple(
            value
            for value in (data.get("context_length"), provider_data.get("context_length"))
            if type(value) is int and value > 0
        )
        completion_values = tuple(
            value
            for value in (
                data.get("max_completion_tokens"),
                provider_data.get("max_completion_tokens"),
            )
            if type(value) is int and value > 0
        )
        if not context_values or not completion_values:
            return None
        context_length = min(context_values)
        maximum = min(completion_values)
        if maximum > context_length:
            return None
        return ModelContextLimits(
            context_length=context_length,
            max_completion_tokens=maximum,
            source=(
                "openrouter-model-metadata"
                if hostname == "openrouter.ai"
                else "openai-compatible-model-metadata"
            ),
        )


OpenRouterModelLimitResolver = ProviderModelLimitResolver


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: str = "0"
    unaccounted_attempts: int = 0
    retry_wait_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ProviderTranscript:
    operation: ResearchOperation
    attempt: int
    request_digest: str
    response_sha256: str | None
    outcome: str
    latency_seconds: float
    retry_wait_seconds: float
    json_bytes: bytes = field(repr=False)
    digest: str


TranscriptSink = Callable[[ProviderTranscript], None]


def _redact(value: object, secret: str) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if str(key).casefold() in {"authorization", "api_key"}
            else _redact(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if type(value) is str:
        result = value.replace(secret, "[REDACTED]") if secret else value
        return _SECRET_PATTERN.sub("[REDACTED]", result)
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SchemaValidationError(f"provider usage {name} must be a non-negative integer")
    return value


def _usage(envelope: Mapping[str, object], pricing: TokenPricing) -> ProviderUsage:
    raw = envelope.get("usage")
    if not isinstance(raw, Mapping):
        raise SchemaValidationError("provider response is missing usage")
    input_tokens = _nonnegative_integer(raw.get("prompt_tokens"), "prompt_tokens")
    output_tokens = _nonnegative_integer(raw.get("completion_tokens"), "completion_tokens")
    total_tokens = _nonnegative_integer(raw.get("total_tokens"), "total_tokens")
    input_details = raw.get("prompt_tokens_details", {})
    output_details = raw.get("completion_tokens_details", {})
    if not isinstance(input_details, Mapping) or not isinstance(output_details, Mapping):
        raise SchemaValidationError("provider token details must be objects")
    cached = _nonnegative_integer(input_details.get("cached_tokens", 0), "cached_tokens")
    reasoning = _nonnegative_integer(output_details.get("reasoning_tokens", 0), "reasoning_tokens")
    if (
        cached > input_tokens
        or reasoning > output_tokens
        or total_tokens != input_tokens + output_tokens
    ):
        raise SchemaValidationError("provider usage token totals are inconsistent")
    cost = (
        Decimal(input_tokens - cached) * Decimal(pricing.input_usd_per_million)
        + Decimal(cached) * Decimal(pricing.cached_input_usd_per_million)
        + Decimal(output_tokens) * Decimal(pricing.output_usd_per_million)
    ) / Decimal(1_000_000)
    return ProviderUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        total_tokens=total_tokens,
        estimated_cost_usd=format(cost.normalize(), "f"),
    )


def _add_usage(left: ProviderUsage, right: ProviderUsage) -> ProviderUsage:
    return ProviderUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated_cost_usd=format(
            (Decimal(left.estimated_cost_usd) + Decimal(right.estimated_cost_usd)).normalize(),
            "f",
        ),
        unaccounted_attempts=left.unaccounted_attempts + right.unaccounted_attempts,
        retry_wait_seconds=left.retry_wait_seconds + right.retry_wait_seconds,
    )


Request = ProposalRequest | ImplementationRequest | RepairRequest | ReflectionRequest
Response = Proposal | GeneratedPackage | Reflection


class OpenAIChatCompletionsModel:
    """Chat Completions implementation of the existing four-operation protocol."""

    def __init__(
        self,
        config: OpenAIChatCompletionsConfig,
        *,
        transport: ResponsesTransport | None = None,
        retry_policy: RetryPolicy | None = None,
        retry_runtime: RetryRuntime | None = None,
        model_limit_resolver: ModelLimitResolver | None = None,
        transcript_sink: TranscriptSink | None = None,
    ) -> None:
        if not isinstance(config, OpenAIChatCompletionsConfig):
            raise ValueError("config must be OpenAIChatCompletionsConfig")
        self.config = config
        self._transport = UrllibResponsesTransport() if transport is None else transport
        self._retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self._retry_runtime = RetryRuntime() if retry_runtime is None else retry_runtime
        self._model_limit_resolver = model_limit_resolver
        self._transcript_sink = transcript_sink
        if not isinstance(self._retry_policy, RetryPolicy):
            raise ValueError("retry_policy must be RetryPolicy")
        if not isinstance(self._retry_runtime, RetryRuntime):
            raise ValueError("retry_runtime must be RetryRuntime")
        if self._model_limit_resolver is not None and not callable(self._model_limit_resolver):
            raise ValueError("model_limit_resolver must be callable or None")
        if self._transcript_sink is not None and not callable(self._transcript_sink):
            raise ValueError("transcript_sink must be callable or None")
        self._transcripts: list[ProviderTranscript] = []
        self._total_usage = ProviderUsage()
        self._context_limits: ModelContextLimits | None = None
        self._context_limits_resolved = False

    @property
    def transcripts(self) -> tuple[ProviderTranscript, ...]:
        return tuple(self._transcripts)

    @property
    def total_usage(self) -> ProviderUsage:
        return self._total_usage

    @property
    def context_limits(self) -> ModelContextLimits | None:
        return self._context_limits

    def _resolve_context_limits(self, secret: str) -> ModelContextLimits | None:
        if self._context_limits_resolved:
            return self._context_limits
        self._context_limits_resolved = True
        if self._model_limit_resolver is None:
            return None
        try:
            limits = self._model_limit_resolver(self.config, secret)
        except Exception:
            return None
        if limits is not None and not isinstance(limits, ModelContextLimits):
            raise ValueError("model_limit_resolver returned invalid limits")
        self._context_limits = limits
        return limits

    def _credential(self, operation: ResearchOperation) -> str:
        key = os.environ.get(self.config.api_key_env)
        if type(key) is not str or not key.strip() or any(ord(char) < 33 for char in key):
            raise OpenAIProviderError(
                ProviderErrorCode.CREDENTIAL_MISSING,
                "required credential environment variable "
                f"{self.config.api_key_env} is unavailable",
                operation=operation,
            )
        return key

    def _payload(
        self,
        operation: ResearchOperation,
        request: Request,
        retry: bool,
        context_limits: ModelContextLimits | None = None,
    ) -> dict[str, object]:
        request_text = canonical_json_bytes(
            {"operation": operation.value, "request": request.to_wire()}
        ).decode("ascii")
        source_policy = (
            request.source_policy
            if isinstance(request, (ProposalRequest, ImplementationRequest, RepairRequest))
            else DEFAULT_CANDIDATE_SOURCE_POLICY
        )
        schema = response_json_schema(operation)
        instructions = instructions_for(
            operation,
            schema_retry=retry,
            source_policy=source_policy,
        )
        if self.config.response_format == "json_schema":
            output_format: dict[str, object] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"kuairand_{operation.value}_v{PROMPT_VERSION}",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            output_format = {"type": "json_object"}
            instructions += (
                "\n\nReturn exactly one JSON object matching this local validation schema: "
                + canonical_json_bytes(schema).decode("ascii")
            )
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": request_text,
                },
            ],
            "response_format": output_format,
            "n": 1,
            "store": False,
            "stream": False,
            "tools": [],
            "tool_choice": "none",
            "parallel_tool_calls": False,
        }
        output_limit = self.config.max_output_tokens
        if context_limits is not None:
            prompt_bytes = len(canonical_json_bytes(payload))
            # A tokenizer-independent upper bound: a text token cannot encode less than one input
            # byte. The fixed allowance covers chat framing that is absent from the JSON body.
            estimated_prompt_tokens = prompt_bytes + 512
            available_context = context_limits.context_length - estimated_prompt_tokens
            output_limit = min(
                output_limit,
                context_limits.max_completion_tokens,
                available_context,
            )
            if output_limit < 1:
                raise OpenAIProviderError(
                    ProviderErrorCode.CONTEXT_LIMIT,
                    "provider-advertised context cannot fit the bounded research request",
                    operation=operation,
                )
        payload[self.config.max_tokens_parameter] = output_limit
        hostname = (urlsplit(self.config.base_url).hostname or "").lower()
        if self.config.send_reasoning_effort:
            if hostname in _GATEWAY_REASONING_HOSTS:
                payload["reasoning"] = {
                    "effort": self.config.reasoning_effort,
                    "exclude": True,
                }
            elif hostname in _THINKING_BUDGET_HOSTS:
                # This host silently ignores both OpenAI-style controls, so the configured
                # effort must be expressed as an explicit budget or the model reasons
                # without bound and the call exceeds its timeout.
                budget = _THINKING_BUDGET_TOKENS.get(self.config.reasoning_effort, 4096)
                payload["thinking"] = (
                    {"type": "enabled", "budget_tokens": budget}
                    if budget > 0
                    else {"type": "disabled"}
                )
            else:
                payload["reasoning_effort"] = self.config.reasoning_effort
        if hostname == "openrouter.ai":
            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
            # OpenRouter's ``require_parameters`` filter compares every supplied optional
            # parameter with endpoint capability metadata.  These four fields are merely their
            # Chat Completions defaults and are not advertised by otherwise compatible strict-
            # schema endpoints, so sending them can incorrectly eliminate every route.
            for defaulted_parameter in ("n", "store", "stream", "parallel_tool_calls"):
                payload.pop(defaulted_parameter, None)
            payload["provider"] = {
                "allow_fallbacks": True,
                "require_parameters": True,
            }
        return payload

    def _record(
        self,
        operation: ResearchOperation,
        attempt: int,
        request: Mapping[str, object],
        response: object,
        secret: str,
        outcome: str,
        response_bytes: bytes | None,
        latency_seconds: float,
        retry_wait_seconds: float = 0.0,
    ) -> None:
        request_digest = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        response_sha256 = (
            None if response_bytes is None else hashlib.sha256(response_bytes).hexdigest()
        )
        wire = {
            "schema_version": 1,
            "operation": operation.value,
            "attempt": attempt,
            "request_digest": request_digest,
            "response_sha256": response_sha256,
            "outcome": outcome,
            "latency_seconds": latency_seconds,
            "retry_wait_seconds": retry_wait_seconds,
            "request": _redact(dict(request), secret),
            "response": _redact(response, secret),
        }
        content = canonical_json_bytes(wire)
        transcript = ProviderTranscript(
            operation=operation,
            attempt=attempt,
            request_digest=request_digest,
            response_sha256=response_sha256,
            outcome=outcome,
            latency_seconds=latency_seconds,
            retry_wait_seconds=retry_wait_seconds,
            json_bytes=content,
            digest=hashlib.sha256(content).hexdigest(),
        )
        self._transcripts.append(transcript)
        if self._transcript_sink is not None:
            self._transcript_sink(transcript)

    def _output_text(self, envelope: Mapping[str, object]) -> tuple[str, str]:
        if envelope.get("error") is not None:
            raise OpenAIProviderError(
                ProviderErrorCode.INCOMPLETE, "provider response was not completed"
            )
        response_id = envelope.get("id")
        choices = envelope.get("choices")
        if (
            envelope.get("object") != "chat.completion"
            or type(response_id) is not str
            or not response_id
            or not isinstance(choices, list)
            or len(choices) != 1
        ):
            raise SchemaValidationError("provider response envelope is malformed")
        choice = choices[0]
        if not isinstance(choice, Mapping) or choice.get("index") != 0:
            raise SchemaValidationError("provider choice is malformed")
        finish_reason = choice.get("finish_reason")
        if finish_reason == "content_filter":
            raise OpenAIProviderError(ProviderErrorCode.REFUSAL, "provider refused the request")
        if finish_reason != "stop":
            raise OpenAIProviderError(
                ProviderErrorCode.INCOMPLETE, "provider response was not completed"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise SchemaValidationError("provider choice contains no assistant message")
        refusal = message.get("refusal")
        if type(refusal) is str and refusal:
            raise OpenAIProviderError(ProviderErrorCode.REFUSAL, "provider refused the request")
        if refusal is not None and type(refusal) is not str:
            raise SchemaValidationError("provider refusal field is malformed")
        if message.get("tool_calls") is not None or message.get("function_call") is not None:
            raise SchemaValidationError("provider returned a forbidden tool call")
        content = message.get("content")
        if type(content) is not str or not content:
            raise SchemaValidationError("provider response contains no message content")
        return content, response_id

    @staticmethod
    def _bounded_response_evidence(body: bytes) -> Mapping[str, object]:
        try:
            return {"body_utf8": body.decode("utf-8", errors="strict")}
        except UnicodeError:
            return {
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "body_encoding": "non_utf8",
            }

    def _parse(self, operation: ResearchOperation, request: Request, text: str) -> Response:
        if operation is ResearchOperation.PROPOSE:
            assert isinstance(request, ProposalRequest)
            proposal = Proposal.from_json(text)
            if proposal.parent_candidate_id != request.parent_candidate_id:
                raise SchemaValidationError("proposal response names the wrong parent candidate")
            try:
                request.source_policy.validate_manifest(
                    proposal.files_expected,
                    require_final_entrypoint=True,
                )
            except CandidateManifestPolicyError as exc:
                raise SchemaValidationError(
                    f"proposal manifest violates candidate-source policy ({exc.fingerprint}): {exc}"
                ) from exc
            return proposal
        if operation in {ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}:
            generated = GeneratedPackage.from_json(text)
            if generated.request_id != request.request_id:
                raise SchemaValidationError("generated response names the wrong request")
            return generated
        return Reflection.from_json(text)

    def _retry_after_seconds(self, value: str | None) -> float | None:
        if value is None:
            return None
        if value.isascii() and value.isdecimal():
            return min(float(int(value)), self._retry_policy.max_retry_after_seconds)
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        now = self._retry_runtime.utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        seconds = max(0.0, (retry_at.astimezone(UTC) - now.astimezone(UTC)).total_seconds())
        return min(seconds, self._retry_policy.max_retry_after_seconds)

    def _rate_limit_wait(self, retry_number: int, retry_after: str | None) -> float | None:
        wait = self._retry_after_seconds(retry_after)
        if wait is None:
            exponential = min(
                self._retry_policy.initial_backoff_seconds * (2 ** max(0, retry_number - 1)),
                self._retry_policy.max_backoff_seconds,
            )
            jitter_room = min(
                exponential * self._retry_policy.jitter_ratio,
                self._retry_policy.max_backoff_seconds - exponential,
            )
            jitter = self._retry_runtime.jitter(0.0, max(0.0, jitter_room))
            if not math.isfinite(jitter) or not 0.0 <= jitter <= max(0.0, jitter_room):
                raise ValueError("retry jitter must remain within its requested bounds")
            wait = exponential + jitter
        remaining = self._remaining_research_seconds()
        if remaining is None:
            return wait
        if wait >= remaining:
            return None
        return wait

    def _remaining_research_seconds(self) -> float | None:
        remaining = self._retry_runtime.remaining_research_seconds()
        if remaining is None:
            return None
        if type(remaining) not in {int, float} or not math.isfinite(remaining) or remaining < 0:
            raise ValueError("remaining research seconds must be finite and non-negative")
        return float(remaining)

    def _call(self, operation: ResearchOperation, request: Request) -> Response:
        secret = self._credential(operation)
        context_limits = self._resolve_context_limits(secret)
        malformed_retries = 0
        transport_retries = 0
        consecutive_rate_limits = 0
        attempt = 0
        while True:
            remaining = self._remaining_research_seconds()
            if remaining is not None and remaining <= 0.0:
                raise OpenAIProviderError(
                    ProviderErrorCode.DEADLINE,
                    "provider research deadline is exhausted",
                    operation=operation,
                    attempts=attempt,
                )
            attempt += 1
            payload = self._payload(
                operation,
                request,
                malformed_retries > 0,
                context_limits,
            )
            body = canonical_json_bytes(payload)
            transport_request = TransportRequest(
                url=self.config.chat_completions_url,
                body=body,
                headers=(
                    ("Authorization", f"Bearer {secret}"),
                    ("Content-Type", "application/json"),
                    ("Accept", "application/json"),
                    ("User-Agent", "kuairand-agent/0.1.0"),
                ),
                timeout_seconds=(
                    self.config.timeout_seconds
                    if remaining is None
                    else min(self.config.timeout_seconds, remaining)
                ),
                max_response_bytes=self.config.max_response_bytes,
            )
            started = self._retry_runtime.monotonic()
            try:
                response = self._transport.send(transport_request)
            except ResponseTooLargeError as exc:
                latency_seconds = self._retry_runtime.monotonic() - started
                self._record(
                    operation,
                    attempt,
                    payload,
                    {"error": "response_too_large"},
                    secret,
                    "failed",
                    None,
                    latency_seconds,
                )
                raise OpenAIProviderError(
                    ProviderErrorCode.RESPONSE_TOO_LARGE,
                    "provider response exceeded the configured byte ceiling",
                    operation=operation,
                    attempts=attempt,
                ) from exc
            except Exception as exc:
                latency_seconds = self._retry_runtime.monotonic() - started
                consecutive_rate_limits = 0
                self._record(
                    operation,
                    attempt,
                    payload,
                    {"error": type(exc).__name__},
                    secret,
                    "failed",
                    None,
                    latency_seconds,
                )
                if transport_retries < self.config.max_transport_retries:
                    transport_retries += 1
                    continue
                raise OpenAIProviderError(
                    ProviderErrorCode.TRANSPORT,
                    f"provider transport failed ({type(exc).__name__})",
                    operation=operation,
                    attempts=attempt,
                ) from exc
            latency_seconds = self._retry_runtime.monotonic() - started
            if len(response.body) > self.config.max_response_bytes:
                self._record(
                    operation,
                    attempt,
                    payload,
                    {"error": "response_too_large"},
                    secret,
                    "failed",
                    response.body,
                    latency_seconds,
                )
                raise OpenAIProviderError(
                    ProviderErrorCode.RESPONSE_TOO_LARGE,
                    "provider response exceeded the configured byte ceiling",
                    operation=operation,
                    attempts=attempt,
                    status_code=response.status_code,
                )
            if response.status_code != 200:
                consecutive_rate_limits = (
                    consecutive_rate_limits + 1 if response.status_code == 429 else 0
                )
                retry_wait_seconds = 0.0
                retryable = response.status_code in {408, 409, 429} or (
                    500 <= response.status_code <= 599
                )
                can_retry = retryable and transport_retries < self.config.max_transport_retries
                if can_retry and response.status_code == 429:
                    if consecutive_rate_limits >= 2:
                        can_retry = False
                    else:
                        wait = self._rate_limit_wait(transport_retries + 1, response.retry_after)
                        if wait is None:
                            can_retry = False
                        else:
                            retry_wait_seconds = wait
                self._record(
                    operation,
                    attempt,
                    payload,
                    {"http_status": response.status_code},
                    secret,
                    "failed",
                    response.body,
                    latency_seconds,
                    retry_wait_seconds,
                )
                if can_retry:
                    transport_retries += 1
                    if retry_wait_seconds:
                        self._total_usage = _add_usage(
                            self._total_usage,
                            ProviderUsage(retry_wait_seconds=retry_wait_seconds),
                        )
                        self._retry_runtime.sleep(retry_wait_seconds)
                    continue
                raise OpenAIProviderError(
                    ProviderErrorCode.HTTP,
                    f"provider returned HTTP {response.status_code}",
                    operation=operation,
                    attempts=attempt,
                    status_code=response.status_code,
                )
            envelope: Mapping[str, object] | None = None
            known_usage: ProviderUsage | None = None
            try:
                envelope = parse_json_object(response.body.decode("utf-8", errors="strict"))
                known_usage = _usage(envelope, self.config.pricing)
                output_text, _ = self._output_text(envelope)
                typed = self._parse(operation, request, output_text)
            except OpenAIProviderError as exc:
                self._record(
                    operation,
                    attempt,
                    payload,
                    dict(envelope or {}),
                    secret,
                    "failed",
                    response.body,
                    latency_seconds,
                )
                if known_usage is not None:
                    self._total_usage = _add_usage(self._total_usage, known_usage)
                raise OpenAIProviderError(
                    exc.code,
                    str(exc),
                    operation=operation,
                    attempts=attempt,
                    status_code=response.status_code,
                ) from exc
            except (SchemaValidationError, UnicodeError, json.JSONDecodeError) as exc:
                self._record(
                    operation,
                    attempt,
                    payload,
                    dict(envelope or self._bounded_response_evidence(response.body)),
                    secret,
                    "malformed",
                    response.body,
                    latency_seconds,
                )
                self._total_usage = _add_usage(
                    self._total_usage,
                    known_usage or ProviderUsage(unaccounted_attempts=1),
                )
                if malformed_retries < self.config.max_malformed_retries:
                    malformed_retries += 1
                    continue
                raise OpenAIProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "provider response failed strict local validation",
                    operation=operation,
                    attempts=attempt,
                ) from exc
            self._record(
                operation,
                attempt,
                payload,
                dict(envelope),
                secret,
                "accepted",
                response.body,
                latency_seconds,
            )
            self._total_usage = _add_usage(self._total_usage, known_usage)
            return typed

    def propose(self, request: ProposalRequest) -> Proposal:
        if not isinstance(request, ProposalRequest):
            raise ResearchModelError("propose requires ProposalRequest")
        return cast(Proposal, self._call(ResearchOperation.PROPOSE, request))

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        if not isinstance(request, ImplementationRequest):
            raise ResearchModelError("implement requires ImplementationRequest")
        return cast(GeneratedPackage, self._call(ResearchOperation.IMPLEMENT, request))

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        if not isinstance(request, RepairRequest):
            raise ResearchModelError("repair requires RepairRequest")
        return cast(GeneratedPackage, self._call(ResearchOperation.REPAIR, request))

    def reflect(self, request: ReflectionRequest) -> Reflection:
        if not isinstance(request, ReflectionRequest):
            raise ResearchModelError("reflect requires ReflectionRequest")
        return cast(Reflection, self._call(ResearchOperation.REFLECT, request))


# Backward-compatible public alias.  Persisted code importing the earlier name still constructs the
# Chat Completions adapter and therefore cannot silently dispatch to ``/responses``.
OpenAIResponsesModel = OpenAIChatCompletionsModel


class OpenAIFailoverModel:
    """Two OpenAI-compatible Chat Completions adapters with bounded, sticky failover.

    Each endpoint retains the original adapter's transport and malformed-response retry policy.
    A terminal typed provider failure switches from ``main`` to ``fallback`` exactly once.  The
    fallback remains active for later operations so a known-bad main endpoint is not retried on
    every proposal, implementation, repair, or reflection call.
    """

    def __init__(
        self,
        main: OpenAIChatCompletionsModel,
        fallback: OpenAIChatCompletionsModel,
    ) -> None:
        if not isinstance(main, OpenAIChatCompletionsModel) or not isinstance(
            fallback, OpenAIChatCompletionsModel
        ):
            raise ValueError("main and fallback must be OpenAIChatCompletionsModel instances")
        if main.config.api_key_env == fallback.config.api_key_env:
            raise ValueError("main and fallback must use dedicated credential variables")
        self.main = main
        self.fallback = fallback
        self._active_slot = "main"
        self._transcripts: list[ProviderTranscript] = []
        self._failover_events: list[ProviderFailoverEvent] = []

    @property
    def active_slot(self) -> str:
        return self._active_slot

    @property
    def active_model(self) -> OpenAIChatCompletionsModel:
        return self.main if self._active_slot == "main" else self.fallback

    @property
    def transcripts(self) -> tuple[ProviderTranscript, ...]:
        return tuple(self._transcripts)

    @property
    def failover_events(self) -> tuple[ProviderFailoverEvent, ...]:
        return tuple(self._failover_events)

    @property
    def total_usage(self) -> ProviderUsage:
        return _add_usage(self.main.total_usage, self.fallback.total_usage)

    @property
    def provider_models(self) -> tuple[tuple[str, OpenAIChatCompletionsModel], ...]:
        return (("main", self.main), ("fallback", self.fallback))

    @staticmethod
    def _failure(slot: str, error: OpenAIProviderError) -> ProviderFailure:
        return ProviderFailure(
            slot=slot,
            code=error.code,
            operation=error.operation,
            attempts=error.attempts,
            status_code=error.status_code,
        )

    def _call_model(
        self,
        model: OpenAIChatCompletionsModel,
        operation: ResearchOperation,
        request: Request,
    ) -> Response:
        before = len(model.transcripts)
        try:
            return model._call(operation, request)
        finally:
            self._transcripts.extend(model.transcripts[before:])

    def _call(self, operation: ResearchOperation, request: Request) -> Response:
        if self._active_slot == "fallback":
            try:
                return self._call_model(self.fallback, operation, request)
            except OpenAIProviderError as error:
                if error.code is ProviderErrorCode.DEADLINE:
                    raise
                failure = self._failure("fallback", error)
                raise OpenAIProviderChainError(error, (failure,)) from error

        try:
            return self._call_model(self.main, operation, request)
        except OpenAIProviderError as main_error:
            if main_error.code is ProviderErrorCode.DEADLINE:
                raise
            main_failure = self._failure("main", main_error)
            self._active_slot = "fallback"
            self._failover_events.append(
                ProviderFailoverEvent(
                    operation=operation,
                    from_slot="main",
                    to_slot="fallback",
                    failure=main_failure,
                )
            )
        try:
            return self._call_model(self.fallback, operation, request)
        except OpenAIProviderError as fallback_error:
            fallback_failure = self._failure("fallback", fallback_error)
            raise OpenAIProviderChainError(
                fallback_error,
                (main_failure, fallback_failure),
            ) from fallback_error

    def propose(self, request: ProposalRequest) -> Proposal:
        if not isinstance(request, ProposalRequest):
            raise ResearchModelError("propose requires ProposalRequest")
        return cast(Proposal, self._call(ResearchOperation.PROPOSE, request))

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        if not isinstance(request, ImplementationRequest):
            raise ResearchModelError("implement requires ImplementationRequest")
        return cast(GeneratedPackage, self._call(ResearchOperation.IMPLEMENT, request))

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        if not isinstance(request, RepairRequest):
            raise ResearchModelError("repair requires RepairRequest")
        return cast(GeneratedPackage, self._call(ResearchOperation.REPAIR, request))

    def reflect(self, request: ReflectionRequest) -> Reflection:
        if not isinstance(request, ReflectionRequest):
            raise ResearchModelError("reflect requires ReflectionRequest")
        return cast(Reflection, self._call(ResearchOperation.REFLECT, request))


__all__ = [
    "ModelContextLimits",
    "ModelLimitResolver",
    "OpenAIChatCompletionsConfig",
    "OpenAIChatCompletionsModel",
    "OpenAIFailoverModel",
    "OpenAIProviderChainError",
    "OpenAIProviderError",
    "OpenAIResponsesConfig",
    "OpenAIResponsesModel",
    "OpenRouterModelLimitResolver",
    "ProviderErrorCode",
    "ProviderFailoverEvent",
    "ProviderFailure",
    "ProviderModelLimitResolver",
    "ProviderTranscript",
    "ProviderUsage",
    "ResponseTooLargeError",
    "ResponsesTransport",
    "ResponsesTransportError",
    "RetryPolicy",
    "RetryRuntime",
    "TokenPricing",
    "TranscriptSink",
    "TransportRequest",
    "TransportResponse",
    "UrllibResponsesTransport",
]

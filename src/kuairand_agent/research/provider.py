"""Bounded OpenAI Responses API adapter for the typed research-model seam.

Credentials are resolved only while a call is being dispatched.  The adapter has no file, shell,
tool, or evaluator authority; its only optional side effect is one bounded HTTPS POST per attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Final, Protocol, cast
from urllib.parse import urlsplit

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

_MODEL_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ENV_RE: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_PRICE_RE: Final = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?\Z")
_REASONING_EFFORTS: Final = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_SECRET_PATTERN: Final = re.compile(r"(?i)(?:bearer\s+|sk-(?:proj-)?)[A-Za-z0-9_.-]{8,}")


class ProviderErrorCode(StrEnum):
    CREDENTIAL_MISSING = "credential_missing"
    TRANSPORT = "transport"
    RESPONSE_TOO_LARGE = "response_too_large"
    HTTP = "http"
    INCOMPLETE = "incomplete"
    REFUSAL = "refusal"
    MALFORMED_RESPONSE = "malformed_response"


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
class OpenAIResponsesConfig:
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
    def response_url(self) -> str:
        return f"{self.base_url}/responses"


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

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("status_code must be an HTTP status")
        if type(self.body) is not bytes:
            raise ValueError("transport response body must be bytes")


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
                )
        except urllib.error.HTTPError as exc:
            return TransportResponse(
                status_code=exc.code,
                body=_bounded_read(exc, request.max_response_bytes),
            )
        except ResponseTooLargeError:
            raise
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            raise ResponsesTransportError(type(exc).__name__) from exc


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: str = "0"
    unaccounted_attempts: int = 0


@dataclass(frozen=True, slots=True)
class ProviderTranscript:
    operation: ResearchOperation
    attempt: int
    request_digest: str
    response_sha256: str | None
    outcome: str
    latency_seconds: float
    json_bytes: bytes = field(repr=False)
    digest: str


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
    input_tokens = _nonnegative_integer(raw.get("input_tokens"), "input_tokens")
    output_tokens = _nonnegative_integer(raw.get("output_tokens"), "output_tokens")
    total_tokens = _nonnegative_integer(raw.get("total_tokens"), "total_tokens")
    input_details = raw.get("input_tokens_details", {})
    output_details = raw.get("output_tokens_details", {})
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
    )


Request = ProposalRequest | ImplementationRequest | RepairRequest | ReflectionRequest
Response = Proposal | GeneratedPackage | Reflection


class OpenAIResponsesModel:
    """OpenAI Responses implementation of the existing four-operation protocol."""

    def __init__(
        self,
        config: OpenAIResponsesConfig,
        *,
        transport: ResponsesTransport | None = None,
    ) -> None:
        if not isinstance(config, OpenAIResponsesConfig):
            raise ValueError("config must be OpenAIResponsesConfig")
        self.config = config
        self._transport = UrllibResponsesTransport() if transport is None else transport
        self._transcripts: list[ProviderTranscript] = []
        self._total_usage = ProviderUsage()

    @property
    def transcripts(self) -> tuple[ProviderTranscript, ...]:
        return tuple(self._transcripts)

    @property
    def total_usage(self) -> ProviderUsage:
        return self._total_usage

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
        self, operation: ResearchOperation, request: Request, retry: bool
    ) -> dict[str, object]:
        request_text = canonical_json_bytes(
            {"operation": operation.value, "request": request.to_wire()}
        ).decode("ascii")
        return {
            "model": self.config.model,
            "instructions": instructions_for(operation, schema_retry=retry),
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request_text}],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"kuairand_{operation.value}_v{PROMPT_VERSION}",
                    "strict": True,
                    "schema": response_json_schema(operation),
                }
            },
            "reasoning": {"effort": self.config.reasoning_effort},
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
            "background": False,
            "stream": False,
            "tools": [],
            "tool_choice": "none",
            "parallel_tool_calls": False,
        }

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
            "request": _redact(dict(request), secret),
            "response": _redact(response, secret),
        }
        content = canonical_json_bytes(wire)
        self._transcripts.append(
            ProviderTranscript(
                operation=operation,
                attempt=attempt,
                request_digest=request_digest,
                response_sha256=response_sha256,
                outcome=outcome,
                latency_seconds=latency_seconds,
                json_bytes=content,
                digest=hashlib.sha256(content).hexdigest(),
            )
        )

    def _output_text(self, envelope: Mapping[str, object]) -> tuple[str, str]:
        if envelope.get("status") != "completed" or envelope.get("error") is not None:
            raise OpenAIProviderError(
                ProviderErrorCode.INCOMPLETE, "provider response was not completed"
            )
        response_id = envelope.get("id")
        output = envelope.get("output")
        if type(response_id) is not str or not response_id or not isinstance(output, list):
            raise SchemaValidationError("provider response envelope is malformed")
        texts: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                raise SchemaValidationError("provider output item must be an object")
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or not isinstance(item.get("content"), list):
                raise SchemaValidationError("provider returned a forbidden non-message output")
            for content in cast(list[object], item["content"]):
                if not isinstance(content, Mapping):
                    raise SchemaValidationError("provider message content must be an object")
                if content.get("type") == "refusal":
                    raise OpenAIProviderError(
                        ProviderErrorCode.REFUSAL, "provider refused the request"
                    )
                if content.get("type") != "output_text" or type(content.get("text")) is not str:
                    raise SchemaValidationError("provider message content is not output_text")
                texts.append(cast(str, content["text"]))
        if not texts:
            raise SchemaValidationError("provider response contains no output_text")
        return "".join(texts), response_id

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
            return proposal
        if operation in {ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}:
            generated = GeneratedPackage.from_json(text)
            if generated.request_id != request.request_id:
                raise SchemaValidationError("generated response names the wrong request")
            return generated
        return Reflection.from_json(text)

    def _call(self, operation: ResearchOperation, request: Request) -> Response:
        secret = self._credential(operation)
        malformed_retries = 0
        transport_retries = 0
        attempt = 0
        while True:
            attempt += 1
            payload = self._payload(operation, request, malformed_retries > 0)
            body = canonical_json_bytes(payload)
            transport_request = TransportRequest(
                url=self.config.response_url,
                body=body,
                headers=(
                    ("Authorization", f"Bearer {secret}"),
                    ("Content-Type", "application/json"),
                    ("Accept", "application/json"),
                    ("User-Agent", "kuairand-agent/0.1.0"),
                ),
                timeout_seconds=self.config.timeout_seconds,
                max_response_bytes=self.config.max_response_bytes,
            )
            started = time.perf_counter()
            try:
                response = self._transport.send(transport_request)
            except ResponseTooLargeError as exc:
                latency_seconds = time.perf_counter() - started
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
                latency_seconds = time.perf_counter() - started
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
            latency_seconds = time.perf_counter() - started
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
                self._record(
                    operation,
                    attempt,
                    payload,
                    {"http_status": response.status_code},
                    secret,
                    "failed",
                    response.body,
                    latency_seconds,
                )
                if (
                    response.status_code in {408, 409, 429} or 500 <= response.status_code <= 599
                ) and transport_retries < self.config.max_transport_retries:
                    transport_retries += 1
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


__all__ = [
    "OpenAIProviderError",
    "OpenAIResponsesConfig",
    "OpenAIResponsesModel",
    "ProviderErrorCode",
    "ProviderTranscript",
    "ProviderUsage",
    "ResponseTooLargeError",
    "ResponsesTransport",
    "ResponsesTransportError",
    "TokenPricing",
    "TransportRequest",
    "TransportResponse",
    "UrllibResponsesTransport",
]

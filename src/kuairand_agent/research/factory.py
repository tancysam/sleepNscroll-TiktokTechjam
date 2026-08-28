"""Fail-closed selection of the configured research-model adapter.

This module is the trusted seam between a frozen ``research.provider`` choice and the
provider-independent :class:`~kuairand_agent.research.interface.ResearchModel` interface.  It
does not call a provider or mutate campaign state.  In particular, credential values are checked
only for availability and are never retained in a result or copied into a diagnostic.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from kuairand_agent.config import HARD_MAX_REPAIRS, ResearchConfig
from kuairand_agent.research.interface import ResearchModel, ResearchModelError
from kuairand_agent.research.provider import (
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    ResponsesTransport,
    TokenPricing,
)
from kuairand_agent.research.scripted import ScriptedResearchModel, ScriptedResponse


class ProviderUnavailableCode(StrEnum):
    """Stable, secret-free reason codes suitable for the durable failure ledger."""

    UNSUPPORTED_PROVIDER = "unsupported_provider"
    INVALID_CONFIG = "invalid_config"
    SCRIPT_MISSING = "script_missing"
    SETTINGS_MISSING = "settings_missing"
    CREDENTIAL_MISSING = "credential_missing"
    CREDENTIAL_LOOKUP_FAILED = "credential_lookup_failed"
    INITIALIZATION_FAILED = "initialization_failed"
    INVALID_MODEL = "invalid_model"


@dataclass(frozen=True, slots=True)
class ProviderUnavailableDiagnostic:
    """Public result when provider research cannot safely start.

    The fixed messages deliberately exclude exception text and credential values so callers may
    persist ``to_wire()`` without making the campaign database a secret-bearing surface.
    """

    provider: str
    code: ProviderUnavailableCode
    message: str
    retryable: bool
    credential_env: str | None

    def __post_init__(self) -> None:
        if self.provider not in {"scripted", "openai", "invalid"}:
            raise ValueError("diagnostic provider must be a stable public provider name")
        if not isinstance(self.code, ProviderUnavailableCode):
            raise ValueError("diagnostic code must be ProviderUnavailableCode")
        if type(self.message) is not str or not self.message or "\x00" in self.message:
            raise ValueError("diagnostic message must be non-empty safe text")
        if type(self.retryable) is not bool:
            raise ValueError("diagnostic retryable flag must be boolean")
        if self.credential_env is not None and (
            type(self.credential_env) is not str or not self.credential_env
        ):
            raise ValueError("diagnostic credential environment name must be non-empty")

    def to_wire(self) -> dict[str, object]:
        """Return the stable, credential-value-free campaign failure payload."""

        return {
            "category": "provider_unavailable",
            "provider": self.provider,
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "credential_env": self.credential_env,
        }


@dataclass(frozen=True, slots=True)
class AvailableResearchProvider:
    """One validated adapter selected through the common research-model interface."""

    provider: str
    model: ResearchModel = field(repr=False)
    live_provider_used: bool

    def __post_init__(self) -> None:
        if self.provider not in {"scripted", "openai"}:
            raise ValueError("available provider name is unsupported")
        if not isinstance(self.model, ResearchModel):
            raise ValueError("available provider must implement the complete ResearchModel seam")
        if type(self.live_provider_used) is not bool:
            raise ValueError("live_provider_used must be boolean")
        if self.live_provider_used != (self.provider == "openai"):
            raise ValueError("live-provider evidence does not match the selected provider")


ResearchProviderSelection = AvailableResearchProvider | ProviderUnavailableDiagnostic
CredentialLookup = Callable[[str], str | None]


class OpenAIModelFactory(Protocol):
    """Injected construction seam; tests can retain the real adapter with a fake transport."""

    def __call__(
        self,
        config: OpenAIResponsesConfig,
        transport: ResponsesTransport | None,
    ) -> object: ...


def _environment_credential(name: str) -> str | None:
    return os.environ.get(name)


def _default_openai_model_factory(
    config: OpenAIResponsesConfig,
    transport: ResponsesTransport | None,
) -> ResearchModel:
    if transport is None:
        return OpenAIResponsesModel(config)
    return OpenAIResponsesModel(config, transport=transport)


def openai_responses_config(research: ResearchConfig) -> OpenAIResponsesConfig | None:
    """Translate the frozen schema-v2 provider table into the transport adapter config."""

    settings = research.openai
    if research.provider != "openai" or settings is None:
        return None
    return OpenAIResponsesConfig(
        model=settings.model,
        base_url=settings.base_url,
        reasoning_effort=settings.reasoning_effort,
        pricing=TokenPricing(
            input_usd_per_million=settings.pricing.input_usd_per_million,
            cached_input_usd_per_million=settings.pricing.cached_input_usd_per_million,
            output_usd_per_million=settings.pricing.output_usd_per_million,
        ),
        api_key_env=settings.api_key_env,
        timeout_seconds=settings.timeout_seconds,
        max_response_bytes=settings.max_response_bytes,
        max_output_tokens=settings.max_output_tokens,
        max_malformed_retries=settings.max_malformed_retries,
        max_transport_retries=settings.max_transport_retries,
    )


def _usable_credential(value: object) -> bool:
    """Match the adapter's dispatch-time credential rules without retaining the value."""

    return (
        type(value) is str
        and bool(value.strip())
        and all(ord(character) >= 33 for character in value)
    )


def _diagnostic(
    provider: str,
    code: ProviderUnavailableCode,
    message: str,
    *,
    retryable: bool,
    credential_env: str | None = None,
) -> ProviderUnavailableDiagnostic:
    return ProviderUnavailableDiagnostic(
        provider=provider,
        code=code,
        message=message,
        retryable=retryable,
        credential_env=credential_env,
    )


def select_research_provider(
    research: object,
    *,
    scripted_responses: Sequence[ScriptedResponse] = (),
    openai_config: OpenAIResponsesConfig | None = None,
    transport: ResponsesTransport | None = None,
    credential_lookup: CredentialLookup = _environment_credential,
    openai_model_factory: OpenAIModelFactory = _default_openai_model_factory,
) -> ResearchProviderSelection:
    """Select the frozen provider without calling it or changing campaign state.

    ``scripted_responses`` is the complete deterministic operation script.  For ``openai``, the
    bounded model/transport settings remain explicit in ``openai_config`` while only the named
    credential is resolved at runtime.  Every construction or availability failure becomes a
    stable public diagnostic; arbitrary exception text is intentionally discarded.
    """

    if not isinstance(research, ResearchConfig):
        return _diagnostic(
            "invalid",
            ProviderUnavailableCode.INVALID_CONFIG,
            "Research provider configuration is invalid.",
            retryable=False,
        )
    if research.provider not in {"scripted", "openai"}:
        return _diagnostic(
            "invalid",
            ProviderUnavailableCode.UNSUPPORTED_PROVIDER,
            "The configured research provider is unsupported.",
            retryable=False,
        )
    if (
        type(research.max_repairs_per_experiment) is not int
        or not 0 <= research.max_repairs_per_experiment <= HARD_MAX_REPAIRS
    ):
        return _diagnostic(
            research.provider,
            ProviderUnavailableCode.INVALID_CONFIG,
            "Research provider configuration is invalid.",
            retryable=False,
        )

    if research.provider == "scripted":
        try:
            scripted_model: ResearchModel = ScriptedResearchModel(scripted_responses)
        except (ResearchModelError, TypeError, ValueError):
            return _diagnostic(
                "scripted",
                ProviderUnavailableCode.SCRIPT_MISSING,
                "The deterministic research script is unavailable or invalid.",
                retryable=False,
            )
        return AvailableResearchProvider(
            provider="scripted",
            model=scripted_model,
            live_provider_used=False,
        )

    if openai_config is None:
        try:
            openai_config = openai_responses_config(research)
        except (TypeError, ValueError):
            openai_config = None
    if not isinstance(openai_config, OpenAIResponsesConfig):
        return _diagnostic(
            "openai",
            ProviderUnavailableCode.SETTINGS_MISSING,
            "OpenAI provider settings are unavailable.",
            retryable=False,
        )
    try:
        credential = credential_lookup(openai_config.api_key_env)
    except Exception:
        return _diagnostic(
            "openai",
            ProviderUnavailableCode.CREDENTIAL_LOOKUP_FAILED,
            "The OpenAI credential could not be resolved.",
            retryable=True,
            credential_env=openai_config.api_key_env,
        )
    if not _usable_credential(credential):
        return _diagnostic(
            "openai",
            ProviderUnavailableCode.CREDENTIAL_MISSING,
            "The required OpenAI credential is unavailable.",
            retryable=True,
            credential_env=openai_config.api_key_env,
        )
    # Do not retain the resolved credential.  OpenAIResponsesModel resolves the named variable
    # afresh only when a typed operation is actually dispatched.
    del credential
    try:
        openai_model = openai_model_factory(openai_config, transport)
    except Exception:
        return _diagnostic(
            "openai",
            ProviderUnavailableCode.INITIALIZATION_FAILED,
            "The OpenAI research adapter could not be initialized.",
            retryable=True,
            credential_env=openai_config.api_key_env,
        )
    if not isinstance(openai_model, ResearchModel):
        return _diagnostic(
            "openai",
            ProviderUnavailableCode.INVALID_MODEL,
            "The initialized OpenAI adapter does not implement the research-model interface.",
            retryable=False,
            credential_env=openai_config.api_key_env,
        )
    return AvailableResearchProvider(
        provider="openai",
        model=openai_model,
        live_provider_used=True,
    )


__all__ = [
    "AvailableResearchProvider",
    "CredentialLookup",
    "OpenAIModelFactory",
    "ProviderUnavailableCode",
    "ProviderUnavailableDiagnostic",
    "ResearchProviderSelection",
    "openai_responses_config",
    "select_research_provider",
]

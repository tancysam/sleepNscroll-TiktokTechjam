from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from kuairand_agent.config import ResearchConfig, load_config
from kuairand_agent.research.factory import (
    AvailableResearchProvider,
    ProviderUnavailableCode,
    ProviderUnavailableDiagnostic,
    openai_chat_completions_configs,
    openai_responses_configs,
    select_research_provider,
)
from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.provider import (
    OpenAIFailoverModel,
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    ResponsesTransport,
    TokenPricing,
)
from kuairand_agent.research.schemas import (
    GeneratedPackage,
    ImplementationRequest,
    Proposal,
    ProposalRequest,
    RepairRequest,
    RequiredField,
    ResearchOperation,
)
from kuairand_agent.research.scripted import ScriptedResearchModel, ScriptedResponse


def _proposal() -> Proposal:
    return Proposal(
        proposal_id="factory-script-v1",
        hypothesis="A legal scalar feature can improve within-user ranking.",
        mechanism="Add one deterministic scalar term to the candidate score.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="fm-seed",
        principal_change="Replace a constant score with one scalar term.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                "trusted/causal_feature_matrix:duration_ms",
                "inference_input",
                "Use one controller-approved current-item field.",
            ),
        ),
        objective="binary long-view ranking",
        sampling="logged impressions",
        grouping="user impression groups",
        weighting="uniform",
        causal_cutoff="No current-row or future outcome is used.",
        estimated_runtime_seconds=1,
        estimated_memory_mb=64,
        smoke_plan="Score a four-row fixture.",
        inner_fold_plan="Screen Fold B before Fold A.",
        falsification_criteria="Reject a structural or metric regression.",
        promotion_criteria="Require all deterministic promotion gates.",
        maximum_repairs=1,
        rollback_parent_id="fm-seed",
        attributions=("factory test fixture",),
    )


def _openai_config() -> OpenAIResponsesConfig:
    return OpenAIResponsesConfig(
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        reasoning_effort="low",
        pricing=TokenPricing(
            input_usd_per_million="1",
            cached_input_usd_per_million="0.25",
            output_usd_per_million="4",
        ),
        api_key_env="KUAIRAND_FACTORY_TEST_KEY",
        timeout_seconds=5.0,
        max_response_bytes=4096,
        max_output_tokens=1024,
    )


def test_scripted_selection_returns_the_existing_typed_deterministic_model() -> None:
    proposal = _proposal()

    selection = select_research_provider(
        ResearchConfig(provider="scripted", max_repairs_per_experiment=1),
        scripted_responses=(ScriptedResponse(ResearchOperation.PROPOSE, proposal),),
    )

    assert isinstance(selection, AvailableResearchProvider)
    assert selection.provider == "scripted"
    assert selection.live_provider_used is False
    assert isinstance(selection.model, ScriptedResearchModel)
    assert isinstance(selection.model, ResearchModel)
    request = ProposalRequest.create(
        request_id="factory-propose",
        campaign_id="factory-campaign",
        scientific_iteration=1,
        parent_candidate_id="fm-seed",
        safe_context={"schema_version": 1},
    )
    assert selection.model.propose(request) is proposal


def test_openai_selection_without_settings_returns_public_fail_closed_diagnostic() -> None:
    selection = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        credential_lookup=lambda _name: "sk-this-must-not-be-consulted",
    )

    assert selection == ProviderUnavailableDiagnostic(
        provider="openai",
        code=ProviderUnavailableCode.SETTINGS_MISSING,
        message="OpenAI provider settings are unavailable.",
        retryable=False,
        credential_env=None,
    )
    assert selection.to_wire()["category"] == "provider_unavailable"


def test_openai_selection_without_credential_never_constructs_a_model() -> None:
    calls = 0

    def model_factory(
        config: OpenAIResponsesConfig,
        transport: ResponsesTransport | None,
    ) -> ResearchModel:
        nonlocal calls
        del config, transport
        calls += 1
        raise AssertionError("missing credentials must stop before model construction")

    selection = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_config=_openai_config(),
        credential_lookup=lambda _name: None,
        openai_model_factory=model_factory,
    )

    assert isinstance(selection, ProviderUnavailableDiagnostic)
    assert selection.code is ProviderUnavailableCode.CREDENTIAL_MISSING
    assert selection.retryable is True
    assert selection.credential_env == "KUAIRAND_FACTORY_TEST_KEY"
    assert calls == 0


def test_openai_selection_constructs_the_real_adapter_only_after_validation() -> None:
    selection = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_config=_openai_config(),
        credential_lookup=lambda _name: "sk-proj-available-only-in-memory",
    )

    assert isinstance(selection, AvailableResearchProvider)
    assert selection.provider == "openai"
    assert selection.live_provider_used is True
    assert isinstance(selection.model, OpenAIResponsesModel)
    assert "sk-proj-available-only-in-memory" not in repr(selection)


def test_schema_v4_resolves_two_dedicated_environment_profiles() -> None:
    research = load_config(Path(__file__).parents[2] / "configs" / "full-pure.toml").research
    values = {
        "INFERENCE_MAIN_BASE_URL": "https://main.example/v1",
        "INFERENCE_MAIN_MODEL": "z-ai/glm-5.3-free",
        "INFERENCE_FALLBACK_BASE_URL": "https://fallback.example/v1",
        "INFERENCE_FALLBACK_MODEL": "deepseek/deepseek-v4-pro-0813",
    }

    configs = openai_chat_completions_configs(research, setting_lookup=values.get)

    assert configs is not None
    assert [(item.model, item.base_url, item.api_key_env) for item in configs] == [
        ("z-ai/glm-5.3-free", "https://main.example/v1", "INFERENCE_MAIN_API_KEY"),
        (
            "deepseek/deepseek-v4-pro-0813",
            "https://fallback.example/v1",
            "INFERENCE_FALLBACK_API_KEY",
        ),
    ]
    assert [item.response_format for item in configs] == ["json_schema", "json_schema"]
    assert [item.max_tokens_parameter for item in configs] == [
        "max_completion_tokens",
        "max_completion_tokens",
    ]
    assert all(item.send_reasoning_effort for item in configs)
    assert openai_responses_configs(research, setting_lookup=values.get) == configs


def test_two_provider_selection_requires_both_credentials_and_builds_chain() -> None:
    main = replace(_openai_config(), api_key_env="KUAIRAND_MAIN_TEST_KEY")
    fallback = replace(
        _openai_config(),
        model="fallback-model",
        base_url="https://fallback.example/v1",
        api_key_env="KUAIRAND_FALLBACK_TEST_KEY",
    )
    missing = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_configs=(main, fallback),
        credential_lookup=lambda name: (
            "sk-proj-main-only" if name == "KUAIRAND_MAIN_TEST_KEY" else None
        ),
    )
    assert isinstance(missing, ProviderUnavailableDiagnostic)
    assert missing.code is ProviderUnavailableCode.CREDENTIAL_MISSING
    assert missing.credential_env == "KUAIRAND_FALLBACK_TEST_KEY"

    available = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_configs=(main, fallback),
        credential_lookup=lambda _name: "sk-proj-dedicated-fixture",
    )
    assert isinstance(available, AvailableResearchProvider)
    assert isinstance(available.model, OpenAIFailoverModel)
    assert available.model.active_slot == "main"


@dataclass
class _IncompleteModel:
    def propose(self, request: ProposalRequest) -> Proposal:
        del request
        return _proposal()

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        raise NotImplementedError(request)

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        raise NotImplementedError(request)

    # Deliberately no reflect method: this is not the complete ResearchModel interface.


def test_invalid_or_failed_model_factory_returns_a_redacted_diagnostic() -> None:
    secret = "sk-proj-do-not-copy-factory-exception"

    def invalid_factory(
        config: OpenAIResponsesConfig,
        transport: ResponsesTransport | None,
    ) -> ResearchModel:
        del config, transport
        raise RuntimeError(f"initialization failed with {secret}")

    failed = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_config=_openai_config(),
        credential_lookup=lambda _name: secret,
        openai_model_factory=invalid_factory,
    )
    assert isinstance(failed, ProviderUnavailableDiagnostic)
    assert failed.code is ProviderUnavailableCode.INITIALIZATION_FAILED
    assert secret not in repr(failed)
    assert secret not in str(failed.to_wire())

    incomplete = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_config=_openai_config(),
        credential_lookup=lambda _name: secret,
        openai_model_factory=lambda _config, _transport: _IncompleteModel(),
    )
    assert isinstance(incomplete, ProviderUnavailableDiagnostic)
    assert incomplete.code is ProviderUnavailableCode.INVALID_MODEL


def test_directly_constructed_unknown_research_config_fails_closed() -> None:
    selection = select_research_provider(
        ResearchConfig(provider="untrusted-provider", max_repairs_per_experiment=1)
    )

    assert isinstance(selection, ProviderUnavailableDiagnostic)
    assert selection.provider == "invalid"
    assert selection.code is ProviderUnavailableCode.UNSUPPORTED_PROVIDER

from __future__ import annotations

import json
import os
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.message import Message
from io import BytesIO
from typing import Any

import pytest

from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.prompts import PROMPT_VERSION
from kuairand_agent.research.provider import (
    ModelContextLimits,
    OpenAIChatCompletionsConfig,
    OpenAIChatCompletionsModel,
    OpenAIFailoverModel,
    OpenAIProviderChainError,
    OpenAIProviderError,
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    OpenRouterModelLimitResolver,
    ProviderErrorCode,
    ProviderModelLimitResolver,
    ResponsesTransportError,
    ResponseTooLargeError,
    RetryPolicy,
    RetryRuntime,
    TokenPricing,
    TransportRequest,
    TransportResponse,
    UrllibResponsesTransport,
)
from kuairand_agent.research.schemas import (
    ExperimentResultSummary,
    GeneratedPackage,
    ImplementationRequest,
    ParentSnapshot,
    ParentSourceFile,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RepairRequest,
)
from kuairand_agent.research.source_policy import DEFAULT_CANDIDATE_SOURCE_POLICY


def _proposal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "proposal_id": "causal-tree-v1",
        "hypothesis": "A strict-past item rate improves logged-impression ordering.",
        "mechanism": "Add one causal item-rate feature to the grouped ranker.",
        "expected_metric_effects": ["GAUC", "nDCG@5"],
        "parent_candidate_id": "fm-seed",
        "principal_change": "One strict-past causal feature.",
        "files_expected": ["candidate.py"],
        "required_fields": [
            {
                "source_field": "data/log_standard_4_08_to_4_21_pure.csv:long_view",
                "role": "strict_past_history_source",
                "purpose": "Build an expanding-prefix item rate.",
            }
        ],
        "objective": "binary long_view ranking",
        "sampling": "logged impressions",
        "grouping": "user impression groups",
        "weighting": "uniform",
        "causal_cutoff": "Only outcomes with smaller validated time_ms are used.",
        "estimated_runtime_seconds": 60,
        "estimated_memory_mb": 512,
        "smoke_plan": "Run a synthetic causal-prefix fixture.",
        "inner_fold_plan": "Screen both train-derived temporal folds.",
        "falsification_criteria": "Close if either fold regresses materially.",
        "promotion_criteria": "Require positive mean primary delta.",
        "maximum_repairs": 1,
        "rollback_parent_id": "fm-seed",
        "attributions": ["WP6 causal feature method card"],
    }


def _response(output: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl_fixture_1",
            "object": "chat.completion",
            "model": "gpt-5.4",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(output, separators=(",", ":")),
                        "refusal": None,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 130,
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass
class FakeTransport:
    responses: list[TransportResponse]
    requests: list[TransportRequest] = field(default_factory=list)

    def send(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _config(env_name: str = "KUAIRAND_TEST_OPENAI_KEY") -> OpenAIChatCompletionsConfig:
    return OpenAIChatCompletionsConfig(
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        reasoning_effort="high",
        pricing=TokenPricing(
            input_usd_per_million="2",
            cached_input_usd_per_million="0.5",
            output_usd_per_million="8",
        ),
        api_key_env=env_name,
        timeout_seconds=12.5,
        max_response_bytes=2_000_000,
        max_output_tokens=4096,
        max_malformed_retries=1,
    )


def _proposal_request() -> ProposalRequest:
    return ProposalRequest.create(
        request_id="iteration-01-propose",
        campaign_id="campaign-1",
        scientific_iteration=1,
        parent_candidate_id="fm-seed",
        safe_context={"schema_version": 1, "remaining_attempts": 12},
    )


def _instructions(payload: dict[str, object]) -> str:
    messages = payload["messages"]
    assert isinstance(messages, list)
    system = messages[0]
    assert isinstance(system, dict)
    content = system["content"]
    assert isinstance(content, str)
    return content


def _json_schema_format(payload: dict[str, object]) -> dict[str, Any]:
    response_format = payload["response_format"]
    assert isinstance(response_format, dict)
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    return json_schema


def test_propose_uses_chat_completions_schema_and_records_redacted_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "sk-proj-fixture-secret-never-persist"
    env_name = "KUAIRAND_TEST_OPENAI_KEY"
    monkeypatch.delenv(env_name, raising=False)
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    config = _config(env_name)
    model = OpenAIChatCompletionsModel(config, transport=transport)
    monkeypatch.setenv(env_name, key)
    request = _proposal_request()

    proposal = model.propose(request)

    assert proposal.parent_candidate_id == "fm-seed"
    sent = transport.requests[0]
    assert sent.url == "https://api.openai.com/v1/chat/completions"
    assert sent.timeout_seconds == 12.5
    assert sent.max_response_bytes == 2_000_000
    assert dict(sent.headers)["Authorization"] == f"Bearer {key}"
    assert key not in repr(sent)
    payload = json.loads(sent.body)
    assert payload["model"] == "gpt-5.4"
    assert payload["reasoning_effort"] == "high"
    assert payload["store"] is False
    assert payload["tools"] == []
    assert payload["tool_choice"] == "none"
    assert payload["max_completion_tokens"] == 4096
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": f"kuairand_propose_v{PROMPT_VERSION}",
            "strict": True,
            "schema": _json_schema_format(payload)["schema"],
        },
    }
    schema = _json_schema_format(payload)["schema"]
    assert schema["title"] == "Proposal"
    assert schema["additionalProperties"] is False
    usage = model.total_usage
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 20
    assert usage.output_tokens == 30
    assert usage.reasoning_tokens == 5
    assert usage.estimated_cost_usd == "0.00041"
    assert usage.unaccounted_attempts == 0
    assert len(model.transcripts) == 1
    assert key.encode() not in model.transcripts[0].json_bytes


@pytest.mark.parametrize(
    "base_url",
    (
        "https://openrouter.ai/api/v1",
        "https://api.tokenrouter.com/v1",
    ),
)
def test_gateway_payload_uses_gateway_reasoning_contract(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "gateway-fixture-secret")
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    model = OpenAIChatCompletionsModel(
        replace(_config(), base_url=base_url, reasoning_effort="low"),
        transport=transport,
    )

    model.propose(_proposal_request())

    payload = json.loads(transport.requests[0].body)
    assert "reasoning_effort" not in payload
    if base_url.startswith("https://api.tokenrouter.com"):
        # This host ignores both reasoning_effort and the gateway reasoning object. Measured:
        # the OpenAI-style controls left 92% of completion tokens as reasoning and pushed
        # implement calls past the configured timeout, so the effort is sent as an explicit
        # budget instead, which the host honours exactly.
        assert "reasoning" not in payload
        assert payload["thinking"] == {"type": "enabled"}
    else:
        assert payload["reasoning"] == {"effort": "low", "exclude": True}
        assert "thinking" not in payload
    if base_url.startswith("https://openrouter.ai/"):
        assert payload["max_tokens"] == 4096
        assert "max_completion_tokens" not in payload
        assert "n" not in payload
        assert "store" not in payload
        assert "stream" not in payload
        assert "parallel_tool_calls" not in payload
        assert payload["tools"] == []
        assert payload["tool_choice"] == "none"
        assert payload["provider"] == {
            "allow_fallbacks": True,
            "require_parameters": True,
        }
    else:
        assert payload["max_completion_tokens"] == 4096
        assert "max_tokens" not in payload
        assert payload["n"] == 1
        assert payload["store"] is False
        assert payload["stream"] is False
        assert payload["parallel_tool_calls"] is False
        assert "provider" not in payload


def test_model_metadata_clamps_completion_budget_to_advertised_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "limits-fixture-secret")
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    resolutions: list[str] = []

    def resolve(config: OpenAIChatCompletionsConfig, _secret: str) -> ModelContextLimits:
        resolutions.append(config.model)
        return ModelContextLimits(
            context_length=128_000,
            max_completion_tokens=2_048,
            source="fixture",
        )

    model = OpenAIChatCompletionsModel(
        _config(),
        transport=transport,
        model_limit_resolver=resolve,
    )

    model.propose(_proposal_request())

    payload = json.loads(transport.requests[0].body)
    assert payload["max_completion_tokens"] == 2_048
    assert resolutions == ["gpt-5.4"]
    assert model.context_limits == ModelContextLimits(
        context_length=128_000,
        max_completion_tokens=2_048,
        source="fixture",
    )


def test_provider_transcript_is_sent_to_durable_sink_before_call_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sink-fixture-secret")
    persisted: list[bytes] = []
    model = OpenAIChatCompletionsModel(
        _config(),
        transport=FakeTransport(
            [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
        ),
        transcript_sink=lambda transcript: persisted.append(transcript.json_bytes),
    )

    proposal = model.propose(_proposal_request())

    assert proposal.proposal_id == "causal-tree-v1"
    assert persisted == [model.transcripts[0].json_bytes]
    assert json.loads(persisted[0])["outcome"] == "accepted"


def test_chat_completions_portable_json_mode_preserves_local_schema_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-portable-chat-secret")
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    config = replace(
        _config(),
        model="third-party-chat-model",
        base_url="https://provider.example/v1",
        response_format="json_object",
        max_tokens_parameter="max_tokens",
        send_reasoning_effort=False,
    )
    model = OpenAIChatCompletionsModel(config, transport=transport)

    proposal = model.propose(_proposal_request())

    assert proposal.proposal_id == "causal-tree-v1"
    sent = transport.requests[0]
    assert sent.url == "https://provider.example/v1/chat/completions"
    payload = json.loads(sent.body)
    assert payload["model"] == "third-party-chat-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert '"title":"Proposal"' in _instructions(payload)


def test_legacy_responses_names_alias_the_chat_completions_adapter() -> None:
    assert OpenAIResponsesConfig is OpenAIChatCompletionsConfig
    assert OpenAIResponsesModel is OpenAIChatCompletionsModel


def test_malformed_typed_output_retries_once_with_schema_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-retry-fixture-secret")
    malformed = _proposal_payload()
    malformed["parent_candidate_id"] = "wrong-parent"
    transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(malformed)),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(_config(), transport=transport)

    proposal = model.propose(_proposal_request())

    assert proposal.proposal_id == "causal-tree-v1"
    assert len(transport.requests) == 2
    first = json.loads(transport.requests[0].body)
    second = json.loads(transport.requests[1].body)
    assert "previous response" not in _instructions(first).lower()
    assert "previous response was rejected" in _instructions(second).lower()
    assert [item.outcome for item in model.transcripts] == ["malformed", "accepted"]
    assert model.total_usage.input_tokens == 200
    assert model.total_usage.output_tokens == 60
    assert model.total_usage.estimated_cost_usd == "0.00082"


def test_proposal_manifest_policy_violation_is_corrected_before_returning_to_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-manifest-retry-secret")
    invalid = _proposal_payload()
    invalid["proposal_id"] = "invalid-hidden-policy-v1"
    invalid["files_expected"] = ["candidate.py", "baseline.py", "submission.csv"]
    transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(invalid)),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(_config(), transport=transport)

    proposal = model.propose(_proposal_request())

    assert proposal.proposal_id == "causal-tree-v1"
    assert [item.outcome for item in model.transcripts] == ["malformed", "accepted"]
    first, corrected = (json.loads(item.body) for item in transport.requests)
    for payload in (first, corrected):
        instructions = _instructions(payload)
        assert (
            f"Candidate source policy digest: {DEFAULT_CANDIDATE_SOURCE_POLICY.digest}."
            in instructions
        )
        assert "must contain candidate.py as its entrypoint" in instructions
        assert "submission.csv" in instructions
        assert "baseline.py" in instructions
        assert "complete-file overlay semantics" in instructions
    assert "previous response was rejected" not in _instructions(first).lower()
    assert "previous response was rejected" in _instructions(corrected).lower()


def test_malformed_retry_exhaustion_is_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-malformed-fixture-secret")
    malformed = _proposal_payload()
    malformed["unknown"] = True
    transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(malformed)),
            TransportResponse(status_code=200, body=_response(malformed)),
        ]
    )
    model = OpenAIResponsesModel(_config(), transport=transport)

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert raised.value.attempts == 2
    assert len(transport.requests) == 2
    assert [item.outcome for item in model.transcripts] == ["malformed", "malformed"]


def _parent() -> ParentSnapshot:
    return ParentSnapshot(
        candidate_id="fm-seed",
        files=(ParentSourceFile.create("candidate.py", "def score(x):\n    return x\n"),),
    )


def _generated_payload(request_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "response_id": f"{request_id}-response",
        "files": [{"path": "candidate.py", "content": "def score(x):\n    return x + 1\n"}],
        "material_change_summary": "Add one score component.",
        "material_symbols": ["score"],
    }


def _reflection_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "response_id": "reflection-response",
        "summary": "The bounded fixture passed.",
        "recommendation": "propose_next",
        "lessons": ["Retain exact typed evidence."],
    }


def test_all_four_operations_implement_the_existing_research_model_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-operations-fixture-secret")
    proposal = Proposal.from_mapping(_proposal_payload())
    implementation = ImplementationRequest.create(
        request_id="implement-1",
        proposal=proposal,
        parent=_parent(),
        safe_context={"schema_version": 1},
    )
    repair = RepairRequest.create(
        request_id="repair-1",
        proposal_id=proposal.proposal_id,
        failed_candidate_id="failed-child",
        failed_child=_parent(),
        failure_category="syntax_error",
        diagnostics="candidate.py:1 invalid syntax",
        remaining_repairs=1,
        safe_context={"schema_version": 1},
    )
    reflection = ReflectionRequest.create(
        request_id="reflect-1",
        proposal_id=proposal.proposal_id,
        candidate_id="candidate-1",
        source_digest="a" * 64,
        diff_digest="b" * 64,
        result=ExperimentResultSummary(
            tier="inner",
            status="passed",
            gauc=0.6,
            ndcg_at_5=0.5,
            primary=0.55,
            runtime_seconds=1.0,
            peak_memory_mb=64.0,
        ),
        safe_context={"schema_version": 1},
    )
    transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(_generated_payload("implement-1"))),
            TransportResponse(status_code=200, body=_response(_generated_payload("repair-1"))),
            TransportResponse(status_code=200, body=_response(_reflection_payload())),
        ]
    )
    model = OpenAIResponsesModel(_config(), transport=transport)

    assert isinstance(model, ResearchModel)
    assert isinstance(model.implement(implementation), GeneratedPackage)
    assert isinstance(model.repair(repair), GeneratedPackage)
    assert isinstance(model.reflect(reflection), Reflection)
    payloads = [json.loads(item.body) for item in transport.requests]
    assert [_json_schema_format(item)["name"] for item in payloads] == [
        f"kuairand_implement_v{PROMPT_VERSION}",
        f"kuairand_repair_v{PROMPT_VERSION}",
        f"kuairand_reflect_v{PROMPT_VERSION}",
    ]
    assert [_json_schema_format(item)["schema"]["title"] for item in payloads] == [
        "GeneratedPackage",
        "GeneratedPackage",
        "Reflection",
    ]


def test_invalid_qualified_material_symbols_use_the_malformed_response_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-symbol-retry-fixture-secret")
    proposal = Proposal.from_mapping(_proposal_payload())
    implementation = ImplementationRequest.create(
        request_id="implement-symbol-retry",
        proposal=proposal,
        parent=_parent(),
        safe_context={"schema_version": 1},
    )
    malformed = _generated_payload(implementation.request_id)
    malformed["material_symbols"] = ["candidate.py.score"]
    transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(malformed)),
            TransportResponse(
                status_code=200,
                body=_response(_generated_payload(implementation.request_id)),
            ),
        ]
    )
    model = OpenAIResponsesModel(_config(), transport=transport)

    generated = model.implement(implementation)

    assert generated.material_symbols == ("score",)
    assert [item.outcome for item in model.transcripts] == ["malformed", "accepted"]
    first, second = (json.loads(item.body) for item in transport.requests)
    assert "bare" in _instructions(first)
    assert "top-level ASCII Python identifiers" in _instructions(first)
    assert "previous response was rejected" in _instructions(second).lower()


def test_provider_parse_preserves_policy_invalid_implementation_for_repair_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-rejected-package-secret")
    proposal = Proposal.from_mapping(_proposal_payload())
    implementation = ImplementationRequest.create(
        request_id="implement-preserve-rejected",
        proposal=proposal,
        parent=_parent(),
        safe_context={"schema_version": 1},
    )
    generated = _generated_payload(implementation.request_id)
    generated["files"] = [
        {"path": "baseline.py", "content": "def train_model():\n    return 1\n"},
        {"path": "candidate.py", "content": "from baseline import train_model\n"},
    ]
    generated["material_symbols"] = ["train_model"]
    model = OpenAIResponsesModel(
        _config(),
        transport=FakeTransport([TransportResponse(status_code=200, body=_response(generated))]),
    )

    package = model.implement(implementation)

    assert tuple(file.path for file in package.files) == ("baseline.py", "candidate.py")
    assert [item.outcome for item in model.transcripts] == ["accepted"]


def test_missing_call_time_credential_has_no_transport_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KUAIRAND_TEST_OPENAI_KEY", raising=False)
    transport = FakeTransport([])
    model = OpenAIResponsesModel(_config(), transport=transport)

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is ProviderErrorCode.CREDENTIAL_MISSING
    assert raised.value.attempts == 0
    assert transport.requests == []
    assert model.transcripts == ()


def test_provider_chain_keeps_fallback_idle_when_main_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_MAIN_KEY", "sk-proj-main-fixture-secret")
    monkeypatch.setenv("KUAIRAND_TEST_FALLBACK_KEY", "sk-proj-fallback-fixture-secret")
    main_transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    fallback_transport = FakeTransport([])
    model = OpenAIFailoverModel(
        OpenAIResponsesModel(_config("KUAIRAND_TEST_MAIN_KEY"), transport=main_transport),
        OpenAIResponsesModel(_config("KUAIRAND_TEST_FALLBACK_KEY"), transport=fallback_transport),
    )

    proposal = model.propose(_proposal_request())

    assert proposal.proposal_id == "causal-tree-v1"
    assert model.active_slot == "main"
    assert model.failover_events == ()
    assert len(main_transport.requests) == 1
    assert fallback_transport.requests == []
    assert len(model.transcripts) == 1


def test_provider_chain_fails_over_once_and_stays_on_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_MAIN_KEY", "sk-proj-main-fixture-secret")
    monkeypatch.setenv("KUAIRAND_TEST_FALLBACK_KEY", "sk-proj-fallback-fixture-secret")
    main_transport = FakeTransport([TransportResponse(status_code=429, body=b"rate limited")])
    fallback_transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIFailoverModel(
        OpenAIResponsesModel(_config("KUAIRAND_TEST_MAIN_KEY"), transport=main_transport),
        OpenAIResponsesModel(_config("KUAIRAND_TEST_FALLBACK_KEY"), transport=fallback_transport),
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"

    assert model.active_slot == "fallback"
    assert len(model.failover_events) == 1
    assert model.failover_events[0].failure.code is ProviderErrorCode.HTTP
    assert len(main_transport.requests) == 1
    assert len(fallback_transport.requests) == 2
    assert [item.outcome for item in model.transcripts] == ["failed", "accepted", "accepted"]
    assert model.total_usage.total_tokens == 260


def test_provider_chain_exhaustion_reports_both_slots_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_secret = "sk-proj-main-chain-secret"
    fallback_secret = "sk-proj-fallback-chain-secret"
    monkeypatch.setenv("KUAIRAND_TEST_MAIN_KEY", main_secret)
    monkeypatch.setenv("KUAIRAND_TEST_FALLBACK_KEY", fallback_secret)
    model = OpenAIFailoverModel(
        OpenAIResponsesModel(
            _config("KUAIRAND_TEST_MAIN_KEY"),
            transport=FakeTransport([TransportResponse(status_code=429, body=b"main")]),
        ),
        OpenAIResponsesModel(
            _config("KUAIRAND_TEST_FALLBACK_KEY"),
            transport=FakeTransport([TransportResponse(status_code=503, body=b"fallback")]),
        ),
    )

    with pytest.raises(OpenAIProviderChainError) as raised:
        model.propose(_proposal_request())

    assert [item.slot for item in raised.value.failures] == ["main", "fallback"]
    assert [item.status_code for item in raised.value.failures] == [429, 503]
    assert raised.value.attempts == 2
    assert main_secret not in repr(raised.value)
    assert fallback_secret not in repr(raised.value)
    assert all(
        main_secret.encode() not in item.json_bytes
        and fallback_secret.encode() not in item.json_bytes
        for item in model.transcripts
    )


@dataclass
class FailingTransport:
    secret: str

    def send(self, request: TransportRequest) -> TransportResponse:
        raise ResponsesTransportError(f"network failed while using {self.secret}")


@dataclass
class FlakyTransport:
    failures_remaining: int
    response: TransportResponse
    requests: list[TransportRequest] = field(default_factory=list)

    def send(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ResponsesTransportError("temporary network failure")
        return self.response


def test_transport_failure_is_typed_and_does_not_persist_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-transport-fixture-secret"
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", secret)
    model = OpenAIResponsesModel(_config(), transport=FailingTransport(secret))

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is ProviderErrorCode.TRANSPORT
    assert secret not in str(raised.value)
    assert len(model.transcripts) == 1
    assert secret.encode() not in model.transcripts[0].json_bytes


def test_transport_retries_are_bounded_and_preserve_attempt_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-retry-fixture-secret")
    transport = FlakyTransport(
        failures_remaining=2,
        response=TransportResponse(status_code=200, body=_response(_proposal_payload())),
    )
    model = OpenAIResponsesModel(
        replace(_config(), max_transport_retries=2),
        transport=transport,
    )

    proposal = model.propose(_proposal_request())

    assert proposal.proposal_id == "causal-tree-v1"
    assert len(transport.requests) == 3
    assert [item.attempt for item in model.transcripts] == [1, 2, 3]
    assert [item.outcome for item in model.transcripts] == ["failed", "failed", "accepted"]


def test_rate_limit_response_uses_the_same_bounded_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-rate-limit-fixture-secret")
    transport = FakeTransport(
        [
            TransportResponse(status_code=429, body=b'{"error":{"type":"rate_limit"}}'),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(
        replace(_config(), max_transport_retries=1),
        transport=transport,
        retry_runtime=RetryRuntime(sleep=lambda _seconds: None),
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert len(transport.requests) == 2
    assert [item.attempt for item in model.transcripts] == [1, 2]


def test_first_rate_limit_honors_bounded_retry_after_and_accounts_wait_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-rate-limit-wait-secret")
    sleeps: list[float] = []
    runtime = RetryRuntime(
        sleep=sleeps.append,
        monotonic=lambda: 100.0,
        utc_now=lambda: datetime(2026, 8, 29, tzinfo=UTC),
        jitter=lambda _lower, upper: upper,
        remaining_research_seconds=lambda: 30.0,
    )
    transport = FakeTransport(
        [
            TransportResponse(
                status_code=429,
                body=b'{"error":{"type":"rate_limit"}}',
                retry_after="7",
            ),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(
        replace(_config(), max_transport_retries=3),
        transport=transport,
        retry_policy=RetryPolicy(max_retry_after_seconds=10.0),
        retry_runtime=runtime,
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert sleeps == [7.0]
    assert [item.retry_wait_seconds for item in model.transcripts] == [7.0, 0.0]
    assert [item.latency_seconds for item in model.transcripts] == [0.0, 0.0]
    assert model.total_usage.retry_wait_seconds == 7.0
    assert model.transcripts[0].request_digest == model.transcripts[1].request_digest
    evidence = json.loads(model.transcripts[0].json_bytes)
    assert evidence["retry_wait_seconds"] == 7.0


def test_rate_limit_http_date_is_bounded_and_missing_header_uses_jittered_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-rate-date-secret")
    sleeps: list[float] = []
    runtime = RetryRuntime(
        sleep=sleeps.append,
        monotonic=lambda: 200.0,
        utc_now=lambda: datetime(2026, 8, 29, 0, 0, tzinfo=UTC),
        jitter=lambda _lower, upper: upper,
    )
    transport = FakeTransport(
        [
            TransportResponse(
                status_code=429,
                body=b"date limited",
                retry_after="Sat, 29 Aug 2026 00:00:20 GMT",
            ),
            TransportResponse(status_code=408, body=b"request timeout"),
            TransportResponse(status_code=429, body=b"no retry header"),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(
        replace(_config(), max_transport_retries=3),
        transport=transport,
        retry_policy=RetryPolicy(
            initial_backoff_seconds=2.0,
            max_backoff_seconds=10.0,
            max_retry_after_seconds=6.0,
            jitter_ratio=0.25,
        ),
        retry_runtime=runtime,
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert sleeps == [6.0, 10.0]
    assert [item.retry_wait_seconds for item in model.transcripts] == [6.0, 0.0, 10.0, 0.0]
    assert model.total_usage.retry_wait_seconds == 16.0


def test_second_consecutive_rate_limit_immediately_switches_to_sticky_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_MAIN_KEY", "sk-proj-main-rate-limit-secret")
    monkeypatch.setenv("KUAIRAND_TEST_FALLBACK_KEY", "sk-proj-fallback-rate-secret")
    sleeps: list[float] = []
    runtime = RetryRuntime(sleep=sleeps.append, jitter=lambda _lower, _upper: 0.0)
    main_transport = FakeTransport(
        [
            TransportResponse(status_code=429, body=b"first", retry_after="1"),
            TransportResponse(status_code=429, body=b"second", retry_after="30"),
        ]
    )
    fallback_transport = FakeTransport(
        [
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIFailoverModel(
        OpenAIResponsesModel(
            replace(_config("KUAIRAND_TEST_MAIN_KEY"), max_transport_retries=3),
            transport=main_transport,
            retry_runtime=runtime,
        ),
        OpenAIResponsesModel(
            _config("KUAIRAND_TEST_FALLBACK_KEY"),
            transport=fallback_transport,
            retry_runtime=runtime,
        ),
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert sleeps == [1.0]
    assert len(main_transport.requests) == 2
    assert len(fallback_transport.requests) == 2
    assert model.active_slot == "fallback"
    assert len(model.failover_events) == 1
    assert model.failover_events[0].failure.attempts == 2
    assert [item.retry_wait_seconds for item in model.transcripts] == [1.0, 0.0, 0.0, 0.0]


def test_rate_limit_wait_never_consumes_remaining_research_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-deadline-rate-secret")
    sleeps: list[float] = []
    transport = FakeTransport(
        [
            TransportResponse(status_code=429, body=b"limited", retry_after="10"),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(
        replace(_config(), max_transport_retries=3),
        transport=transport,
        retry_runtime=RetryRuntime(
            sleep=sleeps.append,
            remaining_research_seconds=lambda: 5.0,
        ),
    )

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.status_code == 429
    assert raised.value.attempts == 1
    assert sleeps == []
    assert len(transport.requests) == 1
    assert model.transcripts[0].retry_wait_seconds == 0.0
    assert model.total_usage.retry_wait_seconds == 0.0


def test_transport_timeout_is_capped_by_remaining_research_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-deadline-timeout-secret")
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    model = OpenAIResponsesModel(
        _config(),
        transport=transport,
        retry_runtime=RetryRuntime(remaining_research_seconds=lambda: 3.25),
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert transport.requests[0].timeout_seconds == 3.25


def test_expired_research_deadline_prevents_transport_and_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_MAIN_KEY", "sk-proj-main-deadline-secret")
    monkeypatch.setenv("KUAIRAND_TEST_FALLBACK_KEY", "sk-proj-fallback-deadline-secret")
    main_transport = FakeTransport([])
    fallback_transport = FakeTransport([])
    runtime = RetryRuntime(remaining_research_seconds=lambda: 0.0)
    model = OpenAIFailoverModel(
        OpenAIResponsesModel(
            _config("KUAIRAND_TEST_MAIN_KEY"),
            transport=main_transport,
            retry_runtime=runtime,
        ),
        OpenAIResponsesModel(
            _config("KUAIRAND_TEST_FALLBACK_KEY"),
            transport=fallback_transport,
            retry_runtime=runtime,
        ),
    )

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is ProviderErrorCode.DEADLINE
    assert raised.value.attempts == 0
    assert model.active_slot == "main"
    assert model.failover_events == ()
    assert main_transport.requests == []
    assert fallback_transport.requests == []
    assert model.transcripts == ()


def test_retryable_http_failure_cannot_start_an_attempt_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-http-deadline-secret")
    remaining = iter((2.0, 0.0))
    transport = FakeTransport(
        [
            TransportResponse(status_code=503, body=b"temporarily unavailable"),
            TransportResponse(status_code=200, body=_response(_proposal_payload())),
        ]
    )
    model = OpenAIResponsesModel(
        replace(_config(), max_transport_retries=2),
        transport=transport,
        retry_runtime=RetryRuntime(remaining_research_seconds=lambda: next(remaining)),
    )

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is ProviderErrorCode.DEADLINE
    assert raised.value.attempts == 1
    assert len(transport.requests) == 1
    assert transport.requests[0].timeout_seconds == 2.0
    assert [item.attempt for item in model.transcripts] == [1]


def test_http_and_response_size_failures_are_typed_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-http-fixture-secret"
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", secret)
    http_model = OpenAIResponsesModel(
        _config(),
        transport=FakeTransport(
            [
                TransportResponse(
                    status_code=429,
                    body=json.dumps({"error": {"message": f"do not retain {secret}"}}).encode(),
                )
            ]
        ),
    )

    with pytest.raises(OpenAIProviderError) as http_raised:
        http_model.propose(_proposal_request())

    assert http_raised.value.code is ProviderErrorCode.HTTP
    assert http_raised.value.status_code == 429
    assert http_raised.value.attempts == 1
    assert secret.encode() not in http_model.transcripts[0].json_bytes

    small_config = replace(_config(), max_response_bytes=1024)
    oversized_model = OpenAIResponsesModel(
        small_config,
        transport=FakeTransport([TransportResponse(status_code=200, body=b"x" * 1025)]),
    )
    with pytest.raises(OpenAIProviderError) as size_raised:
        oversized_model.propose(_proposal_request())
    assert size_raised.value.code is ProviderErrorCode.RESPONSE_TOO_LARGE
    assert size_raised.value.attempts == 1


@pytest.mark.parametrize(
    ("finish_reason", "refusal", "expected"),
    [
        ("length", None, ProviderErrorCode.INCOMPLETE),
        ("stop", "no", ProviderErrorCode.REFUSAL),
    ],
)
def test_incomplete_and_refusal_chat_completions_fail_once_with_typed_context(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
    refusal: str | None,
    expected: ProviderErrorCode,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-incomplete-fixture-secret")
    envelope = json.loads(_response(_proposal_payload()))
    envelope["choices"][0]["finish_reason"] = finish_reason
    envelope["choices"][0]["message"]["refusal"] = refusal
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=json.dumps(envelope).encode("utf-8"))]
    )
    model = OpenAIResponsesModel(_config(), transport=transport)

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is expected
    assert raised.value.operation is not None
    assert raised.value.attempts == 1
    assert len(transport.requests) == 1
    assert model.total_usage.total_tokens == 130


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(_config(), reasoning_effort="ultra"), "reasoning_effort"),
        (lambda: replace(_config(), base_url="http://api.openai.com/v1"), "HTTPS"),
        (
            lambda: replace(_config(), base_url="https://secret@example.com/v1"),
            "credential-free",
        ),
        (lambda: replace(_config(), max_malformed_retries=2), "0 or 1"),
        (lambda: replace(_config(), max_response_bytes=100), "max_response_bytes"),
        (lambda: replace(_config(), max_output_tokens=0), "max_output_tokens"),
        (lambda: replace(_config(), api_key_env="bad-name"), "api_key_env"),
    ],
)
def test_provider_configuration_is_explicit_and_bounded(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_invalid_usage_is_never_silently_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-usage-fixture-secret")
    envelope = json.loads(_response(_proposal_payload()))
    envelope["usage"]["total_tokens"] = 999
    malformed_body = json.dumps(envelope).encode("utf-8")
    model = OpenAIResponsesModel(
        _config(),
        transport=FakeTransport(
            [
                TransportResponse(status_code=200, body=malformed_body),
                TransportResponse(status_code=200, body=malformed_body),
            ]
        ),
    )

    with pytest.raises(OpenAIProviderError) as raised:
        model.propose(_proposal_request())

    assert raised.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert model.total_usage.unaccounted_attempts == 2
    assert model.total_usage.total_tokens == 0


@dataclass
class FakeHTTPResponse:
    body: bytes
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    read_sizes: list[int] = field(default_factory=list)

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.body[:size]


def test_openrouter_model_limit_discovery_uses_current_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(
        json.dumps(
            {
                "data": {
                    "id": "deepseek/deepseek-v4-pro-0813",
                    "context_length": 128_000,
                    "top_provider": {
                        "context_length": 114_688,
                        "max_completion_tokens": 65_536,
                    },
                }
            }
        ).encode("utf-8")
    )
    observed: list[tuple[str, str, float]] = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        assert isinstance(request, urllib.request.Request)
        observed.append((request.full_url, request.get_method(), timeout))
        return response

    monkeypatch.setattr("kuairand_agent.research.provider.urllib.request.urlopen", fake_urlopen)
    config = replace(
        _config(),
        model="deepseek/deepseek-v4-pro-0813",
        base_url="https://openrouter.ai/api/v1",
    )

    limits = OpenRouterModelLimitResolver()(config, "metadata-fixture-secret")

    assert limits == ModelContextLimits(
        context_length=114_688,
        max_completion_tokens=65_536,
        source="openrouter-model-metadata",
    )
    assert observed == [
        (
            "https://openrouter.ai/api/v1/model/deepseek/deepseek-v4-pro-0813",
            "GET",
            10.0,
        )
    ]
    assert response.read_sizes == [256 * 1024 + 1]


def test_compatible_models_endpoint_is_used_only_when_it_publishes_exact_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(
        json.dumps(
            {
                "data": [
                    {"id": "unrelated/model"},
                    {
                        "id": "deepseek/deepseek-v4-pro-0813",
                        "context_length": 96_000,
                        "max_completion_tokens": 24_000,
                    },
                ]
            }
        ).encode("utf-8")
    )
    observed_urls: list[str] = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        assert isinstance(request, urllib.request.Request)
        observed_urls.append(request.full_url)
        return response

    monkeypatch.setattr("kuairand_agent.research.provider.urllib.request.urlopen", fake_urlopen)
    config = replace(
        _config(),
        model="deepseek/deepseek-v4-pro-0813",
        base_url="https://api.tokenrouter.com/v1",
    )

    limits = ProviderModelLimitResolver()(config, "metadata-fixture-secret")

    assert limits == ModelContextLimits(
        context_length=96_000,
        max_completion_tokens=24_000,
        source="openai-compatible-model-metadata",
    )
    assert observed_urls == ["https://api.tokenrouter.com/v1/models"]
    assert response.read_sizes == [2 * 1024 * 1024 + 1]


def test_stdlib_transport_passes_exact_timeout_and_reads_one_byte_past_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(b"{}")
    observed_timeout: list[float] = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        observed_timeout.append(timeout)
        return response

    monkeypatch.setattr("kuairand_agent.research.provider.urllib.request.urlopen", fake_urlopen)
    request = TransportRequest(
        url="https://api.openai.com/v1/chat/completions",
        body=b"{}",
        headers=(("Authorization", "Bearer fixture"),),
        timeout_seconds=7.25,
        max_response_bytes=1024,
    )

    returned = UrllibResponsesTransport().send(request)

    assert returned == TransportResponse(status_code=200, body=b"{}")
    assert observed_timeout == [7.25]
    assert response.read_sizes == [1025]

    oversized = FakeHTTPResponse(b"x" * 1025)

    def oversized_urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        return oversized

    monkeypatch.setattr(
        "kuairand_agent.research.provider.urllib.request.urlopen",
        oversized_urlopen,
    )
    with pytest.raises(ResponseTooLargeError):
        UrllibResponsesTransport().send(request)


def test_stdlib_transport_retains_only_bounded_retry_after_on_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = FakeHTTPResponse(
        b"{}",
        headers={"Retry-After": " 12 ", "Authorization": "secret", "X-Trace": "private"},
    )
    monkeypatch.setattr(
        "kuairand_agent.research.provider.urllib.request.urlopen",
        lambda _request, *, timeout: success,
    )
    request = TransportRequest(
        url="https://api.openai.com/v1/chat/completions",
        body=b"{}",
        headers=(),
        timeout_seconds=1.0,
        max_response_bytes=1024,
    )

    returned = UrllibResponsesTransport().send(request)

    assert returned.retry_after == "12"
    assert not hasattr(returned, "headers")

    error_headers = Message()
    error_headers["Retry-After"] = "Sat, 29 Aug 2026 00:00:20 GMT"
    error_headers["X-Secret"] = "discard-me"
    error = urllib.error.HTTPError(
        request.url,
        429,
        "limited",
        error_headers,
        BytesIO(b"limited"),
    )

    def raise_http_error(_request: object, *, timeout: float) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr(
        "kuairand_agent.research.provider.urllib.request.urlopen",
        raise_http_error,
    )
    returned_error = UrllibResponsesTransport().send(request)
    assert returned_error.retry_after == "Sat, 29 Aug 2026 00:00:20 GMT"
    assert returned_error.body == b"limited"

    oversized = FakeHTTPResponse(b"{}", headers={"Retry-After": "9" * 129})
    monkeypatch.setattr(
        "kuairand_agent.research.provider.urllib.request.urlopen",
        lambda _request, *, timeout: oversized,
    )
    assert UrllibResponsesTransport().send(request).retry_after is None


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is absent",
)
def test_opt_in_real_openai_provider_smoke() -> None:
    if os.environ.get("KUAIRAND_OPENAI_SMOKE") != "1":
        pytest.skip("set KUAIRAND_OPENAI_SMOKE=1 to opt in to the live provider smoke")
    price_names = (
        "KUAIRAND_OPENAI_INPUT_USD_PER_MILLION",
        "KUAIRAND_OPENAI_CACHED_INPUT_USD_PER_MILLION",
        "KUAIRAND_OPENAI_OUTPUT_USD_PER_MILLION",
    )
    if any(not os.environ.get(name) for name in price_names):
        pytest.skip("set all explicit KUAIRAND_OPENAI_*_USD_PER_MILLION prices")
    model = OpenAIResponsesModel(
        OpenAIResponsesConfig(
            model=os.environ.get("KUAIRAND_OPENAI_MODEL", "gpt-5.4"),
            base_url="https://api.openai.com/v1",
            reasoning_effort="low",
            pricing=TokenPricing(*(os.environ[name] for name in price_names)),
            max_output_tokens=8192,
        )
    )

    proposal = model.propose(_proposal_request())

    assert proposal.parent_candidate_id == _proposal_request().parent_candidate_id
    assert model.total_usage.total_tokens > 0
    assert len(model.transcripts) == 1

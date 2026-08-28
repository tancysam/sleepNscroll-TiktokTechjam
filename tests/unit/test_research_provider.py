from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pytest

from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.provider import (
    OpenAIProviderError,
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    ProviderErrorCode,
    ResponsesTransportError,
    ResponseTooLargeError,
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
            "id": "resp_fixture_1",
            "object": "response",
            "status": "completed",
            "model": "gpt-5.4",
            "error": None,
            "incomplete_details": None,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(output, separators=(",", ":")),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 5},
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


def _config(env_name: str = "KUAIRAND_TEST_OPENAI_KEY") -> OpenAIResponsesConfig:
    return OpenAIResponsesConfig(
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


def test_propose_uses_current_responses_schema_and_records_redacted_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "sk-proj-fixture-secret-never-persist"
    env_name = "KUAIRAND_TEST_OPENAI_KEY"
    monkeypatch.delenv(env_name, raising=False)
    transport = FakeTransport(
        [TransportResponse(status_code=200, body=_response(_proposal_payload()))]
    )
    config = _config(env_name)
    model = OpenAIResponsesModel(config, transport=transport)
    monkeypatch.setenv(env_name, key)
    request = _proposal_request()

    proposal = model.propose(request)

    assert proposal.parent_candidate_id == "fm-seed"
    sent = transport.requests[0]
    assert sent.url == "https://api.openai.com/v1/responses"
    assert sent.timeout_seconds == 12.5
    assert sent.max_response_bytes == 2_000_000
    assert dict(sent.headers)["Authorization"] == f"Bearer {key}"
    assert key not in repr(sent)
    payload = json.loads(sent.body)
    assert payload["model"] == "gpt-5.4"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["store"] is False
    assert payload["tools"] == []
    assert payload["tool_choice"] == "none"
    assert payload["max_output_tokens"] == 4096
    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "kuairand_propose_v1",
        "strict": True,
        "schema": payload["text"]["format"]["schema"],
    }
    schema = payload["text"]["format"]["schema"]
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
    assert "previous response" not in first["instructions"].lower()
    assert "previous response was rejected" in second["instructions"].lower()
    assert [item.outcome for item in model.transcripts] == ["malformed", "accepted"]
    assert model.total_usage.input_tokens == 200
    assert model.total_usage.output_tokens == 60
    assert model.total_usage.estimated_cost_usd == "0.00082"


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
    assert [item["text"]["format"]["name"] for item in payloads] == [
        "kuairand_implement_v1",
        "kuairand_repair_v1",
        "kuairand_reflect_v1",
    ]
    assert [item["text"]["format"]["schema"]["title"] for item in payloads] == [
        "GeneratedPackage",
        "GeneratedPackage",
        "Reflection",
    ]


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
    )

    assert model.propose(_proposal_request()).proposal_id == "causal-tree-v1"
    assert len(transport.requests) == 2
    assert [item.attempt for item in model.transcripts] == [1, 2]


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
    ("mutation", "expected"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            ProviderErrorCode.INCOMPLETE,
        ),
        (
            {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]},
            ProviderErrorCode.REFUSAL,
        ),
    ],
)
def test_incomplete_and_refusal_responses_fail_once_with_typed_context(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, object],
    expected: ProviderErrorCode,
) -> None:
    monkeypatch.setenv("KUAIRAND_TEST_OPENAI_KEY", "sk-proj-incomplete-fixture-secret")
    envelope = json.loads(_response(_proposal_payload()))
    envelope.update(mutation)
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
    read_sizes: list[int] = field(default_factory=list)

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.body[:size]


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
        url="https://api.openai.com/v1/responses",
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

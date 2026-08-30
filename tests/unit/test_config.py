from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest

from kuairand_agent.config import (
    ConfigError,
    OpenAIFailoverResearchConfig,
    OpenAIResearchConfig,
    load_config,
    parse_config,
)

ROOT = Path(__file__).parents[2]


def decoded_default() -> dict[str, Any]:
    with (ROOT / "configs/default.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.parametrize(
    ("name", "schema_version"),
    [
        ("default.toml", 2),
        ("smoke.toml", 2),
        ("full-pure.toml", 4),
        ("scripted-demo.toml", 2),
    ],
)
def test_repository_configs_validate(name: str, schema_version: int) -> None:
    config = load_config(ROOT / "configs" / name)
    assert config.schema_version == schema_version
    assert config.benchmark.name == "kuairand-pure"
    assert config.benchmark.target == "long_view"
    assert len(config.digest) == 64


def test_full_config_is_live_autonomous_and_demo_configs_are_explicit() -> None:
    autonomous = load_config(ROOT / "configs/full-pure.toml")
    assert autonomous.research.run_kind == "autonomous"
    assert autonomous.research.provider == "openai"
    assert isinstance(autonomous.research.openai, OpenAIFailoverResearchConfig)
    assert autonomous.research.openai.main.api_key_env == "INFERENCE_MAIN_API_KEY"
    assert autonomous.research.openai.main.base_url_env == "INFERENCE_MAIN_BASE_URL"
    assert autonomous.research.openai.main.model_env == "INFERENCE_MAIN_MODEL"
    assert autonomous.research.openai.main.response_format == "json_schema"
    assert autonomous.research.openai.main.max_tokens_parameter == "max_completion_tokens"
    assert autonomous.research.openai.main.send_reasoning_effort is True
    assert autonomous.research.openai.fallback.api_key_env == "INFERENCE_FALLBACK_API_KEY"
    assert autonomous.research.openai.fallback.base_url_env == "INFERENCE_FALLBACK_BASE_URL"
    assert autonomous.research.openai.fallback.model_env == "INFERENCE_FALLBACK_MODEL"

    demo = load_config(ROOT / "configs/scripted-demo.toml")
    assert demo.research.run_kind == "demo"
    assert demo.research.provider == "scripted"
    assert demo.research.allow_scripted_demo is True
    assert demo.research.openai is None


def test_schema_v1_normalization_remains_backward_compatible() -> None:
    raw = decoded_default()
    raw["schema_version"] = 1
    raw["research"].pop("run_kind")
    raw["research"].pop("allow_scripted_demo")
    config = parse_config(raw)
    assert config.schema_version == 1
    assert config.research.run_kind == "demo"
    assert config.normalized()["research"] == {
        "provider": "scripted",
        "max_repairs_per_experiment": 2,
    }


def test_normalization_and_digest_are_deterministic() -> None:
    first = parse_config(decoded_default())
    second = parse_config(copy.deepcopy(decoded_default()))
    assert first.normalized() == second.normalized()
    assert first.digest == second.digest


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda raw: raw.update({"surprise": 1}), "unknown top-level"),
        (lambda raw: raw["runner"].update({"surprise": 1}), "unknown runner"),
        (lambda raw: raw["benchmark"].update({"max_iterations": 51}), "max_iterations"),
        (lambda raw: raw["benchmark"].update({"wall_clock_seconds": 21_601}), "wall_clock"),
        (lambda raw: raw["validation"].update({"outer_promotion_limit": 7}), "outer"),
        (lambda raw: raw["runner"].update({"max_processes": 2}), "max_processes"),
        (lambda raw: raw["runner"].update({"device": "cuda"}), "device"),
        (lambda raw: raw["benchmark"].update({"epsilon": 0.001}), "epsilon"),
        (lambda raw: raw["benchmark"].update({"epsilon": 0.003}), "epsilon"),
        (lambda raw: raw["benchmark"].update({"convergence_patience": 2}), "patience"),
        (lambda raw: raw["benchmark"].update({"convergence_patience": 4}), "patience"),
        (lambda raw: raw["validation"].update({"confirmation_seeds": [0]}), "seeds"),
        (
            lambda raw: raw["validation"].update({"confirmation_seeds": [0, 2, 1]}),
            "seeds",
        ),
        (lambda raw: raw["research"].update({"max_repairs_per_experiment": 3}), "repairs"),
        (lambda raw: raw["research"].update({"run_kind": "autonomous"}), "provider"),
        (lambda raw: raw["benchmark"].update({"max_iterations": True}), "max_iterations"),
        (
            lambda raw: raw["runner"].update({"finalization_reserve_seconds": 21_600}),
            "finalization reserve",
        ),
        (
            lambda raw: raw["runner"].update({"finalization_reserve_seconds": 599}),
            "finalization_reserve_seconds",
        ),
    ],
)
def test_hard_caps_types_and_unknown_fields_fail(mutator: Any, message: str) -> None:
    raw = decoded_default()
    mutator(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_one_hour_sprint_with_ten_minute_finalization_reserve_is_valid() -> None:
    raw = decoded_default()
    raw["benchmark"]["wall_clock_seconds"] = 3_600
    raw["runner"]["finalization_reserve_seconds"] = 600

    config = parse_config(raw)

    assert config.benchmark.wall_clock_seconds == 3_600
    assert config.runner.finalization_reserve_seconds == 600


def test_temporal_folds_must_be_train_derived_and_ordered() -> None:
    raw = decoded_default()
    raw["validation"]["inner_folds"] = ["20220419:20220421", "20220416:20220418"]
    with pytest.raises(ConfigError, match="chronological"):
        parse_config(raw)


def test_timeout_must_leave_finalization_time() -> None:
    raw = decoded_default()
    raw["runner"]["default_timeout_seconds"] = 18_001
    with pytest.raises(ConfigError, match="timeout plus finalization"):
        parse_config(raw)


def test_live_provider_settings_are_strict_and_demo_cannot_claim_openai() -> None:
    raw = tomllib.loads((ROOT / "configs/full-pure.toml").read_text(encoding="utf-8"))
    raw["research"]["run_kind"] = "demo"
    raw["research"]["allow_scripted_demo"] = True
    with pytest.raises(ConfigError, match="demo runs"):
        parse_config(raw)

    live = tomllib.loads((ROOT / "configs/full-pure.toml").read_text(encoding="utf-8"))
    live["research"]["openai"]["surprise"] = True
    with pytest.raises(ConfigError, match=r"unknown research\.openai"):
        parse_config(live)


def test_schema_v2_single_provider_config_remains_backward_compatible() -> None:
    raw = tomllib.loads((ROOT / "configs/full-pure.toml").read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["research"]["openai"] = {
        "model": "legacy-model",
        "base_url": "https://legacy.example/v1",
        "reasoning_effort": "high",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 120.0,
        "max_response_bytes": 8192,
        "max_output_tokens": 4096,
        "max_malformed_retries": 1,
        "max_transport_retries": 2,
        "pricing": {
            "input_usd_per_million": "1",
            "cached_input_usd_per_million": "0.1",
            "output_usd_per_million": "4",
        },
    }

    config = parse_config(raw)

    assert isinstance(config.research.openai, OpenAIResearchConfig)
    assert config.research.openai.model == "legacy-model"
    assert config.research.openai.api_key_env == "OPENAI_API_KEY"


def test_schema_v4_requires_six_distinct_provider_environment_names() -> None:
    raw = tomllib.loads((ROOT / "configs/full-pure.toml").read_text(encoding="utf-8"))
    raw["research"]["openai"]["fallback"]["api_key_env"] = "INFERENCE_MAIN_API_KEY"

    with pytest.raises(ConfigError, match="six distinct environment names"):
        parse_config(raw)


def test_schema_v3_provider_profiles_remain_backward_compatible() -> None:
    raw = tomllib.loads((ROOT / "configs/full-pure.toml").read_text(encoding="utf-8"))
    raw["schema_version"] = 3
    for slot in ("main", "fallback"):
        raw["research"]["openai"][slot].pop("response_format")
        raw["research"]["openai"][slot].pop("max_tokens_parameter")
        raw["research"]["openai"][slot].pop("send_reasoning_effort")

    config = parse_config(raw)

    assert isinstance(config.research.openai, OpenAIFailoverResearchConfig)
    assert config.research.openai.main.response_format == "json_schema"
    assert config.research.openai.main.max_tokens_parameter == "max_completion_tokens"
    assert config.research.openai.main.send_reasoning_effort is True
    normalized_main = config.normalized()["research"]["openai"]["main"]
    assert "response_format" not in normalized_main
    assert "max_tokens_parameter" not in normalized_main
    assert "send_reasoning_effort" not in normalized_main


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("response_format", "xml", "response_format"),
        ("max_tokens_parameter", "output_tokens", "max_tokens_parameter"),
        ("send_reasoning_effort", "yes", "send_reasoning_effort"),
    ],
)
def test_chat_completions_endpoint_dialect_is_strict(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = tomllib.loads((ROOT / "configs/full-pure.toml").read_text(encoding="utf-8"))
    raw["research"]["openai"]["fallback"][field] = value

    with pytest.raises(ConfigError, match=message):
        parse_config(raw)

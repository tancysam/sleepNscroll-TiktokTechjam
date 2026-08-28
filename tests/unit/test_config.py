from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest

from kuairand_agent.config import ConfigError, load_config, parse_config

ROOT = Path(__file__).parents[2]


def decoded_default() -> dict[str, Any]:
    with (ROOT / "configs/default.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.parametrize(
    "name", ["default.toml", "smoke.toml", "full-pure.toml", "scripted-demo.toml"]
)
def test_repository_configs_validate(name: str) -> None:
    config = load_config(ROOT / "configs" / name)
    assert config.schema_version == 2
    assert config.benchmark.name == "kuairand-pure"
    assert config.benchmark.target == "long_view"
    assert len(config.digest) == 64


def test_full_config_is_live_autonomous_and_demo_configs_are_explicit() -> None:
    autonomous = load_config(ROOT / "configs/full-pure.toml")
    assert autonomous.research.run_kind == "autonomous"
    assert autonomous.research.provider == "openai"
    assert autonomous.research.openai is not None
    assert autonomous.research.openai.model == "gpt-5.6-sol"

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
            lambda raw: raw["runner"].update({"finalization_reserve_seconds": 3_599}),
            "finalization_reserve_seconds",
        ),
    ],
)
def test_hard_caps_types_and_unknown_fields_fail(mutator: Any, message: str) -> None:
    raw = decoded_default()
    mutator(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


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

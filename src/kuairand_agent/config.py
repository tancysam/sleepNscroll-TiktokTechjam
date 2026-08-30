"""Strict, versioned campaign configuration.

The normalized configuration is part of a campaign's frozen scientific identity.  This
module intentionally uses explicit dataclasses and validators instead of permissive object
coercion: booleans are not integers, unknown fields fail, and hard benchmark caps can only be
narrowed by a caller.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

SCHEMA_VERSION: Final = 4
SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1, 2, 3, SCHEMA_VERSION})
HARD_MAX_LAUNCHES: Final = 50
HARD_MAX_WALL_SECONDS: Final = 21_600
HARD_MIN_FINALIZATION_RESERVE_SECONDS: Final = 600
HARD_MAX_OUTER_PROMOTIONS: Final = 6
HARD_MAX_REPAIRS: Final = 2
HARD_MAX_PROPOSAL_BREADTH: Final = 4
FROZEN_CONVERGENCE_EPSILON: Final = 0.002
FROZEN_CONVERGENCE_PATIENCE: Final = 3
FROZEN_CONFIRMATION_SEEDS: Final = (0, 1, 2)
TRAIN_START: Final = 20_220_408
TRAIN_END: Final = 20_220_421


class ConfigError(ValueError):
    """Raised when configuration is not exactly valid for the supported schema."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    data_dir: Path
    starter_dir: Path
    target: str
    max_iterations: int
    wall_clock_seconds: int
    epsilon: float
    convergence_patience: int


@dataclass(frozen=True, slots=True)
class DateFold:
    start: int
    end: int

    def wire_value(self) -> str:
        """Return the stable TOML/manifest representation."""

        return f"{self.start:08d}:{self.end:08d}"


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    outer_promotion_limit: int
    confirmation_seeds: tuple[int, ...]
    inner_folds: tuple[DateFold, ...]


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    device: str
    max_processes: int
    threads: int
    memory_mb: int
    disk_mb: int
    default_timeout_seconds: int
    finalization_reserve_seconds: int


@dataclass(frozen=True, slots=True)
class OpenAITokenPricingConfig:
    input_usd_per_million: str
    cached_input_usd_per_million: str
    output_usd_per_million: str


@dataclass(frozen=True, slots=True)
class OpenAIResearchConfig:
    model: str
    base_url: str
    reasoning_effort: str
    api_key_env: str
    timeout_seconds: float
    max_response_bytes: int
    max_output_tokens: int
    max_malformed_retries: int
    max_transport_retries: int
    pricing: OpenAITokenPricingConfig


@dataclass(frozen=True, slots=True)
class OpenAIEndpointEnvConfig:
    """Environment-variable selectors for one OpenAI-compatible provider endpoint."""

    api_key_env: str
    base_url_env: str
    model_env: str
    pricing: OpenAITokenPricingConfig
    response_format: str = "json_schema"
    max_tokens_parameter: str = "max_completion_tokens"
    send_reasoning_effort: bool = True


@dataclass(frozen=True, slots=True)
class OpenAIFailoverResearchConfig:
    """Two dedicated provider profiles with one shared bounded call policy."""

    main: OpenAIEndpointEnvConfig
    fallback: OpenAIEndpointEnvConfig
    reasoning_effort: str
    timeout_seconds: float
    max_response_bytes: int
    max_output_tokens: int
    max_malformed_retries: int
    max_transport_retries: int


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    provider: str
    max_repairs_per_experiment: int
    run_kind: str = "demo"
    allow_scripted_demo: bool = False
    openai: OpenAIResearchConfig | OpenAIFailoverResearchConfig | None = None
    proposal_breadth: int = 1


@dataclass(frozen=True, slots=True)
class AgentConfig:
    schema_version: int
    benchmark: BenchmarkConfig
    validation: ValidationConfig
    runner: RunnerConfig
    research: ResearchConfig

    def normalized(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with a stable field order."""

        return {
            "schema_version": self.schema_version,
            "benchmark": {
                "name": self.benchmark.name,
                "data_dir": str(self.benchmark.data_dir),
                "starter_dir": str(self.benchmark.starter_dir),
                "target": self.benchmark.target,
                "max_iterations": self.benchmark.max_iterations,
                "wall_clock_seconds": self.benchmark.wall_clock_seconds,
                "epsilon": self.benchmark.epsilon,
                "convergence_patience": self.benchmark.convergence_patience,
            },
            "validation": {
                "outer_promotion_limit": self.validation.outer_promotion_limit,
                "confirmation_seeds": list(self.validation.confirmation_seeds),
                "inner_folds": [fold.wire_value() for fold in self.validation.inner_folds],
            },
            "runner": dataclasses.asdict(self.runner),
            "research": self._normalized_research(),
        }

    def _normalized_research(self) -> dict[str, Any]:
        # Schema v1 campaign manifests must retain their byte-for-byte scientific identity so
        # archived runs remain replayable after autonomous-provider support is added.
        if self.schema_version == 1:
            return {
                "provider": self.research.provider,
                "max_repairs_per_experiment": self.research.max_repairs_per_experiment,
            }
        result: dict[str, Any] = {
            "provider": self.research.provider,
            "max_repairs_per_experiment": self.research.max_repairs_per_experiment,
            "run_kind": self.research.run_kind,
            "allow_scripted_demo": self.research.allow_scripted_demo,
            "proposal_breadth": self.research.proposal_breadth,
        }
        if self.research.openai is not None:
            normalized_openai = dataclasses.asdict(self.research.openai)
            if self.schema_version == 3 and isinstance(
                self.research.openai, OpenAIFailoverResearchConfig
            ):
                for slot in ("main", "fallback"):
                    endpoint = normalized_openai[slot]
                    endpoint.pop("response_format")
                    endpoint.pop("max_tokens_parameter")
                    endpoint.pop("send_reasoning_effort")
            result["openai"] = normalized_openai
        return result

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity of the normalized effective configuration."""

        payload = json.dumps(
            self.normalized(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def _table(raw: Mapping[str, Any], name: str, expected: set[str]) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a TOML table")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ConfigError(f"unknown {name} field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing {name} field(s): {', '.join(sorted(missing))}")
    return cast(Mapping[str, Any], value)


def _exact[T](value: object, expected_type: type[T], location: str) -> T:
    if type(value) is not expected_type:  # exact on purpose: bool must not pass as int
        raise ConfigError(f"{location} must be {expected_type.__name__}")
    return value


def _integer(table: Mapping[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = _exact(table[key], int, key)
    if not minimum <= value <= maximum:
        raise ConfigError(f"{key} must be in [{minimum}, {maximum}], got {value}")
    return value


def _number(table: Mapping[str, Any], key: str, *, minimum: float, maximum: float) -> float:
    raw = table[key]
    if type(raw) not in (int, float):
        raise ConfigError(f"{key} must be a number")
    value = float(raw)
    if not minimum <= value <= maximum:
        raise ConfigError(f"{key} must be in [{minimum}, {maximum}], got {value}")
    return value


def _string(table: Mapping[str, Any], key: str) -> str:
    value = _exact(table[key], str, key)
    if not value or "\x00" in value:
        raise ConfigError(f"{key} must be a non-empty string without NUL bytes")
    return value


def _boolean(table: Mapping[str, Any], key: str) -> bool:
    return _exact(table[key], bool, key)


_PRICE_TEXT_RE: Final = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?$")
_ENV_NAME_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def _price_text(table: Mapping[str, Any], key: str) -> str:
    value = _string(table, key)
    if _PRICE_TEXT_RE.fullmatch(value) is None:
        raise ConfigError(f"{key} must be non-negative decimal text")
    return value


def _environment_name(table: Mapping[str, Any], key: str, location: str) -> str:
    value = _string(table, key)
    if _ENV_NAME_RE.fullmatch(value) is None:
        raise ConfigError(f"{location} must be a portable uppercase environment name")
    return value


_FOLD_RE: Final = re.compile(r"^(\d{8}):(\d{8})$")


def _folds(value: object) -> tuple[DateFold, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("inner_folds must be a non-empty list")
    result: list[DateFold] = []
    for index, raw in enumerate(value):
        if type(raw) is not str:
            raise ConfigError(f"inner_folds[{index}] must be YYYYMMDD:YYYYMMDD")
        match = _FOLD_RE.fullmatch(raw)
        if match is None:
            raise ConfigError(f"inner_folds[{index}] must be YYYYMMDD:YYYYMMDD")
        start, end = (int(part) for part in match.groups())
        if not TRAIN_START <= start <= end <= TRAIN_END:
            raise ConfigError(f"inner_folds[{index}] must be inside the official train period")
        if result and start <= result[-1].end:
            raise ConfigError("inner_folds must be chronological and non-overlapping")
        result.append(DateFold(start=start, end=end))
    return tuple(result)


def _seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("confirmation_seeds must be a non-empty list")
    seeds: list[int] = []
    for index, raw in enumerate(value):
        if type(raw) is not int or raw < 0 or raw > 2**32 - 1:
            raise ConfigError(f"confirmation_seeds[{index}] must be an unsigned 32-bit integer")
        seeds.append(raw)
    if len(seeds) != len(set(seeds)):
        raise ConfigError("confirmation_seeds must be unique")
    normalized = tuple(seeds)
    if normalized != FROZEN_CONFIRMATION_SEEDS:
        raise ConfigError("confirmation_seeds must equal the frozen matched-seed policy [0, 1, 2]")
    return normalized


def parse_config(raw: Mapping[str, Any]) -> AgentConfig:
    """Parse and validate a decoded TOML mapping."""

    expected_top = {"schema_version", "benchmark", "validation", "runner", "research"}
    unknown = set(raw) - expected_top
    missing = expected_top - set(raw)
    if unknown:
        raise ConfigError(f"unknown top-level field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing top-level field(s): {', '.join(sorted(missing))}")
    schema_version = _exact(raw["schema_version"], int, "schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ConfigError(
            f"unsupported schema_version {schema_version}; supported versions are {supported}"
        )

    benchmark_raw = _table(
        raw,
        "benchmark",
        {
            "name",
            "data_dir",
            "starter_dir",
            "target",
            "max_iterations",
            "wall_clock_seconds",
            "epsilon",
            "convergence_patience",
        },
    )
    validation_raw = _table(
        raw,
        "validation",
        {"outer_promotion_limit", "confirmation_seeds", "inner_folds"},
    )
    runner_raw = _table(
        raw,
        "runner",
        {
            "device",
            "max_processes",
            "threads",
            "memory_mb",
            "disk_mb",
            "default_timeout_seconds",
            "finalization_reserve_seconds",
        },
    )
    if schema_version == 1:
        research_raw = _table(raw, "research", {"provider", "max_repairs_per_experiment"})
    else:
        research_value = raw.get("research")
        if not isinstance(research_value, Mapping):
            raise ConfigError("research must be a TOML table")
        research_raw = cast(Mapping[str, Any], research_value)
        expected_research = {
            "provider",
            "max_repairs_per_experiment",
            "run_kind",
            "allow_scripted_demo",
            "openai",
            "proposal_breadth",
        }
        unknown_research = set(research_raw) - expected_research
        missing_research = {
            "provider",
            "max_repairs_per_experiment",
            "run_kind",
            "allow_scripted_demo",
        } - set(research_raw)
        if unknown_research:
            raise ConfigError(f"unknown research field(s): {', '.join(sorted(unknown_research))}")
        if missing_research:
            raise ConfigError(f"missing research field(s): {', '.join(sorted(missing_research))}")

    benchmark = BenchmarkConfig(
        name=_string(benchmark_raw, "name"),
        data_dir=Path(_string(benchmark_raw, "data_dir")),
        starter_dir=Path(_string(benchmark_raw, "starter_dir")),
        target=_string(benchmark_raw, "target"),
        max_iterations=_integer(
            benchmark_raw, "max_iterations", minimum=1, maximum=HARD_MAX_LAUNCHES
        ),
        wall_clock_seconds=_integer(
            benchmark_raw,
            "wall_clock_seconds",
            minimum=60,
            maximum=HARD_MAX_WALL_SECONDS,
        ),
        epsilon=_number(benchmark_raw, "epsilon", minimum=0.0, maximum=1.0),
        convergence_patience=_integer(
            benchmark_raw,
            "convergence_patience",
            minimum=1,
            maximum=FROZEN_CONVERGENCE_PATIENCE,
        ),
    )
    if benchmark.name != "kuairand-pure":
        raise ConfigError("benchmark.name must be 'kuairand-pure'")
    if benchmark.target != "long_view":
        raise ConfigError("benchmark.target must be 'long_view'")
    if benchmark.epsilon != FROZEN_CONVERGENCE_EPSILON:
        raise ConfigError(
            f"epsilon must equal the frozen benchmark value {FROZEN_CONVERGENCE_EPSILON}"
        )
    if benchmark.convergence_patience != FROZEN_CONVERGENCE_PATIENCE:
        raise ConfigError(
            "convergence_patience must equal the frozen benchmark value "
            f"{FROZEN_CONVERGENCE_PATIENCE}"
        )

    validation = ValidationConfig(
        outer_promotion_limit=_integer(
            validation_raw,
            "outer_promotion_limit",
            minimum=0,
            maximum=HARD_MAX_OUTER_PROMOTIONS,
        ),
        confirmation_seeds=_seeds(validation_raw["confirmation_seeds"]),
        inner_folds=_folds(validation_raw["inner_folds"]),
    )

    device = _string(runner_raw, "device")
    if device not in {"cpu", "mps"}:
        raise ConfigError("runner.device must be 'cpu' or 'mps'")
    runner = RunnerConfig(
        device=device,
        max_processes=_integer(runner_raw, "max_processes", minimum=1, maximum=1),
        threads=_integer(runner_raw, "threads", minimum=1, maximum=64),
        memory_mb=_integer(runner_raw, "memory_mb", minimum=256, maximum=262_144),
        disk_mb=_integer(runner_raw, "disk_mb", minimum=256, maximum=1_048_576),
        default_timeout_seconds=_integer(
            runner_raw, "default_timeout_seconds", minimum=1, maximum=HARD_MAX_WALL_SECONDS
        ),
        finalization_reserve_seconds=_integer(
            runner_raw,
            "finalization_reserve_seconds",
            minimum=HARD_MIN_FINALIZATION_RESERVE_SECONDS,
            maximum=HARD_MAX_WALL_SECONDS,
        ),
    )
    if runner.finalization_reserve_seconds >= benchmark.wall_clock_seconds:
        raise ConfigError("finalization reserve must be smaller than the wall-clock budget")
    if (
        runner.default_timeout_seconds + runner.finalization_reserve_seconds
        > benchmark.wall_clock_seconds
    ):
        raise ConfigError("default timeout plus finalization reserve exceeds wall-clock budget")

    provider = _string(research_raw, "provider")
    if provider not in {"scripted", "openai"}:
        raise ConfigError("research.provider must be 'scripted' or 'openai'")
    run_kind = "demo" if schema_version == 1 else _string(research_raw, "run_kind")
    if run_kind not in {"autonomous", "demo", "test"}:
        raise ConfigError("research.run_kind must be 'autonomous', 'demo', or 'test'")
    allow_scripted_demo = (
        False if schema_version == 1 else _boolean(research_raw, "allow_scripted_demo")
    )
    openai: OpenAIResearchConfig | OpenAIFailoverResearchConfig | None = None
    openai_raw = research_raw.get("openai")
    if provider == "openai":
        if schema_version == 1:
            raise ConfigError("schema_version 1 cannot configure the OpenAI provider")
        if not isinstance(openai_raw, Mapping):
            raise ConfigError("research.openai must be a TOML table for the OpenAI provider")
        openai_table = cast(Mapping[str, Any], openai_raw)
        common_openai = {
            "reasoning_effort",
            "timeout_seconds",
            "max_response_bytes",
            "max_output_tokens",
            "max_malformed_retries",
            "max_transport_retries",
        }
        expected_openai = (
            common_openai | {"model", "base_url", "api_key_env", "pricing"}
            if schema_version == 2
            else common_openai | {"main", "fallback"}
        )
        unknown_openai = set(openai_table) - expected_openai
        missing_openai = expected_openai - set(openai_table)
        if unknown_openai:
            raise ConfigError(
                f"unknown research.openai field(s): {', '.join(sorted(unknown_openai))}"
            )
        if missing_openai:
            raise ConfigError(
                f"missing research.openai field(s): {', '.join(sorted(missing_openai))}"
            )
        reasoning_effort = _string(openai_table, "reasoning_effort")
        if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ConfigError("research.openai.reasoning_effort is unsupported")
        timeout_seconds = _number(openai_table, "timeout_seconds", minimum=0.1, maximum=600.0)
        max_response_bytes = _integer(
            openai_table,
            "max_response_bytes",
            minimum=1024,
            maximum=16 * 1024 * 1024,
        )
        max_output_tokens = _integer(openai_table, "max_output_tokens", minimum=1, maximum=200_000)
        max_malformed_retries = _integer(
            openai_table, "max_malformed_retries", minimum=0, maximum=1
        )
        max_transport_retries = _integer(
            openai_table, "max_transport_retries", minimum=0, maximum=3
        )

        def pricing_config(table: Mapping[str, Any]) -> OpenAITokenPricingConfig:
            pricing_raw = _table(
                table,
                "pricing",
                {
                    "input_usd_per_million",
                    "cached_input_usd_per_million",
                    "output_usd_per_million",
                },
            )
            return OpenAITokenPricingConfig(
                input_usd_per_million=_price_text(pricing_raw, "input_usd_per_million"),
                cached_input_usd_per_million=_price_text(
                    pricing_raw, "cached_input_usd_per_million"
                ),
                output_usd_per_million=_price_text(pricing_raw, "output_usd_per_million"),
            )

        if schema_version == 2:
            openai = OpenAIResearchConfig(
                model=_string(openai_table, "model"),
                base_url=_string(openai_table, "base_url"),
                reasoning_effort=reasoning_effort,
                api_key_env=_environment_name(
                    openai_table, "api_key_env", "research.openai.api_key_env"
                ),
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
                max_output_tokens=max_output_tokens,
                max_malformed_retries=max_malformed_retries,
                max_transport_retries=max_transport_retries,
                pricing=pricing_config(openai_table),
            )
        else:
            endpoints: list[OpenAIEndpointEnvConfig] = []
            environment_names: list[str] = []
            for slot in ("main", "fallback"):
                endpoint_fields = {"api_key_env", "base_url_env", "model_env", "pricing"}
                if schema_version >= 4:
                    endpoint_fields |= {
                        "response_format",
                        "max_tokens_parameter",
                        "send_reasoning_effort",
                    }
                endpoint = _table(
                    openai_table,
                    slot,
                    endpoint_fields,
                )
                api_key_env = _environment_name(
                    endpoint, "api_key_env", f"research.openai.{slot}.api_key_env"
                )
                base_url_env = _environment_name(
                    endpoint, "base_url_env", f"research.openai.{slot}.base_url_env"
                )
                model_env = _environment_name(
                    endpoint, "model_env", f"research.openai.{slot}.model_env"
                )
                environment_names.extend((api_key_env, base_url_env, model_env))
                endpoints.append(
                    OpenAIEndpointEnvConfig(
                        api_key_env=api_key_env,
                        base_url_env=base_url_env,
                        model_env=model_env,
                        pricing=pricing_config(endpoint),
                        response_format=(
                            _string(endpoint, "response_format")
                            if schema_version >= 4
                            else "json_schema"
                        ),
                        max_tokens_parameter=(
                            _string(endpoint, "max_tokens_parameter")
                            if schema_version >= 4
                            else "max_completion_tokens"
                        ),
                        send_reasoning_effort=(
                            _boolean(endpoint, "send_reasoning_effort")
                            if schema_version >= 4
                            else True
                        ),
                    )
                )
                if endpoints[-1].response_format not in {"json_schema", "json_object"}:
                    raise ConfigError(
                        f"research.openai.{slot}.response_format must be "
                        "'json_schema' or 'json_object'"
                    )
                if endpoints[-1].max_tokens_parameter not in {
                    "max_completion_tokens",
                    "max_tokens",
                }:
                    raise ConfigError(
                        f"research.openai.{slot}.max_tokens_parameter must be "
                        "'max_completion_tokens' or 'max_tokens'"
                    )
            if len(set(environment_names)) != len(environment_names):
                raise ConfigError(
                    "research.openai main and fallback must use six distinct environment names"
                )
            openai = OpenAIFailoverResearchConfig(
                main=endpoints[0],
                fallback=endpoints[1],
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
                max_output_tokens=max_output_tokens,
                max_malformed_retries=max_malformed_retries,
                max_transport_retries=max_transport_retries,
            )
    elif openai_raw is not None:
        raise ConfigError("research.openai is only valid when research.provider is 'openai'")
    if run_kind == "autonomous" and provider != "openai":
        raise ConfigError("autonomous runs require research.provider = 'openai'")
    if run_kind in {"demo", "test"} and provider != "scripted":
        raise ConfigError(f"{run_kind} runs require research.provider = 'scripted'")
    if schema_version != 1 and run_kind == "demo" and not allow_scripted_demo:
        raise ConfigError("demo runs require research.allow_scripted_demo = true")
    if schema_version != 1 and run_kind != "demo" and allow_scripted_demo:
        raise ConfigError("research.allow_scripted_demo is valid only for demo runs")
    proposal_breadth = (
        _integer(
            research_raw,
            "proposal_breadth",
            minimum=1,
            maximum=HARD_MAX_PROPOSAL_BREADTH,
        )
        if "proposal_breadth" in research_raw
        else 1
    )
    research = ResearchConfig(
        provider=provider,
        max_repairs_per_experiment=_integer(
            research_raw,
            "max_repairs_per_experiment",
            minimum=0,
            maximum=HARD_MAX_REPAIRS,
        ),
        run_kind=run_kind,
        allow_scripted_demo=allow_scripted_demo,
        openai=openai,
        proposal_breadth=proposal_breadth,
    )
    return AgentConfig(
        schema_version=schema_version,
        benchmark=benchmark,
        validation=validation,
        runner=runner,
        research=research,
    )


def load_config(path: str | Path) -> AgentConfig:
    """Load a UTF-8 TOML file and return its validated normalized configuration."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            decoded = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    if not isinstance(decoded, Mapping):  # defensive; tomllib currently always returns a dict
        raise ConfigError("configuration root must be a table")
    return parse_config(cast(Mapping[str, Any], decoded))

"""Deterministic, judge-readable final report and frozen-replay instructions."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from kuairand_agent.finalization.organizer_check import OrganizerCheckEvidence
from kuairand_agent.finalization.replay import CleanReplayEvidence

REPORT_SCHEMA_VERSION: Final = 1
_SAFE_SUBCOMMAND: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")


class FinalReportError(ValueError):
    """Raised when report evidence is incomplete, inconsistent, or unsafe."""


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise FinalReportError(f"{location} must be non-empty text without NUL")
    if "\n" in value or "\r" in value:
        raise FinalReportError(f"{location} must be one line of text")
    return value.strip()


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinalReportError(f"{location} must be a lowercase SHA-256")
    return value


def _metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportError(f"{location} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise FinalReportError(f"{location} must be finite in [0, 1]")
    return rendered


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FinalReportError("report evidence must be finite JSON") from exc


def _unique_lines(
    values: Sequence[str], location: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(_text(value, f"{location}[{index}]") for index, value in enumerate(values))
    if not allow_empty and not normalized:
        raise FinalReportError(f"{location} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise FinalReportError(f"{location} entries must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """One exact aggregate-only metric row; no row-level outcomes are reportable."""

    label: str
    tier: str
    gauc: float
    ndcg_at_5: float
    primary: float
    seeds: tuple[int, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "metric label"))
        object.__setattr__(self, "tier", _text(self.tier, "metric tier"))
        gauc = _metric(self.gauc, "GAUC")
        ndcg = _metric(self.ndcg_at_5, "nDCG@5")
        primary = _metric(self.primary, "primary")
        if not math.isclose(primary, (gauc + ndcg) / 2.0, rel_tol=0.0, abs_tol=2e-7):
            raise FinalReportError("metric primary must be the organizer mean of GAUC and nDCG@5")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise FinalReportError("metric seeds must be non-negative integers")
        if len(self.seeds) != len(set(self.seeds)):
            raise FinalReportError("metric seeds must be unique")
        if self.note:
            object.__setattr__(self, "note", _text(self.note, "metric note"))


@dataclass(frozen=True, slots=True)
class ExperimentNarrative:
    """Aggregate scientific trajectory evidence for one closed experiment."""

    iteration: int
    experiment_id: str
    parent_id: str
    hypothesis: str
    mechanism: str
    material_changes: tuple[str, ...]
    attributions: tuple[str, ...]
    status: str
    inner_primary: float | None = None
    outer_primary: float | None = None

    def __post_init__(self) -> None:
        if type(self.iteration) is not int or self.iteration <= 0:
            raise FinalReportError("experiment iteration must be a positive integer")
        for name in ("experiment_id", "parent_id", "hypothesis", "mechanism", "status"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "material_changes",
            _unique_lines(self.material_changes, "material_changes"),
        )
        object.__setattr__(
            self,
            "attributions",
            _unique_lines(self.attributions, "attributions"),
        )
        for name in ("inner_primary", "outer_primary"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _metric(value, name))


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    """Campaign-level resource and human/provider-use disclosure."""

    wall_seconds: float
    peak_rss_bytes: int
    launch_count: int
    intervention_count: int
    provider_usage: str
    device_usage: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.wall_seconds) is not float
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0
        ):
            raise FinalReportError("wall_seconds must be a finite non-negative float")
        for name in ("peak_rss_bytes", "launch_count", "intervention_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise FinalReportError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "provider_usage", _text(self.provider_usage, "provider_usage"))
        object.__setattr__(self, "device_usage", _unique_lines(self.device_usage, "device_usage"))


@dataclass(frozen=True, slots=True)
class FinalReportContext:
    """All non-row-level facts needed to render the mandatory final report."""

    benchmark_contract: Mapping[str, object]
    baselines: tuple[MetricEvidence, ...]
    selected: MetricEvidence
    experiments: tuple[ExperimentNarrative, ...]
    inner_fold_evidence: tuple[str, ...]
    seed_confirmation: tuple[str, ...]
    failures_and_recoveries: tuple[str, ...]
    leakage_controls: tuple[str, ...]
    test_evidence: tuple[str, ...]
    selection_rationale: str
    resources: ResourceEvidence
    known_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_contract, Mapping) or not self.benchmark_contract:
            raise FinalReportError("benchmark_contract must be a non-empty mapping")
        contract = dict(self.benchmark_contract)
        _canonical_json(contract)
        object.__setattr__(self, "benchmark_contract", MappingProxyType(contract))
        if not self.baselines or any(
            not isinstance(item, MetricEvidence) for item in self.baselines
        ):
            raise FinalReportError("baselines must contain MetricEvidence rows")
        if not isinstance(self.selected, MetricEvidence):
            raise FinalReportError("selected must be MetricEvidence")
        if not self.experiments or any(
            not isinstance(item, ExperimentNarrative) for item in self.experiments
        ):
            raise FinalReportError("experiments must contain at least one narrative")
        iterations = tuple(item.iteration for item in self.experiments)
        if iterations != tuple(sorted(iterations)) or len(iterations) != len(set(iterations)):
            raise FinalReportError("experiments must use unique ascending iteration numbers")
        for name in (
            "inner_fold_evidence",
            "seed_confirmation",
            "failures_and_recoveries",
            "leakage_controls",
            "test_evidence",
        ):
            object.__setattr__(self, name, _unique_lines(getattr(self, name), name))
        object.__setattr__(
            self,
            "known_limitations",
            _unique_lines(self.known_limitations, "known_limitations", allow_empty=True),
        )
        object.__setattr__(
            self,
            "selection_rationale",
            _text(self.selection_rationale, "selection_rationale"),
        )
        if not isinstance(self.resources, ResourceEvidence):
            raise FinalReportError("resources must be ResourceEvidence")


@dataclass(frozen=True, slots=True)
class ReproduceInstructions:
    """Frozen local replay command; research credentials are intentionally absent."""

    expected_data_sha256: str
    replay_subcommand: str = "replay"
    dependency_groups: tuple[str, ...] = ("research-tree",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_data_sha256",
            _digest(self.expected_data_sha256, "expected_data_sha256"),
        )
        if (
            type(self.replay_subcommand) is not str
            or _SAFE_SUBCOMMAND.fullmatch(self.replay_subcommand) is None
        ):
            raise FinalReportError("replay_subcommand is not a safe CLI token")
        if self.dependency_groups not in {
            ("research-tree",),
            ("research-tree", "research-neural"),
        }:
            raise FinalReportError("dependency_groups are unsupported")


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def _number(value: float | None) -> str:
    return "-" if value is None else repr(float(value))


def _bullets(values: Sequence[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _require_report_consistency(
    context: FinalReportContext,
    replay: CleanReplayEvidence,
    organizer_check: OrganizerCheckEvidence,
) -> None:
    if not isinstance(context, FinalReportContext):
        raise FinalReportError("context must be FinalReportContext")
    if not isinstance(replay, CleanReplayEvidence):
        raise FinalReportError("replay must be CleanReplayEvidence")
    if not isinstance(organizer_check, OrganizerCheckEvidence):
        raise FinalReportError("organizer_check must be OrganizerCheckEvidence")
    selected = context.selected
    metrics = replay.validation.metrics
    for label, expected in (
        ("GAUC", selected.gauc),
        ("nDCG@5", selected.ndcg_at_5),
        ("primary", selected.primary),
    ):
        observed = metrics.get(label)
        if observed != expected:
            raise FinalReportError(f"selected {label} differs from clean replay evidence")
    if organizer_check.checker_returncode != 0 or organizer_check.checker_command[-1] != "--check":
        raise FinalReportError("organizer checker evidence is not a successful check-only run")
    if organizer_check.submission_sha256 != replay.final.submission_sha256:
        raise FinalReportError("organizer checker saw different final submission bytes")
    if replay.final.final_outcomes_accessed:
        raise FinalReportError("final replay evidence reports outcome access")


def render_final_report(
    context: FinalReportContext,
    *,
    replay: CleanReplayEvidence,
    organizer_check: OrganizerCheckEvidence,
    fallback_failures: Sequence[str] = (),
) -> str:
    """Render the mandatory report from exact immutable aggregate evidence."""

    _require_report_consistency(context, replay, organizer_check)
    failures = tuple(
        (
            *context.failures_and_recoveries,
            *_unique_lines(fallback_failures, "fallback_failures", allow_empty=True),
        )
    )
    baseline_rows = "\n".join(
        "| {label} | {tier} | {gauc} | {ndcg} | {primary} | {seeds} |".format(
            label=_cell(item.label),
            tier=_cell(item.tier),
            gauc=_number(item.gauc),
            ndcg=_number(item.ndcg_at_5),
            primary=_number(item.primary),
            seeds=", ".join(str(seed) for seed in item.seeds) or "-",
        )
        for item in (*context.baselines, context.selected)
    )
    trajectory_rows = "\n".join(
        f"| {item.iteration} | {_cell(item.experiment_id)} | {_cell(item.parent_id)} | "
        f"{_cell(item.status)} | {_number(item.inner_primary)} | "
        f"{_number(item.outer_primary)} |"
        for item in context.experiments
    )
    experiment_details = "\n\n".join(
        f"### Iteration {item.iteration}: {item.experiment_id}\n\n"
        f"Hypothesis: {item.hypothesis}\n\n"
        f"Mechanism: {item.mechanism}\n\n"
        f"Material executable-source changes:\n\n{_bullets(item.material_changes)}\n\n"
        f"Source attributions:\n\n{_bullets(item.attributions)}"
        for item in context.experiments
    )
    limitations = context.known_limitations or ("No additional engineering limitation recorded.",)
    exact_or_tolerance = (
        "exact same-host float64 prediction bytes"
        if replay.validation.exact_prediction_bytes
        else (
            f"absolute tolerance {replay.absolute_tolerance}, identical per-user top-5, "
            "and protected metric parity"
        )
    )
    final_boundary_statement = (
        "Final inference used only the label-free FINAL capability; "
        "final outcomes were neither accessed nor scored."
    )
    return f"""# Autonomous KuaiRand-Pure final report

Report schema: {REPORT_SCHEMA_VERSION}

## Benchmark contract

The operative task is `long_view` ranking over logged impressions. The primary validation score
is the arithmetic mean of organizer GAUC and organizer nDCG@5.

```json
{_canonical_json(dict(context.benchmark_contract))}
```

## Baseline parity

| Candidate | Evidence tier | GAUC | nDCG@5 | Primary | Seeds |
| --- | --- | ---: | ---: | ---: | --- |
{baseline_rows}

The selected row is local public-validation evidence; it is not a hidden-test claim.

## Experiment trajectory and candidate tree

| Iteration | Candidate | Parent | Outcome | Inner primary | Outer primary |
| ---: | --- | --- | --- | ---: | ---: |
{trajectory_rows}

{experiment_details}

## Inner folds, outer validation, and seed confirmation

Inner-fold evidence:

{_bullets(context.inner_fold_evidence)}

Outer/seed confirmation:

{_bullets(context.seed_confirmation)}

Per-metric tradeoff for the selected candidate: GAUC={_number(context.selected.gauc)},
nDCG@5={_number(context.selected.ndcg_at_5)}, and primary={_number(context.selected.primary)}.

## Failures, repairs, recoveries, and interventions

{_bullets(failures)}

Manual intervention count: {context.resources.intervention_count}.

## Runtime, resources, and device use

- Campaign wall time: {context.resources.wall_seconds} seconds.
- Peak resident memory: {context.resources.peak_rss_bytes} bytes.
- Charged training launches: {context.resources.launch_count}.
- Research-model usage: {context.resources.provider_usage}
- Device use: {"; ".join(context.resources.device_usage)}.

## Leakage controls and tests

{_bullets(context.leakage_controls)}

Verification evidence:

{_bullets(context.test_evidence)}

{final_boundary_statement} The untouched organizer checker received a private masked view and ran
`--check` only.

## Selected-candidate rationale

{context.selection_rationale}

Selected candidate: `{replay.candidate_id}`. Replay standard: {exact_or_tolerance}.
Training replay disclosure: {replay.training_replay}.

## Replay and submission verification

- Restored source SHA-256: `{replay.identity.source_sha256}`.
- Restored config SHA-256: `{replay.identity.config_sha256}`.
- Restored feature/preprocessing SHA-256: `{replay.identity.features_sha256}`.
- Restored checkpoint SHA-256: `{replay.identity.checkpoint_sha256}`.
- Verified data SHA-256: `{replay.identity.data_sha256}`.
- Verified environment SHA-256: `{replay.identity.environment_sha256}`.
- Public prediction digest: `{replay.validation.replay_prediction_digest}`.
- Public CSV serialization preserved top-5 ordering and protected metrics: yes.
- Final prediction digest: `{replay.final.prediction_digest}`.
- Final submission SHA-256: `{replay.final.submission_sha256}`.
- Untouched organizer checker return code: {organizer_check.checker_returncode}.
- Organizer masked {organizer_check.masked_view.final_rows_masked} final rows before loading and
  did not inspect, hash, or score their registered outcome cells.

## Limitations and hidden-test status

{_bullets(limitations)}

Hidden-test improvement is unverified until organizer scoring.
"""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o700 if executable else 0o600)
    except FileExistsError as exc:
        raise FinalReportError(f"refusing to overwrite existing evidence file: {path}") from exc
    committed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
        os.fchmod(descriptor, 0o700 if executable else 0o600)
        _fsync_directory(path.parent)
        committed = True
    finally:
        os.close(descriptor)
        if not committed:
            path.unlink(missing_ok=True)
    return path.resolve()


def write_final_report(
    path: str | Path,
    context: FinalReportContext,
    replay: CleanReplayEvidence,
    organizer_check: OrganizerCheckEvidence,
    *,
    fallback_failures: Sequence[str] = (),
) -> Path:
    """Exclusively write the deterministic UTF-8 report."""

    rendered = render_final_report(
        context,
        replay=replay,
        organizer_check=organizer_check,
        fallback_failures=fallback_failures,
    )
    return _write_exclusive(Path(path), rendered.encode("utf-8"))


def render_reproduce_script(instructions: ReproduceInstructions) -> str:
    """Render deterministic clean-install/frozen-replay instructions without research calls."""

    if not isinstance(instructions, ReproduceInstructions):
        raise FinalReportError("instructions must be ReproduceInstructions")
    sync_groups = " ".join(f"--group {group}" for group in instructions.dependency_groups)
    return f"""#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /absolute/path/to/repository /absolute/path/to/verified/KuaiRand-Pure/data" >&2
  exit 64
fi

REPO_DIR=$1
DATA_DIR=$2
BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

cd -- "$REPO_DIR"
uv sync --locked {sync_groups}
exec uv run --locked --no-sync kuairand-agent {instructions.replay_subcommand} \
  --bundle "$BUNDLE_DIR" \
  --data-dir "$DATA_DIR" \
  --project-root "$REPO_DIR" \
  --expected-data-sha256 {instructions.expected_data_sha256}
"""


def write_reproduce_script(path: str | Path, instructions: ReproduceInstructions) -> Path:
    """Exclusively write an executable frozen replay script."""

    return _write_exclusive(
        Path(path), render_reproduce_script(instructions).encode("ascii"), executable=True
    )


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ExperimentNarrative",
    "FinalReportContext",
    "FinalReportError",
    "MetricEvidence",
    "ReproduceInstructions",
    "ResourceEvidence",
    "render_final_report",
    "render_reproduce_script",
    "write_final_report",
    "write_reproduce_script",
]

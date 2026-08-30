"""Per-iteration research evidence recovered from a closed campaign run directory.

The autonomous loop already persists what a judge-facing report needs: one immutable lineage
record per admitted iteration, and one hash-chained journal entry per rejected branch.  The
finalizer previously read neither and emitted a single hardcoded narrative, so a campaign that
completed five iterations still reported one.  This module reads those durable records and turns
them into :class:`~kuairand_agent.finalization.report.ExperimentNarrative` values.

It is strictly read-only: it opens no database, calls no provider, and writes nothing.  Records
are parsed through the same strict schema constructors the campaign wrote them with, so a
corrupted record fails closed rather than degrading into a plausible-looking narrative.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# The journal reader is module-private but is the only correct decoder for the hash-chained
# ledger; reimplementing the chain verification here would be the worse choice.
from kuairand_agent.campaign.full_campaign_runtime import (
    _load_rejection_journal as load_rejection_journal,
)
from kuairand_agent.finalization.report import ExperimentNarrative
from kuairand_agent.research.context import AggregateRecord
from kuairand_agent.research.schemas import (
    GeneratedPackage,
    Proposal,
    SchemaValidationError,
    parse_json_object,
)

_PRODUCTION_DIR: Final = "production"
_GENERATED_SOURCE_DIR: Final = "generated-source"
_REJECTION_JOURNAL_DIR: Final = "controller-rejection-journal"
_LINEAGE_GLOB: Final = "iteration-*-lineage.json"
_MAX_ITERATIONS: Final = 50

_NO_MATERIAL_CHANGE: Final = (
    "No material executable change; the branch was rejected before evaluation."
)
_NO_ATTRIBUTION: Final = (
    "Controller static gate; a rejected branch retains no provider source attribution."
)


class IterationEvidenceError(ValueError):
    """A durable per-iteration record exists but cannot be read as valid evidence."""


@dataclass(frozen=True, slots=True)
class IterationEvidence:
    """Recovered per-iteration narratives plus the failure lines they imply."""

    narratives: tuple[ExperimentNarrative, ...]
    failure_lines: tuple[str, ...]

    @property
    def iteration_count(self) -> int:
        return len(self.narratives)


def _one_line(value: object, *, limit: int = 2048) -> str:
    """Collapse model-authored text to the single line the report validator requires.

    ``schemas._text`` permits newlines up to 16,384 characters; ``report._text`` rejects any
    string containing one.  Four provider-authored fields cross that boundary, and a lineage
    record is written read-only, so one multi-line hypothesis would otherwise leave the
    campaign unfinalizable without editing files by hand.
    """

    if type(value) is not str:
        return ""
    return " ".join(value.split())[:limit]


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Preserve order while dropping blanks and repeats.

    ``report._unique_lines`` rejects duplicates outright rather than collapsing them, so any
    sequence assembled from model-supplied text must be normalized before it reaches a narrative.
    """

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _one_line(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _text_field(raw: Mapping[str, object], key: str, path_name: str) -> str:
    value = raw.get(key)
    if type(value) is not str or not value.strip():
        raise IterationEvidenceError(f"{path_name} field {key!r} must be non-empty text")
    return value


def _iteration_field(raw: Mapping[str, object], path_name: str) -> int:
    value = raw.get("scientific_iteration")
    if type(value) is not int or not 1 <= value <= _MAX_ITERATIONS:
        raise IterationEvidenceError(
            f"{path_name} scientific_iteration must be an integer in [1, {_MAX_ITERATIONS}]"
        )
    return value


def _scalar_text(record: AggregateRecord, key: str) -> str:
    value = record.values.get(key)
    if value is None:
        return ""
    return str(value)


def _lineage_narrative(
    path: Path,
    *,
    fallback_parent_id: str,
    candidate_outcomes: Mapping[str, str],
) -> tuple[ExperimentNarrative, tuple[str, ...]]:
    """Build one narrative from an admitted iteration's immutable lineage record."""

    try:
        raw = parse_json_object(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SchemaValidationError) as exc:
        raise IterationEvidenceError(f"lineage record {path.name} is unreadable") from exc

    iteration = _iteration_field(raw, path.name)
    candidate_id = _text_field(raw, "candidate_id", path.name)
    proposal_raw = raw.get("proposal")
    package_raw = raw.get("package")
    if not isinstance(proposal_raw, Mapping) or not isinstance(package_raw, Mapping):
        raise IterationEvidenceError(f"lineage record {path.name} lacks a proposal and package")
    try:
        proposal = Proposal.from_mapping(proposal_raw)
        package = GeneratedPackage.from_mapping(package_raw)
    except SchemaValidationError as exc:
        raise IterationEvidenceError(
            f"lineage record {path.name} does not contain a valid proposal and package"
        ) from exc

    repairs_raw = raw.get("repair_calls")
    repair_count = len(repairs_raw) if isinstance(repairs_raw, Sequence) else 0

    material_changes = _dedupe(
        (
            package.material_change_summary,
            *(f"Changed top-level symbol: {name}" for name in package.material_symbols),
        )
    )
    attributions = _dedupe(proposal.attributions) or (
        "Provider-generated candidate source; no explicit attribution supplied.",
    )
    expected = ", ".join(proposal.expected_metric_effects)

    failure_lines: tuple[str, ...] = ()
    if repair_count:
        failure_lines = (
            f"Iteration {iteration} ({candidate_id}): {repair_count} bounded repair "
            "round-trip(s) were required before the candidate satisfied the source contract; "
            "the repaired package was accepted and evaluated.",
        )

    narrative = ExperimentNarrative(
        iteration=iteration,
        experiment_id=candidate_id,
        parent_id=proposal.parent_candidate_id or fallback_parent_id,
        hypothesis=_one_line(proposal.hypothesis),
        mechanism=_one_line(f"{proposal.mechanism} Expected to move: {expected}."),
        material_changes=material_changes,
        attributions=attributions,
        status=candidate_outcomes.get(candidate_id, "evaluated"),
        failures_and_recoveries=failure_lines,
    )
    return narrative, failure_lines


def _rejected_narrative(
    record: AggregateRecord,
    *,
    fallback_parent_id: str,
) -> tuple[ExperimentNarrative, tuple[str, ...]]:
    """Build one narrative from a rejected branch's hash-chained journal entry.

    A rejected branch never materialized, so it has no material change and no retained provider
    attribution.  Saying so explicitly is more useful to a judge than omitting the iteration.
    """

    values = record.values
    iteration = values.get("scientific_iteration")
    if type(iteration) is not int or not 1 <= iteration <= _MAX_ITERATIONS:
        raise IterationEvidenceError("rejection journal entry has no usable iteration number")
    candidate_id = _scalar_text(record, "candidate_id") or f"rejected-iteration-{iteration:02d}"
    outcome = _scalar_text(record, "branch_outcome") or "rejected"
    stage = _scalar_text(record, "root_failure_stage") or "unknown stage"
    category = _scalar_text(record, "root_failure_category") or "unknown category"
    diagnostic = (
        _scalar_text(record, "root_failure_diagnostic")
        or _scalar_text(record, "diagnostic")
        or "No diagnostic was retained."
    )
    repairs = values.get("repairs_attempted")
    repair_count = repairs if type(repairs) is int else 0
    family = _scalar_text(record, "proposal_family") or "unrecorded"

    line = (
        f"Iteration {iteration} ({candidate_id}): {outcome} at {stage} ({category}) after "
        f"{repair_count} repair attempt(s). {diagnostic}"
    )
    narrative = ExperimentNarrative(
        iteration=iteration,
        experiment_id=candidate_id,
        parent_id=fallback_parent_id,
        hypothesis=(
            f"Proposal family {family} was admitted for implementation but did not survive the "
            "controller's static gates."
        ),
        mechanism=f"Rejected at {stage} ({category}). {diagnostic}",
        material_changes=(_NO_MATERIAL_CHANGE,),
        attributions=(_NO_ATTRIBUTION,),
        status=outcome,
        failures_and_recoveries=(line,),
    )
    return narrative, (line,)


def collect_iteration_narratives(
    run_dir: Path,
    *,
    campaign_id: str,
    fallback_parent_id: str,
    candidate_outcomes: Mapping[str, str] | None = None,
    maximum_iterations: int = _MAX_ITERATIONS,
) -> IterationEvidence:
    """Recover every iteration a campaign durably recorded, admitted or rejected.

    Returns empty evidence when a run predates these records or never reached research, so the
    caller can keep its existing single-narrative behaviour instead of failing finalization.
    """

    if not isinstance(run_dir, Path):
        raise IterationEvidenceError("run_dir must be a pathlib.Path")
    outcomes = dict(candidate_outcomes or {})
    generated_root = run_dir / _PRODUCTION_DIR / _GENERATED_SOURCE_DIR
    if not generated_root.is_dir():
        return IterationEvidence((), ())

    narratives: dict[int, ExperimentNarrative] = {}
    failure_lines: list[str] = []

    for path in sorted(generated_root.glob(_LINEAGE_GLOB)):
        if not path.is_file():
            continue
        narrative, lines = _lineage_narrative(
            path,
            fallback_parent_id=fallback_parent_id,
            candidate_outcomes=outcomes,
        )
        narratives[narrative.iteration] = narrative
        failure_lines.extend(lines)

    if (generated_root / _REJECTION_JOURNAL_DIR).is_dir():
        records, _ = load_rejection_journal(
            generated_root,
            campaign_id=campaign_id,
            maximum_iterations=maximum_iterations,
        )
        for record in records:
            narrative, lines = _rejected_narrative(record, fallback_parent_id=fallback_parent_id)
            # An admitted lineage record is richer evidence than a rejection entry, so it wins.
            if narrative.iteration not in narratives:
                narratives[narrative.iteration] = narrative
                failure_lines.extend(lines)

    ordered = tuple(narratives[key] for key in sorted(narratives))
    return IterationEvidence(ordered, _dedupe(failure_lines))


def count_recorded_iterations(run_dir: Path) -> int:
    """Count durably recorded iterations without parsing any record.

    ``_bundle_metadata`` needs only the count and does not build narratives, so this scans
    filenames rather than repeating the strict record parsing done by
    :func:`collect_iteration_narratives`.
    """

    if not isinstance(run_dir, Path):
        raise IterationEvidenceError("run_dir must be a pathlib.Path")
    generated_root = run_dir / _PRODUCTION_DIR / _GENERATED_SOURCE_DIR
    if not generated_root.is_dir():
        return 0
    iterations: set[int] = set()
    for path in generated_root.glob(_LINEAGE_GLOB):
        match = re.fullmatch(r"iteration-(\d{2,})-lineage\.json", path.name)
        if match is not None and path.is_file():
            iterations.add(int(match.group(1)))
    journal = generated_root / _REJECTION_JOURNAL_DIR
    if journal.is_dir():
        for path in journal.glob("rejection-*.json"):
            match = re.fullmatch(r"rejection-(\d{2,})\.json", path.name)
            if match is not None and path.is_file():
                iterations.add(int(match.group(1)))
    return len(iterations)


__all__ = [
    "IterationEvidence",
    "IterationEvidenceError",
    "collect_iteration_narratives",
    "count_recorded_iterations",
]

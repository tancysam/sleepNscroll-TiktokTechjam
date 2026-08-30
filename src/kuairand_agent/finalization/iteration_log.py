"""Per-iteration run log: hypothesis, code diff, metrics, and error/recovery events.

The organizer starter kit requires a per-iteration record covering four things for every turn of
the research loop: what the agent intended to try and why, the code diff it applied, the metrics
that resulted, and any error or recovery event it encountered.  A campaign already persists all
four durably -- the lineage documents under ``production/generated-source``, the generated source
trees beside them, the project-wide research-lineage ledger, and the controller rejection journal.
This module joins those existing records into the single artifact the deliverable asks for; it
derives nothing that the campaign did not already record, and never reads final-period outcomes.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kuairand_agent.campaign.store import LineageEventRecord, ResearchLineageLedger, StoreError

ITERATION_LOG_SCHEMA_VERSION: Final = 1
_LINEAGE_NAME: Final = re.compile(r"iteration-(\d{2,})-lineage\.json\Z")
_MAX_DIFF_LINES: Final = 400


class IterationLogError(ValueError):
    """Raised when a run directory cannot produce a well-formed iteration log."""


@dataclass(frozen=True, slots=True)
class IterationFileDiff:
    """One generated file's unified diff against the parent it was derived from."""

    path: str
    parent_label: str
    diff: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class IterationLogEntry:
    """Everything the run-log deliverable requires for one scientific iteration."""

    iteration: int
    candidate_id: str
    hypothesis: str
    mechanism: str
    objective: str
    principal_change: str
    expected_metric_effects: tuple[str, ...]
    attributions: tuple[str, ...]
    diffs: tuple[IterationFileDiff, ...]
    metrics: Mapping[str, float | bool | None]
    repairs_attempted: int
    events: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": ITERATION_LOG_SCHEMA_VERSION,
            "iteration": self.iteration,
            "candidate_id": self.candidate_id,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "objective": self.objective,
            "principal_change": self.principal_change,
            "expected_metric_effects": list(self.expected_metric_effects),
            "attributions": list(self.attributions),
            "code_diff": [
                {
                    "path": item.path,
                    "parent": item.parent_label,
                    "diff": item.diff,
                    "truncated": item.truncated,
                }
                for item in self.diffs
            ],
            "metrics": dict(self.metrics),
            "repairs_attempted": self.repairs_attempted,
            "events": list(self.events),
        }


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IterationLogError(f"{path.name} is unreadable or malformed") from exc
    if not isinstance(decoded, Mapping):
        raise IterationLogError(f"{path.name} is not a JSON object")
    return decoded


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _lines(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(_text(item) for item in value if _text(item))


def _candidate_dir(generated_root: Path, candidate_id: str) -> Path | None:
    if not candidate_id or "/" in candidate_id or candidate_id.startswith("."):
        return None
    candidate = generated_root / candidate_id
    return candidate if candidate.is_dir() else None


def _parent_dir(
    *,
    generated_root: Path,
    project_root: Path,
    parent_candidate_id: str,
) -> tuple[Path | None, str]:
    """Resolve the tree this iteration was derived from.

    A follow-up iteration is parented on an earlier promoted candidate; the first is parented on
    the organizer candidate seed.  Falling back to the seed keeps the diff meaningful rather than
    rendering the whole file as an addition when an intermediate tree is unavailable.
    """

    generated_parent = _candidate_dir(generated_root, parent_candidate_id)
    if generated_parent is not None:
        return generated_parent, parent_candidate_id
    seed = project_root / "candidate_seed"
    if seed.is_dir():
        return seed, "candidate_seed"
    return None, parent_candidate_id or "unknown"


def _unified_diff(
    *,
    path: str,
    parent_dir: Path | None,
    candidate_dir: Path | None,
    parent_label: str,
) -> IterationFileDiff:
    def read(directory: Path | None) -> list[str]:
        if directory is None:
            return []
        target = directory / path
        if not target.is_file():
            return []
        try:
            return target.read_text(encoding="utf-8").splitlines(keepends=True)
        except (OSError, UnicodeDecodeError):
            return []

    rendered = list(
        difflib.unified_diff(
            read(parent_dir),
            read(candidate_dir),
            fromfile=f"{parent_label}/{path}",
            tofile=f"candidate/{path}",
            n=3,
        )
    )
    truncated = len(rendered) > _MAX_DIFF_LINES
    if truncated:
        rendered = rendered[:_MAX_DIFF_LINES]
        rendered.append(f"... diff truncated at {_MAX_DIFF_LINES} lines ...\n")
    return IterationFileDiff(
        path=path,
        parent_label=parent_label,
        diff="".join(rendered),
        truncated=truncated,
    )


def _metrics_of(event: LineageEventRecord | None) -> dict[str, float | bool | None]:
    if event is None:
        return {}
    metrics: dict[str, float | bool | None] = {}
    for label, fold in (("fold_a", event.inner_fold_a), ("fold_b", event.inner_fold_b)):
        if fold is None:
            continue
        metrics[f"{label}_gauc"] = fold.gauc
        metrics[f"{label}_ndcg_at_5"] = fold.ndcg_at_5
        metrics[f"{label}_primary"] = fold.primary
    if event.parent_fold_a_primary is not None:
        metrics["parent_fold_a_primary"] = event.parent_fold_a_primary
    if event.parent_fold_b_primary is not None:
        metrics["parent_fold_b_primary"] = event.parent_fold_b_primary
    if event.inner_fold_a is not None and event.parent_fold_a_primary is not None:
        metrics["fold_a_delta"] = event.inner_fold_a.primary - event.parent_fold_a_primary
    if event.inner_fold_b is not None and event.parent_fold_b_primary is not None:
        metrics["fold_b_delta"] = event.inner_fold_b.primary - event.parent_fold_b_primary
    metrics["promoted"] = event.promoted
    return metrics


def _lineage_events(ledger_path: Path, campaign_id: str) -> dict[str, LineageEventRecord]:
    """Index this campaign's durable admission evidence by candidate id.

    A missing or unreadable ledger is not fatal: the log still carries hypothesis, diff, and
    recovery evidence, and simply reports no metrics rather than refusing to render.
    """

    if not ledger_path.is_file():
        return {}
    try:
        ledger = ResearchLineageLedger.open(ledger_path, read_only=True)
    except (StoreError, OSError):
        return {}
    try:
        rows = ledger.events_for_campaign(campaign_id)
    except (StoreError, OSError, AttributeError):
        return {}
    finally:
        ledger.close()
    return {event.candidate_id: event for event in rows}


def _rejection_events(generated_root: Path) -> tuple[str, ...]:
    journal = generated_root / "controller-rejection-journal"
    if not journal.is_dir():
        return ()
    events: list[str] = []
    for path in sorted(journal.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            record = _read_json(path)
        except IterationLogError:
            continue
        values = record.get("values")
        source = values if isinstance(values, Mapping) else record
        candidate = _text(source.get("candidate_id"))
        code = _text(source.get("root_failure_code")) or _text(source.get("terminal_failure_code"))
        diagnostic = _text(source.get("diagnostic"))[:200]
        events.append(
            f"pre-admission rejection: candidate={candidate or 'unknown'} "
            f"code={code or 'unknown'} diagnostic={diagnostic or 'none'}"
        )
    return tuple(events)


def _candidate_outcomes(run_dir: Path) -> dict[str, tuple[str, str]]:
    """Map candidate id to its (outcome, reason) as the scientific loop recorded it.

    This is what makes an execution failure visible in the log.  A candidate can be admitted,
    trained, and still never produce inner metrics -- ``callback_failed`` is the clearest case --
    and the run-log deliverable asks for exactly those error events.  Reading the retained
    ``scientific_result`` row keeps the log honest about them instead of inferring silence from
    absent metrics.
    """

    for name in ("production/finalization-support", "final"):
        path = run_dir / name / "experiments.jsonl"
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping) or record.get("record_type") != "scientific_result":
                continue
            evidence = record.get("evidence")
            outcomes = evidence.get("candidate_outcomes") if isinstance(evidence, Mapping) else None
            if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
                continue
            resolved: dict[str, tuple[str, str]] = {}
            for item in outcomes:
                if not isinstance(item, Mapping):
                    continue
                candidate = _text(item.get("candidate_id"))
                if candidate:
                    resolved[candidate] = (
                        _text(item.get("outcome")) or "unknown",
                        _text(item.get("reason")) or "none",
                    )
            if resolved:
                return resolved
    return {}


def _iteration_documents(generated_root: Path) -> Iterator[tuple[int, Mapping[str, object]]]:
    if not generated_root.is_dir():
        raise IterationLogError("run directory has no production/generated-source tree")
    for path in sorted(generated_root.iterdir()):
        if not path.is_file():
            continue
        matched = _LINEAGE_NAME.fullmatch(path.name)
        if matched is None:
            continue
        yield int(matched.group(1)), _read_json(path)


def build_iteration_log(run_dir: Path, *, project_root: Path) -> tuple[IterationLogEntry, ...]:
    """Join a completed or in-flight campaign's durable records into the run-log deliverable."""

    generated_root = run_dir / "production" / "generated-source"
    documents = list(_iteration_documents(generated_root))
    if not documents:
        raise IterationLogError("run directory contains no scientific-iteration lineage documents")

    campaign_id = _text(documents[0][1].get("campaign_id"))
    events = _lineage_events(run_dir.parent / "research-lineage-ledger.sqlite3", campaign_id)
    rejections = _rejection_events(generated_root)
    outcomes = _candidate_outcomes(run_dir)

    entries: list[IterationLogEntry] = []
    for iteration, document in documents:
        proposal = document.get("proposal")
        if not isinstance(proposal, Mapping):
            raise IterationLogError(f"iteration {iteration} lineage lacks a proposal")
        package = document.get("package")
        files = package.get("files") if isinstance(package, Mapping) else None
        paths = (
            tuple(
                _text(item.get("path"))
                for item in files
                if isinstance(item, Mapping) and _text(item.get("path"))
            )
            if isinstance(files, Sequence) and not isinstance(files, (str, bytes))
            else ()
        )
        candidate_id = _text(document.get("candidate_id"))
        candidate_dir = _candidate_dir(generated_root, candidate_id)
        parent_dir, parent_label = _parent_dir(
            generated_root=generated_root,
            project_root=project_root,
            parent_candidate_id=_text(proposal.get("parent_candidate_id")),
        )
        repair_calls = document.get("repair_calls")
        repairs = len(repair_calls) if isinstance(repair_calls, Sequence) else 0
        iteration_events = list(rejections) if iteration == documents[0][0] else []
        if repairs:
            iteration_events.append(
                f"bounded repair loop: {repairs} repair call(s) accepted before admission"
            )
        outcome, reason = outcomes.get(candidate_id, ("", ""))
        if outcome:
            iteration_events.append(f"scientific outcome: {outcome} ({reason})")
        if outcome in {"callback_failed", "budget_rejected"} and iteration != documents[-1][0]:
            iteration_events.append(
                "the campaign recovered from this failure and continued to the next iteration "
                "rather than stalling or aborting"
            )
        entries.append(
            IterationLogEntry(
                iteration=iteration,
                candidate_id=candidate_id,
                hypothesis=_text(proposal.get("hypothesis")),
                mechanism=_text(proposal.get("mechanism")),
                objective=_text(proposal.get("objective")),
                principal_change=_text(proposal.get("principal_change")),
                expected_metric_effects=_lines(proposal.get("expected_metric_effects")),
                attributions=_lines(proposal.get("attributions")),
                diffs=tuple(
                    _unified_diff(
                        path=path,
                        parent_dir=parent_dir,
                        candidate_dir=candidate_dir,
                        parent_label=parent_label,
                    )
                    for path in paths
                ),
                metrics=_metrics_of(events.get(candidate_id)),
                repairs_attempted=repairs,
                events=tuple(iteration_events),
            )
        )
    return tuple(entries)


def render_jsonl(entries: Sequence[IterationLogEntry]) -> str:
    return "".join(
        json.dumps(entry.to_wire(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for entry in entries
    )


def _render_metrics(metrics: Mapping[str, float | bool | None]) -> str:
    if not metrics:
        return "No inner-fold metrics were recorded for this iteration.\n"
    rows = ["| Metric | Value |", "|---|---|"]
    for key, value in metrics.items():
        if value is None:
            rendered = "not applicable"
        elif isinstance(value, bool):
            rendered = "yes" if value else "no"
        else:
            rendered = f"{value:+.7f}" if key.endswith("_delta") else f"{value:.7f}"
        rows.append(f"| {key} | {rendered} |")
    return "\n".join(rows) + "\n"


def render_markdown(entries: Sequence[IterationLogEntry]) -> str:
    sections = [
        "# Run and iteration log",
        "",
        "One section per scientific iteration, covering the four facts the starter kit's run-log",
        "requirement asks for: the hypothesis, the code diff applied, the resulting metrics, and",
        "any error or recovery event. Every value is read from the campaign's own durable records.",
        "",
    ]
    for entry in entries:
        sections.extend(
            [
                f"## Iteration {entry.iteration} — `{entry.candidate_id}`",
                "",
                "### Hypothesis",
                "",
                entry.hypothesis or "_not recorded_",
                "",
                f"**Objective.** {entry.objective or '_not recorded_'}",
                "",
                f"**Mechanism.** {entry.mechanism or '_not recorded_'}",
                "",
                f"**Principal change.** {entry.principal_change or '_not recorded_'}",
                "",
            ]
        )
        if entry.expected_metric_effects:
            sections.extend(
                [
                    "**Expected metric effects.** " + ", ".join(entry.expected_metric_effects),
                    "",
                ]
            )
        if entry.attributions:
            sections.extend(["**Attributions.**", ""])
            sections.extend(f"- {item}" for item in entry.attributions)
            sections.append("")
        sections.extend(["### Code diff", ""])
        if not entry.diffs:
            sections.extend(["_No generated files were recorded for this iteration._", ""])
        for item in entry.diffs:
            sections.extend(
                [
                    f"`{item.path}` (against `{item.parent_label}`)",
                    "",
                    "```diff",
                    item.diff.rstrip("\n") or "(no textual difference)",
                    "```",
                    "",
                ]
            )
        sections.extend(["### Resulting metrics", "", _render_metrics(entry.metrics), ""])
        sections.extend(["### Errors and recovery", ""])
        if entry.events:
            sections.extend(f"- {item}" for item in entry.events)
        else:
            sections.append("- No error, rejection, or repair event was recorded.")
        sections.append("")
    return "\n".join(sections)

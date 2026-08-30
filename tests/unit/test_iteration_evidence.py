"""Recover per-iteration research evidence from a closed run directory.

Every assertion here traces to something the finalizer previously could not report. A live
campaign generated candidates 2, 3 and 4 plus repair variants inside 47 minutes while the final
report still claimed a single hardcoded iteration, and the run directory holding the only copy of
those proposals was later overwritten. The collector exists so that trajectory reaches a judge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kuairand_agent.campaign.full_campaign_runtime import _persist_rejection_journal_entry
from kuairand_agent.finalization.iteration_evidence import (
    IterationEvidenceError,
    collect_iteration_narratives,
    count_recorded_iterations,
)
from kuairand_agent.research.context import AggregateRecord
from kuairand_agent.research.schemas import (
    GeneratedFile,
    GeneratedPackage,
    Proposal,
    RequiredField,
    canonical_json_bytes,
)

CAMPAIGN = "campaign-iteration-evidence"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _proposal(
    iteration: int,
    *,
    parent: str = "official-fm-fallback-seed-4",
    hypothesis: str | None = None,
    mechanism: str | None = None,
) -> Proposal:
    return Proposal(
        proposal_id=f"proposal-{iteration:02d}",
        hypothesis=(
            hypothesis
            or f"Iteration {iteration} replaces the pointwise objective with a pairwise one."
        ),
        mechanism=(
            mechanism
            or f"Sample same-user pairs and optimise softplus over the margin, run {iteration}."
        ),
        expected_metric_effects=("GAUC",),
        parent_candidate_id=parent,
        principal_change="Replace the training objective.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                source_field="log_standard:user_id",
                role="grouping",
                purpose="Group impressions within a user.",
            ),
        ),
        objective="pairwise logistic",
        sampling="uniform positive, then uniform same-user negative",
        grouping="user_id",
        weighting="GAUC positive-count weighting",
        causal_cutoff="strictly earlier events only",
        estimated_runtime_seconds=600,
        estimated_memory_mb=4096,
        smoke_plan="Fit on a bounded subset.",
        inner_fold_plan="Score both train-derived folds.",
        falsification_criteria="Reject if fold primary regresses beyond noise.",
        promotion_criteria="Promote only above epsilon on matched seeds.",
        maximum_repairs=1,
        rollback_parent_id=parent,
        attributions=(f"Organizer briefing, direction {iteration}",),
    )


def _package(iteration: int, *, symbols: tuple[str, ...] = ("fit_scores",)) -> GeneratedPackage:
    return GeneratedPackage(
        request_id=f"iteration-{iteration:02d}-implement",
        response_id=f"response-{iteration:02d}",
        files=(GeneratedFile(path="candidate.py", content="def fit_scores():\n    return 0\n"),),
        material_change_summary=f"Rewrote the objective for iteration {iteration}.",
        material_symbols=symbols,
    )


def _write_lineage(
    generated_root: Path,
    iteration: int,
    *,
    candidate_id: str | None = None,
    repair_calls: tuple[object, ...] = (),
    symbols: tuple[str, ...] = ("fit_scores",),
    hypothesis: str | None = None,
    mechanism: str | None = None,
) -> Path:
    generated_root.mkdir(parents=True, exist_ok=True)
    proposal = _proposal(iteration, hypothesis=hypothesis, mechanism=mechanism)
    package = _package(iteration, symbols=symbols)
    payload = {
        "schema_version": 1,
        "provider": "openai",
        "campaign_id": CAMPAIGN,
        "scientific_iteration": iteration,
        "parent_digest": _digest(f"parent-{iteration}"),
        "safe_context_digest": _digest(f"context-{iteration}"),
        "candidate_id": candidate_id or f"candidate-{iteration:02d}",
        "proposal": proposal.to_wire(),
        "implementation_package": package.to_wire(),
        "repair_calls": list(repair_calls),
        "package": package.to_wire(),
    }
    path = generated_root / f"iteration-{iteration:02d}-lineage.json"
    path.write_bytes(canonical_json_bytes(payload))
    return path


def _rejection_record(iteration: int) -> AggregateRecord:
    return AggregateRecord(
        name=f"rejected_lineage_{iteration:02d}",
        values={
            "scientific_iteration": iteration,
            "candidate_id": f"rejected-{iteration:02d}",
            "branch_outcome": "rejected_before_execution",
            "repairs_attempted": 2,
            "diagnostic": "reserved candidate filename is forbidden: 'baseline.py'",
            "root_failure_stage": "materialize",
            "root_failure_category": "static_policy",
            "root_failure_code": "forbidden_basename",
            "root_failure_subject": "baseline.py",
            "root_failure_fingerprint": _digest(f"fingerprint-{iteration}"),
            "root_failure_diagnostic": "reserved candidate filename is forbidden: 'baseline.py'",
        },
    )


def _generated_root(run_dir: Path) -> Path:
    return run_dir / "production" / "generated-source"


def test_every_admitted_iteration_becomes_a_narrative_in_order(tmp_path: Path) -> None:
    """A three-iteration campaign must report three iterations, not one."""

    root = _generated_root(tmp_path)
    for iteration in (3, 1, 2):  # written out of order on purpose
        _write_lineage(root, iteration)

    evidence = collect_iteration_narratives(
        tmp_path,
        campaign_id=CAMPAIGN,
        fallback_parent_id="official-fm-fallback-seed-4",
    )

    assert [item.iteration for item in evidence.narratives] == [1, 2, 3]
    assert evidence.iteration_count == 3
    first = evidence.narratives[0]
    assert "pointwise objective" in first.hypothesis
    assert first.experiment_id == "candidate-01"
    assert first.parent_id == "official-fm-fallback-seed-4"
    assert "Changed top-level symbol: fit_scores" in first.material_changes
    assert any("Organizer briefing" in item for item in first.attributions)


def test_model_authored_text_is_collapsed_to_one_line(tmp_path: Path) -> None:
    """A multi-line hypothesis must not make a finished campaign unfinalizable.

    ``schemas._text`` accepts newlines up to 16,384 characters and ``report._text`` rejects any
    string containing one, so four provider-authored fields cross a boundary that would raise
    during report construction.  Lineage records are written read-only, so the only recovery from
    one bad hypothesis would be editing files by hand after the run.
    """

    _write_lineage(
        _generated_root(tmp_path),
        1,
        hypothesis="Replace the pointwise objective.\n\n  Step 1: sample pairs.\tStep 2: fit.",
        mechanism="Optimise softplus\nover the margin.",
    )

    evidence = collect_iteration_narratives(
        tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="parent"
    )
    narrative = evidence.narratives[0]

    assert narrative.hypothesis == (
        "Replace the pointwise objective. Step 1: sample pairs. Step 2: fit."
    )
    assert "\n" not in narrative.mechanism
    assert "Optimise softplus over the margin." in narrative.mechanism


def test_expected_metric_effects_reach_the_mechanism_text(tmp_path: Path) -> None:
    """A judge grading target selection needs to see which metric the agent aimed at."""

    _write_lineage(_generated_root(tmp_path), 1)
    evidence = collect_iteration_narratives(
        tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="parent"
    )
    assert "Expected to move: GAUC." in evidence.narratives[0].mechanism


def test_repair_round_trips_are_reported_as_recovery_events(tmp_path: Path) -> None:
    """Repairs are the robustness evidence the deliverable asks for; they were being dropped."""

    root = _generated_root(tmp_path)
    _write_lineage(root, 1)
    _write_lineage(root, 2, repair_calls=({"request": {}, "response": {}},))

    evidence = collect_iteration_narratives(
        tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="parent"
    )

    clean, repaired = evidence.narratives
    assert clean.failures_and_recoveries == ()
    assert len(repaired.failures_and_recoveries) == 1
    assert "1 bounded repair" in repaired.failures_and_recoveries[0]
    assert any("Iteration 2" in line for line in evidence.failure_lines)


def test_rejected_branches_are_reported_rather_than_omitted(tmp_path: Path) -> None:
    """A rejected iteration is a real research event; silence would overstate the trajectory."""

    root = _generated_root(tmp_path)
    root.mkdir(parents=True)
    _write_lineage(root, 1)
    _persist_rejection_journal_entry(
        root,
        campaign_id=CAMPAIGN,
        scientific_iteration=1,
        record=_rejection_record(1),
        previous_journal_digest=None,
    )

    evidence = collect_iteration_narratives(
        tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="parent"
    )

    # Iteration 1 has both a lineage record and a journal entry; the richer one must win.
    assert evidence.iteration_count == 1
    assert evidence.narratives[0].experiment_id == "candidate-01"


def test_a_rejection_only_campaign_still_produces_a_trajectory(tmp_path: Path) -> None:
    """This is the shape of a live run whose candidates never pass the static gates."""

    root = _generated_root(tmp_path)
    root.mkdir(parents=True)
    previous: str | None = None
    for iteration in (1, 2):
        previous = _persist_rejection_journal_entry(
            root,
            campaign_id=CAMPAIGN,
            scientific_iteration=iteration,
            record=_rejection_record(iteration),
            previous_journal_digest=previous,
        )

    evidence = collect_iteration_narratives(
        tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="official-fm"
    )

    assert evidence.iteration_count == 2
    narrative = evidence.narratives[0]
    assert narrative.status == "rejected_before_execution"
    assert narrative.material_changes == (
        "No material executable change; the branch was rejected before evaluation.",
    )
    assert "static_policy" in narrative.mechanism
    assert "baseline.py" in narrative.failures_and_recoveries[0]
    assert len(evidence.failure_lines) == 2


def test_a_run_without_research_records_yields_empty_evidence(tmp_path: Path) -> None:
    """The caller falls back to its single literal narrative; it must not fail finalization."""

    evidence = collect_iteration_narratives(
        tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="parent"
    )
    assert evidence.narratives == ()
    assert evidence.failure_lines == ()
    assert count_recorded_iterations(tmp_path) == 0


def test_a_corrupt_lineage_record_fails_closed(tmp_path: Path) -> None:
    """A silently skipped record would understate the trajectory; that is worse than an error."""

    root = _generated_root(tmp_path)
    root.mkdir(parents=True)
    (root / "iteration-01-lineage.json").write_text(
        json.dumps({"schema_version": 1, "scientific_iteration": 1, "candidate_id": "c"}),
        encoding="utf-8",
    )

    with pytest.raises(IterationEvidenceError, match="proposal and package"):
        collect_iteration_narratives(tmp_path, campaign_id=CAMPAIGN, fallback_parent_id="parent")


def test_counting_iterations_does_not_require_parsing(tmp_path: Path) -> None:
    """`_bundle_metadata` needs only the count, so it must not repeat the strict parse."""

    root = _generated_root(tmp_path)
    root.mkdir(parents=True)
    _write_lineage(root, 1)
    _write_lineage(root, 2)
    _persist_rejection_journal_entry(
        root,
        campaign_id=CAMPAIGN,
        scientific_iteration=3,
        record=_rejection_record(3),
        previous_journal_digest=None,
    )

    assert count_recorded_iterations(tmp_path) == 3

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuairand_agent.campaign.store import LineageFoldMetrics, ResearchLineageLedger
from kuairand_agent.finalization.iteration_log import (
    IterationLogError,
    build_iteration_log,
    render_jsonl,
    render_markdown,
)

_CAMPAIGN = "kuairand-iteration-log-fixture"


def _digest(character: str) -> str:
    return character * 64


def _write_lineage(
    generated_root: Path,
    *,
    iteration: int,
    candidate_id: str,
    parent_candidate_id: str,
    repair_calls: int = 0,
) -> None:
    document = {
        "schema_version": 1,
        "campaign_id": _CAMPAIGN,
        "candidate_id": candidate_id,
        "scientific_iteration": iteration,
        "provider": "openai",
        "repair_calls": [{"request": {}, "response": {}} for _ in range(repair_calls)],
        "proposal": {
            "parent_candidate_id": parent_candidate_id,
            "hypothesis": f"Hypothesis for iteration {iteration}.",
            "mechanism": f"Mechanism for iteration {iteration}.",
            "objective": f"Objective for iteration {iteration}.",
            "principal_change": f"Principal change for iteration {iteration}.",
            "expected_metric_effects": ["GAUC", "nDCG@5"],
            "attributions": ["A cited source."],
        },
        "package": {"files": [{"path": "model_impl.py"}]},
    }
    (generated_root / f"iteration-{iteration:02d}-lineage.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


@pytest.fixture
def campaign(tmp_path: Path) -> tuple[Path, Path]:
    """A run directory holding two iterations, the second parented on the first."""

    run_dir = tmp_path / "runs" / "fixture-run"
    generated_root = run_dir / "production" / "generated-source"
    generated_root.mkdir(parents=True)

    seed = tmp_path / "candidate_seed"
    seed.mkdir()
    (seed / "model_impl.py").write_text("value = 1\n", encoding="utf-8")

    first = generated_root / "candidate-01-aaaa"
    first.mkdir()
    (first / "model_impl.py").write_text("value = 2\n", encoding="utf-8")
    second = generated_root / "candidate-02-bbbb"
    second.mkdir()
    (second / "model_impl.py").write_text("value = 3\n", encoding="utf-8")

    _write_lineage(
        generated_root,
        iteration=1,
        candidate_id="candidate-01-aaaa",
        parent_candidate_id="official-fm-fallback-seed-4",
    )
    _write_lineage(
        generated_root,
        iteration=2,
        candidate_id="candidate-02-bbbb",
        parent_candidate_id="candidate-01-aaaa",
        repair_calls=1,
    )
    return run_dir, tmp_path


def _record_admission(run_dir: Path, *, candidate_id: str, promoted: bool | None) -> None:
    ledger_path = run_dir.parent / "research-lineage-ledger.sqlite3"
    ledger = (
        ResearchLineageLedger.open(ledger_path)
        if ledger_path.exists()
        else ResearchLineageLedger.create(ledger_path)
    )
    try:
        ledger.record_admission(
            campaign_id=_CAMPAIGN,
            benchmark_digest=_digest("1"),
            starter_digest=_digest("2"),
            source_digest=_digest("3"),
            candidate_id=candidate_id,
            proposal_family="pairwise",
            proposal_signature=_digest("4"),
            inner_fold_a=(
                LineageFoldMetrics(gauc=0.66, ndcg_at_5=0.55, primary=0.605)
                if promoted is not None
                else None
            ),
            inner_fold_b=LineageFoldMetrics(gauc=0.65, ndcg_at_5=0.49, primary=0.57),
            parent_fold_a_primary=0.60 if promoted is not None else None,
            parent_fold_b_primary=0.575,
            promoted=promoted,
        )
    finally:
        ledger.close()


def test_iteration_log_carries_every_required_field(campaign: tuple[Path, Path]) -> None:
    run_dir, project_root = campaign
    _record_admission(run_dir, candidate_id="candidate-01-aaaa", promoted=True)

    entries = build_iteration_log(run_dir, project_root=project_root)

    assert [entry.iteration for entry in entries] == [1, 2]
    first, second = entries
    # Hypothesis.
    assert first.hypothesis == "Hypothesis for iteration 1."
    assert first.attributions == ("A cited source.",)
    # Code diff, against the organizer seed for the first iteration.
    assert first.diffs[0].parent_label == "candidate_seed"
    assert "-value = 1" in first.diffs[0].diff
    assert "+value = 2" in first.diffs[0].diff
    # Resulting metrics, including the delta against the candidate's real parent.
    assert first.metrics["fold_a_primary"] == pytest.approx(0.605)
    assert first.metrics["fold_a_delta"] == pytest.approx(0.005)
    assert first.metrics["promoted"] is True
    # Error and recovery events.
    assert first.events == ()
    assert second.repairs_attempted == 1
    assert any("bounded repair loop" in item for item in second.events)


def test_iteration_log_diffs_a_followup_against_its_real_parent(
    campaign: tuple[Path, Path],
) -> None:
    """Iteration 2 is parented on iteration 1's tree, not on the organizer seed."""

    run_dir, project_root = campaign

    second = build_iteration_log(run_dir, project_root=project_root)[1]

    assert second.diffs[0].parent_label == "candidate-01-aaaa"
    assert "-value = 2" in second.diffs[0].diff
    assert "+value = 3" in second.diffs[0].diff


def test_iteration_log_reports_a_screen_rejected_candidate_without_fold_a(
    campaign: tuple[Path, Path],
) -> None:
    run_dir, project_root = campaign
    _record_admission(run_dir, candidate_id="candidate-02-bbbb", promoted=None)

    second = build_iteration_log(run_dir, project_root=project_root)[1]

    assert "fold_a_primary" not in second.metrics
    assert second.metrics["fold_b_primary"] == pytest.approx(0.57)
    assert second.metrics["promoted"] is None


def test_iteration_log_renders_without_any_recorded_metrics(
    campaign: tuple[Path, Path],
) -> None:
    """A campaign whose ledger is absent still produces the other three required fields."""

    run_dir, project_root = campaign

    entries = build_iteration_log(run_dir, project_root=project_root)

    assert all(entry.metrics == {} for entry in entries)
    markdown = render_markdown(entries)
    assert "No inner-fold metrics were recorded" in markdown
    assert "### Hypothesis" in markdown
    assert "### Code diff" in markdown
    assert "### Errors and recovery" in markdown


def test_iteration_log_jsonl_is_one_canonical_object_per_iteration(
    campaign: tuple[Path, Path],
) -> None:
    run_dir, project_root = campaign

    lines = render_jsonl(build_iteration_log(run_dir, project_root=project_root)).splitlines()

    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert [item["iteration"] for item in decoded] == [1, 2]
    assert all(item["schema_version"] == 1 for item in decoded)
    assert all({"hypothesis", "code_diff", "metrics", "events"} <= set(item) for item in decoded)


def test_iteration_log_refuses_a_run_directory_without_lineage(tmp_path: Path) -> None:
    (tmp_path / "production" / "generated-source").mkdir(parents=True)

    with pytest.raises(IterationLogError, match="no scientific-iteration lineage"):
        build_iteration_log(tmp_path, project_root=tmp_path)


def _write_scientific_result(run_dir: Path, outcomes: list[tuple[str, str, str]]) -> None:
    support = run_dir / "production" / "finalization-support"
    support.mkdir(parents=True, exist_ok=True)
    (support / "experiments.jsonl").write_text(
        json.dumps(
            {
                "record_type": "scientific_result",
                "record_id": _digest("e"),
                "evidence": {
                    "candidate_outcomes": [
                        {"candidate_id": candidate, "outcome": outcome, "reason": reason}
                        for candidate, outcome, reason in outcomes
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_iteration_log_surfaces_an_execution_failure_and_its_recovery(
    campaign: tuple[Path, Path],
) -> None:
    """A candidate can train and still produce no metrics; that is an error the log must report.

    Reproduces `runs/postfix-20260830T105450Z`, where candidate-01 hit
    callback_failed:CandidateExecutionError. Inferring "no errors" from absent metrics would have
    hidden the single clearest piece of recovery evidence the campaign produced.
    """

    run_dir, project_root = campaign
    _write_scientific_result(
        run_dir,
        [
            ("candidate-01-aaaa", "callback_failed", "callback_failed:CandidateExecutionError"),
            ("candidate-02-bbbb", "screen_rejected", "fold_b_screen_failed"),
        ],
    )

    first, second = build_iteration_log(run_dir, project_root=project_root)

    assert any("callback_failed" in item for item in first.events)
    assert any("recovered from this failure" in item for item in first.events)
    assert any("screen_rejected" in item for item in second.events)
    # The final iteration must not claim a recovery that never happened.
    assert not any("recovered from this failure" in item for item in second.events)


def test_iteration_log_claims_no_recovery_when_the_failure_ended_the_campaign(
    campaign: tuple[Path, Path],
) -> None:
    run_dir, project_root = campaign
    _write_scientific_result(
        run_dir,
        [("candidate-02-bbbb", "callback_failed", "callback_failed:CandidateExecutionError")],
    )

    last = build_iteration_log(run_dir, project_root=project_root)[-1]

    assert any("callback_failed" in item for item in last.events)
    assert not any("recovered from this failure" in item for item in last.events)

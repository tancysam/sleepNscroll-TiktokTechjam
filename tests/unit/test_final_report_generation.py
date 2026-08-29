from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from kuairand_agent.finalization.organizer_check import (
    MaskedFileEvidence,
    MaskedViewEvidence,
    OrganizerCheckEvidence,
)
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    FinalReplayEvidence,
    FrozenReplayIdentity,
    ReplayEquality,
    ValidationReplayEvidence,
)
from kuairand_agent.finalization.report import (
    ExperimentNarrative,
    FinalReportContext,
    MetricEvidence,
    ReproduceInstructions,
    ResourceEvidence,
    render_final_report,
    render_reproduce_script,
    write_final_report,
    write_reproduce_script,
)


def _replay() -> CleanReplayEvidence:
    identity = FrozenReplayIdentity(
        source_sha256="1" * 64,
        config_sha256="2" * 64,
        features_sha256="3" * 64,
        checkpoint_sha256="4" * 64,
        validation_prediction_artifact_sha256="5" * 64,
        validation_prediction_digest="6" * 64,
        data_sha256="7" * 64,
        environment_sha256="8" * 64,
    )
    validation = ValidationReplayEvidence(
        row_count=3,
        reference_prediction_digest="6" * 64,
        replay_prediction_digest="6" * 64,
        replay_prediction_file_sha256="9" * 64,
        exact_prediction_bytes=True,
        maximum_absolute_difference=0.0,
        top5_order_identical=True,
        protected_metrics_identical=True,
        metrics={"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
        public_submission_sha256="a" * 64,
        public_submission_prediction_digest="6" * 64,
        csv_round_trip_identity=True,
        csv_within_user_order_preserved=True,
        csv_top5_preserved=True,
        csv_protected_metrics_preserved=True,
    )
    final = FinalReplayEvidence(
        row_count=2,
        prediction_digest="b" * 64,
        prediction_file_sha256="c" * 64,
        submission_sha256="d" * 64,
        submission_prediction_digest="b" * 64,
        finite_scores=True,
        csv_round_trip_identity=True,
    )
    return CleanReplayEvidence(
        candidate_id="pairwise-confirmed",
        identity=identity,
        equality=ReplayEquality.EXACT,
        absolute_tolerance=0.0,
        training_replay="from_scratch_and_checkpoint",
        validation=validation,
        final=final,
        validation_capability_digest="e" * 64,
        final_capability_digest="f" * 64,
    )


def _check() -> OrganizerCheckEvidence:
    masked = MaskedViewEvidence(
        files=(MaskedFileEvidence("late.csv", "0" * 64, 12, 2, 2),),
        final_rows_masked=2,
        final_outcome_cells_replaced=22,
        digest="1" * 64,
    )
    return OrganizerCheckEvidence(
        starter_manifest_sha256="2" * 64,
        submission_sha256="d" * 64,
        submission_size_bytes=100,
        masked_view=masked,
        checker_command=("python", "submit.py", "--split", "test", "--check"),
        checker_returncode=0,
        checker_stdout="ok\n",
        checker_stderr="",
        checker_stdout_sha256=hashlib.sha256(b"ok\n").hexdigest(),
        checker_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )


def _context() -> FinalReportContext:
    return FinalReportContext(
        benchmark_contract={
            "dataset": "KuaiRand-Pure",
            "target": "long_view",
            "primary": "mean(GAUC,nDCG@5)",
        },
        baselines=(
            MetricEvidence("official FM mean", "outer validation", 0.6674, 0.5357, 0.60155),
        ),
        selected=MetricEvidence(
            "pairwise confirmed",
            "outer validation matched seeds",
            0.67,
            0.54,
            0.605,
            seeds=(0, 1, 2),
        ),
        experiments=(
            ExperimentNarrative(
                iteration=1,
                experiment_id="pairwise-screen",
                parent_id="official-fm",
                hypothesis="User-balanced pairs align training with GAUC.",
                mechanism="Sample users by eligible pair mass.",
                material_changes=("candidate.py: added pairwise loss",),
                attributions=("organizer evaluator semantics",),
                status="promoted after matched-seed confirmation",
                inner_primary=0.604,
                outer_primary=0.605,
            ),
        ),
        inner_fold_evidence=("Fold A primary 0.603", "Fold B primary 0.604"),
        seed_confirmation=("Seeds 0, 1, and 2 used for incumbent and challenger.",),
        failures_and_recoveries=("One syntax-invalid child was repaired without incumbent loss.",),
        leakage_controls=(
            "Final inputs are built through the FINAL capability without targets.",
            "Protected validation labels remain inside the trusted scorer.",
        ),
        test_evidence=("Focused replay and submission tests passed.",),
        selection_rationale="Highest eligible confirmed primary under the frozen selector.",
        resources=ResourceEvidence(
            wall_seconds=123.5,
            peak_rss_bytes=1_000_000,
            launch_count=12,
            intervention_count=0,
            provider_usage="No provider call during frozen replay.",
            device_usage=("CPU qualification and replay", "MPS optional candidate screening"),
        ),
        known_limitations=("From-scratch replay was performed only on the selected seed.",),
    )


def test_report_is_deterministic_complete_and_honest(tmp_path: Path) -> None:
    rendered = render_final_report(_context(), replay=_replay(), organizer_check=_check())

    assert rendered == render_final_report(_context(), replay=_replay(), organizer_check=_check())
    for heading in (
        "Benchmark contract",
        "Baseline parity",
        "Experiment trajectory and candidate tree",
        "Inner folds, outer validation, and seed confirmation",
        "Failures, repairs, recoveries, and interventions",
        "Leakage controls and tests",
        "Replay and submission verification",
        "Limitations and hidden-test status",
    ):
        assert f"## {heading}" in rendered
    assert "Hidden-test improvement is unverified until organizer scoring." in rendered
    assert "final outcomes were neither accessed nor scored" in rendered
    assert "0.67" in rendered and "0.54" in rendered and "0.605" in rendered

    path = write_final_report(tmp_path / "report.md", _context(), _replay(), _check())
    assert path.read_text(encoding="utf-8") == rendered


def test_reproduce_script_is_frozen_explicit_and_never_calls_an_llm(tmp_path: Path) -> None:
    instructions = ReproduceInstructions(
        expected_data_sha256="7" * 64,
        replay_subcommand="replay",
        dependency_groups=("research-tree", "research-neural"),
    )
    script = render_reproduce_script(instructions)

    assert "uv sync --locked --group research-tree --group research-neural" in script
    assert "uv run --locked --no-sync kuairand-agent replay" in script
    assert '--bundle "$BUNDLE_DIR"' in script
    assert '--project-root "$REPO_DIR"' in script
    assert "--bundle-dir" not in script
    assert '"$BUNDLE_DIR"' in script
    assert '"$DATA_DIR"' in script
    assert "OPENAI" not in script
    assert "provider" not in script.lower()
    assert "LLM" not in script

    path = write_reproduce_script(tmp_path / "reproduce.sh", instructions)
    assert path.stat().st_mode & 0o111
    assert path.read_text(encoding="utf-8") == script


def test_report_renders_every_iteration_and_its_recovery_events() -> None:
    """A multi-iteration campaign must render as a multi-iteration trajectory.

    The finalizer previously emitted one hardcoded narrative regardless of how many iterations
    ran, so the per-iteration run log -- a required submission deliverable -- could not be
    produced at all.
    """

    context = replace(
        _context(),
        experiments=(
            ExperimentNarrative(
                iteration=1,
                experiment_id="candidate-01",
                parent_id="official-fm",
                hypothesis="Pairwise sampling aligns training with GAUC.",
                mechanism="Sample same-user pairs.",
                material_changes=("Changed top-level symbol: fit_scores",),
                attributions=("organizer briefing",),
                status="rejected",
            ),
            ExperimentNarrative(
                iteration=2,
                experiment_id="candidate-02",
                parent_id="candidate-01",
                hypothesis="Listwise softmax over the user's impressions.",
                mechanism="Softmax cross-entropy per user.",
                material_changes=("Changed top-level symbol: fit_scores",),
                attributions=("organizer briefing",),
                status="promoted",
                outer_primary=0.605,
                failures_and_recoveries=(
                    "Iteration 2 (candidate-02): 1 bounded repair round-trip was required.",
                ),
            ),
        ),
    )

    rendered = render_final_report(context, replay=_replay(), organizer_check=_check())

    assert "### Iteration 1: candidate-01" in rendered
    assert "### Iteration 2: candidate-02" in rendered
    assert "Listwise softmax" in rendered
    # The recovery block appears only for the iteration that actually recovered.
    assert rendered.count("Errors and recoveries:") == 1
    assert "1 bounded repair round-trip was required" in rendered

from __future__ import annotations

from pathlib import Path

from kuairand_agent.contract import verify_starter_kit
from kuairand_agent.data.audit import DataAuditReport
from kuairand_agent.data.canonical import CanonicalDataset
from tests.support.scripted_campaign import run_scripted_three_iteration_acceptance

ROOT = Path(__file__).parents[2]


def test_verified_data_gate_includes_three_iteration_failure_and_recovery_campaign(
    tmp_path: Path,
    official_audit: DataAuditReport,
    official_dataset: CanonicalDataset,
) -> None:
    """The full-data gate carries the complete scripted autonomous controller acceptance.

    The generated children intentionally use a four-row input-only scorer fixture: robustness
    failures must never be manufactured inside a scored official-data run.  Requiring the official
    audit fixture here proves this controller gate is exercised in the same qualified acceptance
    invocation as the real-data model tests, while the fixture independently verifies three
    scientific iterations, one syntax repair, one failed material runtime child, protected
    scoring, reflection, durable failure records, and incumbent preservation.
    """

    assert official_audit.final_outcome_trace.manifest()["outcome_cells_scored"] == 0
    starter = verify_starter_kit(ROOT / "kuairand-starter-kit")
    evidence = run_scripted_three_iteration_acceptance(
        tmp_path / "scripted-three-iteration",
        starter_dir=ROOT / "kuairand-starter-kit",
        verified_audit_digest=official_audit.digest,
        verified_starter_digest=starter.manifest_sha256,
        verified_dataset_digest=official_dataset.digest,
    )
    assert evidence.statuses == ("promoted", "promoted", "failed")
    assert evidence.selected_candidate_id == "iteration-02-repair-1"
    assert evidence.fallback_candidate_id == "fm-fallback"
    assert evidence.repairs == (0, 1, 0)
    assert evidence.trusted_evaluation_present == (True, True, False)
    assert evidence.evaluator_calls == 2
    assert evidence.model_responses_remaining == 0
    assert evidence.incumbent_id == "iteration-02-repair-1"
    assert evidence.incumbent_primary == 1.0
    assert evidence.launches_used == 3
    assert evidence.completed_iterations == 3
    assert evidence.best_primary == 1.0
    assert {"propose", "implement", "repair", "run", "evaluate", "reflect"} <= (evidence.operations)
    assert {
        "proposal_transcript",
        "implementation_transcript",
        "repair_transcript",
        "reflection_transcript",
        "source_diff",
        "source_manifest",
    } <= evidence.artifact_roles
    assert {"syntax_error", "runtime_error"} <= evidence.failure_categories
    assert all(0.0 <= value <= 1.0 for row in evidence.stored_metrics for value in row)
    assert evidence.workspaces_cleaned
    assert evidence.execution_count == 3
    assert evidence.environment_receipts_complete
    assert evidence.parent_source_unchanged
    assert evidence.campaign_benchmark_digest == official_audit.digest
    assert evidence.campaign_starter_digest == starter.manifest_sha256
    assert evidence.campaign_dataset_digest == official_dataset.digest

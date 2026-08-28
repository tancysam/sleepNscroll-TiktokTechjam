from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.data.audit import DataAuditReport
from kuairand_agent.data.canonical import CanonicalDataset

ROOT = Path(__file__).parents[2]
RUN_ENV = "KUAIRAND_COMPLETED_RUN_DIR"
BUNDLE_ENV = "KUAIRAND_FINAL_BUNDLE_DIR"

pytestmark = [
    pytest.mark.skipif(
        RUN_ENV not in os.environ,
        reason=f"set {RUN_ENV} to run completed production acceptance",
    ),
    pytest.mark.skipif(
        BUNDLE_ENV not in os.environ,
        reason=f"set {BUNDLE_ENV} to run closed-bundle replay acceptance",
    ),
]


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name} to run completed production acceptance")
    try:
        path = Path(value).resolve(strict=True)
    except OSError as exc:
        pytest.skip(f"{name} does not resolve to an existing path: {exc}")
    return path


def test_completed_campaign_and_closed_bundle_replay_are_exact_and_label_free(
    official_data_dir: Path,
    official_dataset: CanonicalDataset,
    official_audit: DataAuditReport,
) -> None:
    # Import late so the default suite can collect this independently of optional tree tooling and
    # an in-progress production implementation. The public functions remain the acceptance seam.
    from kuairand_agent.finalization.production import (
        finalize_provider_free_campaign,
        replay_final_bundle,
    )

    run_dir = _required_path(RUN_ENV)
    bundle_dir = _required_path(BUNDLE_ENV)
    retained_outcome = run_dir / "production" / "finalization" / "outcome.json"
    if not retained_outcome.is_file():
        pytest.skip(f"{RUN_ENV} is not completed: missing production/finalization/outcome.json")
    outcome_bytes = retained_outcome.read_bytes()
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()

    completed = finalize_provider_free_campaign(run_dir, project_root=ROOT)
    assert completed.run_dir == run_dir
    assert completed.bundle_root == bundle_dir
    assert completed.campaign_status == "COMPLETED"
    assert completed.campaign_revision > 0
    assert completed.fallback_count == len(completed.failures)
    assert completed.bundle_manifest_sha256
    assert completed.submission_sha256
    assert retained_outcome.read_bytes() == outcome_bytes
    assert (bundle_dir / "manifest.json").read_bytes() == manifest_bytes

    replay = replay_final_bundle(
        bundle_dir,
        official_data_dir,
        official_dataset.digest,
        project_root=ROOT,
    )
    replay_manifest = replay.manifest()
    assert replay.bundle_manifest_sha256 == completed.bundle_manifest_sha256
    assert replay.bundled_submission_sha256 == completed.submission_sha256
    assert replay.reproduced_submission_sha256 == completed.submission_sha256
    assert replay_manifest["submission_bytes_identical"] is True
    assert replay_manifest["provider_used"] is False
    assert replay_manifest["network_used"] is False
    assert replay_manifest["final_outcomes_accessed"] is False
    assert replay_manifest["final_outcomes_scored"] is False
    assert official_dataset.final.targets is None
    assert official_dataset.final.outcome_trace.parsed_cell_count == 0
    assert official_audit.final_outcome_trace.manifest()["outcome_cells_scored"] == 0

    bundle_manifest = cast(
        dict[str, object],
        json.loads((bundle_dir / "manifest.json").read_text(encoding="ascii")),
    )
    data_identity = cast(dict[str, object], bundle_manifest["data_identity"])
    assert data_identity["canonical_digest"] == official_dataset.digest
    assert data_identity["final_outcomes_accessed"] is False
    assert data_identity["final_outcomes_scored"] is False
    selection = cast(dict[str, object], bundle_manifest["selection"])
    assert selection["selected_experiment"] == completed.selected_candidate_id
    assert selection["status"] in {
        "baseline_reproduced",
        "validation_improved",
        "materially_confirmed",
    }
    assert stat.S_IMODE(bundle_dir.stat().st_mode) & 0o222 == 0

from __future__ import annotations

import hashlib
import json
import os
import statistics
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.performance import PerformanceProfile
from tests.support.completed_run_resources import (
    CAMPAIGN_LIMIT_SECONDS,
    FINALIZATION_COVERAGE,
    FINALIZATION_RESERVE_SECONDS,
    validate_completed_finalization_resources,
)

RUN_ENV = "KUAIRAND_COMPLETED_RUN_DIR"
BUNDLE_ENV = "KUAIRAND_FINAL_BUNDLE_DIR"

pytestmark = [
    pytest.mark.skipif(
        RUN_ENV not in os.environ,
        reason=f"set {RUN_ENV} to the completed production campaign run",
    ),
    pytest.mark.skipif(
        BUNDLE_ENV not in os.environ,
        reason=f"set {BUNDLE_ENV} to the closed production bundle",
    ),
]


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AssertionError(f"{name} must be a real number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AssertionError(f"{name} must be an integer")
    return int(value)


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name} to run completed production resource acceptance")
    try:
        path = Path(value).resolve(strict=True)
    except OSError as exc:
        pytest.skip(f"{name} does not resolve to an existing path: {exc}")
    if not path.is_dir():
        pytest.skip(f"{name} is not a directory: {path}")
    return path


def _component_digest(manifest: dict[str, object], relative_path: str) -> str:
    components = cast(dict[str, object], manifest["components"])
    files = cast(list[object], components["files"])
    matches = [
        cast(dict[str, object], item)
        for item in files
        if isinstance(item, dict) and item.get("path") == relative_path
    ]
    assert len(matches) == 1
    return _digest(matches[0]["sha256"], f"component {relative_path} SHA-256")


def test_completed_run_uses_actual_terminal_resource_receipts_and_fits_both_limits() -> None:
    # Import late: the default suite must still collect and skip before a production run exists.
    from kuairand_agent.campaign.controller import CampaignEngine
    from kuairand_agent.finalization.production import ProductionFinalizationOutcome

    run_dir = _required_path(RUN_ENV)
    bundle = _required_path(BUNDLE_ENV)
    assert bundle == run_dir / "final"
    outcome_path = run_dir / "production" / "finalization" / "outcome.json"
    if not outcome_path.is_file():
        pytest.fail(f"completed run lacks terminal production receipt: {outcome_path}")
    outcome = ProductionFinalizationOutcome.from_bytes(outcome_path.read_bytes())
    assert outcome.run_dir == run_dir
    assert outcome.bundle_root == bundle

    manifest_path = bundle / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == outcome.bundle_manifest_sha256
    manifest = cast(dict[str, object], json.loads(manifest_bytes.decode("ascii")))
    selection = cast(dict[str, object], manifest["selection"])
    status = selection["status"]
    generated = status != "baseline_reproduced"
    assert status == outcome.selected_status.value

    replay_path = bundle / "replay" / "evidence.json"
    verification_path = bundle / "verification.json"
    submission_path = bundle / "submission.csv"
    final_predictions = bundle / "replay" / "final-predictions.npy"
    assert _sha256(replay_path) == outcome.replay_evidence_sha256
    assert _sha256(verification_path) == outcome.organizer_verification_sha256
    assert _sha256(submission_path) == outcome.submission_sha256
    assert _component_digest(manifest, "replay/evidence.json") == outcome.replay_evidence_sha256
    assert _component_digest(manifest, "verification.json") == outcome.organizer_verification_sha256
    assert _component_digest(manifest, "submission.csv") == outcome.submission_sha256

    replay = cast(
        dict[str, object],
        json.loads(replay_path.read_text(encoding="ascii")),
    )
    final = cast(dict[str, object], replay["final"])
    assert _integer(final["row_count"], "final replay row_count") == 170_588
    assert _sha256(final_predictions) == _digest(
        final["prediction_file_sha256"],
        "final prediction file SHA-256",
    )
    assert _component_digest(manifest, "replay/final-predictions.npy") == _sha256(final_predictions)
    assert final["prediction_digest"] == final["submission_prediction_digest"]
    assert final["submission_sha256"] == outcome.submission_sha256
    assert final["final_outcomes_accessed"] is False
    assert final["final_outcomes_scored"] is False

    verification = cast(
        dict[str, object],
        json.loads(verification_path.read_text(encoding="ascii")),
    )
    organizer = cast(dict[str, object], verification["organizer_check"])
    organizer_submission = cast(dict[str, object], organizer["submission"])
    assert organizer["mode"] == "check_only"
    assert organizer["returncode"] == 0
    assert organizer_submission["sha256"] == outcome.submission_sha256

    resources = validate_completed_finalization_resources(
        outcome.resource_evidence,
        generated_selection=generated,
        expected_bundle_manifest_sha256=outcome.bundle_manifest_sha256,
        training_replay=outcome.training_replay,
    )
    assert resources.aggregate.rows == 170_588

    engine = CampaignEngine()
    request = engine.load_request(run_dir)
    assert request.config.benchmark.wall_clock_seconds == int(CAMPAIGN_LIMIT_SECONDS)
    assert request.config.runner.finalization_reserve_seconds == int(FINALIZATION_RESERVE_SECONDS)
    deadline_files = sorted((run_dir / "controller" / "deadline").glob("checkpoint-*.json"))
    assert deadline_files
    terminal_deadline = cast(
        dict[str, object],
        json.loads(deadline_files[-1].read_text(encoding="ascii")),
    )
    deadline_state = cast(dict[str, object], terminal_deadline["state"])
    assert (
        _real(
            deadline_state["last_elapsed_seconds"],
            "terminal signed deadline elapsed_seconds",
        )
        == resources.campaign_elapsed_seconds
    )

    environment = cast(dict[str, object], manifest["environment_and_resource_usage"])
    advertised = cast(dict[str, object], environment["production_resource_receipt"])
    assert advertised == {
        "schema_version": 1,
        "terminal_receipt": "production/finalization/outcome.json",
        "aggregate_evidence_binding": "closed_bundle_manifest_sha256",
        "coverage": list(FINALIZATION_COVERAGE),
    }
    totals = cast(dict[str, object], manifest["campaign_totals"])
    assert _real(totals["elapsed_seconds"], "pre-finalization elapsed_seconds") <= (
        resources.campaign_elapsed_seconds
    )

    candidate_wall_seconds: list[float] = []
    candidate_peak_rss_bytes: list[int] = []
    if generated:
        validation = cast(dict[str, object], manifest["validation"])
        seed_summary = cast(dict[str, object], validation["seed_summary"])
        per_seed = cast(list[object], seed_summary["per_seed"])
        assert len(per_seed) == 3
        for raw in per_seed:
            row = cast(dict[str, object], raw)
            seed_resources = cast(dict[str, object], row["candidate_resources"])
            candidate_wall_seconds.append(
                _real(seed_resources["wall_seconds"], "candidate wall_seconds")
            )
            candidate_peak_rss_bytes.append(
                _integer(
                    seed_resources["peak_rss_bytes"],
                    "candidate peak_rss_bytes",
                )
            )
            _digest(row["checkpoint_sha256"], "candidate checkpoint_sha256")
            assert _integer(seed_resources["disk_bytes"], "candidate disk_bytes") > 0
            assert seed_resources["device"] == "cpu"
        assert resources.final_training is not None
        assert environment["seed_resource_deltas_in_seed_summary"] is True
        assert _integer(
            environment["retained_training_peak_rss_bytes"],
            "retained_training_peak_rss_bytes",
        ) == max(candidate_peak_rss_bytes)
        ordered = sorted(candidate_wall_seconds)
        assert len(ordered) == 3
        candidate_p50 = float(statistics.median(ordered))
        candidate_p95 = ordered[-1]
        assert 0.0 < candidate_p50 <= candidate_p95 < FINALIZATION_RESERVE_SECONDS
    else:
        assert resources.final_training is None

    profile = PerformanceProfile.create(
        receipts=(resources.aggregate,),
        controller_overhead_seconds=0.0,
        model_runtime_seconds=resources.aggregate.wall_seconds,
        finalization_reserve_seconds=FINALIZATION_RESERVE_SECONDS,
        finalization_families=("production_finalization_total",),
        projected_campaign_seconds=resources.campaign_elapsed_seconds,
        campaign_limit_seconds=CAMPAIGN_LIMIT_SECONDS,
    )
    assert profile.finalization_p95_seconds == resources.finalization_elapsed_seconds
    assert profile.finalization_reserve_sufficient
    assert profile.campaign_time_sufficient

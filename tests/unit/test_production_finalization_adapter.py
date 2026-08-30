from __future__ import annotations

import hashlib
import json
import os
import stat
import statistics
import threading
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

import kuairand_agent.finalization.production as production
from kuairand_agent.campaign.budgets import LaunchCategory
from kuairand_agent.campaign.candidate_journal import CandidateJournalPolicy
from kuairand_agent.campaign.controller import CampaignCreateRequest, CampaignEngine
from kuairand_agent.campaign.full_campaign import (
    FinalizationSelectionPlan,
    FullCampaignStage,
)
from kuairand_agent.campaign.models import CampaignState
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.candidate_executor import GeneratedTrainRequest
from kuairand_agent.execution.policy import SplitRole
from kuairand_agent.finalization.finalize import FinalizationCancelledError
from kuairand_agent.finalization.production import (
    ProductionFinalizationError,
    ProductionFinalizationOutcome,
)
from kuairand_agent.finalization.recipe import GeneratedLambdaRankReplayRecipe
from kuairand_agent.finalization.report import MetricEvidence
from kuairand_agent.finalization.submission_bundle import (
    FINAL_BUNDLE_SCHEMA_VERSION,
    REQUIRED_DIRECTORY_PATHS,
    REQUIRED_FILE_PATHS,
    FinalStatus,
)


def _digest(character: str) -> str:
    return character * 64


def _metrics(gauc: float, ndcg: float) -> dict[str, float]:
    return {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2.0}


def _fallback_resource_evidence(
    bundle_digest: str,
    *,
    hard_wall_seconds: int = 21_600,
    finalization_reserve_seconds: int = 3_600,
) -> MappingProxyType[str, object]:
    return MappingProxyType(
        {
            "schema_version": 1,
            "clock_basis": "durable_max_of_monotonic_and_utc_elapsed",
            "campaign_elapsed_seconds": 20.0,
            "hard_wall_seconds": hard_wall_seconds,
            "finalization_reserve_seconds": finalization_reserve_seconds,
            "finalization_started_elapsed_seconds": 10.0,
            "finalization_elapsed_seconds": 10.0,
            "coverage": list(production._FINALIZATION_COVERAGE),
            "within_reserve": True,
            "within_hard_wall": True,
            "aggregate": {
                "family": "production_finalization_total",
                "wall_seconds": 10.0,
                "local_monotonic_wall_seconds": 9.0,
                "cpu_seconds": 1.0,
                "peak_rss_bytes": 1024,
                "rows": 10,
                "evidence_digest": bundle_digest,
                "rss_accounting": (
                    "conservative_sum_of_process_lifetime_self_and_child_high_water_marks"
                ),
            },
        }
    )


def _generated_manifest(delta: float) -> dict[str, object]:
    candidate_rows: list[dict[str, float]] = []
    fm_rows: list[dict[str, float]] = []
    records: list[dict[str, object]] = []
    primary_deltas: list[Decimal] = []
    for seed in (0, 1, 2):
        fm = _metrics(0.6 + seed * 0.001, 0.4 + seed * 0.001)
        candidate = _metrics(fm["GAUC"] + delta, fm["nDCG@5"] + delta)
        deltas = {name: candidate[name] - fm[name] for name in ("GAUC", "nDCG@5", "primary")}
        primary_deltas.append(Decimal(str(deltas["primary"])))
        candidate_rows.append(candidate)
        fm_rows.append(fm)
        records.append(
            {
                "seed": seed,
                "candidate_metrics": candidate,
                "official_fm_metrics": fm,
                "paired_deltas": deltas,
                "candidate_resources": {
                    "wall_seconds": 2.0 + seed,
                    "peak_rss_bytes": 2000 + seed,
                    "disk_bytes": 100 + seed,
                    "device": "cpu",
                },
                "official_fm_resources": {
                    "wall_seconds": 1.0 + seed,
                    "cpu_seconds": 0.5 + seed,
                    "peak_rss_bytes": 1000 + seed,
                    "disk_bytes": 50 + seed,
                    "device": "cpu",
                },
                "resource_deltas": {"wall_seconds": 1.0, "peak_rss_bytes": 1000},
                "candidate_prediction_artifact_sha256": _digest("a"),
                "candidate_prediction_digest": _digest("b"),
                "official_fm_prediction_artifact_sha256": _digest("c"),
                "official_fm_prediction_digest": _digest("d"),
                "scientific_request_digest": _digest("e"),
                "scientific_record_digest": _digest("f"),
                "checkpoint_sha256": _digest("1"),
            }
        )
    candidate_mean = {
        name: sum(row[name] for row in candidate_rows) / 3 for name in ("GAUC", "nDCG@5", "primary")
    }
    fm_mean = {
        name: sum(row[name] for row in fm_rows) / 3 for name in ("GAUC", "nDCG@5", "primary")
    }
    mean_primary = sum(primary_deltas, Decimal(0)) / len(primary_deltas)
    primary_summary = {
        "mean": float(mean_primary),
        "median": float(statistics.median(primary_deltas)),
        "minimum": float(min(primary_deltas)),
        "population_std": float(statistics.pstdev(primary_deltas)),
    }
    inner_rows: list[dict[str, object]] = []
    inner_deltas: list[Decimal] = []
    for fold, fold_delta in (("A", 0.001), ("B", 0.002)):
        reference = _metrics(0.55, 0.45)
        parent = _metrics(0.56, 0.46)
        candidate = _metrics(
            reference["GAUC"] + fold_delta,
            reference["nDCG@5"] + fold_delta,
        )
        delta_reference = candidate["primary"] - reference["primary"]
        inner_deltas.append(Decimal(str(delta_reference)))
        inner_rows.append(
            {
                "fold": fold,
                "candidate_metrics": candidate,
                "parent_metrics": parent,
                "official_fm_reference_metrics": reference,
                "primary_delta_to_parent": candidate["primary"] - parent["primary"],
                "primary_delta_to_reference": delta_reference,
                "evidence_digest": _digest("2" if fold == "A" else "3"),
                "weights_selected_on_public_validation": False,
            }
        )
    mean_inner = sum(inner_deltas, Decimal(0)) / len(inner_deltas)
    status = (
        FinalStatus.MATERIALLY_CONFIRMED
        if mean_primary > Decimal("0.002")
        else FinalStatus.VALIDATION_IMPROVED
    )
    representative = candidate_rows[0]
    bootstrap_metrics = {
        name: {
            "candidate": representative[name],
            "control": fm_rows[0][name],
            "delta": representative[name] - fm_rows[0][name],
            "confidence_interval": {
                "lower": -0.001,
                "upper": 0.01,
                "confidence_level": 0.95,
                "method": "percentile-linear",
            },
        }
        for name in ("GAUC", "nDCG@5", "primary")
    }
    return {
        "selection": {"status": status.value},
        "validation": {
            "metrics": representative,
            "seed_summary": {
                "schema_version": 1,
                "seeds": [0, 1, 2],
                "representative_seed": 0,
                "candidate_mean": candidate_mean,
                "official_fm_mean": fm_mean,
                "per_seed": records,
                "primary_delta_summary": primary_summary,
                "inner_delta_summary": {
                    "mean_primary_delta": float(mean_inner),
                    "worst_primary_delta": float(min(inner_deltas)),
                },
                "paired_user_cluster_bootstrap": {
                    "schema_version": 2,
                    "decision_use": "diagnostic_only",
                    "gating_eligible": False,
                    "phase": "outer_valid",
                    "rows": 10,
                    "users": 2,
                    "gauc_eligible_users": 2,
                    "resamples": 2000,
                    "seed": 0,
                    "point_estimate_source": "protected_organizer_scorer",
                    "point_estimate_provenance": {
                        "scorer_digest": STARTER_FILE_SHA256["evaluate.py"],
                        "candidate_prediction_digest": _digest("b"),
                        "control_prediction_digest": _digest("d"),
                        "rows": 10,
                        "users": 2,
                        "primary_aggregation": (
                            "selector_decimal_mean_of_protected_gauc_and_ndcg_at_5"
                        ),
                    },
                    "metrics": bootstrap_metrics,
                },
                "derived_status": status.value,
                "status_thresholds": {
                    "material_primary_delta_strictly_greater_than": 0.002,
                    "mean_inner_primary_delta_strictly_greater_than": 0.0,
                    "worst_inner_primary_delta_minimum": -0.002,
                },
                "confirmation_is_controller_derived": True,
            },
            "inner_fold_results": inner_rows,
        },
    }


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0.004, FinalStatus.MATERIALLY_CONFIRMED),
        (0.001, FinalStatus.VALIDATION_IMPROVED),
    ],
)
def test_bundle_status_is_independently_derived_from_all_matched_seeds(
    delta: float,
    expected: FinalStatus,
) -> None:
    assert production._derive_bundle_status(_generated_manifest(delta)) is expected


def test_material_status_forgery_and_incomplete_confirmation_are_rejected() -> None:
    forged = _generated_manifest(0.001)
    cast(dict[str, object], forged["selection"])["status"] = "materially_confirmed"
    with pytest.raises(ProductionFinalizationError, match="not independently derivable"):
        production._derive_bundle_status(forged)

    incomplete = _generated_manifest(0.004)
    summary = cast(
        dict[str, object],
        cast(dict[str, object], incomplete["validation"])["seed_summary"],
    )
    cast(list[object], summary["per_seed"]).pop()
    with pytest.raises(ProductionFinalizationError, match="incomplete"):
        production._derive_bundle_status(incomplete)


def test_generated_bundle_requires_protected_bootstrap_point_provenance() -> None:
    manifest = _generated_manifest(0.001)
    bootstrap = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], manifest["validation"])["seed_summary"],
        )["paired_user_cluster_bootstrap"],
    )
    provenance = cast(dict[str, object], bootstrap["point_estimate_provenance"])

    provenance["candidate_prediction_digest"] = _digest("9")
    with pytest.raises(ProductionFinalizationError, match="candidate prediction"):
        production._derive_bundle_status(manifest)

    manifest = _generated_manifest(0.001)
    bootstrap = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], manifest["validation"])["seed_summary"],
        )["paired_user_cluster_bootstrap"],
    )
    provenance = cast(dict[str, object], bootstrap["point_estimate_provenance"])
    provenance["scorer_digest"] = _digest("9")
    with pytest.raises(ProductionFinalizationError, match="scorer"):
        production._derive_bundle_status(manifest)

    manifest = _generated_manifest(0.001)
    bootstrap = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], manifest["validation"])["seed_summary"],
        )["paired_user_cluster_bootstrap"],
    )
    provenance = cast(dict[str, object], bootstrap["point_estimate_provenance"])
    provenance["primary_aggregation"] = "raw_score_result_primary"
    with pytest.raises(ProductionFinalizationError, match="primary aggregation"):
        production._derive_bundle_status(manifest)

    manifest = _generated_manifest(0.001)
    bootstrap = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], manifest["validation"])["seed_summary"],
        )["paired_user_cluster_bootstrap"],
    )
    bootstrap["point_estimate_source"] = "independent_diagnostic_reconstruction"
    with pytest.raises(ProductionFinalizationError, match="protected"):
        production._derive_bundle_status(manifest)


def test_generated_bundle_rejects_tampered_protected_bootstrap_point() -> None:
    manifest = _generated_manifest(0.001)
    bootstrap = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], manifest["validation"])["seed_summary"],
        )["paired_user_cluster_bootstrap"],
    )
    metric = cast(dict[str, object], cast(dict[str, object], bootstrap["metrics"])["GAUC"])
    metric["control"] = float(cast(float, metric["control"])) + 0.0001

    with pytest.raises(ProductionFinalizationError, match="bootstrap control GAUC"):
        production._derive_bundle_status(manifest)


def test_generated_resource_provenance_separates_selected_training_from_report_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_peaks = (948_830_208, 941_211_648, 948_617_216)
    fm_peaks = (1_843_101_696, 1_700_000_000, 1_650_000_000)
    seed_rows = tuple(
        MappingProxyType(
            {
                "seed": seed,
                "candidate_resources": {
                    "wall_seconds": 2.0 + seed,
                    "peak_rss_bytes": candidate_peaks[seed],
                    "disk_bytes": 100 + seed,
                    "device": "cpu",
                },
                "official_fm_resources": {
                    "wall_seconds": 1.0 + seed,
                    "cpu_seconds": 0.5 + seed,
                    "peak_rss_bytes": fm_peaks[seed],
                    "disk_bytes": 50 + seed,
                    "device": "cpu",
                },
                "resource_deltas": {
                    "wall_seconds": 1.0,
                    "peak_rss_bytes": candidate_peaks[seed] - fm_peaks[seed],
                },
            }
        )
        for seed in (0, 1, 2)
    )
    confirmation = production._GeneratedConfirmation(
        status=FinalStatus.VALIDATION_IMPROVED,
        representative_metrics=MappingProxyType(_metrics(0.61, 0.41)),
        candidate_mean=MappingProxyType(_metrics(0.61, 0.41)),
        fm_mean=MappingProxyType(_metrics(0.60, 0.40)),
        seed_rows=seed_rows,
        primary_summary=MappingProxyType({}),
        inner_rows=(),
        inner_summary=MappingProxyType({}),
        bootstrap=MappingProxyType({}),
    )

    # The selected-lineage value is derived only from retained generated training rows. A
    # larger matched-FM or qualification peak must not be relabelled as generated usage.
    assert confirmation.retained_training_peak_rss_bytes == max(candidate_peaks)

    qualification = SimpleNamespace(
        outer_runs=tuple(
            SimpleNamespace(
                seed=seed,
                resources=SimpleNamespace(
                    wall_seconds=1.0,
                    cpu_seconds=0.5,
                    peak_rss_bytes=peak,
                    disk_bytes=50,
                    device="cpu",
                ),
            )
            for seed, peak in enumerate((*fm_peaks, 1_600_000_000, 1_500_000_000))
        )
    )
    assert production._report_peak_rss_bytes(cast(Any, qualification), confirmation) == max(
        fm_peaks
    )

    environment_digest = _digest("e")
    monkeypatch.setattr(
        production,
        "environment_identity_digest",
        lambda _environment: environment_digest,
    )
    monkeypatch.setattr(
        production,
        "_judge_progress_facts",
        lambda _outcome: production._JudgeProgressFacts(
            provider_usage="Research-model calls=0; network/API calls=0.",
            portfolio_count=1,
            portfolio_cap=1,
            portfolio_cap_reason="bounded_high_value_lambdarank_branch_prioritized",
            advanced_branch_disposition="Advanced branches were not entered.",
        ),
    )
    identity = SimpleNamespace(
        data_sha256=_digest("1"),
        source_sha256=_digest("2"),
        config_sha256=_digest("3"),
        features_sha256=_digest("4"),
        checkpoint_sha256=_digest("5"),
        validation_prediction_artifact_sha256=_digest("6"),
        environment_sha256=environment_digest,
    )
    request = SimpleNamespace(
        benchmark_digest=_digest("7"),
        starter_manifest_digest=_digest("8"),
        source_digest=_digest("9"),
        environment_digest=environment_digest,
    )
    outcome = SimpleNamespace(
        selection=SimpleNamespace(representative_seed=0),
        scientific_result_digest=_digest("a"),
        manual_interventions=0,
    )
    metadata = production._bundle_metadata(
        candidate_id="generated-candidate",
        lineage=("official-fm", "generated-candidate"),
        status=FinalStatus.VALIDATION_IMPROVED,
        metrics=_metrics(0.61, 0.41),
        identity=cast(Any, identity),
        environment={
            "uv_lock_sha256": _digest("b"),
            "packages": {
                "lightgbm": "4.7.0",
                "numpy": "2.5.2",
                "psutil": "7.2.2",
                "torch": None,
            },
        },
        request=cast(Any, request),
        outcome=cast(Any, outcome),
        seeds=(0, 1, 2),
        generated=True,
        confirmation=confirmation,
        campaign_wall_seconds=20.0,
        launch_count=14,
    )
    assert metadata.environment_and_resource_usage["retained_training_peak_rss_bytes"] == max(
        candidate_peaks
    )


def _write_baseline_bundle(root: Path) -> None:
    root.mkdir(mode=0o700)
    for relative in REQUIRED_DIRECTORY_PATHS:
        (root / relative).mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    for relative in REQUIRED_FILE_PATHS:
        payload = f"{relative}\n".encode("ascii")
        path = root / relative
        path.write_bytes(payload)
        os.chmod(path, 0o444)
        records.append(
            {
                "path": relative,
                "component": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    submission_digest = next(
        cast(str, item["sha256"]) for item in records if item["path"] == "submission.csv"
    )
    prepublication_resource_digest = next(
        cast(str, item["sha256"])
        for item in records
        if item["path"] == "prepublication-resource.json"
    )
    manifest = {
        "schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
        "benchmark_identity": {"name": "KuaiRand-Pure"},
        "starter_identity": {"manifest_sha256": _digest("a")},
        "data_identity": {"canonical_digest": _digest("b")},
        "selection": {
            "selected_experiment": "official-fm-fallback-seed-4",
            "lineage": ["official-fm-fallback-seed-4"],
            "status": "baseline_reproduced",
        },
        "validation": {
            "metrics": _metrics(0.6, 0.4),
            "seed_summary": {
                "schema_version": 1,
                "seeds": [4],
                "representative_seed": 4,
                "matched_confirmation": False,
                "derived_status": "baseline_reproduced",
                "confirmation_is_controller_derived": True,
            },
            "inner_fold_results": [{"fold": "official qualification"}],
        },
        "scientific_artifact_hashes": {
            "source": _digest("c"),
            "config": _digest("d"),
            "features": _digest("e"),
            "checkpoint": _digest("f"),
            "predictions": _digest("1"),
            "submission": submission_digest,
        },
        "prepublication_resource_receipt": {
            "path": "prepublication-resource.json",
            "sha256": prepublication_resource_digest,
        },
        "environment_and_resource_usage": {"environment_sha256": _digest("2")},
        "campaign_totals": {"launch_count": 6},
        "components": {
            "required_paths": [
                "manifest.json",
                *REQUIRED_FILE_PATHS,
                *REQUIRED_DIRECTORY_PATHS,
            ],
            "roots": {},
            "files": records,
        },
        "known_limitations": [],
        "unresolved_organizer_questions": [],
    }
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")
    (root / "manifest.json").write_bytes(payload)
    os.chmod(root / "manifest.json", 0o444)


def test_bundle_close_removes_all_directory_write_bits_and_reopens_exactly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "final"
    _write_baseline_bundle(root)

    first = production._close_bundle_directories(root)
    second = production._verify_closed_bundle(root)

    assert first.manifest_sha256 == second.manifest_sha256
    assert root.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in root.rglob("*") if path.is_dir())


def test_closed_bundle_rejects_unbound_prepublication_resource_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "final"
    _write_baseline_bundle(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["prepublication_resource_receipt"]["sha256"] = _digest("9")
    os.chmod(manifest_path, 0o644)
    manifest_path.write_bytes(
        (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    )
    os.chmod(manifest_path, 0o444)
    with pytest.raises(ProductionFinalizationError, match="prepublication resource"):
        production._close_bundle_directories(root)


def test_closed_bundle_rejects_directory_permission_and_content_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "final"
    _write_baseline_bundle(root)
    production._close_bundle_directories(root)

    os.chmod(root / "model", 0o755)
    with pytest.raises(ProductionFinalizationError, match="write bits"):
        production._verify_closed_bundle(root)
    os.chmod(root / "model", 0o555)

    os.chmod(root, 0o755)
    extra = root / "unlisted.txt"
    extra.write_text("tamper", encoding="ascii")
    os.chmod(extra, 0o444)
    os.chmod(root, 0o555)
    with pytest.raises(ProductionFinalizationError, match="inventory"):
        production._verify_closed_bundle(root)


def test_production_outcome_round_trip_is_signed_and_tamper_evident(tmp_path: Path) -> None:
    outcome = ProductionFinalizationOutcome(
        run_dir=(tmp_path / "run").absolute(),
        campaign_id="campaign",
        research_outcome_digest=_digest("a"),
        selected_candidate_id="official-fm-fallback-seed-4",
        selected_status=FinalStatus.BASELINE_REPRODUCED,
        fallback_count=0,
        failures=(),
        training_replay=MappingProxyType({"required": False, "completed": True}),
        resource_evidence=_fallback_resource_evidence(_digest("b")),
        bundle_root=(tmp_path / "run" / "final").absolute(),
        bundle_manifest_sha256=_digest("b"),
        submission_sha256=_digest("c"),
        replay_evidence_sha256=_digest("d"),
        organizer_verification_sha256=_digest("e"),
        campaign_revision=42,
    )

    assert ProductionFinalizationOutcome.from_bytes(outcome.canonical_bytes) == outcome
    decoded = json.loads(outcome.canonical_bytes)
    decoded["campaign_revision"] = 43
    tampered = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(ProductionFinalizationError, match="digest mismatch"):
        ProductionFinalizationOutcome.from_bytes(tampered)


def test_production_outcome_accepts_one_hour_sprint_resource_limits(tmp_path: Path) -> None:
    outcome = ProductionFinalizationOutcome(
        run_dir=(tmp_path / "run").absolute(),
        campaign_id="campaign",
        research_outcome_digest=_digest("a"),
        selected_candidate_id="official-fm-fallback-seed-4",
        selected_status=FinalStatus.BASELINE_REPRODUCED,
        fallback_count=0,
        failures=(),
        training_replay=MappingProxyType({"required": False, "completed": True}),
        resource_evidence=_fallback_resource_evidence(
            _digest("b"),
            hard_wall_seconds=3_600,
            finalization_reserve_seconds=600,
        ),
        bundle_root=(tmp_path / "run" / "final").absolute(),
        bundle_manifest_sha256=_digest("b"),
        submission_sha256=_digest("c"),
        replay_evidence_sha256=_digest("d"),
        organizer_verification_sha256=_digest("e"),
        campaign_revision=42,
    )

    assert outcome.resource_evidence["hard_wall_seconds"] == 3_600
    assert outcome.resource_evidence["finalization_reserve_seconds"] == 600
    assert ProductionFinalizationOutcome.from_bytes(outcome.canonical_bytes) == outcome


def test_production_outcome_rejects_invalid_dynamic_resource_limits(tmp_path: Path) -> None:
    with pytest.raises(ProductionFinalizationError, match="finalization reserve is invalid"):
        ProductionFinalizationOutcome(
            run_dir=(tmp_path / "run").absolute(),
            campaign_id="campaign",
            research_outcome_digest=_digest("a"),
            selected_candidate_id="official-fm-fallback-seed-4",
            selected_status=FinalStatus.BASELINE_REPRODUCED,
            fallback_count=0,
            failures=(),
            training_replay=MappingProxyType({"required": False, "completed": True}),
            resource_evidence=_fallback_resource_evidence(
                _digest("b"),
                hard_wall_seconds=3_600,
                finalization_reserve_seconds=3_600,
            ),
            bundle_root=(tmp_path / "run" / "final").absolute(),
            bundle_manifest_sha256=_digest("b"),
            submission_sha256=_digest("c"),
            replay_evidence_sha256=_digest("d"),
            organizer_verification_sha256=_digest("e"),
            campaign_revision=42,
        )


def test_final_training_resources_bind_tree_file_not_model_directory_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree_checkpoint = _digest("a")
    model_directory = _digest("b")
    recipe = GeneratedLambdaRankReplayRecipe(
        source_artifact_sha256=_digest("1"),
        candidate_source_sha256=_digest("2"),
        candidate_config_sha256=_digest("3"),
        feature_artifact_sha256=_digest("4"),
        validation_features_sha256=_digest("5"),
        final_features_sha256=_digest("6"),
        checkpoint_artifact_sha256=model_directory,
        tree_checkpoint_sha256=tree_checkpoint,
        data_sha256=_digest("7"),
        validation_inputs_digest=_digest("8"),
        final_inputs_digest=_digest("9"),
        feature_count=3,
        timeout_seconds=30,
        memory_limit_bytes=1024**3,
        threads=1,
    )
    bundle_root = tmp_path / "final"
    (bundle_root / "config").mkdir(parents=True)
    bundle = cast(
        Any,
        SimpleNamespace(
            root=bundle_root,
            manifest={
                "scientific_artifact_hashes": {
                    "config": recipe.digest,
                    "checkpoint": model_directory,
                }
            },
        ),
    )
    research = cast(
        Any,
        SimpleNamespace(
            selection=SimpleNamespace(tree_checkpoint=SimpleNamespace(sha256=tree_checkpoint))
        ),
    )
    outcome = cast(
        Any,
        SimpleNamespace(
            resource_evidence={"final_training": {"evidence_digest": tree_checkpoint}},
            training_replay={"checkpoint_sha256": tree_checkpoint},
        ),
    )
    monkeypatch.setattr(production, "load_replay_recipe", lambda *_args, **_kwargs: recipe)

    production._verify_generated_final_training_binding(
        bundle=bundle,
        research=research,
        outcome=outcome,
    )

    outcome.resource_evidence["final_training"]["evidence_digest"] = model_directory
    with pytest.raises(ProductionFinalizationError, match="selected tree checkpoint"):
        production._verify_generated_final_training_binding(
            bundle=bundle,
            research=research,
            outcome=outcome,
        )


def test_fresh_training_replay_uses_only_official_train_capabilities_and_charges_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "candidate.py").write_text("pass\n", encoding="ascii")
    (source_dir / "config.json").write_text("{}\n", encoding="ascii")
    (source_dir / "README.md").write_text("generated\n", encoding="ascii")
    source = artifacts.put_directory(source_dir, kind=ArtifactKind.SOURCE)
    source_entries = {entry.path: entry.artifact for entry in source.entries}
    training_features = artifacts.put_bytes(b"train-features", kind=ArtifactKind.INPUT)
    training_targets = artifacts.put_bytes(b"train-targets", kind=ArtifactKind.INPUT)
    training_groups = artifacts.put_bytes(b"train-groups", kind=ArtifactKind.INPUT)
    validation_features = artifacts.put_bytes(b"validation-features", kind=ArtifactKind.INPUT)
    checkpoint = artifacts.put_bytes(b"checkpoint", kind=ArtifactKind.CHECKPOINT)
    selection = SimpleNamespace(
        digest=_digest("9"),
        experiment_id="iteration-01",
        candidate_id="generated",
        timeout_seconds=30,
        memory_limit_bytes=1024**3,
        threads=1,
        representative_seed=0,
        source_snapshot=source,
        source_digest=_digest("8"),
        config_digest=source_entries["config.json"].sha256,
        dataset_digest=_digest("7"),
        training_policy_digest=_digest("6"),
        training_features=training_features,
        training_targets=training_targets,
        training_user_groups=training_groups,
        validation_features=validation_features,
        tree_checkpoint=checkpoint,
    )
    request = SimpleNamespace(config=SimpleNamespace(runner=SimpleNamespace(disk_mb=64)))
    captured: dict[str, object] = {}
    active_interpreter = tmp_path / "production-venv" / "bin" / "python"

    class FakeJournal:
        def __init__(self, **kwargs: object) -> None:
            captured["policy"] = kwargs["policy"]

    class FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured["executor"] = kwargs

        def train(self, train: object, **kwargs: object) -> object:
            captured["train"] = train
            captured["train_kwargs"] = kwargs
            return SimpleNamespace(
                checkpoint=checkpoint,
                execution=SimpleNamespace(
                    wall_seconds=1.25,
                    peak_tree_rss_bytes=4096,
                    peak_workspace_bytes=2048,
                ),
            )

    monkeypatch.setattr(production, "CampaignStoreCandidateJournal", FakeJournal)
    monkeypatch.setattr(production, "GeneratedCandidateExecutor", FakeExecutor)
    monkeypatch.setattr(
        production,
        "active_python_interpreter",
        lambda: active_interpreter,
    )
    monkeypatch.setattr(production, "WorkspaceMaterializer", lambda *args, **kwargs: object())
    engine = SimpleNamespace(observe_deadline=lambda _run: object())
    campaign_store = SimpleNamespace(execution=lambda _execution_id: None)

    actual, evidence = production._fresh_training_replay(
        run_dir=tmp_path,
        request=cast(CampaignCreateRequest, request),
        selection=cast(FinalizationSelectionPlan, selection),
        artifact_store=artifacts,
        engine=cast(CampaignEngine, engine),
        store=cast(CampaignStore, campaign_store),
        cancel_event=None,
    )

    train = cast(GeneratedTrainRequest, captured["train"])
    policy = cast(CandidateJournalPolicy, captured["policy"])
    executor = cast(dict[str, object], captured["executor"])
    assert actual == checkpoint
    assert executor["interpreter"] == active_interpreter
    assert train.split_role is SplitRole.TRAIN
    assert train.features == training_features
    assert train.targets == training_targets
    assert train.user_groups == training_groups
    assert train.features != validation_features
    assert policy.category is LaunchCategory.FINAL_TRAINING_REPLAY
    assert policy.experiment_id == "iteration-01"
    assert evidence["charged_launch"] is True
    assert evidence["training_rows_include_public_validation"] is False


def test_fresh_training_checkpoint_parity_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    expected = artifacts.put_bytes(b"expected", kind=ArtifactKind.CHECKPOINT)
    actual = artifacts.put_bytes(b"actual", kind=ArtifactKind.CHECKPOINT)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "candidate.py").write_text("pass\n", encoding="ascii")
    (source_dir / "config.json").write_text("{}\n", encoding="ascii")
    source = artifacts.put_directory(source_dir, kind=ArtifactKind.SOURCE)
    entries = {entry.path: entry.artifact for entry in source.entries}
    refs = [artifacts.put_bytes(name.encode(), kind=ArtifactKind.INPUT) for name in ("x", "y", "z")]
    selection = SimpleNamespace(
        digest=_digest("1"),
        experiment_id="iteration-01",
        candidate_id="generated",
        timeout_seconds=30,
        memory_limit_bytes=1024**3,
        threads=1,
        representative_seed=0,
        source_snapshot=source,
        source_digest=_digest("2"),
        config_digest=entries["config.json"].sha256,
        dataset_digest=_digest("3"),
        training_policy_digest=_digest("4"),
        training_features=refs[0],
        training_targets=refs[1],
        training_user_groups=refs[2],
        tree_checkpoint=expected,
    )

    class FakeExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def train(self, _train: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                checkpoint=actual,
                execution=SimpleNamespace(
                    wall_seconds=1.0,
                    peak_tree_rss_bytes=1,
                    peak_workspace_bytes=1,
                ),
            )

    monkeypatch.setattr(production, "CampaignStoreCandidateJournal", lambda **_kwargs: object())
    monkeypatch.setattr(production, "GeneratedCandidateExecutor", FakeExecutor)
    monkeypatch.setattr(production, "WorkspaceMaterializer", lambda *args, **kwargs: object())
    with pytest.raises(ProductionFinalizationError, match="checkpoint bytes differ"):
        production._fresh_training_replay(
            run_dir=tmp_path,
            request=cast(
                CampaignCreateRequest,
                SimpleNamespace(config=SimpleNamespace(runner=SimpleNamespace(disk_mb=64))),
            ),
            selection=cast(FinalizationSelectionPlan, selection),
            artifact_store=artifacts,
            engine=cast(
                CampaignEngine,
                SimpleNamespace(observe_deadline=lambda _run: object()),
            ),
            store=cast(
                CampaignStore,
                SimpleNamespace(execution=lambda _execution_id: None),
            ),
            cancel_event=None,
        )


def test_generated_completion_requires_exact_charged_final_training_terminal() -> None:
    checkpoint_sha = _digest("a")
    selection = SimpleNamespace(
        digest=_digest("b"),
        experiment_id="iteration-01",
        representative_seed=0,
        source_digest=_digest("c"),
        config_digest=_digest("d"),
        dataset_digest=_digest("e"),
        tree_checkpoint=SimpleNamespace(sha256=checkpoint_sha),
    )
    execution = SimpleNamespace(
        status="SUCCEEDED",
        experiment_id="iteration-01",
        seed=0,
        source_digest=selection.source_digest,
        config_digest=selection.config_digest,
        data_digest=selection.dataset_digest,
        launch_category=LaunchCategory.FINAL_TRAINING_REPLAY.value,
        original_launch_category=LaunchCategory.FINAL_TRAINING_REPLAY.value,
    )
    artifact = SimpleNamespace(
        digest=checkpoint_sha,
        kind=ArtifactKind.CHECKPOINT.value,
    )
    store = SimpleNamespace(
        execution=lambda _execution_id: execution,
        artifacts_for=lambda **_kwargs: (("checkpoint", artifact),),
    )

    production._verify_charged_final_training_terminal(
        cast(CampaignStore, store),
        cast(FinalizationSelectionPlan, selection),
    )

    execution.launch_category = LaunchCategory.RECOVERY_RESERVE.value
    with pytest.raises(ProductionFinalizationError, match="charged successful training"):
        production._verify_charged_final_training_terminal(
            cast(CampaignStore, store),
            cast(FinalizationSelectionPlan, selection),
        )

    execution.launch_category = LaunchCategory.FINAL_TRAINING_REPLAY.value
    execution.experiment_id = "generated-causal-lambdarank-v1"
    with pytest.raises(ProductionFinalizationError, match="charged successful training"):
        production._verify_charged_final_training_terminal(
            cast(CampaignStore, store),
            cast(FinalizationSelectionPlan, selection),
        )


def test_generated_incumbent_links_to_durable_experiment_not_candidate_id(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "final"
    bundle_root.mkdir()
    (bundle_root / "manifest.json").write_text("{}\n", encoding="ascii")
    manifest = _generated_manifest(0.001)
    manifest["selection"] = {
        "selected_experiment": "generated-causal-lambdarank-v1",
        "status": FinalStatus.VALIDATION_IMPROVED.value,
    }
    manifest["scientific_artifact_hashes"] = {
        "checkpoint": _digest("a"),
        "submission": _digest("b"),
    }
    bundle = SimpleNamespace(
        root=bundle_root,
        manifest=manifest,
        manifest_sha256=_digest("c"),
    )
    selection = SimpleNamespace(
        experiment_id="iteration-01",
        candidate_id="generated-causal-lambdarank-v1",
        source_digest=_digest("d"),
    )
    outcome = SimpleNamespace(
        selection=selection,
        fallback_candidate_id="official-fm-fallback-seed-4",
        digest=_digest("e"),
    )
    captured: dict[str, object] = {}
    incumbent = object()

    class FakeStore:
        def snapshot(self) -> object:
            return SimpleNamespace(revision=10, status=CampaignState.FINALIZING.value)

        def record_incumbent(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return incumbent

        def current_incumbent(self) -> object:
            return incumbent

        def set_campaign_phase(self, **_kwargs: object) -> object:
            return SimpleNamespace(revision=11, status=CampaignState.COMPLETED.value)

    revision = production._complete_campaign(
        store=cast(CampaignStore, FakeStore()),
        request=cast(CampaignCreateRequest, SimpleNamespace()),
        outcome=cast(Any, outcome),
        result=None,
        bundle=cast(Any, bundle),
    )

    assert revision == 11
    assert captured["experiment_id"] == "iteration-01"


def test_generated_replay_requires_exact_rebuilt_candidate_capability_digests() -> None:
    dataset_digest = _digest("1")
    validation_capability_digest = _digest("2")
    final_capability_digest = _digest("3")
    raw_canonical_digest = _digest("4")
    context = SimpleNamespace(
        dataset=SimpleNamespace(digest=dataset_digest),
        capabilities=SimpleNamespace(
            validation_inputs=SimpleNamespace(digest=validation_capability_digest),
            final_inputs=SimpleNamespace(digest=final_capability_digest),
        ),
    )
    selection = SimpleNamespace(
        dataset_digest=dataset_digest,
        validation_inputs_digest=validation_capability_digest,
        final_inputs_digest=final_capability_digest,
        representative_seed=0,
    )

    class ReachedQualificationAfterCapabilityGate(RuntimeError):
        pass

    class Qualification:
        def outer_seed(self, _seed: int) -> object:
            raise ReachedQualificationAfterCapabilityGate

    kwargs: dict[str, object] = {
        "selection": selection,
        "checkpoint": None,
        "qualification": Qualification(),
        "context": context,
        "request": None,
        "environment": {},
        "artifact_store": None,
        "outcome": None,
        "report_failures": (),
        "campaign_wall_seconds": 1.0,
        "launch_count": 1,
        "cancel_event": None,
    }
    with pytest.raises(ReachedQualificationAfterCapabilityGate):
        production._generated_replay_candidate(**cast(Any, kwargs))

    selection.final_inputs_digest = raw_canonical_digest
    with pytest.raises(ProductionFinalizationError, match="capability identity changed"):
        production._generated_replay_candidate(**cast(Any, kwargs))


class _FakeCampaignStore:
    def __init__(self) -> None:
        self.launches_used = 11

    def __enter__(self) -> _FakeCampaignStore:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def snapshot(self) -> object:
        return SimpleNamespace(
            launches_used=self.launches_used,
            status="FINALIZING",
        )


class _DeadlineEngine:
    def __init__(self, *, elapsed_seconds: float) -> None:
        self.elapsed_seconds = elapsed_seconds
        self.resume_calls = 0

    @property
    def expired(self) -> bool:
        return self.elapsed_seconds >= 21_600.0

    def expire(self) -> None:
        self.elapsed_seconds = 21_600.0

    def resume(self, _run_dir: Path) -> object:
        self.resume_calls += 1
        return SimpleNamespace(
            status=("INCOMPLETE" if self.expired else "FINALIZATION_REQUIRED"),
            phase=("deadline_exhausted" if self.expired else "finalization_required"),
            deadline_elapsed_seconds=min(self.elapsed_seconds, 21_600.0),
            deadline_remaining_seconds=max(0.0, 21_600.0 - self.elapsed_seconds),
            unfinished_execution_ids=(),
        )

    def inspect_deadline(self, _run_dir: Path) -> object:
        return SimpleNamespace(hard_expired=self.expired)

    def observe_deadline(self, _run_dir: Path) -> object:
        return SimpleNamespace(
            hard_expired=self.expired,
            elapsed_seconds=min(self.elapsed_seconds, 21_600.0),
        )


@pytest.mark.parametrize("elapsed_seconds", [21_600.0, 21_601.0])
def test_public_finalizer_reconciles_exact_or_beyond_hard_deadline_before_loading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elapsed_seconds: float,
) -> None:
    run_dir = tmp_path / "run"
    project_root = tmp_path / "project"
    run_dir.mkdir()
    project_root.mkdir()
    engine = _DeadlineEngine(elapsed_seconds=elapsed_seconds)

    monkeypatch.setattr(
        production,
        "_load_research_outcome",
        lambda *_args, **_kwargs: pytest.fail(
            "expired finalization must stop before data or research evidence is loaded"
        ),
    )
    monkeypatch.setattr(
        production,
        "run_finalization",
        lambda *_args, **_kwargs: pytest.fail("expired finalization must never publish"),
    )

    with pytest.raises(ProductionFinalizationError, match=r"hard deadline.*INCOMPLETE"):
        production.finalize_provider_free_campaign(
            run_dir,
            project_root=project_root,
            engine=cast(CampaignEngine, engine),
        )

    assert engine.resume_calls == 1
    assert not (run_dir / "final").exists()


@pytest.mark.parametrize("crossing_stage", ["final inference", "bundle publication"])
def test_public_finalizer_clock_crossing_during_final_work_marks_incomplete_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crossing_stage: str,
) -> None:
    run_dir = tmp_path / "run"
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    starter_dir = project_root / "starter"
    run_dir.mkdir()
    data_dir.mkdir(parents=True)
    starter_dir.mkdir()
    artifact_store = ArtifactStore(run_dir / "artifacts")
    research = SimpleNamespace(
        selection=None,
        fallback_candidate_id="official-fm-fallback-seed-4",
        campaign_id="campaign",
        digest=_digest("a"),
    )
    request = SimpleNamespace(
        campaign_id="campaign",
        dataset_manifest_digest=_digest("b"),
        config=SimpleNamespace(
            benchmark=SimpleNamespace(
                data_dir=Path("data"),
                starter_dir=Path("starter"),
                wall_clock_seconds=21_600,
            ),
            runner=SimpleNamespace(finalization_reserve_seconds=3_600),
        ),
    )
    engine = _DeadlineEngine(elapsed_seconds=100.0)
    store = _FakeCampaignStore()
    fallback_candidate = SimpleNamespace(
        candidate_id="official-fm-fallback-seed-4",
        bundle_metadata=SimpleNamespace(status=FinalStatus.BASELINE_REPRODUCED),
    )

    monkeypatch.setattr(
        production,
        "_load_research_outcome",
        lambda *_args, **_kwargs: (request, artifact_store, research),
    )
    monkeypatch.setattr(
        production,
        "_trusted_replay_context",
        lambda **_kwargs: SimpleNamespace(capabilities=object(), score=lambda _scores: {}),
    )
    monkeypatch.setattr(production, "_environment", lambda *_args: {})
    monkeypatch.setattr(production, "_qualification", lambda *_args: object())
    monkeypatch.setattr(production, "_phase_for_finalization", lambda _store: None)
    monkeypatch.setattr(
        CampaignStore,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: store),
    )
    monkeypatch.setattr(
        production,
        "_fallback_replay_candidate",
        lambda **_kwargs: fallback_candidate,
    )

    def ledgers(**_kwargs: object) -> tuple[Path, Path]:
        jsonl = run_dir / "experiments.jsonl"
        csv_path = run_dir / "experiments.csv"
        jsonl.write_text("{}\n", encoding="ascii")
        csv_path.write_text("candidate_id\n", encoding="ascii")
        return jsonl, csv_path

    monkeypatch.setattr(production, "_write_experiment_ledgers", ledgers)
    monkeypatch.setattr(
        production,
        "FinalizationRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def cross_deadline(_request: object, **kwargs: object) -> object:
        cancellation = cast(threading.Event, kwargs["cancel_event"])
        assert not (run_dir / "final").exists()
        engine.expire()
        assert cancellation.is_set(), f"deadline was not rechecked during {crossing_stage}"
        raise FinalizationCancelledError(f"deadline crossed during {crossing_stage}")

    monkeypatch.setattr(production, "run_finalization", cross_deadline)
    monkeypatch.setattr(
        production,
        "_complete_campaign",
        lambda **_kwargs: pytest.fail("expired campaign must never complete"),
    )

    with pytest.raises(ProductionFinalizationError, match=r"hard deadline.*INCOMPLETE"):
        production.finalize_provider_free_campaign(
            run_dir,
            project_root=project_root,
            engine=cast(CampaignEngine, engine),
        )

    assert engine.resume_calls == 2
    assert not (run_dir / "final").exists()


@pytest.mark.parametrize("generated_succeeds", [True, False])
def test_public_finalizer_uses_generated_candidate_or_walks_to_official_fm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_succeeds: bool,
) -> None:
    run_dir = tmp_path / "run"
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    starter_dir = project_root / "starter"
    run_dir.mkdir()
    data_dir.mkdir(parents=True)
    starter_dir.mkdir()
    artifact_store = ArtifactStore(run_dir / "artifacts")
    checkpoint = artifact_store.put_bytes(b"checkpoint", kind=ArtifactKind.CHECKPOINT)
    selection = SimpleNamespace(candidate_id="generated-candidate")
    research = SimpleNamespace(
        selection=selection,
        fallback_candidate_id="official-fm-fallback-seed-4",
        campaign_id="campaign",
        digest=_digest("a"),
    )
    request = SimpleNamespace(
        campaign_id="campaign",
        dataset_manifest_digest=_digest("b"),
        config=SimpleNamespace(
            benchmark=SimpleNamespace(
                data_dir=Path("data"),
                starter_dir=Path("starter"),
                wall_clock_seconds=21_600,
            ),
            runner=SimpleNamespace(finalization_reserve_seconds=3_600),
        ),
    )
    engine = SimpleNamespace(
        resume=lambda _run: SimpleNamespace(
            status="FINALIZATION_REQUIRED",
            deadline_elapsed_seconds=100.0,
            deadline_remaining_seconds=21_500.0,
        ),
        inspect_deadline=lambda _run: SimpleNamespace(hard_expired=False),
        observe_deadline=lambda _run: SimpleNamespace(
            elapsed_seconds=123.0,
            hard_expired=False,
        ),
    )
    store = _FakeCampaignStore()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        production,
        "_load_research_outcome",
        lambda *_args, **_kwargs: (request, artifact_store, research),
    )
    replay_context = SimpleNamespace(
        capabilities=object(),
        score=lambda _scores: {},
        dataset=SimpleNamespace(final=SimpleNamespace(row_count=10)),
    )
    monkeypatch.setattr(
        production,
        "_trusted_replay_context",
        lambda **_kwargs: replay_context,
    )
    monkeypatch.setattr(production, "_environment", lambda *_args: {})
    monkeypatch.setattr(production, "_qualification", lambda *_args: object())
    monkeypatch.setattr(production, "_phase_for_finalization", lambda _store: None)
    monkeypatch.setattr(
        CampaignStore,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: store),
    )

    def fresh(**_kwargs: object) -> tuple[object, MappingProxyType[str, object]]:
        if not generated_succeeds:
            raise RuntimeError("deterministic checkpoint mismatch")
        return checkpoint, MappingProxyType(
            {
                "required": True,
                "completed": True,
                "charged_launch": True,
                "training_rows_include_public_validation": False,
                "checkpoint_sha256": checkpoint.sha256,
                "device": "cpu",
                "resources": {
                    "wall_seconds": 1.0,
                    "peak_rss_bytes": 1024,
                    "disk_bytes": 512,
                },
            }
        )

    generated_candidate = SimpleNamespace(
        candidate_id="generated-candidate",
        bundle_metadata=SimpleNamespace(status=FinalStatus.MATERIALLY_CONFIRMED),
    )
    fallback_candidate = SimpleNamespace(
        candidate_id="official-fm-fallback-seed-4",
        bundle_metadata=SimpleNamespace(status=FinalStatus.BASELINE_REPRODUCED),
    )
    monkeypatch.setattr(production, "_fresh_training_replay", fresh)
    monkeypatch.setattr(
        production,
        "_verify_charged_final_training_terminal",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        production,
        "_generated_replay_candidate",
        lambda **_kwargs: generated_candidate,
    )
    monkeypatch.setattr(
        production,
        "_fallback_replay_candidate",
        lambda **_kwargs: fallback_candidate,
    )

    def ledgers(**kwargs: object) -> tuple[Path, Path]:
        captured["selected_metadata"] = kwargs["selected_metadata"]
        jsonl = run_dir / "experiments.jsonl"
        csv_path = run_dir / "experiments.csv"
        jsonl.write_text("{}\n", encoding="ascii")
        csv_path.write_text("candidate_id\n", encoding="ascii")
        return jsonl, csv_path

    monkeypatch.setattr(production, "_write_experiment_ledgers", ledgers)

    def finalization_request(**kwargs: object) -> object:
        captured["request"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(production, "FinalizationRequest", finalization_request)
    manifest_sha = _digest("c")
    submission_sha = _digest("d")

    def run_finalization(request_value: object, **_kwargs: object) -> object:
        destination = cast(Any, request_value).destination
        destination.mkdir()
        (destination / "submission.csv").write_text("score\n", encoding="ascii")
        selected = generated_candidate if generated_succeeds else fallback_candidate
        return SimpleNamespace(
            selected_candidate_id=selected.candidate_id,
            failures=(),
            selected_status=selected.bundle_metadata.status,
            bundle=SimpleNamespace(
                root=destination,
                manifest_sha256=manifest_sha,
                submission_sha256=submission_sha,
            ),
        )

    monkeypatch.setattr(production, "run_finalization", run_finalization)
    verified = SimpleNamespace(
        root=run_dir / "final",
        manifest_sha256=manifest_sha,
        manifest={},
    )
    monkeypatch.setattr(production, "_close_bundle_directories", lambda _root: verified)
    monkeypatch.setattr(production, "sha256_file", lambda _path: submission_sha)
    monkeypatch.setattr(production, "_complete_campaign", lambda **_kwargs: 99)
    terminal = cast(
        ProductionFinalizationOutcome,
        SimpleNamespace(canonical_bytes=b"{}"),
    )

    def outcome_from_bundle(**kwargs: object) -> object:
        captured["failures"] = kwargs["failures"]
        captured["training_replay"] = kwargs["training_replay"]
        return terminal

    monkeypatch.setattr(production, "_outcome_from_bundle", outcome_from_bundle)

    result = production.finalize_provider_free_campaign(
        run_dir,
        project_root=project_root,
        engine=cast(CampaignEngine, engine),
    )

    assert result is terminal
    final_request = cast(dict[str, object], captured["request"])
    final_candidates = cast(tuple[Any, ...], final_request["candidates"])
    if generated_succeeds:
        assert [item.candidate_id for item in final_candidates] == [
            "generated-candidate",
            "official-fm-fallback-seed-4",
        ]
        assert captured["selected_metadata"] is generated_candidate.bundle_metadata
        assert captured["failures"] == []
    else:
        assert [item.candidate_id for item in final_candidates] == ["official-fm-fallback-seed-4"]
        assert captured["selected_metadata"] is None
        failures = cast(list[Any], captured["failures"])
        assert len(failures) == 1
        assert failures[0].stage == "fresh official-train replay"
        training_replay = cast(dict[str, object], captured["training_replay"])
        assert training_replay["mode"] == "failed_closed_to_official_fm"


def test_public_finalizer_exact_terminal_retry_returns_retained_outcome_without_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    starter_dir = project_root / "starter"
    (run_dir / "production" / "finalization").mkdir(parents=True)
    (run_dir / "production" / "finalization" / "outcome.json").write_text("{}", encoding="ascii")
    data_dir.mkdir(parents=True)
    starter_dir.mkdir()
    artifact_store = ArtifactStore(run_dir / "artifacts")
    request = SimpleNamespace(
        campaign_id="campaign",
        dataset_manifest_digest=_digest("a"),
        config=SimpleNamespace(
            benchmark=SimpleNamespace(
                data_dir=Path("data"),
                starter_dir=Path("starter"),
                wall_clock_seconds=21_600,
            ),
            runner=SimpleNamespace(finalization_reserve_seconds=3_600),
        ),
    )
    research = SimpleNamespace(selection=None, campaign_id="campaign", digest=_digest("b"))
    engine = SimpleNamespace(
        resume=lambda _run: SimpleNamespace(
            status="COMPLETED",
            deadline_elapsed_seconds=1.0,
            deadline_remaining_seconds=21_599.0,
        ),
        inspect_deadline=lambda _run: SimpleNamespace(hard_expired=False),
        observe_deadline=lambda _run: SimpleNamespace(elapsed_seconds=1.0, hard_expired=False),
    )
    retained = cast(ProductionFinalizationOutcome, object())

    monkeypatch.setattr(
        production,
        "_load_research_outcome",
        lambda *_args, **_kwargs: (request, artifact_store, research),
    )
    monkeypatch.setattr(production, "_trusted_replay_context", lambda **_kwargs: object())
    monkeypatch.setattr(production, "_environment", lambda *_args: {})
    monkeypatch.setattr(production, "_qualification", lambda *_args: object())
    monkeypatch.setattr(
        CampaignStore,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: _FakeCampaignStore()),
    )
    monkeypatch.setattr(
        production,
        "_load_production_outcome",
        lambda *_args, **_kwargs: retained,
    )
    monkeypatch.setattr(
        production,
        "run_finalization",
        lambda *_args, **_kwargs: pytest.fail("exact retry must not republish"),
    )

    assert (
        production.finalize_provider_free_campaign(
            run_dir,
            project_root=project_root,
            engine=cast(CampaignEngine, engine),
        )
        is retained
    )


def test_positive_submaterial_delta_is_explicitly_reported_as_unconfirmed() -> None:
    improved = production._derive_bundle_status(_generated_manifest(0.001))
    rationale, limitations = production._selection_report_language(
        generated=True,
        status=improved,
    )

    assert improved is FinalStatus.VALIDATION_IMPROVED
    assert "materially unconfirmed" in rationale
    assert ">0.002" in rationale
    assert "official FM remains retained" in rationale
    assert len(limitations) == 1
    assert "materially unconfirmed" in limitations[0]

    material_rationale, material_limitations = production._selection_report_language(
        generated=True,
        status=FinalStatus.MATERIALLY_CONFIRMED,
    )
    assert "materially confirmed" in material_rationale
    assert "unconfirmed" not in material_rationale
    assert material_limitations == ()

    baseline_rationale, baseline_limitations = production._selection_report_language(
        generated=False,
        status=FinalStatus.BASELINE_REPRODUCED,
    )
    assert "Immutable official FM seed 4" in baseline_rationale
    assert "not agent-generated" in baseline_rationale
    assert baseline_limitations == ()


@pytest.mark.parametrize(
    ("torch_version", "expected"),
    [
        (None, ("research-tree",)),
        ("2.13.0", ("research-tree", "research-neural")),
    ],
)
def test_dependency_groups_are_derived_from_the_locked_package_profile(
    torch_version: str | None,
    expected: tuple[str, ...],
) -> None:
    environment = {
        "packages": {
            "lightgbm": "4.7.0",
            "numpy": "2.5.2",
            "psutil": "7.2.2",
            "torch": torch_version,
        }
    }

    assert production._dependency_groups(environment) == expected


def test_judge_report_quantifies_scripted_calls_and_manifest_limitations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_digest = _digest("a")
    reflected_digest = _digest("b")
    reflection_request = _digest("c")
    reflection_response = _digest("d")
    lineage = SimpleNamespace(
        stage=FullCampaignStage.LINEAGE_READY,
        evidence={
            "lineage": {
                "provider": "scripted",
                "live_provider_used": False,
                "model_calls": [
                    {
                        "operation": "propose",
                        "request_digest": _digest("1"),
                        "response_digest": _digest("2"),
                    },
                    {
                        "operation": "implement",
                        "request_digest": _digest("3"),
                        "response_digest": _digest("4"),
                    },
                ],
            }
        },
    )
    science = SimpleNamespace(
        stage=FullCampaignStage.SCIENCE_COMPLETE,
        evidence={
            "portfolio_count": 1,
            "portfolio_cap": 1,
            "portfolio_cap_reason": "bounded_high_value_lambdarank_branch_prioritized",
            "research_stage_counts": {
                "branches_attempted": 3,
                "proposal_responses_accepted": 3,
                "implementation_responses_accepted": 2,
                "repair_responses_accepted": 1,
                "branches_rejected_pre_execution": 2,
                "candidates_admitted": 1,
                "training_started": 1,
                "inner_evaluations_completed": 1,
                "outer_evaluations_completed": 0,
            },
            "research_rejection_summary": {
                "branches_rejected_pre_execution": 2,
                "root_counts": [
                    {
                        "fingerprint": (_digest("1")),
                        "stage": "implementation",
                        "category": "candidate_path_policy",
                        "code": "forbidden_basename",
                        "subject": "baseline.py",
                        "count": 2,
                    }
                ],
                "terminal_counts": [
                    {
                        "fingerprint": _digest("2"),
                        "stage": "repair",
                        "category": "materiality",
                        "code": "declared_symbol_unchanged",
                        "subject": "main",
                        "count": 2,
                    }
                ],
                "examples": [
                    {
                        "scientific_iteration": 2,
                        "candidate_id": "candidate-02",
                        "proposal_family": "pairwise:bpr",
                        "proposal_signature": _digest("e"),
                        "role": "root",
                        "fingerprint": (_digest("1")),
                        "diagnostic": ("reserved candidate filename is forbidden: baseline.py"),
                    },
                    {
                        "scientific_iteration": 2,
                        "candidate_id": "candidate-02",
                        "proposal_family": "pairwise:bpr",
                        "proposal_signature": _digest("e"),
                        "role": "terminal",
                        "fingerprint": _digest("2"),
                        "diagnostic": "declared material symbol did not change: main",
                    },
                ],
                "counts_truncated": False,
                "examples_truncated": False,
            },
        },
    )
    reflected = SimpleNamespace(
        stage=FullCampaignStage.REFLECTED,
        digest=reflected_digest,
        evidence={
            "reflection_request_digest": reflection_request,
            "reflection_response_digest": reflection_response,
            "portfolio_count": 1,
            "portfolio_cap_reason": "bounded_high_value_lambdarank_branch_prioritized",
        },
    )
    terminal = SimpleNamespace(
        stage=FullCampaignStage.FINALIZATION_REQUIRED,
        request_digest=request_digest,
    )
    progress = SimpleNamespace(checkpoints=lambda: (lineage, science, reflected, terminal))
    monkeypatch.setattr(
        production,
        "FullCampaignProgressLedger",
        lambda *_args, **_kwargs: progress,
    )
    outcome = cast(
        Any,
        SimpleNamespace(
            run_dir=tmp_path,
            request_digest=request_digest,
            progress_predecessor_digest=reflected_digest,
            reflection_request_digest=reflection_request,
            reflection_response_digest=reflection_response,
        ),
    )

    facts = production._judge_progress_facts(outcome)

    assert facts.provider_usage == (
        "Research-model calls=3 (PROPOSE+IMPLEMENT+REFLECT); provider=scripted; "
        "network/API calls=0; replay provider calls=0."
    )
    assert "Advanced WP7 branches were not entered" in facts.advanced_branch_disposition
    assert "not an advanced-branch failure" in facts.advanced_branch_disposition
    assert facts.research_progress == (
        "Research admission: branches attempted=3; proposal responses accepted=3; "
        "implementation responses accepted=2; repair responses accepted=1; "
        "rejected pre-execution=2; candidates admitted=1; training started=1; "
        "inner evaluations completed=1; outer evaluations completed=0."
    )
    assert facts.research_rejections == (
        "Research rejection roots: implementation/candidate_path_policy/"
        f"forbidden_basename/baseline.py [{_digest('1')}] x2.",
        "Research rejection terminals: repair/materiality/declared_symbol_unchanged/"
        f"main [{_digest('2')}] x2.",
        f"Research rejection example (root): {_digest('1')}; "
        "reserved candidate filename is forbidden: baseline.py",
        f"Research rejection example (terminal): {_digest('2')}; "
        "declared material symbol did not change: main",
    )
    limitations = production._bundle_known_limitations(
        generated=True,
        status=FinalStatus.VALIDATION_IMPROVED,
        judge_facts=facts,
    )
    assert any("materially unconfirmed" in item for item in limitations)
    assert any("inclusive production resource receipt" in item for item in limitations)
    assert facts.advanced_branch_disposition in limitations


def test_provider_unavailable_report_records_zero_research_and_replay_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_digest = _digest("a")
    reflected_digest = _digest("b")
    checkpoints = (
        SimpleNamespace(
            stage=FullCampaignStage.LINEAGE_READY,
            evidence={"provider_diagnostic": {"code": "credential_unavailable"}},
        ),
        SimpleNamespace(
            stage=FullCampaignStage.SCIENCE_COMPLETE,
            evidence={"scientific_result_digest": None},
        ),
        SimpleNamespace(
            stage=FullCampaignStage.REFLECTED,
            digest=reflected_digest,
            evidence={
                "reflection_request_digest": None,
                "reflection_response_digest": None,
                "portfolio_count": 0,
                "portfolio_cap_reason": "configured_provider_unavailable",
            },
        ),
        SimpleNamespace(
            stage=FullCampaignStage.FINALIZATION_REQUIRED,
            request_digest=request_digest,
        ),
    )
    monkeypatch.setattr(
        production,
        "FullCampaignProgressLedger",
        lambda *_args, **_kwargs: SimpleNamespace(checkpoints=lambda: checkpoints),
    )
    outcome = cast(
        Any,
        SimpleNamespace(
            run_dir=tmp_path,
            request_digest=request_digest,
            progress_predecessor_digest=reflected_digest,
            reflection_request_digest=None,
            reflection_response_digest=None,
        ),
    )

    facts = production._judge_progress_facts(outcome)

    assert facts.provider_usage == (
        "Research-model calls=0 (none); provider=configured_provider_unavailable; "
        "network/API calls=0; replay provider calls=0."
    )
    assert facts.portfolio_count == 0
    assert facts.portfolio_cap_reason == "configured_provider_unavailable"
    assert facts.research_outcome == (
        "Research did not start because the configured provider was unavailable."
    )

    checkpoints[1].evidence["provider_usage"] = {
        "model": "gpt-5.6-sol",
        "input_tokens": 300,
        "cached_input_tokens": 0,
        "output_tokens": 120,
        "reasoning_tokens": 60,
        "total_tokens": 420,
        "estimated_cost_usd": "0.0036",
        "unaccounted_attempts": 0,
        "transcript_count": 3,
        "provider_wall_seconds": 30.0,
    }

    attempted = production._judge_progress_facts(outcome)

    assert "Research-model attempts=3" in attempted.provider_usage
    assert "provider=configured_provider_unavailable" in attempted.provider_usage
    assert "total tokens=420" in attempted.provider_usage

    checkpoints[1].evidence["portfolio_count"] = 0
    checkpoints[1].evidence["portfolio_cap_reason"] = "runtime_provider_unavailable"
    checkpoints[2].evidence["portfolio_cap_reason"] = "runtime_provider_unavailable"
    checkpoints[0].evidence["provider_diagnostic"]["attempts"] = 3

    runtime_failure = production._judge_progress_facts(outcome)

    assert "provider=runtime_provider_failure" in runtime_failure.provider_usage
    assert runtime_failure.research_outcome == (
        "Research started, but the provider failed at runtime after durable attempts."
    )

    checkpoints[0].evidence = {
        "admission_closed": True,
        "reason": "repeated_pre_admission_failure",
    }
    checkpoints[1].evidence = {
        "admission_closed": True,
        "reason": "repeated_pre_admission_failure",
        "portfolio_count": 0,
        "portfolio_cap_reason": "repeated_pre_admission_failure",
    }
    checkpoints[2].evidence.update(
        {
            "admission_closed": True,
            "reason": "repeated_pre_admission_failure",
            "portfolio_cap_reason": "repeated_pre_admission_failure",
        }
    )

    admission_closed = production._judge_progress_facts(outcome)

    assert admission_closed.research_outcome == (
        "Research admission closed before a candidate was admitted; controller reason="
        "repeated_pre_admission_failure."
    )


def test_judge_report_accepts_live_provider_and_quantifies_tokens_and_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_digest = _digest("a")
    reflected_digest = _digest("b")
    reflection_request = _digest("c")
    reflection_response = _digest("d")
    usage = {
        "model": "gpt-5.6-sol",
        "input_tokens": 1200,
        "cached_input_tokens": 200,
        "output_tokens": 500,
        "reasoning_tokens": 300,
        "total_tokens": 1700,
        "estimated_cost_usd": "0.014",
        "unaccounted_attempts": 0,
        "transcript_count": 6,
        "provider_wall_seconds": 12.5,
    }
    checkpoints = (
        SimpleNamespace(
            stage=FullCampaignStage.LINEAGE_READY,
            evidence={
                "lineage": {
                    "provider": "openai",
                    "live_provider_used": True,
                    "model_calls": [
                        {
                            "operation": "propose",
                            "request_digest": _digest("1"),
                            "response_digest": _digest("2"),
                        },
                        {
                            "operation": "implement",
                            "request_digest": _digest("3"),
                            "response_digest": _digest("4"),
                        },
                    ],
                }
            },
        ),
        SimpleNamespace(
            stage=FullCampaignStage.SCIENCE_COMPLETE,
            evidence={
                "portfolio_count": 2,
                "portfolio_cap": 50,
                "portfolio_cap_reason": "exact_terminal_condition_reached",
                "provider_usage": usage,
            },
        ),
        SimpleNamespace(
            stage=FullCampaignStage.REFLECTED,
            digest=reflected_digest,
            evidence={
                "reflection_request_digest": reflection_request,
                "reflection_response_digest": reflection_response,
                "portfolio_count": 2,
                "portfolio_cap_reason": "exact_terminal_condition_reached",
                "provider_usage": usage,
            },
        ),
        SimpleNamespace(
            stage=FullCampaignStage.FINALIZATION_REQUIRED,
            request_digest=request_digest,
        ),
    )
    monkeypatch.setattr(
        production,
        "FullCampaignProgressLedger",
        lambda *_args, **_kwargs: SimpleNamespace(checkpoints=lambda: checkpoints),
    )
    outcome = cast(
        Any,
        SimpleNamespace(
            run_dir=tmp_path,
            request_digest=request_digest,
            progress_predecessor_digest=reflected_digest,
            reflection_request_digest=reflection_request,
            reflection_response_digest=reflection_response,
        ),
    )

    facts = production._judge_progress_facts(outcome)

    assert "provider=openai" in facts.provider_usage
    assert "model=gpt-5.6-sol" in facts.provider_usage
    assert "total tokens=1700" in facts.provider_usage
    assert "estimated API cost USD=0.014" in facts.provider_usage
    assert "replay provider calls=0" in facts.provider_usage

    usage.update(
        {
            "base_url": "https://fallback.example/v1",
            "context_limits": {
                "context_length": 1_050_000,
                "max_completion_tokens": 128_000,
                "source": "openrouter-model-metadata",
            },
            "active_slot": "fallback",
            "failover_count": 1,
            "failover_events": [
                {
                    "operation": "implement",
                    "from_slot": "main",
                    "to_slot": "fallback",
                    "failure": {
                        "slot": "main",
                        "code": "http",
                        "operation": "implement",
                        "attempts": 3,
                        "status_code": 429,
                    },
                }
            ],
            "provider_chain": [
                {
                    "slot": "main",
                    "model": "gpt-5.6-sol",
                    "base_url": "https://main.example/v1",
                    "credential_env": "INFERENCE_MAIN_API_KEY",
                    "context_limits": None,
                    "input_tokens": 1000,
                    "cached_input_tokens": 100,
                    "output_tokens": 300,
                    "reasoning_tokens": 200,
                    "total_tokens": 1300,
                    "estimated_cost_usd": "0.010",
                    "unaccounted_attempts": 0,
                    "transcript_count": 4,
                },
                {
                    "slot": "fallback",
                    "model": "fallback-model",
                    "base_url": "https://fallback.example/v1",
                    "credential_env": "INFERENCE_FALLBACK_API_KEY",
                    "context_limits": {
                        "context_length": 1_050_000,
                        "max_completion_tokens": 128_000,
                        "source": "openrouter-model-metadata",
                    },
                    "input_tokens": 200,
                    "cached_input_tokens": 100,
                    "output_tokens": 200,
                    "reasoning_tokens": 100,
                    "total_tokens": 400,
                    "estimated_cost_usd": "0.004",
                    "unaccounted_attempts": 0,
                    "transcript_count": 2,
                },
            ],
        }
    )

    chained_facts = production._judge_progress_facts(outcome)

    assert "active slot=fallback" in chained_facts.provider_usage
    assert "failovers=1" in chained_facts.provider_usage
    assert "active base URL=https://fallback.example/v1" in chained_facts.provider_usage

    usage["retry_wait_seconds"] = 3.5
    cast(list[dict[str, object]], usage["provider_chain"])[0]["retry_wait_seconds"] = 2.0
    cast(list[dict[str, object]], usage["provider_chain"])[1]["retry_wait_seconds"] = 1.5

    retry_facts = production._judge_progress_facts(outcome)

    assert "retry wait seconds=3.5" in retry_facts.provider_usage

    usage["context_limits"] = {
        "context_length": 128_000,
        "max_completion_tokens": 1_050_000,
        "source": "invalid",
    }
    with pytest.raises(
        production.ProductionFinalizationError,
        match="live provider usage evidence is malformed",
    ):
        production._judge_progress_facts(outcome)


def test_fallback_report_includes_research_admission_and_rejection_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = production._JudgeProgressFacts(
        provider_usage="Research-model attempts=6; replay provider calls=0.",
        portfolio_count=0,
        portfolio_cap=50,
        portfolio_cap_reason="repeated_pre_admission_failure",
        advanced_branch_disposition="Advanced branches were not entered.",
        research_progress=(
            "Research admission: branches attempted=3; proposal responses accepted=3; "
            "implementation responses accepted=2; repair responses accepted=2; "
            "rejected pre-execution=3; candidates admitted=0; training started=0; "
            "inner evaluations completed=0; outer evaluations completed=0."
        ),
        research_outcome=(
            "Research admission closed before a candidate was admitted; "
            "controller reason=repeated_pre_admission_failure."
        ),
        research_rejections=(
            "Research rejection roots: candidate_path_policy:forbidden_basename:baseline.py x3.",
            "Research rejection terminals: materiality:declared_symbol_unchanged:main x2.",
        ),
    )
    monkeypatch.setattr(production, "_judge_progress_facts", lambda _outcome: facts)
    monkeypatch.setattr(
        production,
        "_baseline_mean",
        lambda _qualification: MetricEvidence(
            label="official-fm",
            tier="qualified baseline",
            gauc=0.6,
            ndcg_at_5=0.4,
            primary=0.5,
            seeds=(0, 1, 2, 3, 4),
            note="Official fallback.",
        ),
    )
    monkeypatch.setattr(production, "_report_peak_rss_bytes", lambda *_args: 1024)

    context = production._report_context(
        candidate_id="official-fm-fallback-seed-4",
        parent_id="official-fm-fallback-seed-4",
        metrics=_metrics(0.6, 0.4),
        qualification=cast(Any, SimpleNamespace(benchmark_digest=_digest("a"))),
        outcome=cast(
            Any,
            SimpleNamespace(
                selection=None,
                manual_interventions=0,
                fallback_receipt_digest=_digest("b"),
            ),
        ),
        generated=False,
        failures=(),
        confirmation=None,
        campaign_wall_seconds=10.0,
        launch_count=0,
    )

    assert context.failures_and_recoveries[:4] == (
        facts.research_outcome,
        facts.research_progress,
        *facts.research_rejections,
    )
    assert "not agent-generated" in context.selection_rationale


def test_current_runtime_reproof_rejects_source_or_lock_drift_and_is_path_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "checkout-a"
    second = tmp_path / "different" / "checkout-b"
    first.mkdir()
    second.mkdir(parents=True)
    source_digest = _digest("a")
    environment_digest = _digest("b")
    lock_digest = _digest("c")
    closure = production._RuntimeIdentityClosure(
        project_source_digest=source_digest,
        environment_digest=environment_digest,
        uv_lock_sha256=lock_digest,
        dependency_groups=("research-tree",),
    )
    manifest = {
        "schema_version": 2,
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "Test", "release": "1", "machine": "arm64"},
        "packages": {
            "lightgbm": "4.7.0",
            "numpy": "2.5.2",
            "psutil": "7.2.2",
            "torch": None,
        },
        "uv_lock_sha256": lock_digest,
        "digest": environment_digest,
    }
    monkeypatch.setattr(
        production,
        "hash_source_tree",
        lambda _root: SimpleNamespace(digest=source_digest),
    )
    monkeypatch.setattr(
        production,
        "capture_environment_identity",
        lambda _root: SimpleNamespace(
            digest=environment_digest,
            manifest=lambda: dict(manifest),
        ),
    )
    monkeypatch.setattr(
        production,
        "environment_identity_digest",
        lambda _manifest: environment_digest,
    )

    assert production._verify_current_runtime(first, closure)[0] == first.resolve()
    assert production._verify_current_runtime(second, closure)[0] == second.resolve()

    monkeypatch.setattr(
        production,
        "hash_source_tree",
        lambda _root: SimpleNamespace(digest=_digest("d")),
    )
    with pytest.raises(ProductionFinalizationError, match="differs from bundle"):
        production._verify_current_runtime(first, closure)

    monkeypatch.setattr(
        production,
        "hash_source_tree",
        lambda _root: SimpleNamespace(digest=source_digest),
    )
    changed = dict(manifest)
    changed["uv_lock_sha256"] = _digest("e")
    monkeypatch.setattr(
        production,
        "capture_environment_identity",
        lambda _root: SimpleNamespace(
            digest=environment_digest,
            manifest=lambda: changed,
        ),
    )
    with pytest.raises(ProductionFinalizationError, match="differs from bundle"):
        production._verify_current_runtime(first, closure)


def _install_closed_replay_scratch_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    organizer_hook: Any,
) -> tuple[Path, Path, Path, Path, str]:
    bundle_root = tmp_path / "final"
    data_dir = tmp_path / "data"
    project_root = tmp_path / "project"
    starter_dir = tmp_path / "starter"
    for directory in (bundle_root, data_dir, project_root, starter_dir):
        directory.mkdir()
    submission = bundle_root / "submission.csv"
    submission.write_text("row_id,prediction\n0,0.5\n", encoding="ascii")
    submission_sha256 = hashlib.sha256(submission.read_bytes()).hexdigest()
    scratch = tmp_path / "private-replay-scratch"
    digests = {name: _digest(name) for name in "12345678"}
    expected_data_sha256 = digests["7"]
    bundle = SimpleNamespace(
        root=bundle_root,
        manifest={"selection": {"selected_experiment": "official-fm-fallback-seed-4"}},
        manifest_sha256=digests["8"],
    )
    environment = {"trusted": True}
    closure = SimpleNamespace(
        project_source_digest=digests["1"],
        environment_digest=digests["2"],
        uv_lock_sha256=digests["3"],
        dependency_groups=("research-tree",),
    )
    base_identity = SimpleNamespace(
        source_sha256=digests["1"],
        config_sha256=digests["2"],
        features_sha256=digests["3"],
        checkpoint_sha256=digests["4"],
        validation_prediction_artifact_sha256=digests["5"],
        data_sha256=expected_data_sha256,
        environment_sha256=digests["2"],
    )

    class FakeArtifactStore:
        def __init__(self, _root: Path) -> None:
            pass

        def put_directory(self, path: Path, **_kwargs: object) -> object:
            digest = (
                base_identity.source_sha256
                if path.name == "source"
                else (
                    base_identity.features_sha256
                    if path.name == "preprocessing"
                    else base_identity.checkpoint_sha256
                )
            )
            return SimpleNamespace(sha256=digest)

        def put_file(self, path: Path, **_kwargs: object) -> object:
            parent = path.parent.name
            digest = {
                "config": base_identity.config_sha256,
                "preprocessing": base_identity.features_sha256,
                "model": base_identity.checkpoint_sha256,
                "validation-evidence": base_identity.validation_prediction_artifact_sha256,
            }[parent]
            return SimpleNamespace(sha256=digest)

    class FakeOfficialRecipe:
        data_sha256 = expected_data_sha256

    def fake_mkdtemp(**_kwargs: object) -> str:
        scratch.mkdir(mode=0o700)
        return str(scratch)

    def fake_clean_replay(*_args: object, **_kwargs: object) -> object:
        replayed = scratch / "replayed-submission.csv"
        replayed.write_bytes(submission.read_bytes())
        return SimpleNamespace(final_submission=replayed, evidence=object())

    def fake_checker(_submission: Path, **kwargs: object) -> object:
        organizer_hook(cast(Path, kwargs["scratch_dir"]))
        return SimpleNamespace(submission_sha256=submission_sha256)

    monkeypatch.setattr(production, "_verify_closed_bundle", lambda _path: bundle)
    monkeypatch.setattr(production, "_strict_json", lambda *_args, **_kwargs: environment)
    monkeypatch.setattr(production, "_runtime_identity_closure", lambda *_args: closure)
    monkeypatch.setattr(
        production,
        "_verify_current_runtime",
        lambda *_args: (project_root.resolve(), environment),
    )
    monkeypatch.setattr(production, "_find_starter", lambda *_args: starter_dir.resolve())
    monkeypatch.setattr(
        production,
        "_trusted_replay_context",
        lambda **_kwargs: SimpleNamespace(
            dataset=SimpleNamespace(valid=SimpleNamespace(row_count=1)),
            capabilities=object(),
            score=lambda _scores: {},
        ),
    )
    monkeypatch.setattr(production, "_bundle_identity", lambda _manifest: base_identity)
    monkeypatch.setattr(
        production,
        "environment_identity_digest",
        lambda _environment: base_identity.environment_sha256,
    )
    monkeypatch.setattr(cast(Any, production).tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(production, "ArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(
        production,
        "load_replay_recipe",
        lambda *_args, **_kwargs: FakeOfficialRecipe(),
    )
    monkeypatch.setattr(production, "OfficialFMReplayRecipe", FakeOfficialRecipe)
    monkeypatch.setattr(production, "_load_npy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(production, "prediction_digest", lambda _scores: digests["6"])
    monkeypatch.setattr(
        production,
        "FrozenReplayIdentity",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        production,
        "ReplayArtifacts",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        production,
        "CleanReplayRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(production, "load_replay_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(production, "run_clean_replay", fake_clean_replay)
    monkeypatch.setattr(production, "check_final_submission", fake_checker)
    monkeypatch.setattr(
        production,
        "FinalBundleReplayOutcome",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return bundle_root, data_dir, project_root, scratch, expected_data_sha256


def test_closed_replay_creates_private_organizer_checker_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def require_safe_scratch(path: Path) -> None:
        assert path.is_dir()
        assert not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o700

    bundle, data, project, scratch, expected = _install_closed_replay_scratch_harness(
        tmp_path,
        monkeypatch,
        organizer_hook=require_safe_scratch,
    )

    result = production.replay_final_bundle(
        bundle,
        data,
        expected,
        project_root=project,
    )

    assert result.candidate_id == "official-fm-fallback-seed-4"
    assert not scratch.exists()


def test_closed_replay_removes_read_only_artifacts_from_private_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def close_replay_artifacts(path: Path) -> None:
        retained = path / "read-only" / "nested"
        retained.mkdir(parents=True)
        artifact = retained / "artifact.bin"
        artifact.write_bytes(b"immutable replay evidence")
        artifact.chmod(0o400)
        retained.chmod(0o500)
        retained.parent.chmod(0o500)

    bundle, data, project, scratch, expected = _install_closed_replay_scratch_harness(
        tmp_path,
        monkeypatch,
        organizer_hook=close_replay_artifacts,
    )

    production.replay_final_bundle(
        bundle,
        data,
        expected,
        project_root=project,
    )

    assert not scratch.exists()


def test_judge_ledger_exports_exact_lineage_commands_resources_and_portfolio_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request_digest = _digest("1")
    reflected_digest = _digest("2")
    scientific_digest = _digest("3")
    source_snapshot = SimpleNamespace(manifest=lambda: {"kind": "source", "entries": []})
    selection = SimpleNamespace(
        candidate_id="generated-candidate",
        parent_source_digest=_digest("4"),
        representative_seed=0,
        source_snapshot=source_snapshot,
        manifest=lambda: {
            "candidate_id": "generated-candidate",
            "source_digest": _digest("5"),
        },
    )
    outcome = SimpleNamespace(
        progress_predecessor_digest=reflected_digest,
        fallback_candidate_id="official-fm-fallback-seed-4",
        fallback_receipt_digest=_digest("6"),
        qualification_manifest_digest=_digest("7"),
        scientific_result_digest=scientific_digest,
        reflection_request_digest=_digest("8"),
        reflection_response_digest=_digest("9"),
        reflection_transcript=SimpleNamespace(manifest=lambda: {"sha256": _digest("a")}),
        selection=selection,
        manual_interventions=0,
    )
    request = SimpleNamespace(
        digest=request_digest,
        campaign_id="campaign",
        benchmark_digest=_digest("b"),
        starter_manifest_digest=_digest("c"),
        dataset_manifest_digest=_digest("d"),
        source_digest=_digest("e"),
        environment_digest=_digest("f"),
        config=SimpleNamespace(digest=_digest("0")),
    )

    evidence_by_stage: dict[FullCampaignStage, dict[str, object]] = {
        stage: {"stage_evidence": stage.value} for stage in FullCampaignStage
    }
    evidence_by_stage[FullCampaignStage.LINEAGE_READY] = {
        "material_generated_source": True,
        "lineage": {
            "candidate_id": "generated-candidate",
            "proposal_request_digest": _digest("a"),
            "proposal_digest": _digest("b"),
            "implementation_request_digest": _digest("c"),
            "diff_sha256": _digest("d"),
            "model_calls": [
                {
                    "operation": "propose",
                    "request_digest": _digest("a"),
                    "response_digest": _digest("b"),
                },
                {
                    "operation": "implement",
                    "request_digest": _digest("c"),
                    "response_digest": _digest("d"),
                },
            ],
        },
    }
    evidence_by_stage[FullCampaignStage.SCIENCE_COMPLETE] = {
        "scientific_result_digest": scientific_digest,
        "portfolio_count": 1,
        "portfolio_cap": 1,
        "portfolio_cap_reason": "bounded_high_value_lambdarank_branch_prioritized",
        "declared_portfolio_cap_reached": True,
    }
    evidence_by_stage[FullCampaignStage.REFLECTED] = {
        "scientific_result_digest": scientific_digest,
        "selected_candidate_id": "generated-candidate",
        "portfolio_count": 1,
        "portfolio_cap_reason": "bounded_high_value_lambdarank_branch_prioritized",
    }

    checkpoints: list[SimpleNamespace] = []
    previous: str | None = None
    for sequence, stage in enumerate(FullCampaignStage, start=1):
        digest = (
            reflected_digest
            if stage is FullCampaignStage.REFLECTED
            else _digest(format(sequence, "x"))
        )
        evidence = evidence_by_stage[stage]
        checkpoint = SimpleNamespace(
            sequence=sequence,
            stage=stage,
            request_digest=request_digest,
            digest=digest,
            evidence=evidence,
        )
        checkpoint.manifest = lambda item=checkpoint, prior=previous: {
            "sequence": item.sequence,
            "stage": item.stage.value,
            "request_digest": item.request_digest,
            "previous_digest": prior,
            "evidence": item.evidence,
            "digest": item.digest,
        }
        checkpoints.append(checkpoint)
        previous = digest

    progress = SimpleNamespace(checkpoints=lambda: tuple(checkpoints))
    monkeypatch.setattr(
        production,
        "FullCampaignProgressLedger",
        lambda *_args, **_kwargs: progress,
    )
    scientific_result = {
        "digest": scientific_digest,
        "incumbent_candidate_id": "generated-candidate",
        "candidate_outcomes": [],
        "convergence": {"best_primary": 0.51},
        "stop_reason": "candidates_exhausted",
        "elapsed_seconds": 12.5,
    }
    monkeypatch.setattr(
        production,
        "_scientific_result_document",
        lambda **_kwargs: scientific_result,
    )
    scientific_row = production._ledger_row(
        "scientific_run",
        _digest("a"),
        {
            "request_digest": _digest("a"),
            "metrics": _metrics(0.61, 0.41),
            "resources": {"wall_seconds": 4.0, "peak_rss_bytes": 4096},
        },
        summary={
            "candidate_id": "generated-candidate",
            "tier": "outer_matched_seed",
            "seed": 0,
            "status": "succeeded",
            "primary": 0.51,
            "wall_seconds": 4.0,
            "peak_rss_bytes": 4096,
        },
    )
    monkeypatch.setattr(
        production,
        "_scientific_record_rows",
        lambda **_kwargs: [scientific_row],
    )

    identity = SimpleNamespace(
        campaign_id="campaign",
        config_digest=request.config.digest,
        benchmark_digest=request.benchmark_digest,
        starter_manifest_digest=request.starter_manifest_digest,
        dataset_manifest_digest=request.dataset_manifest_digest,
        source_digest=request.source_digest,
        environment_digest=request.environment_digest,
        hard_deadline_utc="2026-08-28T00:00:00+00:00",
        max_launches=50,
        outer_query_limit=6,
        created_at="2026-08-27T00:00:00+00:00",
    )
    health = SimpleNamespace(
        journal_mode="wal",
        foreign_keys=True,
        synchronous=2,
        user_version=1,
        quick_check="ok",
        schema_digest=_digest("a"),
        catalog_digest=_digest("b"),
    )
    snapshot = SimpleNamespace(
        campaign_id="campaign",
        revision=20,
        status="FINALIZING",
        phase="finalizing",
        max_launches=50,
        launches_used=10,
        launches_remaining=40,
        outer_query_limit=6,
        outer_queries_used=1,
        outer_queries_remaining=5,
        convergence_state={"best_primary": 0.51},
    )
    launch = SimpleNamespace(
        launch_id="launch-001",
        launch_number=1,
        reservation_key="reservation",
        category="diverse_inner_screen",
        original_category="diverse_inner_screen",
        purpose="screen generated candidate",
        state="SUCCEEDED",
        charged=True,
        event_seq=2,
        experiment_id="iteration-01",
        scientific_iteration=1,
        seed=0,
        start_receipt_digest=_digest("c"),
    )
    execution_values: dict[str, object] = {
        "execution_id": "scientific-train-001",
        "experiment_id": "iteration-01",
        "launch_id": "launch-001",
        "launch_number": 1,
        "launch_category": "diverse_inner_screen",
        "original_launch_category": "diverse_inner_screen",
        "kind": "train",
        "tier": "fold_b_screen",
        "seed": 0,
        "command": ("python", "candidate.py", "train"),
        "status": "SUCCEEDED",
        "nonce": "nonce",
        "source_digest": _digest("d"),
        "config_digest": _digest("e"),
        "capability_digest": _digest("f"),
        "environment_digest": request.environment_digest,
        "data_digest": _digest("1"),
        "checkpoint_digest": _digest("2"),
        "process_record_digest": _digest("3"),
        "process_record": {"wall_seconds": 3.0, "peak_tree_rss_bytes": 2048},
        "process_id": 123,
        "process_create_time": 1.0,
        "process_group_id": 123,
        "process_command_digest": _digest("4"),
        "process_environment_digest": _digest("5"),
        "result_digest": _digest("6"),
        "created_at": "2026-08-27T00:00:00+00:00",
        "updated_at": "2026-08-27T00:01:00+00:00",
        "started_at": "2026-08-27T00:00:00+00:00",
        "finished_at": "2026-08-27T00:01:00+00:00",
        "metadata": {"device": "cpu", "threads": 1},
    }
    execution = SimpleNamespace(**execution_values)
    experiment = SimpleNamespace(
        experiment_id="iteration-01",
        iteration_number=1,
        parent_experiment_id=None,
        hypothesis="Causal LambdaRank improves long-view ranking.",
        mechanism="Train grouped trees with causal history features.",
        method_attribution="LightGBM LambdaRank",
        status="REFLECTED",
        metadata={"proposal_id": "proposal-01", "parent_candidate_id": "official-fm"},
        created_at="2026-08-27T00:00:00+00:00",
    )
    proposal = SimpleNamespace(
        proposal_id="proposal-01",
        experiment_id="iteration-01",
        request_digest=_digest("7"),
        response_digest=_digest("8"),
        provider="scripted",
        metadata={"operation": "propose"},
        created_at="2026-08-27T00:00:00+00:00",
    )

    class FakeStore:
        def identity(self) -> object:
            return identity

        def health(self) -> object:
            return health

        def snapshot(self) -> object:
            return snapshot

        def launches(self) -> tuple[object, ...]:
            return (launch,)

        def executions(self) -> tuple[object, ...]:
            return (execution,)

        def experiment(self, experiment_id: str) -> object | None:
            return experiment if experiment_id == "iteration-01" else None

        def proposal(self, proposal_id: str) -> object | None:
            return proposal if proposal_id == "proposal-01" else None

        def artifacts_for(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

        def reallocations(self) -> tuple[object, ...]:
            return ()

    selected_metadata = SimpleNamespace(
        status=FinalStatus.VALIDATION_IMPROVED,
        seed_summary={"derived_status": "validation_improved"},
        inner_fold_results=({"fold": "A"}, {"fold": "B"}),
        validation_metrics={"primary": 0.51},
    )
    kwargs = {
        "run_dir": run_dir,
        "request": request,
        "artifact_store": object(),
        "store": FakeStore(),
        "outcome": outcome,
        "failures": (),
        "selected_metadata": selected_metadata,
    }
    jsonl, csv_path = production._write_experiment_ledgers(**cast(Any, kwargs))
    first_jsonl = jsonl.read_bytes()
    first_csv = csv_path.read_bytes()
    production._write_experiment_ledgers(**cast(Any, kwargs))

    assert jsonl.read_bytes() == first_jsonl
    assert csv_path.read_bytes() == first_csv
    rows = [json.loads(line) for line in first_jsonl.splitlines()]
    by_type = {row["record_type"]: row for row in rows}
    assert {
        "campaign_identity",
        "campaign_progress",
        "campaign_budget_and_convergence",
        "fallback_incumbent",
        "generated_lineage",
        "scientific_result",
        "scientific_run",
        "research_experiment",
        "research_proposal",
        "execution",
        "reflection",
        "portfolio_closure",
        "finalization_selection",
    }.issubset(by_type)
    assert by_type["research_experiment"]["evidence"]["hypothesis"].startswith("Causal")
    assert by_type["research_proposal"]["evidence"]["request_digest"] == _digest("7")
    assert by_type["execution"]["evidence"]["command"] == [
        "python",
        "candidate.py",
        "train",
    ]
    assert by_type["execution"]["summary"]["wall_seconds"] == 3.0
    portfolio = by_type["portfolio_closure"]["evidence"]
    assert portfolio["portfolio_count"] == 1
    assert portfolio["portfolio_cap_reason"] == "bounded_high_value_lambdarank_branch_prioritized"
    assert portfolio["advanced_wp7_branches_entered"] is False
    assert portfolio["advanced_wp7_disposition"].startswith("not_entered")
    assert first_csv.startswith(b"record_type,record_id,candidate_id,parent_id,tier,seed,status")


def _fake_scientific_record(
    *,
    request_digest: str,
    source_snapshot_digest: str,
    source_digest: str,
    config_digest: str,
    environment_digest: str,
    primary: float,
) -> SimpleNamespace:
    """A scientific run record reduced to exactly the fields the exporter reads."""

    empty_artifacts = SimpleNamespace(entries=())
    return SimpleNamespace(
        digest=request_digest,
        request_digest=request_digest,
        source_snapshot_digest=source_snapshot_digest,
        checkpoint=None,
        raw_prediction=None,
        replay_prediction=None,
        scored_prediction=None,
        train_artifacts=empty_artifacts,
        prediction_artifacts=empty_artifacts,
        replay_artifacts=empty_artifacts,
        manifest=lambda: {"request_digest": request_digest},
        evidence=SimpleNamespace(
            digest=request_digest,
            replay_verified=True,
            metrics=SimpleNamespace(primary=primary),
            resources=SimpleNamespace(wall_seconds=4.0, peak_rss_bytes=4096),
            identities=SimpleNamespace(
                source_digest=source_digest,
                config_digest=config_digest,
                environment_digest=environment_digest,
            ),
        ),
    )


def _scientific_record_rows_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: dict[str, SimpleNamespace],
    *,
    selected_request_digest: str,
    selection_snapshot: str,
    selection_source: str,
    selection_config: str,
    environment_digest: str,
) -> list[dict[str, object]]:
    record_root = tmp_path / "production" / "scientific-records"
    record_root.mkdir(parents=True)
    for request_digest in records:
        (record_root / f"{request_digest}.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        production,
        "FileScientificRunEvidenceRepository",
        lambda _root: SimpleNamespace(load=records.get),
    )
    selection = SimpleNamespace(
        candidate_id="generated-candidate",
        source_snapshot=SimpleNamespace(sha256=selection_snapshot),
        source_digest=selection_source,
        config_digest=selection_config,
        matched_seeds=(
            SimpleNamespace(
                seed=0,
                scientific_request_digest=selected_request_digest,
                scientific_record_digest=selected_request_digest,
            ),
        ),
    )
    return production._scientific_record_rows(
        run_dir=tmp_path,
        request=cast(CampaignCreateRequest, SimpleNamespace(environment_digest=environment_digest)),
        outcome=cast(Any, SimpleNamespace(selection=selection)),
        artifact_store=cast(ArtifactStore, SimpleNamespace(verify=lambda _reference: None)),
        scientific_result={"candidate_outcomes": [{"run_digests": [selected_request_digest]}]},
    )


def test_scientific_record_export_keeps_other_candidates_own_source_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A campaign that evaluates several generated candidates still exports.

    Reproduces the live `hard-block-verify-20260830T103424Z` failure: once the campaign promoted
    one candidate, the repository held records from three distinct generated candidates (three
    distinct source-snapshot/source/config triples). The exporter required *every* record to carry
    the selected candidate's triple, so a campaign crashed in finalization precisely when it
    succeeded. Only the environment digest is campaign-wide.
    """

    selected = _digest("a")
    other = _digest("b")
    records = {
        selected: _fake_scientific_record(
            request_digest=selected,
            source_snapshot_digest=_digest("1"),
            source_digest=_digest("2"),
            config_digest=_digest("3"),
            environment_digest=_digest("9"),
            primary=0.61,
        ),
        other: _fake_scientific_record(
            request_digest=other,
            source_snapshot_digest=_digest("4"),
            source_digest=_digest("5"),
            config_digest=_digest("6"),
            environment_digest=_digest("9"),
            primary=0.58,
        ),
    }

    rows = _scientific_record_rows_harness(
        monkeypatch,
        tmp_path,
        records,
        selected_request_digest=selected,
        selection_snapshot=_digest("1"),
        selection_source=_digest("2"),
        selection_config=_digest("3"),
        environment_digest=_digest("9"),
    )

    assert len(rows) == 2
    decisions = {
        row["record_id"]: cast(dict[str, object], row["summary"])["decision"] for row in rows
    }
    assert decisions[selected] == "replay_verified_scientific_result_run"
    assert decisions[other] == "retained_unreferenced_scientific_run"


def test_scientific_record_export_rejects_a_forged_source_identity_mix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Claiming the selected snapshot while carrying a different source digest still fails."""

    selected = _digest("a")
    forged = _digest("b")
    records = {
        selected: _fake_scientific_record(
            request_digest=selected,
            source_snapshot_digest=_digest("1"),
            source_digest=_digest("2"),
            config_digest=_digest("3"),
            environment_digest=_digest("9"),
            primary=0.61,
        ),
        forged: _fake_scientific_record(
            request_digest=forged,
            source_snapshot_digest=_digest("1"),
            source_digest=_digest("5"),
            config_digest=_digest("3"),
            environment_digest=_digest("9"),
            primary=0.58,
        ),
    }

    with pytest.raises(ProductionFinalizationError, match="source, config, or environment"):
        _scientific_record_rows_harness(
            monkeypatch,
            tmp_path,
            records,
            selected_request_digest=selected,
            selection_snapshot=_digest("1"),
            selection_source=_digest("2"),
            selection_config=_digest("3"),
            environment_digest=_digest("9"),
        )


def test_scientific_record_export_rejects_a_changed_environment_on_any_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The environment digest is campaign-wide, so it is still enforced on every record."""

    selected = _digest("a")
    other = _digest("b")
    records = {
        selected: _fake_scientific_record(
            request_digest=selected,
            source_snapshot_digest=_digest("1"),
            source_digest=_digest("2"),
            config_digest=_digest("3"),
            environment_digest=_digest("9"),
            primary=0.61,
        ),
        other: _fake_scientific_record(
            request_digest=other,
            source_snapshot_digest=_digest("4"),
            source_digest=_digest("5"),
            config_digest=_digest("6"),
            environment_digest=_digest("8"),
            primary=0.58,
        ),
    }

    with pytest.raises(ProductionFinalizationError, match="environment identity changed"):
        _scientific_record_rows_harness(
            monkeypatch,
            tmp_path,
            records,
            selected_request_digest=selected,
            selection_snapshot=_digest("1"),
            selection_source=_digest("2"),
            selection_config=_digest("3"),
            environment_digest=_digest("9"),
        )

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.support.completed_run_resources import (
    FINALIZATION_COVERAGE,
    CompletedRunResourceError,
    validate_completed_finalization_resources,
)

BUNDLE_DIGEST = "a" * 64
CHECKPOINT_DIGEST = "b" * 64


def _training_replay() -> dict[str, object]:
    return {
        "required": True,
        "completed": True,
        "mode": "fresh_subprocess_official_train_replay",
        "execution_id": "final-train-fixture",
        "seed": 1,
        "training_rows_include_public_validation": False,
        "training_period": "official_train_20220408_to_20220421",
        "checkpoint_sha256": CHECKPOINT_DIGEST,
        "selected_checkpoint_sha256": CHECKPOINT_DIGEST,
        "exact_checkpoint_bytes": True,
        "charged_launch": True,
        "device": "cpu",
        "resources": {
            "wall_seconds": 40.0,
            "peak_rss_bytes": 2_000_000_000,
            "disk_bytes": 50_000_000,
        },
    }


def _resource_evidence(*, generated: bool = True) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": 1,
        "clock_basis": "durable_max_of_monotonic_and_utc_elapsed",
        "campaign_elapsed_seconds": 17_300.0,
        "hard_wall_seconds": 21_600,
        "finalization_reserve_seconds": 3_600,
        "finalization_started_elapsed_seconds": 17_000.0,
        "finalization_elapsed_seconds": 300.0,
        "coverage": list(FINALIZATION_COVERAGE),
        "within_reserve": True,
        "within_hard_wall": True,
        "aggregate": {
            "family": "production_finalization_total",
            "wall_seconds": 300.0,
            "local_monotonic_wall_seconds": 299.5,
            "cpu_seconds": 280.0,
            "peak_rss_bytes": 4_000_000_000,
            "rows": 1_436_609,
            "evidence_digest": BUNDLE_DIGEST,
            "rss_accounting": (
                "conservative_sum_of_process_lifetime_self_and_child_high_water_marks"
            ),
        },
    }
    if generated:
        evidence["final_training"] = {
            "family": "generated_final_training_replay",
            "wall_seconds": 40.0,
            "peak_rss_bytes": 2_000_000_000,
            "disk_bytes": 50_000_000,
            "device": "cpu",
            "evidence_digest": CHECKPOINT_DIGEST,
        }
    return evidence


def _validate(
    evidence: object,
    *,
    generated: bool = True,
    training_replay: object | None = None,
) -> None:
    replay: object = (
        _training_replay()
        if training_replay is None and generated
        else (
            {"required": False, "completed": False} if training_replay is None else training_replay
        )
    )
    validate_completed_finalization_resources(
        evidence,
        generated_selection=generated,
        expected_bundle_manifest_sha256=BUNDLE_DIGEST,
        training_replay=replay,
    )


def test_generated_and_fallback_accept_only_actual_bounded_aggregate_receipts() -> None:
    generated = validate_completed_finalization_resources(
        _resource_evidence(),
        generated_selection=True,
        expected_bundle_manifest_sha256=BUNDLE_DIGEST,
        training_replay=_training_replay(),
    )
    assert generated.campaign_elapsed_seconds == 17_300.0
    assert generated.finalization_elapsed_seconds == 300.0
    assert generated.aggregate.evidence_digest == BUNDLE_DIGEST
    assert generated.final_training is not None
    assert generated.final_training.evidence_digest == CHECKPOINT_DIGEST

    fallback = validate_completed_finalization_resources(
        _resource_evidence(generated=False),
        generated_selection=False,
        expected_bundle_manifest_sha256=BUNDLE_DIGEST,
        training_replay={"required": False, "completed": False},
    )
    assert fallback.final_training is None


@pytest.mark.parametrize("missing", ["aggregate", "coverage", "within_reserve"])
def test_missing_required_finalization_receipt_fields_fail_closed(missing: str) -> None:
    evidence = _resource_evidence()
    del evidence[missing]
    with pytest.raises(CompletedRunResourceError, match="fields differ"):
        _validate(evidence)


def test_forged_bundle_or_checkpoint_receipt_digest_fails_closed() -> None:
    aggregate_forgery = _resource_evidence()
    aggregate = aggregate_forgery["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["evidence_digest"] = "c" * 64
    with pytest.raises(CompletedRunResourceError, match="closed bundle"):
        _validate(aggregate_forgery)

    checkpoint_forgery = _resource_evidence()
    final_training = checkpoint_forgery["final_training"]
    assert isinstance(final_training, dict)
    final_training["evidence_digest"] = "c" * 64
    with pytest.raises(CompletedRunResourceError, match="checkpoint-bound"):
        _validate(checkpoint_forgery)


def test_reserve_and_hard_wall_overruns_fail_even_when_booleans_are_honest() -> None:
    reserve_overrun = _resource_evidence()
    reserve_overrun["campaign_elapsed_seconds"] = 19_600.5
    reserve_overrun["finalization_started_elapsed_seconds"] = 16_000.0
    reserve_overrun["finalization_elapsed_seconds"] = 3_600.5
    reserve_overrun["within_reserve"] = False
    aggregate = reserve_overrun["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["wall_seconds"] = 3_600.5
    aggregate["local_monotonic_wall_seconds"] = 3_600.0
    with pytest.raises(CompletedRunResourceError, match="3,600-second reserve"):
        _validate(reserve_overrun)

    hard_wall_overrun = _resource_evidence()
    hard_wall_overrun["campaign_elapsed_seconds"] = 21_600.5
    hard_wall_overrun["finalization_started_elapsed_seconds"] = 18_000.0
    hard_wall_overrun["finalization_elapsed_seconds"] = 3_600.5
    hard_wall_overrun["within_reserve"] = False
    hard_wall_overrun["within_hard_wall"] = False
    aggregate = hard_wall_overrun["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["wall_seconds"] = 3_600.5
    aggregate["local_monotonic_wall_seconds"] = 3_600.0
    with pytest.raises(CompletedRunResourceError, match="21,600-second hard wall"):
        _validate(hard_wall_overrun)


def test_research_cannot_consume_the_protected_finalization_reserve() -> None:
    evidence = _resource_evidence()
    evidence["campaign_elapsed_seconds"] = 18_100.0
    evidence["finalization_started_elapsed_seconds"] = 18_000.1
    evidence["finalization_elapsed_seconds"] = 99.9
    aggregate = evidence["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["wall_seconds"] = 99.9
    aggregate["local_monotonic_wall_seconds"] = 99.8
    with pytest.raises(CompletedRunResourceError, match="protected reserve"):
        _validate(evidence)


def test_generated_selection_requires_cross_bound_non_synthetic_training_resources() -> None:
    missing = _resource_evidence()
    del missing["final_training"]
    with pytest.raises(CompletedRunResourceError, match="fields differ"):
        _validate(missing)

    mismatched = _resource_evidence()
    replay = deepcopy(_training_replay())
    resources = replay["resources"]
    assert isinstance(resources, dict)
    resources["wall_seconds"] = 39.0
    with pytest.raises(CompletedRunResourceError, match="wall receipts differ"):
        _validate(mismatched, training_replay=replay)

    rehydrated_without_resources = deepcopy(_training_replay())
    del rehydrated_without_resources["resources"]
    with pytest.raises(CompletedRunResourceError, match=r"training_replay\.resources"):
        _validate(_resource_evidence(), training_replay=rehydrated_without_resources)


def test_measured_process_or_final_training_rss_above_eight_gib_fails() -> None:
    aggregate_overrun = _resource_evidence()
    aggregate = aggregate_overrun["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["peak_rss_bytes"] = 8 * 1024**3
    with pytest.raises(CompletedRunResourceError, match="finalization exceeded the 8-GiB"):
        _validate(aggregate_overrun)

    training_overrun = _resource_evidence()
    final_training = training_overrun["final_training"]
    assert isinstance(final_training, dict)
    final_training["peak_rss_bytes"] = 8 * 1024**3
    replay = _training_replay()
    replay_resources = replay["resources"]
    assert isinstance(replay_resources, dict)
    replay_resources["peak_rss_bytes"] = 8 * 1024**3
    with pytest.raises(CompletedRunResourceError, match="final training exceeded the 8-GiB"):
        _validate(training_overrun, training_replay=replay)


def test_fallback_cannot_claim_a_generated_final_training_receipt() -> None:
    with pytest.raises(CompletedRunResourceError, match="fields differ"):
        _validate(_resource_evidence(), generated=False)

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from kuairand_agent.campaign.store import (
    ArtifactSpec,
    CampaignExistsError,
    CampaignNotFoundError,
    CampaignStore,
    LaunchLimitError,
    RevisionConflictError,
    StoreInvariantError,
    StoreVersionError,
)

_CONFIG = "a" * 64
_BENCHMARK = "b" * 64
_STARTER = "c" * 64
_DATASET = "d" * 64
_ENVIRONMENT = "e" * 64
_MANIFEST = "f" * 64
_SOURCE = "1" * 64
_CHECKPOINT = "2" * 64
_CLOSURE = "3" * 64
_ARTIFACT_A = "4" * 64
_ARTIFACT_B = "5" * 64
_RECEIPT = "6" * 64


def _convergence(
    *,
    best: float = 0.6016,
    streak: int = 0,
    iterations: int = 0,
    pending: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "best_primary": best,
        "non_material_streak": streak,
        "unmeasured_streak": 0,
        "completed_iterations": iterations,
        "required_completion_pending": pending,
    }


def _create_campaign(
    path: Path,
    *,
    campaign_id: str = "campaign-001",
    max_launches: int = 50,
    outer_query_limit: int = 6,
) -> CampaignStore:
    return CampaignStore.create(
        path,
        campaign_id=campaign_id,
        config_digest=_CONFIG,
        benchmark_digest=_BENCHMARK,
        starter_digest=_STARTER,
        dataset_digest=_DATASET,
        environment_digest=_ENVIRONMENT,
        source_digest=_SOURCE,
        hard_deadline_utc="2030-01-01T00:00:00Z",
        max_launches=max_launches,
        outer_query_limit=outer_query_limit,
        initial_convergence=_convergence(),
    )


def _qualification_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "launch_number": 1,
            "kind": "official_fm_training",
            "seed": 0,
            "charged": True,
        },
        {
            "launch_number": 2,
            "kind": "official_fm_training",
            "seed": 1,
            "charged": True,
        },
        {
            "launch_number": 3,
            "kind": "official_fm_training",
            "seed": 2,
            "charged": True,
        },
        {
            "launch_number": 4,
            "kind": "official_fm_training",
            "seed": 3,
            "charged": True,
        },
        {
            "launch_number": 5,
            "kind": "official_fm_training",
            "seed": 4,
            "charged": True,
        },
        {
            "launch_number": 6,
            "kind": "clean_source_retrain",
            "seed": 0,
            "charged": True,
        },
    )


def _runner_command_digest(command: tuple[str, ...]) -> str:
    payload = json.dumps(
        list(command),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"kuairand-runner-command-v1\0" + payload).hexdigest()


def test_create_open_strict_schema_and_existing_run_refusal(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path) as store:
        health = store.health()
        assert health.journal_mode == "wal"
        assert health.foreign_keys is True
        assert health.synchronous == 2
        assert health.user_version == 1
        assert health.quick_check == "ok"
        assert len(health.schema_digest) == 64
        assert len(health.catalog_digest) == 64

        snapshot = store.snapshot()
        assert snapshot.campaign_id == "campaign-001"
        assert snapshot.revision == 0
        assert snapshot.launches_used == 0
        assert snapshot.outer_queries_used == 0
        assert snapshot.convergence_state == _convergence()
        identity = store.identity()
        assert identity.campaign_id == snapshot.campaign_id
        assert identity.config_digest == _CONFIG
        assert identity.benchmark_digest == _BENCHMARK
        assert identity.starter_manifest_digest == _STARTER
        assert identity.dataset_manifest_digest == _DATASET
        assert identity.source_digest == _SOURCE
        assert identity.environment_digest == _ENVIRONMENT
        assert identity.hard_deadline_utc == "2030-01-01T00:00:00.000000Z"
        assert identity.max_launches == 50
        assert identity.outer_query_limit == 6

        with pytest.raises(CampaignExistsError, match="already exists"):
            _create_campaign(path)

    assert path.stat().st_mode & 0o777 == 0o600
    with CampaignStore.open(path, read_only=True, campaign_id="campaign-001") as reopened:
        assert reopened.snapshot() == snapshot
        with pytest.raises(StoreInvariantError, match="read-only"):
            reopened.set_campaign_phase(
                phase="researching",
                status="RUNNING",
                expected_revision=0,
                reason="must not mutate",
            )

    with pytest.raises(CampaignNotFoundError, match="not 'another-campaign'"):
        CampaignStore.open(path, campaign_id="another-campaign")

    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA user_version = 999")
    with pytest.raises(StoreVersionError, match="unsupported database identity"):
        CampaignStore.open(path)


def test_execution_projection_reconciles_starting_running_and_terminal_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite"
    command = ("/usr/bin/python3", "-I", "candidate.py", "--seed", "13")
    nonce = "nonce-0123456789abcdef"
    locked_environment_digest = "0" * 64
    process_environment_digest = "9" * 64
    with _create_campaign(path) as store:
        launch = store.reserve_launch(
            launch_id="launch-resume-001",
            reservation_key="resume:001",
            category="inner_screen",
            original_category="inner_screen",
            purpose="resume seam test",
            expected_revision=0,
            scientific_iteration=1,
            seed=13,
        )
        assert launch.launch_number == 1
        assert (
            store.create_execution(
                execution_id="execution-resume-001",
                launch_id=launch.launch_id,
                kind="train_evaluate",
                tier="inner_fold_a",
                command=command,
                seed=13,
                status="STARTING",
                nonce=nonce,
                source_digest="1" * 64,
                config_digest="2" * 64,
                capability_digest="3" * 64,
                environment_digest=locked_environment_digest,
                data_digest="4" * 64,
                checkpoint_digest="5" * 64,
                expected_revision=1,
                metadata={"family": "pairwise_fm"},
            )
            == 2
        )
        starting = store.unfinished_executions()
        assert len(starting) == 1
        assert starting[0].status == "STARTING"
        assert starting[0].launch_number == 1
        assert starting[0].launch_category == "inner_screen"
        assert starting[0].original_launch_category == "inner_screen"
        assert starting[0].nonce == nonce
        assert starting[0].command == command
        assert starting[0].environment_digest == locked_environment_digest
        assert starting[0].process_environment_digest is None
        assert starting[0].process_record is None
        assert starting[0].started_at is None

    process_record: dict[str, object] = {
        "schema_version": 2,
        "execution_id": "execution-resume-001",
        "nonce": nonce,
        "pid": 43123,
        "process_create_time": 1_777_777_777.25,
        "process_group_id": 43124,
        "command": list(command),
        "command_digest": _runner_command_digest(command),
        "environment_digest": process_environment_digest,
        "interpreter_real_path": "/usr/bin/python3",
        "workspace": str((tmp_path / "workspace").absolute()),
        "control_dir": str((tmp_path / "control").absolute()),
        "source_digest": "1" * 64,
        "config_digest": "2" * 64,
        "data_digest": "4" * 64,
        "checkpoint_digest": "5" * 64,
        "started_at_utc": "2030-01-01T01:00:00+00:00",
        "launcher_sha256": "6" * 64,
    }
    process_payload = json.dumps(
        process_record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    process_record_digest = hashlib.sha256(
        b"kuairand-runner-process-record-v2\0" + process_payload
    ).hexdigest()

    with CampaignStore.open(path) as resumed:
        assert resumed.unfinished_executions()[0].status == "STARTING"
        invalid_process = dict(process_record)
        invalid_process["nonce"] = "different-0123456789"
        with pytest.raises(StoreInvariantError, match="nonce does not match"):
            resumed.transition_execution(
                "execution-resume-001",
                from_state="STARTING",
                to_state="RUNNING",
                expected_revision=2,
                reason="runner launch commit",
                process_record_digest=process_record_digest,
                process_record=invalid_process,
            )
        with pytest.raises(StoreInvariantError, match="does not match the exact runner-v2"):
            resumed.transition_execution(
                "execution-resume-001",
                from_state="STARTING",
                to_state="RUNNING",
                expected_revision=2,
                reason="runner launch commit",
                process_record_digest="f" * 64,
                process_record=process_record,
            )
        assert resumed.snapshot().revision == 2

        running = resumed.transition_execution(
            "execution-resume-001",
            from_state="STARTING",
            to_state="RUNNING",
            expected_revision=2,
            reason="runner launch commit",
            process_record_digest=process_record_digest,
            process_record=process_record,
        )
        assert resumed.snapshot().revision == 3
        assert running.status == "RUNNING"
        assert running.process_record_digest == process_record_digest
        assert running.process_id == 43123
        assert running.process_group_id == 43124
        assert running.process_create_time == 1_777_777_777.25
        assert running.process_command_digest == _runner_command_digest(command)
        assert running.environment_digest == locked_environment_digest
        assert running.process_environment_digest == process_environment_digest
        assert running.process_record is not None
        assert running.process_record["command"] == command
        assert running.started_at == "2030-01-01T01:00:00.000000Z"
        assert running.finished_at is None

    with CampaignStore.open(path) as resumed_again:
        running_after_restart = resumed_again.unfinished_executions()
        assert running_after_restart == (running,)
        finished = resumed_again.transition_execution(
            "execution-resume-001",
            from_state="RUNNING",
            to_state="SUCCEEDED",
            expected_revision=3,
            reason="trusted runner completed",
            result_digest="7" * 64,
            finished_at="2030-01-01T01:02:03Z",
        )
        assert resumed_again.snapshot().revision == 4
        assert finished.status == "SUCCEEDED"
        assert finished.result_digest == "7" * 64
        assert finished.finished_at == "2030-01-01T01:02:03.000000Z"
        assert finished.process_record == running.process_record
        assert resumed_again.unfinished_executions() == ()

        exact_retry = resumed_again.transition_execution(
            "execution-resume-001",
            from_state="RUNNING",
            to_state="SUCCEEDED",
            expected_revision=3,
            reason="trusted runner completed",
            result_digest="7" * 64,
            finished_at="2030-01-01T01:02:03Z",
        )
        assert exact_retry == finished
        assert resumed_again.snapshot().revision == 4

    with CampaignStore.open(path, read_only=True) as read_only:
        assert read_only.executions() == (finished,)
        assert read_only.unfinished_executions() == ()

    with sqlite3.connect(path) as raw:
        raw.execute(
            'UPDATE executions SET command_json = \'{"not":"an array"}\' '
            "WHERE execution_id = 'execution-resume-001'"
        )
    with (
        CampaignStore.open(path, read_only=True) as corrupted,
        pytest.raises(StoreVersionError, match="command_json"),
    ):
        corrupted.executions()


def test_exact_qualification_import_and_conservative_launch_reservations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path, max_launches=7) as store:
        invalid = [dict(record) for record in _qualification_records()]
        invalid[-1]["seed"] = 1
        with pytest.raises(StoreInvariantError, match="frozen six"):
            store.import_qualification_launches(
                invalid,
                manifest_digest=_MANIFEST,
                expected_revision=0,
            )
        assert store.snapshot().revision == 0
        assert store.launches() == ()

        qualification = store.import_qualification_launches(
            _qualification_records(),
            manifest_digest=_MANIFEST,
            expected_revision=0,
        )
        assert [record.launch_number for record in qualification] == [1, 2, 3, 4, 5, 6]
        assert [record.seed for record in qualification] == [0, 1, 2, 3, 4, 0]
        assert {record.state for record in qualification} == {"FINISHED"}
        assert all(record.charged for record in qualification)
        assert store.snapshot().launches_used == 6
        assert store.snapshot().qualification_digest == _MANIFEST

        reserved = store.reserve_launch(
            launch_id="screen-001",
            reservation_key="screen:001",
            category="inner_screen",
            purpose="pairwise FM fold A",
            expected_revision=1,
            scientific_iteration=1,
            seed=13,
        )
        assert reserved.state == "RESERVED"
        assert reserved.charged is True
        assert reserved.launch_number == 7
        assert store.snapshot().launches_used == 7

        retry = store.reserve_launch(
            launch_id="screen-001",
            reservation_key="screen:001",
            category="inner_screen",
            purpose="pairwise FM fold A",
            expected_revision=1,
            scientific_iteration=1,
            seed=13,
        )
        assert retry == reserved
        assert store.snapshot().revision == 2

        with pytest.raises(LaunchLimitError, match="frozen campaign limit"):
            store.reserve_launch(
                launch_id="screen-over-limit",
                reservation_key="screen:over-limit",
                category="inner_screen",
                purpose="must not launch",
                expected_revision=2,
            )

        released = store.transition_launch(
            "screen-001",
            to_state="NOT_STARTED",
            expected_revision=2,
            metadata={"executor_receipt": "explicit-not-started"},
        )
        assert released.charged is False
        assert store.snapshot().launches_used == 6

        second = store.reserve_launch(
            launch_id="screen-002",
            reservation_key="screen:002",
            category="inner_screen",
            purpose="causal FM fold A",
            expected_revision=3,
        )
        assert second.launch_number == 8
        with pytest.raises(StoreInvariantError, match="durable start receipt"):
            store.transition_launch(
                "screen-002",
                to_state="STARTED",
                expected_revision=4,
            )
        assert store.snapshot().revision == 4

        started = store.transition_launch(
            "screen-002",
            to_state="STARTED",
            expected_revision=4,
            start_receipt_digest=_RECEIPT,
        )
        assert started.charged is True
        uncertain = store.transition_launch(
            "screen-002",
            to_state="START_UNCERTAIN",
            expected_revision=5,
        )
        assert uncertain.charged is True
        assert uncertain.start_receipt_digest == _RECEIPT
        assert store.snapshot().launches_used == 7


def test_optimistic_revision_and_atomic_transition_artifact_links(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite"
    first = _create_campaign(path)
    second = CampaignStore.open(path)
    try:
        revision = first.create_experiment(
            experiment_id="experiment-001",
            iteration_number=1,
            hypothesis="Pairwise loss better matches GAUC.",
            mechanism="Logged positive-negative pairs.",
            method_attribution="generated_source",
            expected_revision=0,
        )
        assert revision == 1
        with pytest.raises(RevisionConflictError, match="expected 0, found 1"):
            second.create_experiment(
                experiment_id="experiment-stale",
                iteration_number=2,
                hypothesis="stale",
                mechanism="stale",
                method_attribution="stale",
                expected_revision=0,
            )

        artifact = ArtifactSpec(
            digest=_ARTIFACT_A,
            kind="source_snapshot",
            relative_path="objects/44/source.py",
            size_bytes=123,
            metadata={"validated": True},
        )
        assert (
            first.transition_entity(
                "experiment",
                "experiment-001",
                from_state="PLANNED",
                to_state="SCREENED",
                expected_revision=1,
                reason="static and inner checks passed",
                artifacts=(("generated_source", artifact),),
            )
            == 2
        )

        conflicting_artifact = ArtifactSpec(
            digest=_ARTIFACT_B,
            kind="report",
            relative_path="objects/55/report.json",
            size_bytes=456,
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            first.transition_entity(
                "experiment",
                "experiment-001",
                from_state="SCREENED",
                to_state="PROMOTED",
                expected_revision=2,
                reason="force an atomic rollback",
                artifacts=(("generated_source", conflicting_artifact),),
            )
        assert first.snapshot().revision == 2

        with sqlite3.connect(path) as raw:
            status = raw.execute(
                "SELECT status FROM experiments WHERE experiment_id = 'experiment-001'"
            ).fetchone()
            absent = raw.execute(
                "SELECT COUNT(*) FROM artifacts WHERE digest = ?", (_ARTIFACT_B,)
            ).fetchone()
            assert status == ("SCREENED",)
            assert absent == (0,)
            assert raw.execute("SELECT COUNT(*) FROM artifact_links").fetchone() == (1,)

        assert (
            first.record_proposal(
                proposal_id="proposal-001",
                experiment_id="experiment-001",
                request_digest="7" * 64,
                response_digest="8" * 64,
                provider="fixture-provider",
                expected_revision=2,
                artifacts=(("response", conflicting_artifact),),
            )
            == 3
        )
        assert (
            first.record_source_snapshot(
                snapshot_id="source-001",
                experiment_id="experiment-001",
                source_digest="9" * 64,
                parent_source_digest=_SOURCE,
                diff_digest="a" * 64,
                expected_revision=3,
            )
            == 4
        )
        assert (
            first.create_execution(
                execution_id="execution-001",
                experiment_id="experiment-001",
                kind="train_evaluate",
                tier="inner_fold_a",
                command=("python", "candidate.py", "--seed", "13"),
                seed=13,
                expected_revision=4,
            )
            == 5
        )
        assert (
            first.record_metric(
                metric_id="inner-001",
                experiment_id="experiment-001",
                execution_id="execution-001",
                split_role="inner_fold_a",
                seed=13,
                gauc=0.66,
                ndcg_at_5=0.54,
                primary=0.60,
                primary_delta=0.001,
                scorer_digest="b" * 64,
                prediction_digest="c" * 64,
                expected_revision=5,
            )
            == 6
        )
        assert (
            first.record_intervention(
                intervention_id="intervention-001",
                category="credential",
                description="User supplied an already-authorized local provider credential.",
                expected_revision=6,
            )
            == 7
        )
        sample_id = first.record_runtime_sample(
            family="pairwise_fm",
            elapsed_seconds=12.5,
            peak_rss_bytes=12_345,
            disk_bytes=6_789,
            outcome="completed",
            execution_id="execution-001",
            expected_revision=7,
        )
        assert sample_id > 0
        assert (
            first.record_reallocation(
                reallocation_id="reallocation-001",
                from_category="blend",
                to_category="inner_screen",
                launch_count=1,
                reason="no complementary blend survived screening",
                expected_revision=8,
            )
            == 9
        )
        assert first.snapshot().revision == 9

        with sqlite3.connect(path) as raw:
            assert raw.execute("SELECT COUNT(*) FROM proposals").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM source_snapshots").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM executions").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM metrics").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM interventions").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM runtime_samples").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM reallocations").fetchone() == (1,)
            assert raw.execute("SELECT COUNT(*) FROM transitions").fetchone() == (9,)
    finally:
        second.close()
        first.close()

    raw = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(
                "UPDATE artifacts SET size_bytes = 999 WHERE digest = ?",
                (_ARTIFACT_A,),
            )
    finally:
        raw.close()


def test_failure_cannot_demote_incumbent_and_convergence_survives_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path) as store:
        fallback = store.record_incumbent(
            incumbent_id="official-fm-fallback",
            eligibility="QUALIFIED",
            source_digest=_SOURCE,
            checkpoint_digest=_CHECKPOINT,
            artifact_closure_digest=_CLOSURE,
            replay_verified=True,
            is_fallback=True,
            expected_revision=0,
            reason="official FM qualification passed",
            outer_primary_mean=0.6016,
        )
        assert fallback.is_fallback is True

        assert (
            store.record_failure(
                failure_id="failure-001",
                category="timeout",
                fingerprint="timeout:pairwise:fold-a",
                retry_ordinal=0,
                expected_revision=1,
                repair_action="close identical config",
                recovery_outcome="incumbent preserved",
            )
            == 2
        )
        assert store.current_incumbent() == fallback

        challenger = store.record_incumbent(
            incumbent_id="candidate-001",
            eligibility="ELIGIBLE_UNCONFIRMED",
            source_digest="7" * 64,
            checkpoint_digest="8" * 64,
            artifact_closure_digest="9" * 64,
            replay_verified=True,
            is_fallback=False,
            expected_revision=2,
            reason="higher protected primary under frozen policy",
            outer_primary_mean=0.6030,
        )
        assert store.current_incumbent() == challenger

        with pytest.raises(StoreInvariantError, match="replay verified"):
            store.record_incumbent(
                incumbent_id="unreplayable",
                eligibility="INELIGIBLE",
                source_digest="a" * 64,
                checkpoint_digest="b" * 64,
                artifact_closure_digest="c" * 64,
                replay_verified=False,
                is_fallback=False,
                expected_revision=3,
                reason="must reject",
            )
        assert store.snapshot().revision == 3
        assert store.current_incumbent() == challenger

        persisted_state: Mapping[str, object] = _convergence(
            best=0.6030,
            streak=1,
            iterations=1,
            pending=True,
        )
        updated = store.set_convergence_state(persisted_state, expected_revision=3)
        assert updated.revision == 4
        assert updated.convergence_state == persisted_state
        no_op = store.set_convergence_state(persisted_state, expected_revision=4)
        assert no_op.revision == 4

    with CampaignStore.open(path, read_only=True) as reopened:
        resumed = reopened.snapshot()
        assert resumed.revision == 4
        assert resumed.convergence_state == persisted_state
        assert resumed.incumbent == challenger

    raw = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(
                "UPDATE incumbent_history SET eligibility = 'MUTATED' WHERE incumbent_id = ?",
                (challenger.incumbent_id,),
            )
    finally:
        raw.close()

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kuairand_agent.campaign.controller import (
    CAMPAIGN_DATABASE_NAME,
    CampaignCreateRequest,
    CampaignEngine,
    CampaignIntegrityError,
    CampaignRunExistsError,
    FallbackIdentity,
)
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.config import AgentConfig, parse_config

BENCHMARK_DIGEST = "b" * 64
STARTER_DIGEST = "a" * 64
DATASET_DIGEST = "d" * 64
SOURCE_DIGEST = "1" * 64
ENVIRONMENT_DIGEST = "e" * 64
CHECKPOINT_DIGEST = "c" * 64
CONFIG_DIGEST = "f" * 64
ENCODING_DIGEST = "7" * 64


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _signed(value: dict[str, object]) -> dict[str, object]:
    return value | {"digest": _digest(value)}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value) + b"\n")


def _config() -> AgentConfig:
    raw: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": {
            "name": "kuairand-pure",
            "data_dir": "unused-data",
            "starter_dir": "unused-starter",
            "target": "long_view",
            "max_iterations": 4,
            "wall_clock_seconds": 7200,
            "epsilon": 0.002,
            "convergence_patience": 3,
        },
        "validation": {
            "outer_promotion_limit": 2,
            "confirmation_seeds": [0, 1, 2],
            "inner_folds": ["20220416:20220418", "20220419:20220421"],
        },
        "runner": {
            "device": "cpu",
            "max_processes": 1,
            "threads": 1,
            "memory_mb": 512,
            "disk_mb": 512,
            "default_timeout_seconds": 30,
            "finalization_reserve_seconds": 3600,
        },
        "research": {"provider": "scripted", "max_repairs_per_experiment": 2},
    }
    return parse_config(raw)


@dataclass(slots=True)
class FakeClock:
    now: datetime = datetime(2030, 1, 1, tzinfo=UTC)
    monotonic: int = 100_000_000_000
    boot: str = "fixture-boot"

    def monotonic_ns(self) -> int:
        return self.monotonic

    def utc_now(self) -> datetime:
        return self.now

    def boot_identity(self) -> str:
        return self.boot

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)
        self.monotonic += int(seconds * 1_000_000_000)


def build_request(
    root: Path,
    *,
    final_outcomes_accessed: bool = False,
) -> CampaignCreateRequest:
    qualification = root / "qualification"
    model = qualification / "fallback" / "model"
    model.mkdir(parents=True)
    files = {
        "checkpoint.bin": b"qualified-checkpoint",
        "encoding.bin": b"qualified-encoding",
        "run.json": b'{"trusted":"aggregate-only"}\n',
    }
    tree_records: list[dict[str, object]] = []
    root_artifacts: list[dict[str, object]] = []
    for name, payload in sorted(files.items()):
        path = model / name
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        tree_records.append({"path": name, "size": len(payload), "sha256": digest})
        root_artifacts.append(
            {
                "path": f"fallback/model/{name}",
                "size": len(payload),
                "sha256": digest,
            }
        )
    tree_digest = _digest(tree_records)
    fallback_body: dict[str, object] = {
        "schema_version": 1,
        "kind": "immutable_official_fm_fallback",
        "seed": 4,
        "validation_metrics": {"GAUC": 0.66, "nDCG@5": 0.54, "primary": 0.60},
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "encoding_digest": ENCODING_DIGEST,
        "config_digest": CONFIG_DIGEST,
        "validation_prediction_digest": "8" * 64,
        "validation_submission": {
            "path": "validation/submission.csv",
            "sha256": "9" * 64,
            "prediction_digest": "8" * 64,
            "round_trip_identity": True,
            "protected_metrics_preserved": True,
        },
        "final_submission": {
            "path": "final/submission.csv",
            "sha256": "6" * 64,
            "prediction_digest": "5" * 64,
            "round_trip_identity": True,
            "final_outcomes_accessed": False,
        },
        "source_model_tree_digest": tree_digest,
        "fallback_model_tree_digest": tree_digest,
        "replay_verified": True,
        "clean_seed_zero_retrain_verified": True,
    }
    fallback = _signed(fallback_body)
    _write_json(qualification / "fallback" / "manifest.json", fallback)
    launch_records = [
        {
            "launch_number": number,
            "kind": kind,
            "seed": seed,
            "charged": True,
        }
        for number, kind, seed in (
            (1, "official_fm_training", 0),
            (2, "official_fm_training", 1),
            (3, "official_fm_training", 2),
            (4, "official_fm_training", 3),
            (5, "official_fm_training", 4),
            (6, "clean_source_retrain", 0),
        )
    ]
    root_body: dict[str, object] = {
        "schema_version": 1,
        "status": "baseline_reproduced",
        "benchmark_digest": BENCHMARK_DIGEST,
        "qualification_input_digest": "4" * 64,
        "double_build_identity": True,
        "final_period": {
            "input_rows": 10,
            "outcomes_accessed": final_outcomes_accessed,
            "outcomes_scored": False,
            "target_capability": None,
        },
        "launch_accounting": {
            "charged_launches": 6,
            "expected_launches": 6,
            "checkpoint_replays_charged": False,
            "random_rungs_charged": False,
            "popularity_rung_charged": False,
            "records": launch_records,
        },
        "fallback": fallback,
        "fm": {
            "runs": [
                {
                    "seed": 4,
                    "starter_manifest_digest": STARTER_DIGEST,
                    "checkpoint_digest": CHECKPOINT_DIGEST,
                    "config_digest": CONFIG_DIGEST,
                    "encoding_digest": ENCODING_DIGEST,
                    "organizer_parity_passed": True,
                }
            ]
        },
        "artifacts": root_artifacts,
    }
    root_manifest = _signed(root_body)
    _write_json(qualification / "manifest.json", root_manifest)
    return CampaignCreateRequest(
        run_dir=root / "campaign-run",
        campaign_id="fixture-campaign",
        config=_config(),
        qualification_run_dir=qualification,
        qualification_manifest_digest=root_manifest["digest"],  # type: ignore[arg-type]
        fallback=FallbackIdentity(
            manifest_digest=fallback["digest"],  # type: ignore[arg-type]
            source_digest=tree_digest,
            checkpoint_digest=CHECKPOINT_DIGEST,
            artifact_closure_digest=tree_digest,
            config_digest=CONFIG_DIGEST,
            encoding_digest=ENCODING_DIGEST,
            outer_primary=0.60,
        ),
        benchmark_digest=BENCHMARK_DIGEST,
        starter_manifest_digest=STARTER_DIGEST,
        dataset_manifest_digest=DATASET_DIGEST,
        source_digest=SOURCE_DIGEST,
        environment_digest=ENVIRONMENT_DIGEST,
    )


def test_create_binds_exact_qualification_and_fallback_and_status_is_read_only(
    tmp_path: Path,
) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)

    created = engine.create(request)

    assert created.status == "RUNNING"
    assert created.phase == "researching"
    assert created.launches_used == 6
    assert created.launches_remaining == 44
    assert created.scientific_iterations == 0
    assert created.outer_queries_remaining == 2
    assert created.qualification_manifest_digest == request.qualification_manifest_digest
    assert created.incumbent_id == "official-fm-fallback-seed-4"
    assert created.incumbent_is_fallback
    assert created.incumbent_replay_verified
    assert created.deadline_remaining_seconds == 7200.0
    assert not created.finalization_required

    with CampaignStore.open(
        request.run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=request.campaign_id,
    ) as store:
        identity = store.identity()
        assert identity.config_digest == request.config.digest
        assert identity.benchmark_digest == BENCHMARK_DIGEST
        assert identity.starter_manifest_digest == STARTER_DIGEST
        assert identity.dataset_manifest_digest == DATASET_DIGEST
        assert identity.source_digest == SOURCE_DIGEST
        assert identity.environment_digest == ENVIRONMENT_DIGEST
        assert store.snapshot().qualification_digest == request.qualification_manifest_digest
        assert tuple(item.launch_number for item in store.launches()) == tuple(range(1, 7))

    deadline_dir = request.run_dir / "controller" / "deadline"
    checkpoints_before = tuple(deadline_dir.iterdir())
    database_mtime = (request.run_dir / CAMPAIGN_DATABASE_NAME).stat().st_mtime_ns
    assert engine.status(request.run_dir).manifest() == created.manifest()
    assert tuple(deadline_dir.iterdir()) == checkpoints_before
    assert (request.run_dir / CAMPAIGN_DATABASE_NAME).stat().st_mtime_ns == database_mtime


def test_create_refuses_every_existing_destination_without_overwriting(tmp_path: Path) -> None:
    request = build_request(tmp_path)
    request.run_dir.mkdir()
    marker = request.run_dir / "user-owned.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(CampaignRunExistsError, match="already exists"):
        CampaignEngine(clock=FakeClock()).create(request)

    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert tuple(request.run_dir.iterdir()) == (marker,)


def test_create_rejects_qualification_that_accessed_final_outcomes(tmp_path: Path) -> None:
    request = build_request(tmp_path, final_outcomes_accessed=True)

    with pytest.raises(CampaignIntegrityError, match="final-period outcome boundary"):
        CampaignEngine(clock=FakeClock()).create(request)

    assert not request.run_dir.exists()


def test_status_fails_closed_on_tampered_signed_request(tmp_path: Path) -> None:
    request = build_request(tmp_path)
    engine = CampaignEngine(clock=FakeClock())
    engine.create(request)
    manifest_path = request.run_dir / "controller" / "create-request.json"
    manifest_path.chmod(0o600)
    value = json.loads(manifest_path.read_text(encoding="ascii"))
    value["environment_digest"] = "0" * 64
    _write_json(manifest_path, value)

    with pytest.raises(CampaignIntegrityError, match="digest does not match"):
        engine.status(request.run_dir)


def test_observe_deadline_persists_chain_and_preserves_original_clock_across_restart(
    tmp_path: Path,
) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)
    engine.create(request)
    deadline_dir = request.run_dir / "controller" / "deadline"
    original = json.loads((deadline_dir / "checkpoint-00000000.json").read_text(encoding="ascii"))[
        "state"
    ]
    before = tuple(sorted(deadline_dir.iterdir()))

    clock.advance(900)
    observed = engine.observe_deadline(request.run_dir)

    assert observed.elapsed_seconds == 900.0
    assert observed.remaining_seconds == 6300.0
    assert observed.same_boot
    after_first = tuple(sorted(deadline_dir.iterdir()))
    assert len(after_first) == len(before) + 1
    assert json.loads(after_first[-1].read_text(encoding="ascii"))["state"] == (
        observed.state.manifest()
    )

    rebooted_clock = FakeClock(
        now=clock.now + timedelta(seconds=600),
        monotonic=1_000_000,
        boot="fixture-second-boot",
    )
    restarted = CampaignEngine(clock=rebooted_clock).observe_deadline(request.run_dir)

    assert not restarted.same_boot
    assert restarted.elapsed_seconds == 1500.0
    assert restarted.remaining_seconds == 5700.0
    assert len(tuple(deadline_dir.iterdir())) == len(after_first) + 1
    for immutable_field in (
        "wall_clock_seconds",
        "finalization_reserve_seconds",
        "started_utc",
        "utc_deadline",
        "original_boot_identity",
        "monotonic_started_ns",
        "monotonic_deadline_ns",
    ):
        assert restarted.state.manifest()[immutable_field] == original[immutable_field]


def test_inspect_deadline_returns_verified_observation_without_writing(tmp_path: Path) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)
    engine.create(request)
    deadline_dir = request.run_dir / "controller" / "deadline"
    before = tuple(sorted(deadline_dir.iterdir()))

    clock.advance(75)
    observed = engine.inspect_deadline(request.run_dir)

    assert observed.elapsed_seconds == 75.0
    assert observed.remaining_seconds == 7125.0
    assert tuple(sorted(deadline_dir.iterdir())) == before


def test_load_request_returns_verified_immutable_request_without_writing(tmp_path: Path) -> None:
    request = build_request(tmp_path)
    engine = CampaignEngine(clock=FakeClock())
    engine.create(request)
    controller_dir = request.run_dir / "controller"
    files_before = tuple(sorted(controller_dir.rglob("*")))
    database_mtime = (request.run_dir / CAMPAIGN_DATABASE_NAME).stat().st_mtime_ns

    loaded = engine.load_request(request.run_dir)

    assert loaded == request
    assert loaded.digest == request.digest
    assert loaded.run_dir == request.run_dir
    assert tuple(sorted(controller_dir.rglob("*"))) == files_before
    assert (request.run_dir / CAMPAIGN_DATABASE_NAME).stat().st_mtime_ns == database_mtime


@pytest.mark.parametrize(
    ("tamper_target", "expected_error"),
    (
        ("request", "campaign create request digest does not match"),
        ("run", "run manifest digest does not match"),
        ("deadline", "deadline checkpoint 1 digest does not match"),
        ("store", "campaign database source_digest differs"),
    ),
)
def test_load_request_rejects_any_inconsistent_runtime_identity(
    tmp_path: Path,
    tamper_target: str,
    expected_error: str,
) -> None:
    request = build_request(tmp_path)
    engine = CampaignEngine(clock=FakeClock())
    engine.create(request)

    if tamper_target == "request":
        path = request.run_dir / "controller" / "create-request.json"
        path.chmod(0o600)
        value = json.loads(path.read_text(encoding="ascii"))
        value["source_digest"] = "0" * 64
        _write_json(path, value)
    elif tamper_target == "run":
        path = request.run_dir / "controller" / "run-manifest.json"
        path.chmod(0o600)
        value = json.loads(path.read_text(encoding="ascii"))
        value["request_digest"] = "0" * 64
        _write_json(path, value)
    elif tamper_target == "deadline":
        path = request.run_dir / "controller" / "deadline" / "checkpoint-00000001.json"
        path.chmod(0o600)
        value = json.loads(path.read_text(encoding="ascii"))
        value["state"]["last_elapsed_seconds"] = 1.0
        _write_json(path, value)
    else:
        with sqlite3.connect(request.run_dir / CAMPAIGN_DATABASE_NAME) as connection:
            connection.execute("UPDATE campaigns SET source_digest = ?", ("0" * 64,))

    controller_files_before = tuple(sorted((request.run_dir / "controller").rglob("*")))
    with pytest.raises(CampaignIntegrityError, match=expected_error):
        engine.load_request(request.run_dir)
    assert tuple(sorted((request.run_dir / "controller").rglob("*"))) == (controller_files_before)


@pytest.mark.parametrize(
    ("tamper_target", "expected_error"),
    (
        ("run", "run manifest digest does not match"),
        ("deadline", "deadline checkpoint 1 digest does not match"),
        ("store", "campaign database source_digest differs"),
    ),
)
def test_observe_deadline_rejects_tampered_run_deadline_or_store_without_appending(
    tmp_path: Path,
    tamper_target: str,
    expected_error: str,
) -> None:
    request = build_request(tmp_path)
    engine = CampaignEngine(clock=FakeClock())
    engine.create(request)
    deadline_dir = request.run_dir / "controller" / "deadline"
    checkpoints_before = tuple(sorted(deadline_dir.iterdir()))

    if tamper_target == "run":
        path = request.run_dir / "controller" / "run-manifest.json"
        path.chmod(0o600)
        value = json.loads(path.read_text(encoding="ascii"))
        value["request_digest"] = "0" * 64
        _write_json(path, value)
    elif tamper_target == "deadline":
        path = deadline_dir / "checkpoint-00000001.json"
        path.chmod(0o600)
        value = json.loads(path.read_text(encoding="ascii"))
        value["state"]["last_elapsed_seconds"] = 1.0
        _write_json(path, value)
    else:
        with sqlite3.connect(request.run_dir / CAMPAIGN_DATABASE_NAME) as connection:
            connection.execute("UPDATE campaigns SET source_digest = ?", ("0" * 64,))

    with pytest.raises(CampaignIntegrityError, match=expected_error):
        engine.observe_deadline(request.run_dir)

    assert tuple(sorted(deadline_dir.iterdir())) == checkpoints_before

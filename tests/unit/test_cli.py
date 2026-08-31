from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import kuairand_agent.cli as cli_module
from kuairand_agent.baselines.qualification import QualificationError, QualificationMetrics
from kuairand_agent.campaign import CampaignIntegrityError
from kuairand_agent.cli import build_parser, main
from kuairand_agent.config import load_config
from kuairand_agent.data.acquire import ArchiveIntegrityError
from kuairand_agent.data.audit import DataAuditError

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["data", "--help"],
        ["data", "prepare", "--help"],
        ["data", "audit", "--help"],
        ["qualify", "--help"],
        ["run", "--help"],
        ["resume", "--help"],
        ["status", "--help"],
        ["finalize", "--help"],
        ["replay", "--help"],
        ["validate-submission", "--help"],
        ["config", "validate", "--help"],
        ["provider", "preflight", "--help"],
        ["contract", "verify-starter", "--help"],
    ],
)
def test_every_command_has_help(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(argv)
    assert raised.value.code == 0


def test_status_json_emits_one_stable_object_without_noise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    observed: list[Path] = []

    class FakeEngine:
        def status(self, path: Path) -> SimpleNamespace:
            observed.append(path)
            return SimpleNamespace(
                manifest=lambda: {
                    "schema_version": 1,
                    "campaign_id": "campaign-fixture",
                    "status": "RUNNING",
                    "phase": "researching",
                }
            )

    monkeypatch.setattr("kuairand_agent.cli.CampaignEngine", FakeEngine)

    code = main(["status", "--run-dir", str(run_dir), "--json"])
    output = capsys.readouterr()

    assert code == 0
    assert output.err == ""
    assert output.out == (
        '{"campaign_id":"campaign-fixture","phase":"researching",'
        '"schema_version":1,"status":"RUNNING"}\n'
    )
    assert observed == [run_dir.resolve()]


def test_provider_preflight_reports_live_availability_without_dispatching_api_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = {
        "INFERENCE_MAIN_API_KEY": "sk-test-main-provider-preflight",
        "INFERENCE_MAIN_BASE_URL": "https://main.example/v1",
        "INFERENCE_MAIN_MODEL": "main-model",
        "INFERENCE_FALLBACK_API_KEY": "sk-test-fallback-provider-preflight",
        "INFERENCE_FALLBACK_BASE_URL": "https://fallback.example/v1",
        "INFERENCE_FALLBACK_MODEL": "fallback-model",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    code = main(["provider", "preflight", "--config", str(ROOT / "configs/full-pure.toml")])
    output = capsys.readouterr()
    payload = json.loads(output.out)

    assert code == 0
    assert output.err == ""
    assert payload == {
        "api_request_sent": False,
        "config_digest": load_config(ROOT / "configs/full-pure.toml").digest,
        "credential_env": None,
        "live_provider_used": True,
        "model": None,
        "provider": "openai",
        "provider_profiles": [
            {
                "base_url": "https://main.example/v1",
                "credential_env": "INFERENCE_MAIN_API_KEY",
                "model": "main-model",
                "slot": "main",
            },
            {
                "base_url": "https://fallback.example/v1",
                "credential_env": "INFERENCE_FALLBACK_API_KEY",
                "model": "fallback-model",
                "slot": "fallback",
            },
        ],
        "run_kind": "autonomous",
        "schema_version": 1,
        "status": "available",
    }
    assert "sk-test-main-provider-preflight" not in output.out
    assert "sk-test-fallback-provider-preflight" not in output.out


def test_provider_preflight_missing_credential_is_redacted_and_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = {
        "INFERENCE_MAIN_API_KEY": "sk-test-main-provider-preflight",
        "INFERENCE_MAIN_BASE_URL": "https://main.example/v1",
        "INFERENCE_MAIN_MODEL": "main-model",
        "INFERENCE_FALLBACK_BASE_URL": "https://fallback.example/v1",
        "INFERENCE_FALLBACK_MODEL": "fallback-model",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("INFERENCE_FALLBACK_API_KEY", raising=False)

    code = main(["provider", "preflight", "--config", str(ROOT / "configs/full-pure.toml")])
    output = capsys.readouterr()

    assert code == cli_module.EXIT_INVALID
    assert output.out == ""
    assert output.err.startswith("PROVIDER_PREFLIGHT_FAILED: {")
    diagnostic = json.loads(output.err.removeprefix("PROVIDER_PREFLIGHT_FAILED: "))
    assert diagnostic == {
        "category": "provider_unavailable",
        "code": "credential_missing",
        "credential_env": "INFERENCE_FALLBACK_API_KEY",
        "message": "A required OpenAI-compatible credential is unavailable.",
        "provider": "openai",
        "retryable": True,
    }


def test_status_missing_campaign_is_nonzero_and_stderr_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class FakeEngine:
        def status(self, _path: Path) -> None:
            raise CampaignIntegrityError("campaign run directory does not exist")

    monkeypatch.setattr("kuairand_agent.cli.CampaignEngine", FakeEngine)

    code = main(["status", "--run-dir", str(tmp_path / "missing"), "--json"])
    output = capsys.readouterr()

    assert code != 0
    assert output.out == ""
    assert output.err == ("CAMPAIGN_STATUS_FAILED: campaign run directory does not exist\n")


def test_status_does_not_resolve_away_a_rejected_campaign_symlink(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "real-campaign"
    target.mkdir()
    link = tmp_path / "campaign-link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    code = main(["status", "--run-dir", link.name, "--json"])
    output = capsys.readouterr()

    assert code != 0
    assert output.out == ""
    assert output.err == "CAMPAIGN_STATUS_FAILED: campaign run path must be a real directory\n"


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["run", "--config", "configs/default.toml"], "run"),
        (["resume", "--run-dir", "runs/legacy"], "resume"),
        (["finalize", "--run-dir", "runs/legacy"], "finalize"),
    ],
)
def test_legacy_mutating_commands_fail_closed_before_touching_either_authority(
    argv: list[str],
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "CampaignEngine",
        lambda: pytest.fail("legacy CampaignEngine must not be constructed"),
    )
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda _path: pytest.fail("legacy run must not load campaign configuration"),
    )

    code = main(argv)
    output = capsys.readouterr()

    assert code == cli_module.EXIT_CONTRACT
    assert output.out == ""
    assert output.err.startswith(f"LEGACY_CAMPAIGN_COMMAND_DISABLED: {command}: ")
    assert "StateRepository authority" in output.err
    assert "kuairand-agent compete" in output.err
    assert "inspect" in output.err
    assert "replay" in output.err
    assert list(tmp_path.iterdir()) == []


def test_replay_delegates_only_to_the_trusted_closed_bundle_facade(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "final"
    data_dir = tmp_path / "data"
    digest = "c" * 64
    observed: dict[str, object] = {}
    result = SimpleNamespace(
        manifest=lambda: {
            "schema_version": 1,
            "candidate_id": "generated-lambdarank",
            "final": {"final_outcomes_accessed": False},
        }
    )

    def replay(
        candidate_bundle: Path,
        *,
        project_root: Path,
        data_dir: Path,
        expected_data_sha256: str,
        cancel_event: threading.Event,
    ) -> object:
        observed.update(
            bundle=candidate_bundle,
            project_root=project_root,
            data_dir=data_dir,
            expected_data_sha256=expected_data_sha256,
            cancel_event=cancel_event,
        )
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_replay_final_bundle", replay)

    code = main(
        [
            "replay",
            "--bundle",
            "final",
            "--project-root",
            ".",
            "--data-dir",
            "data",
            "--expected-data-sha256",
            digest,
        ]
    )
    output = capsys.readouterr()

    assert code == 0
    assert output.err == ""
    assert json.loads(output.out) == result.manifest()
    assert output.out.count("\n") == 1
    assert observed == {
        "bundle": bundle,
        "project_root": tmp_path,
        "data_dir": data_dir,
        "expected_data_sha256": digest,
        "cancel_event": observed["cancel_event"],
    }
    assert isinstance(observed["cancel_event"], threading.Event)


def test_replay_rejects_malformed_expected_data_digest_during_parsing() -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(
            [
                "replay",
                "--bundle",
                "final",
                "--project-root",
                ".",
                "--data-dir",
                "data",
                "--expected-data-sha256",
                "not-a-sha256",
            ]
        )
    assert raised.value.code == cli_module.EXIT_INVALID


def test_replay_integrity_failure_is_nonzero_and_stderr_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail(
        _bundle: Path,
        *,
        project_root: Path,
        data_dir: Path,
        expected_data_sha256: str,
        cancel_event: threading.Event,
    ) -> None:
        del project_root, data_dir, expected_data_sha256, cancel_event
        raise RuntimeError("closed bundle member digest mismatch")

    monkeypatch.setattr(cli_module, "_replay_final_bundle", fail)
    code = main(
        [
            "replay",
            "--bundle",
            str(tmp_path / "final"),
            "--project-root",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--expected-data-sha256",
            "d" * 64,
        ]
    )
    output = capsys.readouterr()

    assert code == cli_module.EXIT_CONTRACT
    assert output.out == ""
    assert output.err == "CAMPAIGN_REPLAY_FAILED: closed bundle member digest mismatch\n"


def test_legacy_finalize_and_replay_handlers_remain_parse_compatible() -> None:
    finalize = build_parser().parse_args(["finalize", "--run-dir", "campaign"])
    replay = build_parser().parse_args(
        [
            "replay",
            "--bundle",
            "final",
            "--project-root",
            ".",
            "--data-dir",
            "data",
            "--expected-data-sha256",
            "e" * 64,
        ]
    )

    assert finalize.handler is cli_module._finalize
    assert replay.handler is cli_module._replay


def test_config_validate_emits_one_stable_json_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["config", "validate", str(ROOT / "configs/default.toml")])
    output = capsys.readouterr()
    assert code == 0
    assert output.err == ""
    assert output.out.count("\n") == 1
    assert '"digest"' in output.out


def test_contract_verify_starter_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["contract", "verify-starter", "--starter-dir", str(ROOT / "kuairand-starter-kit")])
    output = capsys.readouterr()
    assert code == 0
    assert output.err == ""
    assert '"manifest_sha256"' in output.out


def test_data_prepare_handler_emits_verified_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "prepared"
    dataset_root = destination / "KuaiRand-Pure"
    fake = SimpleNamespace(
        destination=destination,
        dataset_root=dataset_root,
        integrity_manifest=destination / "acquisition-integrity.json",
        manifest_sha256="a" * 64,
        verification=SimpleNamespace(
            archive_sha256="b" * 64,
            members=tuple(range(10)),
            payload_size=203_539_133,
        ),
    )
    monkeypatch.setattr("kuairand_agent.cli.prepare_archive", lambda archive, target: fake)

    code = main(
        [
            "data",
            "prepare",
            "--archive",
            str(tmp_path / "KuaiRand-Pure.tar.gz"),
            "--data-dir",
            str(destination),
        ]
    )
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert code == 0
    assert output.err == ""
    assert payload["data_dir"] == str(dataset_root / "data")
    assert payload["member_count"] == 10


def test_data_prepare_failure_is_nonzero_and_stderr_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail(_archive: Path, _destination: Path) -> None:
        raise ArchiveIntegrityError("fixture mismatch")

    monkeypatch.setattr("kuairand_agent.cli.prepare_archive", fail)
    code = main(
        [
            "data",
            "prepare",
            "--archive",
            str(tmp_path / "bad.tar.gz"),
            "--data-dir",
            str(tmp_path / "prepared"),
        ]
    )
    output = capsys.readouterr()
    assert code != 0
    assert output.out == ""
    assert "DATA_PREPARATION_FAILED" in output.err


def test_data_audit_json_handler_is_one_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    fake = SimpleNamespace(
        to_json=lambda: '{"schema_version":1,"digest":"fixture"}',
        readable_report=lambda: "# fixture\n",
    )
    monkeypatch.setattr("kuairand_agent.cli.audit_dataset", lambda _path: fake)
    code = main(["data", "audit", "--data-dir", str(tmp_path), "--json"])
    output = capsys.readouterr()
    assert code == 0
    assert output.err == ""
    assert json.loads(output.out)["digest"] == "fixture"


def test_data_audit_failure_is_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail(_path: Path) -> None:
        raise DataAuditError("unsafe fixture")

    monkeypatch.setattr("kuairand_agent.cli.audit_dataset", fail)
    code = main(["data", "audit", "--data-dir", str(tmp_path)])
    output = capsys.readouterr()
    assert code != 0
    assert output.out == ""
    assert "DATA_AUDIT_FAILED" in output.err


def test_qualify_handler_emits_one_stable_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "qualification"
    fake = SimpleNamespace(
        run_dir=run_dir,
        manifest_digest="a" * 64,
        fallback_seed=4,
        launch_count=6,
        validation_metrics=QualificationMetrics(0.6, 0.5, 0.55),
        validation_submission=SimpleNamespace(
            path=run_dir / "validation" / "submission.csv",
            submission_digest="b" * 64,
        ),
        final_submission=SimpleNamespace(
            path=run_dir / "final" / "submission.csv",
            submission_digest="c" * 64,
        ),
    )
    monkeypatch.setattr(
        "kuairand_agent.baselines.qualification.run_qualification", lambda _request: fake
    )

    code = main(
        [
            "qualify",
            "--data-dir",
            str(tmp_path / "data"),
            "--starter-dir",
            str(tmp_path / "starter"),
            "--run-dir",
            str(run_dir),
        ]
    )
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert code == 0
    assert output.err == ""
    assert payload["launch_count"] == 6
    assert payload["fallback_seed"] == 4
    assert payload["final_outcomes_accessed"] is False


def test_qualify_failure_is_nonzero_and_does_not_claim_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail(_request: object) -> None:
        raise QualificationError("fixture contract mismatch")

    monkeypatch.setattr("kuairand_agent.baselines.qualification.run_qualification", fail)
    code = main(
        [
            "qualify",
            "--data-dir",
            str(tmp_path / "data"),
            "--run-dir",
            str(tmp_path / "qualification"),
        ]
    )
    output = capsys.readouterr()
    assert code != 0
    assert output.out == ""
    assert "QUALIFICATION_FAILED" in output.err


def test_validate_submission_uses_trusted_canonical_alignment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    alignment = SimpleNamespace(
        row_id=(0, 1),
        user_id=("u", "u"),
        video_id=("v1", "v2"),
    )
    fake = SimpleNamespace(
        valid=SimpleNamespace(alignment=alignment, targets=object()),
        final=SimpleNamespace(alignment=alignment, targets=None),
    )
    monkeypatch.setattr("kuairand_agent.cli.load_canonical_dataset", lambda _path: fake)
    submission = tmp_path / "submission.csv"
    submission.write_text(
        "row_id,user_id,video_id,score\n0,u,v1,0.1\n1,u,v2,0.9\n",
        encoding="utf-8",
    )

    code = main(
        [
            "validate-submission",
            "--split",
            "test",
            "--data-dir",
            str(tmp_path / "data"),
            str(submission),
        ]
    )
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert code == 0
    assert output.err == ""
    assert payload["row_count"] == 2
    assert payload["canonical_alignment"] is True
    assert payload["organizer_check"] is False
    assert payload["final_outcomes_accessed"] is False


def test_validate_submission_requires_data_dir(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["validate-submission", "--split", "test", "submission.csv"])
    output = capsys.readouterr()
    assert code != 0
    assert output.out == ""
    assert "--data-dir is required" in output.err

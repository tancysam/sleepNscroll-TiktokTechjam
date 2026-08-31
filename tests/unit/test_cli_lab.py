from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kuairand_agent.cli import build_parser, main


def test_compete_cli_uses_public_admission_and_never_runs_the_test_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Path.cwd() / "configs/competition-cpu.toml"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    data_dir = tmp_path / "data"
    starter_dir = tmp_path / "starter"
    qualification_run_dir = tmp_path / "qualification"
    performance_profile = tmp_path / "performance.json"

    assert (
        main(
            [
                "compete",
                "--config",
                str(config),
                "--state-root",
                str(state_root),
                "--run-root",
                str(run_root),
                "--data-dir",
                str(data_dir),
                "--starter-dir",
                str(starter_dir),
                "--qualification-run-dir",
                str(qualification_run_dir),
                "--performance-profile",
                str(performance_profile),
                "--idempotency-key",
                "cli-offline-fixture",
            ]
        )
        == 3
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "before CampaignId creation" in output.err
    assert not state_root.exists()
    assert not run_root.exists()


def test_compete_cli_requires_all_production_evidence_paths() -> None:
    parser = build_parser()
    base = [
        "compete",
        "--config",
        "config.toml",
        "--run-root",
        "run",
    ]
    evidence_arguments = (
        ("--data-dir", "data"),
        ("--starter-dir", "starter"),
        ("--qualification-run-dir", "qualification"),
        ("--performance-profile", "performance.json"),
    )

    for omitted_flag, _ in evidence_arguments:
        arguments = [*base]
        for flag, value in evidence_arguments:
            if flag != omitted_flag:
                arguments.extend((flag, value))
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(arguments)
        assert exc_info.value.code == 2


def test_compete_cli_transports_production_evidence_into_campaign_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from kuairand_agent import resource_profiles
    from kuairand_agent.lab import AutonomousExperimentLab

    captured: dict[str, Any] = {}

    class _Result:
        def manifest(self) -> dict[str, object]:
            return {"status": "admitted"}

    class _Lab:
        def compete(self, *, options: object, idempotency_key: str) -> _Result:
            captured["options"] = options
            captured["idempotency_key"] = idempotency_key
            return _Result()

    def _open(**kwargs: object) -> _Lab:
        captured["open"] = kwargs
        return _Lab()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        resource_profiles,
        "load_resource_profile",
        lambda path: SimpleNamespace(name="competition-cpu"),
    )
    monkeypatch.setattr(AutonomousExperimentLab, "open", staticmethod(_open))

    assert (
        main(
            [
                "compete",
                "--config",
                "config.toml",
                "--state-root",
                "state",
                "--run-root",
                "run",
                "--data-dir",
                "data",
                "--starter-dir",
                "starter",
                "--qualification-run-dir",
                "qualification",
                "--performance-profile",
                "performance.json",
                "--idempotency-key",
                "production-evidence",
            ]
        )
        == 0
    )

    options = captured["options"]
    assert options.config_path == tmp_path / "config.toml"
    assert options.data_root == tmp_path / "data"
    assert options.starter_root == tmp_path / "starter"
    assert options.qualification_receipt == tmp_path / "qualification"
    assert options.performance_profile == tmp_path / "performance.json"
    assert captured["idempotency_key"] == "production-evidence"
    assert captured["open"] == {
        "repository_root": tmp_path,
        "state_root": tmp_path / "state",
        "run_root": tmp_path / "run",
        "profile": "competition-cpu",
    }
    assert json.loads(capsys.readouterr().out) == {"status": "admitted"}


def test_legacy_replay_parser_still_accepts_original_arguments() -> None:
    digest = "a" * 64

    args = build_parser().parse_args(
        [
            "replay",
            "--bundle",
            "bundle",
            "--project-root",
            ".",
            "--data-dir",
            "data",
            "--expected-data-sha256",
            digest,
        ]
    )

    assert args.bundle == Path("bundle")
    assert args.campaign_id is None


def test_campaign_replay_defaults_to_the_honest_fixture_grade() -> None:
    digest = "a" * 64

    args = build_parser().parse_args(
        [
            "replay",
            "--campaign-id",
            digest,
            "--state-root",
            "state",
        ]
    )

    assert args.grade == "experiment-same-backend"

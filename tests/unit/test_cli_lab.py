from __future__ import annotations

from pathlib import Path

import pytest

from kuairand_agent.cli import build_parser, main


def test_compete_cli_uses_public_admission_and_never_runs_the_test_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Path.cwd() / "configs/competition-cpu.toml"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"

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

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "run_full_campaign.sh"


def test_full_campaign_script_is_valid_posix_shell() -> None:
    checked = subprocess.run(
        ("sh", "-n", str(SCRIPT)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0
    assert checked.stdout == ""
    assert checked.stderr == ""


def test_full_campaign_script_selects_tree_group_and_forwards_explicit_paths(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_uv.chmod(0o700)
    config = tmp_path / "config.toml"
    qualification = tmp_path / "qualification"
    run_dir = tmp_path / "campaign"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")

    completed = subprocess.run(
        (str(SCRIPT), str(config), str(qualification), str(run_dir)),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        "sync",
        "--locked",
        "--group",
        "research-tree",
        "--no-group",
        "research-neural",
        "run",
        "--locked",
        "--group",
        "research-tree",
        "--no-group",
        "research-neural",
        "kuairand-agent",
        "run",
        "--config",
        str(config),
        "--qualification-run-dir",
        str(qualification),
        "--run-dir",
        str(run_dir),
    ]


def test_full_campaign_script_rejects_extra_arguments_before_uv(tmp_path: Path) -> None:
    completed = subprocess.run(
        (str(SCRIPT), "one", "two", "three", "four"),
        check=False,
        capture_output=True,
        text=True,
        env=os.environ | {"UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("usage: ")
    assert "[CONFIG_PATH [QUALIFICATION_RUN_DIR [RUN_DIR]]]" in completed.stderr

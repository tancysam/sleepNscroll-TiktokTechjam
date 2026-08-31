from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).parents[2]
CPU_SCRIPT = ROOT / "scripts" / "acceptance_cpu.sh"
GPU_SCRIPT = ROOT / "scripts" / "qualify_gpu.sh"


def _toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project_lock_package() -> dict[str, object]:
    packages = cast(list[dict[str, object]], _toml(ROOT / "uv.lock")["package"])
    return next(package for package in packages if package["name"] == "kuairand-agent")


def test_dependency_groups_keep_legacy_names_and_add_honest_cpu_gpu_profiles() -> None:
    project = _toml(ROOT / "pyproject.toml")
    groups = cast(dict[str, list[str]], project["dependency-groups"])

    assert groups["research-tree"] == ["lightgbm==4.7.0"]
    assert groups["research-neural"] == ["torch==2.13.0"]
    assert groups["tree-cpu"] == groups["research-tree"]
    # PyPI LightGBM's version does not prove its native library was built with GPU support.
    assert groups["tree-gpu"] == []


def test_lock_records_new_profiles_without_claiming_stock_lightgbm_is_gpu_enabled() -> None:
    project_package = _project_lock_package()
    metadata = cast(dict[str, object], project_package["metadata"])
    locked_groups = cast(dict[str, list[dict[str, str]]], metadata["requires-dev"])

    assert locked_groups["research-tree"] == [{"name": "lightgbm", "specifier": "==4.7.0"}]
    assert locked_groups["tree-cpu"] == locked_groups["research-tree"]
    assert locked_groups["tree-gpu"] == []


@pytest.mark.parametrize("script", [CPU_SCRIPT, GPU_SCRIPT])
def test_profile_scripts_are_valid_posix_shell_and_refuse_implicit_execution(
    script: Path,
    tmp_path: Path,
) -> None:
    syntax = subprocess.run(
        ("sh", "-n", str(script)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0
    assert syntax.stdout == ""
    assert syntax.stderr == ""

    environment = os.environ.copy()
    environment.pop("KUAIRAND_ENABLE_CPU_ACCEPTANCE", None)
    environment.pop("KUAIRAND_ENABLE_GPU_QUALIFICATION", None)
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    refused = subprocess.run(
        (str(script),),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert refused.returncode == 2
    assert refused.stdout == ""
    assert "opt-in" in refused.stderr


@pytest.mark.parametrize(
    ("script", "opt_in_name", "expected_module_action"),
    [
        (CPU_SCRIPT, "KUAIRAND_ENABLE_CPU_ACCEPTANCE", "validate"),
        (GPU_SCRIPT, "KUAIRAND_ENABLE_GPU_QUALIFICATION", "qualify-gpu"),
    ],
)
def test_profile_scripts_only_dispatch_local_resource_checks(
    script: Path,
    opt_in_name: str,
    expected_module_action: str,
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_uv.chmod(0o700)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    environment[opt_in_name] = "1"

    completed = subprocess.run(
        (str(script),),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "kuairand_agent.resource_profiles" in completed.stdout
    assert expected_module_action in completed.stdout
    assert "kuairand-agent" not in completed.stdout
    assert "protected" not in completed.stdout.lower()
    assert "provider" not in completed.stdout.lower()

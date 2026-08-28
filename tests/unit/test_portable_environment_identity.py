from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from kuairand_agent.campaign.provenance import capture_environment_identity, hash_source_tree
from kuairand_agent.finalization.replay import ReplayError, environment_identity_digest

_DECLARED_PACKAGES = {
    "lightgbm": "4.7.0",
    "numpy": "2.5.2",
    "psutil": "7.2.2",
    "torch": "2.13.0",
}
_PORTABLE_V2_BODY: dict[str, object] = {
    "schema_version": 2,
    "python": {"implementation": "CPython", "version": "3.12.13"},
    "platform": {"system": "Darwin", "release": "25.6.0", "machine": "arm64"},
    "packages": dict(_DECLARED_PACKAGES),
    "uv_lock_sha256": "f" * 64,
}


def _signed_environment(body: dict[str, object], *, domain: bytes) -> dict[str, object]:
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return body | {"digest": hashlib.sha256(domain + encoded).hexdigest()}


def _capture_fake_environment(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    implementation: str = "CPython",
    python_version: str = "3.12.13",
    system: str = "Darwin",
    release: str = "25.6.0",
    machine: str = "arm64",
    packages: Mapping[str, str] = _DECLARED_PACKAGES,
    lock: str = "version = 1\n",
) -> str:
    (root / "uv.lock").write_text(lock, encoding="utf-8")
    monkeypatch.setattr(platform, "python_implementation", lambda: implementation)
    monkeypatch.setattr(platform, "python_version", lambda: python_version)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "release", lambda: release)
    monkeypatch.setattr(platform, "machine", lambda: machine)

    def installed_version(package: str) -> str:
        try:
            return packages[package]
        except KeyError as exc:
            raise importlib.metadata.PackageNotFoundError(package) from exc

    monkeypatch.setattr(importlib.metadata, "version", installed_version)
    return capture_environment_identity(root).digest


def _write_minimal_project(root: Path) -> None:
    for directory in ("candidate_templates", "configs", "scripts", "src"):
        (root / directory).mkdir(parents=True)
    (root / ".python-version").write_text("3.12.13\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='portable'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "candidate_templates" / "candidate.py").write_text(
        "def predict():\n    return 1\n", encoding="utf-8"
    )
    (root / "configs" / "default.toml").write_text("schema_version = 1\n", encoding="utf-8")
    (root / "scripts" / "run.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (root / "src" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")


def test_source_identity_is_independent_of_checkout_location_and_binds_trusted_bytes(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "checkout-a"
    second_root = tmp_path / "unrelated" / "checkout-b"
    _write_minimal_project(first_root)
    _write_minimal_project(second_root)

    first = hash_source_tree(first_root)
    second = hash_source_tree(second_root)
    assert first.root != second.root
    assert first.manifest() == second.manifest()
    assert first.digest == second.digest

    (second_root / "src" / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert hash_source_tree(second_root).digest != first.digest


def test_environment_identity_is_independent_of_interpreter_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("PRIVATE_RESEARCH_TOKEN", "must-not-appear")

    first_root = "/private/tmp/clean-a/.venv/bin/python"
    monkeypatch.setattr(sys, "executable", first_root)
    monkeypatch.setattr(sys, "prefix", "/private/tmp/clean-a/.venv")
    monkeypatch.setattr(sys, "exec_prefix", "/private/tmp/clean-a/.venv")
    first = capture_environment_identity(tmp_path)

    second_root = "/opt/replay/clean-b/.venv/bin/python"
    monkeypatch.setattr(sys, "executable", second_root)
    monkeypatch.setattr(sys, "prefix", "/opt/replay/clean-b/.venv")
    monkeypatch.setattr(sys, "exec_prefix", "/opt/replay/clean-b/.venv")
    second = capture_environment_identity(tmp_path)

    assert first.digest == second.digest
    assert first.manifest() == second.manifest()
    assert first.manifest_data["schema_version"] == 2
    rendered = json.dumps(first.manifest(), sort_keys=True)
    assert first_root not in rendered
    assert second_root not in rendered
    assert "/private/tmp/clean-a/.venv" not in rendered
    assert "/opt/replay/clean-b/.venv" not in rendered
    assert "executable" not in rendered
    assert "must-not-appear" not in rendered


def test_replay_verifier_accepts_legacy_v1_and_portable_v2_identities(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    current = capture_environment_identity(tmp_path).manifest()
    assert environment_identity_digest(current) == current["digest"]

    legacy_body: dict[str, object] = {
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.12.0",
            "executable": "/legacy/location/.venv/bin/python",
        },
        "platform": {"system": "fixture", "release": "1", "machine": "test"},
        "packages": {"lightgbm": "4.7.0", "numpy": "2.5.2"},
        "uv_lock_sha256": "f" * 64,
    }
    legacy = _signed_environment(legacy_body, domain=b"kuairand-environment-v1\0")
    assert environment_identity_digest(legacy) == legacy["digest"]


def test_environment_identity_binds_runtime_facts_and_declared_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _capture_fake_environment(tmp_path, monkeypatch)
    manifest = capture_environment_identity(tmp_path).manifest_data
    assert manifest["packages"] == _DECLARED_PACKAGES

    assert _capture_fake_environment(tmp_path, monkeypatch, implementation="PyPy") != baseline
    assert _capture_fake_environment(tmp_path, monkeypatch, python_version="3.12.14") != baseline
    assert _capture_fake_environment(tmp_path, monkeypatch, system="Linux") != baseline
    assert _capture_fake_environment(tmp_path, monkeypatch, release="25.6.1") != baseline
    assert _capture_fake_environment(tmp_path, monkeypatch, machine="x86_64") != baseline
    assert (
        _capture_fake_environment(
            tmp_path,
            monkeypatch,
            packages=_DECLARED_PACKAGES | {"numpy": "2.5.3"},
        )
        != baseline
    )
    assert (
        _capture_fake_environment(
            tmp_path,
            monkeypatch,
            packages={key: value for key, value in _DECLARED_PACKAGES.items() if key != "torch"},
        )
        != baseline
    )
    assert _capture_fake_environment(tmp_path, monkeypatch, lock="version = 2\n") != baseline


@pytest.mark.parametrize(
    ("schema_version", "domain"),
    [
        (1, b"kuairand-environment-v2\0"),
        (2, b"kuairand-environment-v1\0"),
        (3, b"kuairand-environment-v2\0"),
        (True, b"kuairand-environment-v1\0"),
    ],
)
def test_replay_verifier_rejects_schema_domain_mismatch_or_unknown_schema(
    schema_version: object,
    domain: bytes,
) -> None:
    body: dict[str, object] = {
        "schema_version": schema_version,
        "python": {"implementation": "CPython", "version": "3.12.13"},
        "platform": {"system": "fixture", "release": "1", "machine": "test"},
        "packages": dict(_DECLARED_PACKAGES),
        "uv_lock_sha256": "f" * 64,
    }
    identity = _signed_environment(body, domain=domain)
    with pytest.raises(ReplayError, match=r"schema_version|digest"):
        environment_identity_digest(identity)


@pytest.mark.parametrize(
    "body",
    [
        _PORTABLE_V2_BODY | {"unexpected": "field"},
        _PORTABLE_V2_BODY
        | {
            "python": {
                "implementation": "CPython",
                "version": "3.12.13",
                "executable": "/private/tmp/.venv/bin/python",
            }
        },
        _PORTABLE_V2_BODY | {"python": {"implementation": "CPython"}},
        _PORTABLE_V2_BODY
        | {
            "platform": {
                "system": "Darwin",
                "release": "25.6.0",
                "machine": "arm64",
                "venv_root": "/private/tmp/.venv",
            }
        },
        _PORTABLE_V2_BODY | {"packages": _DECLARED_PACKAGES | {"pytest": "9.1.1"}},
        _PORTABLE_V2_BODY
        | {"packages": {key: value for key, value in _DECLARED_PACKAGES.items() if key != "torch"}},
        _PORTABLE_V2_BODY | {"packages": _DECLARED_PACKAGES | {"torch": 2}},
        _PORTABLE_V2_BODY | {"uv_lock_sha256": "F" * 64},
    ],
)
def test_v2_replay_verifier_rejects_extraneous_or_malformed_runtime_facts(
    body: dict[str, object],
) -> None:
    identity = _signed_environment(body, domain=b"kuairand-environment-v2\0")
    with pytest.raises(ReplayError):
        environment_identity_digest(identity)

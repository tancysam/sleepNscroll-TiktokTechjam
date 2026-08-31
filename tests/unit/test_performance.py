from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.performance import (
    LoadedPerformanceProfile,
    PerformanceAcceptanceError,
    PerformanceProfile,
    TimingReceipt,
    load_performance_profile,
    write_performance_profile,
)

type JsonObject = dict[str, object]


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def _profile() -> PerformanceProfile:
    receipts = tuple(
        TimingReceipt(
            family=family,
            wall_seconds=wall_seconds,
            cpu_seconds=wall_seconds / 2.0,
            peak_rss_bytes=4096 + index,
            rows=100 + index,
            evidence_digest=_digest(f"{family}-{index}"),
        )
        for index, (family, wall_seconds) in enumerate(
            (
                ("tree", 1.0),
                ("tree", 4.0),
                ("final_replay", 10.0),
                ("submission_check", 5.0),
            )
        )
    )
    return PerformanceProfile.create(
        receipts=receipts,
        controller_overhead_seconds=0.25,
        model_runtime_seconds=25.0,
        finalization_reserve_seconds=20.0,
        finalization_families=("final_replay", "submission_check"),
        projected_campaign_seconds=120.0,
        campaign_limit_seconds=360.0,
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _object(root: JsonObject, name: str) -> JsonObject:
    return cast(JsonObject, root[name])


def _receipt(root: JsonObject) -> JsonObject:
    return cast(JsonObject, cast(list[object], root["receipts"])[0])


def _family(root: JsonObject, name: str = "tree") -> JsonObject:
    return cast(JsonObject, cast(JsonObject, root["families"])[name])


def _mutated_profile(
    tmp_path: Path, mutate: Callable[[JsonObject], None], name: str = "mutated.json"
) -> Path:
    manifest = _profile().manifest()
    mutate(manifest)
    path = tmp_path / name
    path.write_bytes(_canonical(manifest))
    return path


def test_load_performance_profile_reconstructs_and_exposes_physical_identity(
    tmp_path: Path,
) -> None:
    profile = _profile()
    path = write_performance_profile(profile, tmp_path / "performance.json")

    loaded = load_performance_profile(path)

    assert isinstance(loaded, LoadedPerformanceProfile)
    assert loaded.profile == profile
    assert loaded.digest == profile.digest
    assert loaded.physical_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.file_sha256 == loaded.physical_sha256
    assert loaded.size_bytes == path.stat().st_size


@pytest.mark.parametrize(
    "mutate",
    [
        lambda root: root.__setitem__("unknown", True),
        lambda root: root.__setitem__("schema_version", 2),
        lambda root: _object(root, "controller").__setitem__("unknown", 0),
        lambda root: _receipt(root).__setitem__("unknown", 0),
        lambda root: _family(root).__setitem__("unknown", 0),
        lambda root: _object(root, "finalization").__setitem__("unknown", 0),
        lambda root: _object(root, "campaign").__setitem__("unknown", 0),
    ],
)
def test_load_rejects_unknown_schema_or_fields(
    tmp_path: Path, mutate: Callable[[JsonObject], None]
) -> None:
    path = _mutated_profile(tmp_path, mutate)

    with pytest.raises(PerformanceAcceptanceError, match=r"schema_version|fields differ"):
        load_performance_profile(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda root: _family(root).__setitem__("p95_seconds", 1.0),
        lambda root: _family(root).__setitem__("sample_count", 3),
        lambda root: _object(root, "controller").__setitem__("overhead_ratio", 0.5),
        lambda root: _object(root, "finalization").__setitem__("p95_seconds", 1.0),
        lambda root: _object(root, "finalization").__setitem__("reserve_sufficient", False),
        lambda root: _object(root, "campaign").__setitem__("time_sufficient", False),
        lambda root: root.__setitem__("digest", "0" * 64),
    ],
)
def test_load_rejects_inconsistent_derived_fields_or_digest(
    tmp_path: Path, mutate: Callable[[JsonObject], None]
) -> None:
    path = _mutated_profile(tmp_path, mutate)

    with pytest.raises(PerformanceAcceptanceError, match="inconsistent derived evidence"):
        load_performance_profile(path)


@pytest.mark.parametrize(
    "payload_transform",
    [
        lambda payload: b" " + payload,
        lambda payload: payload.rstrip(b"\n"),
        lambda payload: payload.replace(b'"schema_version":1', b'"schema_version":1.0'),
    ],
)
def test_load_rejects_noncanonical_physical_json(
    tmp_path: Path, payload_transform: Callable[[bytes], bytes]
) -> None:
    payload = _canonical(_profile().manifest())
    path = tmp_path / "noncanonical.json"
    path.write_bytes(payload_transform(payload))

    with pytest.raises(PerformanceAcceptanceError):
        load_performance_profile(path)


def test_load_rejects_symlink_nonregular_hardlinked_empty_and_oversized_files(
    tmp_path: Path,
) -> None:
    target = write_performance_profile(_profile(), tmp_path / "target.json")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(PerformanceAcceptanceError, match="non-symlink"):
        load_performance_profile(symlink)

    with pytest.raises(PerformanceAcceptanceError, match="regular non-symlink"):
        load_performance_profile(tmp_path)

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(PerformanceAcceptanceError, match="hard link"):
        load_performance_profile(target)

    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(PerformanceAcceptanceError, match="size"):
        load_performance_profile(empty)

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(PerformanceAcceptanceError, match="size"):
        load_performance_profile(oversized)

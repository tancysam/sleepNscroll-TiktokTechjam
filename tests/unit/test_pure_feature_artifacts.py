from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

import kuairand_agent.campaign.pure_feature_artifacts as artifact_module
from kuairand_agent.campaign.pure_feature_artifacts import (
    PureFeatureArtifactError,
    load_pure_feature_pair,
    save_pure_feature_pair,
    verify_pure_feature_artifact,
)
from kuairand_agent.campaign.pure_features import PureFeaturePair, build_pure_feature_pair
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.causal_features import FeatureMatrix


def _pair() -> PureFeaturePair:
    prefix = FeatureMatrix(
        np.asarray(
            [
                [np.float64(-0.0), np.nextafter(np.float64(1.0), np.float64(2.0))],
                [2.5, -3.25],
            ],
            dtype=np.float64,
        ),
        ("safe_a", "safe_b"),
    )
    query = FeatureMatrix(
        np.asarray([[4.5, np.nextafter(np.float64(0.0), np.float64(1.0))]]),
        prefix.feature_names,
    )
    return PureFeaturePair(
        prefix=prefix,
        query=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        causal_cache_key="b" * 64,
        categorical_encoding_digest="c" * 64,
        auxiliary_history_cache_key="d" * 64,
        feature_schema_version=7,
        feature_policy="pure-causal-tree-features-v7",
    )


def _current_v8_pair() -> PureFeaturePair:
    prefix = CanonicalInputs(
        user_id=("u", "u"),
        video_id=("v1", "v2"),
        date=(20220408, 20220408),
        duration_ms=(10_000.0, 20_000.0),
        tab=("0", "0"),
        author_id=("a", "a"),
        time_ms=(1, 2),
        video_type=("NORMAL", "AD"),
    )
    query = CanonicalInputs(
        user_id=("u",),
        video_id=("v1",),
        date=(20220409,),
        duration_ms=(10_000.0,),
        tab=("0",),
        author_id=("a",),
        time_ms=(3,),
        video_type=("NORMAL",),
    )
    return build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1, 0),
        prefix_click_labels=(1, 0),
        prefix_watch_progress=(1.2, 0.4),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        builder_source_digest="b" * 64,
    )


def test_pair_round_trip_is_byte_deterministic_and_float64_exact(tmp_path: Path) -> None:
    pair = _pair()
    first = save_pure_feature_pair(tmp_path / "first.npz", pair)
    second = save_pure_feature_pair(tmp_path / "second.npz", pair)

    restored = load_pure_feature_pair(
        first.npz_path,
        expected_manifest_sha256=first.manifest_sha256,
    )

    assert first.npz_path.read_bytes() == second.npz_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.npz_sha256 == second.npz_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest() == second.manifest()
    assert first.npz_sha256 == "61fdf1ce3ca7add3d950670dec87ed8f1b0e45c4276ee9ebfe3dbd8be7ae8c6f"
    assert (
        first.manifest_sha256 == "d15daf1fcd51088f3e353f4dd4e284fafc038f025b69c7cd4d4cb9b5883f6630"
    )
    assert restored.digest == pair.digest
    assert restored.prefix.values.tobytes() == pair.prefix.values.tobytes()
    assert restored.query.values.tobytes() == pair.query.values.tobytes()
    assert restored.prefix.feature_names == pair.prefix.feature_names
    assert not restored.prefix.values.flags.writeable
    assert not restored.query.values.flags.writeable
    assert first.npz_path.stat().st_nlink == 1
    assert first.manifest_path.stat().st_nlink == 1
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_schema_v1_feature_artifact_remains_replayable(tmp_path: Path) -> None:
    current = _pair()
    historical = PureFeaturePair(
        prefix=current.prefix,
        query=current.query,
        dataset_digest=current.dataset_digest,
        split_role=current.split_role,
        causal_cache_key=current.causal_cache_key,
        feature_schema_version=1,
        feature_policy="pure-causal-tree-features-v1",
    )
    artifact = save_pure_feature_pair(tmp_path / "historical.npz", historical)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == 1
    assert restored.feature_policy == "pure-causal-tree-features-v1"
    assert "recency" not in restored.manifest()
    assert restored.digest == historical.digest


def test_schema_v2_feature_artifact_remains_replayable(tmp_path: Path) -> None:
    current = _pair()
    historical = PureFeaturePair(
        prefix=current.prefix,
        query=current.query,
        dataset_digest=current.dataset_digest,
        split_role=current.split_role,
        causal_cache_key=current.causal_cache_key,
        feature_schema_version=2,
        feature_policy="pure-causal-tree-features-v2",
    )
    artifact = save_pure_feature_pair(tmp_path / "historical-v2.npz", historical)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == 2
    assert restored.feature_policy == "pure-causal-tree-features-v2"
    assert "recency" in restored.manifest()
    assert "categorical_codes" not in restored.manifest()
    assert restored.digest == historical.digest


def test_schema_v3_feature_artifact_remains_replayable(tmp_path: Path) -> None:
    current = _pair()
    historical = PureFeaturePair(
        prefix=current.prefix,
        query=current.query,
        dataset_digest=current.dataset_digest,
        split_role=current.split_role,
        causal_cache_key=current.causal_cache_key,
        categorical_encoding_digest="c" * 64,
        feature_schema_version=3,
        feature_policy="pure-causal-tree-features-v3",
    )
    artifact = save_pure_feature_pair(tmp_path / "historical-v3.npz", historical)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == 3
    assert restored.feature_policy == "pure-causal-tree-features-v3"
    assert restored.manifest()["recency"]["half_life_days"] == 3.0  # type: ignore[index]
    assert "categorical_codes" in restored.manifest()
    assert restored.digest == historical.digest


def test_schema_v4_feature_artifact_remains_replayable_without_auxiliary_history(
    tmp_path: Path,
) -> None:
    current = _pair()
    historical = PureFeaturePair(
        prefix=current.prefix,
        query=current.query,
        dataset_digest=current.dataset_digest,
        split_role=current.split_role,
        causal_cache_key=current.causal_cache_key,
        categorical_encoding_digest="c" * 64,
        feature_schema_version=4,
        feature_policy="pure-causal-tree-features-v4",
    )
    artifact = save_pure_feature_pair(tmp_path / "historical-v4.npz", historical)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == 4
    assert restored.feature_policy == "pure-causal-tree-features-v4"
    assert restored.manifest()["recency"]["half_life_days"] == [1.0, 3.0, 7.0]  # type: ignore[index]
    assert "categorical_codes" in restored.manifest()
    assert "auxiliary_history" not in restored.manifest()
    assert restored.digest == historical.digest


def test_schema_v5_feature_artifact_remains_replayable_without_watch_progress(
    tmp_path: Path,
) -> None:
    current = _pair()
    historical = PureFeaturePair(
        prefix=current.prefix,
        query=current.query,
        dataset_digest=current.dataset_digest,
        split_role=current.split_role,
        causal_cache_key=current.causal_cache_key,
        categorical_encoding_digest="c" * 64,
        auxiliary_history_cache_key="d" * 64,
        feature_schema_version=5,
        feature_policy="pure-causal-tree-features-v5",
    )
    artifact = save_pure_feature_pair(tmp_path / "historical-v5.npz", historical)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == 5
    assert restored.feature_policy == "pure-causal-tree-features-v5"
    assert "auxiliary_history" in restored.manifest()
    assert "watch_progress_history" not in restored.manifest()
    assert restored.digest == historical.digest


@pytest.mark.parametrize("version", (6, 7))
def test_schema_v6_and_v7_feature_artifacts_remain_replayable(
    tmp_path: Path,
    version: int,
) -> None:
    current = _pair()
    historical = PureFeaturePair(
        prefix=current.prefix,
        query=current.query,
        dataset_digest=current.dataset_digest,
        split_role=current.split_role,
        causal_cache_key=current.causal_cache_key,
        categorical_encoding_digest="c" * 64,
        auxiliary_history_cache_key="d" * 64,
        feature_schema_version=version,
        feature_policy=f"pure-causal-tree-features-v{version}",
    )
    artifact = save_pure_feature_pair(tmp_path / f"historical-v{version}.npz", historical)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == version
    assert "watch_progress_history" in restored.manifest()
    assert ("video_type_code" in restored.manifest()) is (version == 7)
    assert "input_exposure" not in restored.manifest()
    assert restored.digest == historical.digest


def test_schema_v8_input_exposure_round_trip_is_exact(tmp_path: Path) -> None:
    pair = _current_v8_pair()
    artifact = save_pure_feature_pair(tmp_path / "current-v8.npz", pair)

    restored = load_pure_feature_pair(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )

    assert restored.feature_schema_version == 8
    assert restored.input_exposure is not None
    assert restored.input_exposure.digest == pair.input_exposure.digest  # type: ignore[union-attr]
    assert restored.manifest()["input_exposure"] == pair.manifest()["input_exposure"]
    assert restored.digest == pair.digest
    assert restored.prefix.values.tobytes() == pair.prefix.values.tobytes()


def test_schema_v8_rejects_missing_input_exposure_manifest(tmp_path: Path) -> None:
    artifact = save_pure_feature_pair(tmp_path / "current-v8.npz", _current_v8_pair())
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    del manifest["pair"]["input_exposure"]
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match="logical pair manifest"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )


def test_save_never_overwrites_either_existing_artifact_file(tmp_path: Path) -> None:
    pair = _pair()
    artifact = save_pure_feature_pair(tmp_path / "features.npz", pair)
    npz_bytes = artifact.npz_path.read_bytes()
    manifest_bytes = artifact.manifest_path.read_bytes()

    with pytest.raises(PureFeatureArtifactError, match="cannot be overwritten"):
        save_pure_feature_pair(artifact.npz_path, pair)

    assert artifact.npz_path.read_bytes() == npz_bytes
    assert artifact.manifest_path.read_bytes() == manifest_bytes
    assert not tuple(tmp_path.glob(".*.tmp"))

    manifest_only = tmp_path / "occupied.manifest.json"
    manifest_only.write_bytes(b"incumbent")
    with pytest.raises(PureFeatureArtifactError, match="cannot be overwritten"):
        save_pure_feature_pair(tmp_path / "occupied.npz", pair)
    assert manifest_only.read_bytes() == b"incumbent"
    assert not (tmp_path / "occupied.npz").exists()


def test_load_rejects_symlinked_npz_or_manifest(tmp_path: Path) -> None:
    artifact = save_pure_feature_pair(tmp_path / "source.npz", _pair())

    linked_npz = tmp_path / "linked-npz.npz"
    linked_npz.symlink_to(artifact.npz_path)
    shutil.copyfile(artifact.manifest_path, linked_npz.with_suffix(".manifest.json"))
    with pytest.raises(PureFeatureArtifactError, match="regular non-symlink"):
        load_pure_feature_pair(
            linked_npz,
            expected_manifest_sha256=artifact.manifest_sha256,
        )

    linked_manifest_npz = tmp_path / "linked-manifest.npz"
    shutil.copyfile(artifact.npz_path, linked_manifest_npz)
    linked_manifest_npz.with_suffix(".manifest.json").symlink_to(artifact.manifest_path)
    with pytest.raises(PureFeatureArtifactError, match="regular non-symlink"):
        load_pure_feature_pair(
            linked_manifest_npz,
            expected_manifest_sha256=artifact.manifest_sha256,
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def test_load_rejects_corrupt_npz_and_tampered_manifest(tmp_path: Path) -> None:
    corrupt = save_pure_feature_pair(tmp_path / "corrupt.npz", _pair())
    corrupt.npz_path.chmod(0o600)
    corrupt.npz_path.write_bytes(b"corrupt")
    with pytest.raises(PureFeatureArtifactError, match="NPZ SHA-256 mismatch"):
        load_pure_feature_pair(
            corrupt.npz_path,
            expected_manifest_sha256=corrupt.manifest_sha256,
        )

    tampered = save_pure_feature_pair(tmp_path / "tampered.npz", _pair())
    tampered.manifest_path.chmod(0o600)
    tampered.manifest_path.write_bytes(tampered.manifest_path.read_bytes() + b" ")
    with pytest.raises(PureFeatureArtifactError, match="manifest SHA-256 mismatch"):
        load_pure_feature_pair(
            tampered.npz_path,
            expected_manifest_sha256=tampered.manifest_sha256,
        )


def test_load_rejects_self_consistent_transport_with_forged_logical_metadata(
    tmp_path: Path,
) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    manifest["pair"]["prefix"]["row_count"] = 99
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match=r"logical manifest|shape"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )


def test_load_rejects_malformed_npz_even_when_transport_hashes_are_updated(
    tmp_path: Path,
) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    malformed = b"not a NumPy archive"
    artifact.npz_path.chmod(0o600)
    artifact.npz_path.write_bytes(malformed)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    manifest["npz"]["sha256"] = hashlib.sha256(malformed).hexdigest()
    manifest["npz"]["size_bytes"] = len(malformed)
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match="cannot decode pure feature NPZ"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )


def test_verify_checks_declared_transport_logical_shape_and_split_identities(
    tmp_path: Path,
) -> None:
    pair = _pair()
    artifact = save_pure_feature_pair(tmp_path / "features.npz", pair)

    verified = verify_pure_feature_artifact(
        artifact.npz_path,
        expected_manifest_sha256=artifact.manifest_sha256,
        expected_npz_sha256=artifact.npz_sha256,
        expected_pair_digest=pair.digest,
        expected_dataset_digest=pair.dataset_digest,
        expected_split_role=pair.split_role,
        expected_prefix_row_count=2,
        expected_query_row_count=1,
        expected_feature_count=2,
    )

    assert verified == artifact


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"expected_npz_sha256": "0" * 64}, "expected NPZ SHA-256 mismatch"),
        ({"expected_pair_digest": "0" * 64}, "expected pair digest mismatch"),
        ({"expected_dataset_digest": "0" * 64}, "dataset digest mismatch"),
        ({"expected_split_role": "fold-a"}, "split role mismatch"),
        ({"expected_prefix_row_count": 3}, "prefix row count mismatch"),
        ({"expected_query_row_count": 3}, "query row count mismatch"),
        ({"expected_feature_count": 3}, "feature count mismatch"),
    ],
)
def test_verify_rejects_expected_identity_or_shape_mismatch(
    tmp_path: Path,
    expected: dict[str, object],
    message: str,
) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())

    with pytest.raises(PureFeatureArtifactError, match=message):
        verify_pure_feature_artifact(
            artifact.npz_path,
            expected_manifest_sha256=artifact.manifest_sha256,
            **expected,  # type: ignore[arg-type]
        )


def test_failed_directory_durability_check_removes_both_new_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "features.npz"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr("os.fsync", fail_directory_fsync)

    with pytest.raises(PureFeatureArtifactError, match="atomically install"):
        save_pure_feature_pair(destination, _pair())

    assert not destination.exists()
    assert not destination.with_suffix(".manifest.json").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_retry_recovers_matching_npz_left_before_manifest_commit(tmp_path: Path) -> None:
    pair = _pair()
    source = save_pure_feature_pair(tmp_path / "source.npz", pair)
    destination = tmp_path / "recovered.npz"
    shutil.copyfile(source.npz_path, destination)
    destination.chmod(0o444)
    orphan_bytes = destination.read_bytes()

    recovered = save_pure_feature_pair(destination, pair)

    assert recovered.npz_path.read_bytes() == orphan_bytes
    assert recovered.manifest_path.is_file()
    restored = load_pure_feature_pair(
        recovered.npz_path,
        expected_manifest_sha256=recovered.manifest_sha256,
    )
    assert restored.digest == pair.digest


def test_save_never_publishes_an_artifact_its_loader_size_bound_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "features.npz"
    monkeypatch.setattr(artifact_module, "_MAX_NPZ_BYTES", 1)

    with pytest.raises(PureFeatureArtifactError, match="supported bound"):
        save_pure_feature_pair(destination, _pair())

    assert not destination.exists()
    assert not destination.with_suffix(".manifest.json").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_load_rejects_in_place_npz_change_after_initial_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    real_zip_file = zipfile.ZipFile
    mutated = False

    def mutate_before_decode(
        file: object,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> zipfile.ZipFile:
        nonlocal mutated
        if mode == "r" and not mutated:
            artifact.npz_path.chmod(0o600)
            with artifact.npz_path.open("ab") as writable:
                writable.write(b"changed after initial hash")
            mutated = True
        return real_zip_file(  # type: ignore[no-any-return,call-overload]
            file,
            mode,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(zipfile, "ZipFile", mutate_before_decode)

    with pytest.raises(PureFeatureArtifactError, match="changed while"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=artifact.manifest_sha256,
        )


def test_load_rejects_json_number_and_boolean_type_confusion(tmp_path: Path) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    manifest["schema_version"] = True
    manifest["pair"]["schema_version"] = True
    manifest["pair"]["prefix"]["row_count"] = 2.0
    manifest["pair"]["query"]["row_count"] = True
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match=r"schema|logical manifest"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )


def test_load_rejects_reanchored_noncanonical_npz_encoding(tmp_path: Path) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    artifact.npz_path.chmod(0o600)
    with zipfile.ZipFile(artifact.npz_path, mode="a") as archive:
        archive.comment = b"alternate but otherwise valid encoding"
    npz_payload = artifact.npz_path.read_bytes()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    manifest["npz"]["sha256"] = hashlib.sha256(npz_payload).hexdigest()
    manifest["npz"]["size_bytes"] = len(npz_payload)
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match="comment is not canonical"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )


def test_load_rejects_reanchored_npz_without_canonical_zip64_members(tmp_path: Path) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    with zipfile.ZipFile(artifact.npz_path, mode="r") as original:
        members = tuple((name, original.read(name)) for name in ("prefix.npy", "query.npy"))
    artifact.npz_path.chmod(0o600)
    with zipfile.ZipFile(artifact.npz_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    npz_payload = artifact.npz_path.read_bytes()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    manifest["npz"]["sha256"] = hashlib.sha256(npz_payload).hexdigest()
    manifest["npz"]["size_bytes"] = len(npz_payload)
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match="member encoding is invalid"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )


def test_huge_claimed_npy_shape_is_rejected_before_array_allocation(tmp_path: Path) -> None:
    artifact = save_pure_feature_pair(tmp_path / "features.npz", _pair())
    with zipfile.ZipFile(artifact.npz_path, mode="r") as original:
        query_member = original.read("query.npy")
    huge_header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        huge_header,
        {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (10**15, 2),
        },
    )
    artifact.npz_path.chmod(0o600)
    with zipfile.ZipFile(artifact.npz_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in (
            ("prefix.npy", huge_header.getvalue()),
            ("query.npy", query_member),
        ):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                member.write(payload)
    npz_payload = artifact.npz_path.read_bytes()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="ascii"))
    manifest["npz"]["sha256"] = hashlib.sha256(npz_payload).hexdigest()
    manifest["npz"]["size_bytes"] = len(npz_payload)
    forged_payload = _canonical_json(manifest)
    artifact.manifest_path.chmod(0o600)
    artifact.manifest_path.write_bytes(forged_payload)

    with pytest.raises(PureFeatureArtifactError, match="NPY shape"):
        load_pure_feature_pair(
            artifact.npz_path,
            expected_manifest_sha256=hashlib.sha256(forged_payload).hexdigest(),
        )

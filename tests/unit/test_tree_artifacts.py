from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO, cast

import pytest

from kuairand_agent.candidates.tree_artifacts import (
    TreeCheckpointArtifactError,
    deserialize_tree_checkpoint,
    load_tree_checkpoint,
    save_tree_checkpoint,
    serialize_tree_checkpoint,
)
from kuairand_agent.candidates.tree_ranker import TreeRankerCheckpoint
from kuairand_agent.data.capabilities import DataPhase


def _checkpoint() -> TreeRankerCheckpoint:
    return TreeRankerCheckpoint(
        training_phase=DataPhase.INNER_TRAIN,
        feature_names=("causal_user_rate", "duration_log", "causal_item_count"),
        training_feature_digest="f" * 64,
        training_grouping_digest="a" * 64,
        training_target_digest="d" * 64,
        inner_validation_digest="e" * 64,
        config_digest="c" * 64,
        backend_identity="lightgbm:4.7.0:cpu",
        best_iteration=37,
        model_text="tree\nversion=v4\nfeature_names=causal_user_rate duration_log\n",
    )


def test_tree_checkpoint_round_trip_is_byte_deterministic_and_identity_checked(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint()

    first = save_tree_checkpoint(tmp_path / "first.tree", checkpoint)
    second = save_tree_checkpoint(tmp_path / "second.tree", checkpoint)
    restored = load_tree_checkpoint(
        first.path,
        expected_file_sha256=first.file_sha256,
        expected_checkpoint_digest=checkpoint.digest,
        expected_model_sha256=checkpoint.model_sha256,
        expected_training_feature_digest=checkpoint.training_feature_digest,
        expected_training_grouping_digest=checkpoint.training_grouping_digest,
        expected_training_target_digest=checkpoint.training_target_digest,
        expected_inner_validation_digest=checkpoint.inner_validation_digest,
        expected_config_digest=checkpoint.config_digest,
        expected_training_phase=DataPhase.INNER_TRAIN,
        expected_feature_names=checkpoint.feature_names,
        expected_backend_identity=checkpoint.backend_identity,
        expected_best_iteration=checkpoint.best_iteration,
    )

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.file_sha256 == second.file_sha256
    assert first.file_sha256 == "0c64e02b7c5bd3b997ec4f1ba8c184d5f26dc4818a212871ed31a5ee24482991"
    assert first.checkpoint_digest == checkpoint.digest
    assert first.model_sha256 == checkpoint.model_sha256
    assert restored == checkpoint
    assert restored.model_text == checkpoint.model_text
    assert restored.feature_names == checkpoint.feature_names


def test_tree_checkpoint_rejects_model_and_identity_metadata_tampering() -> None:
    payload = serialize_tree_checkpoint(_checkpoint())

    changed_model = payload[:-1] + bytes([payload[-1] ^ 1])
    with pytest.raises(TreeCheckpointArtifactError, match="model SHA-256 mismatch"):
        deserialize_tree_checkpoint(changed_model)

    changed_feature_order = payload.replace(
        b'["causal_user_rate","duration_log","causal_item_count"]',
        b'["causal_item_count","duration_log","causal_user_rate"]',
    )
    assert changed_feature_order != payload
    with pytest.raises(TreeCheckpointArtifactError, match="logical digest mismatch"):
        deserialize_tree_checkpoint(changed_feature_order)

    changed_backend = payload.replace(b"lightgbm:4.7.0:cpu", b"lightgbm:4.7.0:gpu")
    assert changed_backend != payload
    with pytest.raises(TreeCheckpointArtifactError, match="logical digest mismatch"):
        deserialize_tree_checkpoint(changed_backend)

    changed_iteration = payload.replace(b'"best_iteration":37', b'"best_iteration":38')
    assert changed_iteration != payload
    with pytest.raises(TreeCheckpointArtifactError, match="logical digest mismatch"):
        deserialize_tree_checkpoint(changed_iteration)

    changed_phase = payload.replace(
        b'"training_phase":"inner_train"',
        b'"training_phase":"outer_valid"',
    )
    assert changed_phase != payload
    with pytest.raises(TreeCheckpointArtifactError, match="training_phase"):
        deserialize_tree_checkpoint(changed_phase)


def test_tree_checkpoint_rejects_every_caller_supplied_identity_mismatch() -> None:
    payload = serialize_tree_checkpoint(_checkpoint())

    with pytest.raises(TreeCheckpointArtifactError, match="logical digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_checkpoint_digest="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="model digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_model_sha256="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="training feature digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_training_feature_digest="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="training grouping digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_training_grouping_digest="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="training target digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_training_target_digest="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="inner validation digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_inner_validation_digest="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="config digest mismatch"):
        deserialize_tree_checkpoint(payload, expected_config_digest="0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="training phase mismatch"):
        deserialize_tree_checkpoint(payload, expected_training_phase=DataPhase.TRAIN)
    with pytest.raises(TreeCheckpointArtifactError, match="feature names/order mismatch"):
        deserialize_tree_checkpoint(
            payload,
            expected_feature_names=(
                "duration_log",
                "causal_user_rate",
                "causal_item_count",
            ),
        )
    with pytest.raises(TreeCheckpointArtifactError, match="backend identity mismatch"):
        deserialize_tree_checkpoint(payload, expected_backend_identity="lightgbm:4.7.0:gpu")
    with pytest.raises(TreeCheckpointArtifactError, match="best iteration mismatch"):
        deserialize_tree_checkpoint(payload, expected_best_iteration=38)


def test_tree_checkpoint_publish_is_no_overwrite_and_load_rejects_symlinks(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkpoint.tree"
    artifact = save_tree_checkpoint(destination, _checkpoint())
    original = destination.read_bytes()

    with pytest.raises(TreeCheckpointArtifactError, match="cannot be overwritten"):
        save_tree_checkpoint(destination, _checkpoint())
    assert destination.read_bytes() == original

    with pytest.raises(TreeCheckpointArtifactError, match="file SHA-256 mismatch"):
        load_tree_checkpoint(destination, expected_file_sha256="0" * 64)

    link = tmp_path / "checkpoint-link.tree"
    link.symlink_to(artifact.path)
    with pytest.raises(TreeCheckpointArtifactError, match="non-symlink"):
        load_tree_checkpoint(link)


def test_tree_checkpoint_load_decodes_the_same_inode_that_was_hash_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _checkpoint()
    replacement = TreeRankerCheckpoint(
        training_phase=DataPhase.TRAIN,
        feature_names=("replacement_feature",),
        training_feature_digest="1" * 64,
        training_grouping_digest="2" * 64,
        training_target_digest="3" * 64,
        inner_validation_digest=None,
        config_digest="4" * 64,
        backend_identity="lightgbm:4.7.0:cpu",
        best_iteration=2,
        model_text="replacement-model\n",
    )
    original_artifact = save_tree_checkpoint(tmp_path / "original.tree", original)
    replacement_artifact = save_tree_checkpoint(tmp_path / "replacement.tree", replacement)
    real_fdopen = os.fdopen
    replaced = False

    def replace_path_after_open(descriptor: int, mode: str) -> IO[bytes]:
        nonlocal replaced
        if not replaced:
            os.replace(replacement_artifact.path, original_artifact.path)
            replaced = True
        return cast(IO[bytes], real_fdopen(descriptor, mode))

    monkeypatch.setattr(os, "fdopen", replace_path_after_open)

    restored = load_tree_checkpoint(
        original_artifact.path,
        expected_file_sha256=original_artifact.file_sha256,
    )

    assert restored.digest == original.digest
    assert restored.digest != replacement.digest


def test_tree_checkpoint_publish_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync
    observed_directory_flags: list[bool] = []

    def recording_fsync(descriptor: int) -> None:
        observed_directory_flags.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    save_tree_checkpoint(tmp_path / "checkpoint.tree", _checkpoint())

    assert observed_directory_flags == [False, True]


def test_failed_tree_checkpoint_directory_fsync_removes_new_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.tree"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(TreeCheckpointArtifactError, match="directory durable"):
        save_tree_checkpoint(destination, _checkpoint())

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".checkpoint.tree.*.tmp"))


def test_tree_checkpoint_codec_rejects_malformed_or_noncanonical_payloads(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint()
    payload = serialize_tree_checkpoint(checkpoint)

    with pytest.raises(TreeCheckpointArtifactError, match="payload must be bytes"):
        deserialize_tree_checkpoint(bytearray(payload))  # type: ignore[arg-type]
    with pytest.raises(TreeCheckpointArtifactError, match="magic or schema"):
        deserialize_tree_checkpoint(b"X" + payload[1:])
    with pytest.raises(TreeCheckpointArtifactError, match="valid UTF-8"):
        deserialize_tree_checkpoint(payload[:-1] + b"\xff")

    magic_end = payload.index(b"\n") + 1
    length_end = magic_end + 8
    metadata_size = int.from_bytes(payload[magic_end:length_end], "big")
    metadata_end = length_end + metadata_size
    noncanonical_metadata = b" " + payload[length_end:metadata_end]
    noncanonical = b"".join(
        (
            payload[:magic_end],
            len(noncanonical_metadata).to_bytes(8, "big"),
            noncanonical_metadata,
            payload[metadata_end:],
        )
    )
    with pytest.raises(TreeCheckpointArtifactError, match="canonical JSON"):
        deserialize_tree_checkpoint(noncanonical)

    with pytest.raises(TreeCheckpointArtifactError, match=r"must end in \.tree"):
        save_tree_checkpoint(tmp_path / "checkpoint.txt", checkpoint)

    object.__setattr__(checkpoint, "digest", "0" * 64)
    with pytest.raises(TreeCheckpointArtifactError, match="identity does not match"):
        serialize_tree_checkpoint(checkpoint)


def test_tree_checkpoint_codec_wraps_json_integer_limit_failures() -> None:
    payload = serialize_tree_checkpoint(_checkpoint())
    magic_end = payload.index(b"\n") + 1
    length_end = magic_end + 8
    metadata_size = int.from_bytes(payload[magic_end:length_end], "big")
    metadata_end = length_end + metadata_size
    metadata = payload[length_end:metadata_end]
    original_size = str(len(_checkpoint().model_text.encode("utf-8"))).encode("ascii")
    oversized_integer = metadata.replace(
        b'"model_size_bytes":' + original_size,
        b'"model_size_bytes":' + b"9" * 5000,
    )
    assert oversized_integer != metadata
    malformed = b"".join(
        (
            payload[:magic_end],
            len(oversized_integer).to_bytes(8, "big"),
            oversized_integer,
            payload[metadata_end:],
        )
    )

    with pytest.raises(TreeCheckpointArtifactError, match="canonical JSON"):
        deserialize_tree_checkpoint(malformed)

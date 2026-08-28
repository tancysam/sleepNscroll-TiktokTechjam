from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO, cast

import numpy as np
import pytest

from kuairand_agent.baselines.artifacts import (
    BaselineArtifactError,
    PredictionVector,
    StarterFMCheckpoint,
    load_checkpoint,
    load_predictions,
    save_checkpoint,
    save_predictions,
)


def _checkpoint() -> StarterFMCheckpoint:
    factors = np.zeros((2, 16), dtype=np.float32)
    factors[0, 0] = np.float32(0.1)
    factors[0, 1] = np.float32(-0.2)
    factors[1, 0] = np.nextafter(np.float32(1), np.float32(2))
    return StarterFMCheckpoint(
        V=factors,
        W=np.array([np.float32(-0.0), np.float32(0.25)]),
        b=np.nextafter(np.float32(0.0), np.float32(1.0)),
        encoding_digest="e" * 64,
        config_digest="c" * 64,
        starter_manifest_digest="a" * 64,
        seed=0,
        best_epoch=2,
        epochs_completed=5,
        optimizer_steps=10,
    )


def test_checkpoint_round_trip_preserves_exact_float32_state_and_bytes(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    first = save_checkpoint(tmp_path / "first.npz", checkpoint)
    second = save_checkpoint(tmp_path / "second.npz", checkpoint)
    restored = load_checkpoint(
        first.path,
        expected_file_sha256=first.file_sha256,
        expected_checkpoint_digest=checkpoint.digest,
        expected_encoding_digest=checkpoint.encoding_digest,
        expected_starter_manifest_digest=checkpoint.starter_manifest_digest,
    )

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.file_sha256 == second.file_sha256
    assert restored.digest == checkpoint.digest
    assert restored.V.tobytes() == checkpoint.V.tobytes()
    assert restored.W.tobytes() == checkpoint.W.tobytes()
    assert restored.b.tobytes() == checkpoint.b.tobytes()
    assert not restored.V.flags.writeable
    assert not restored.W.flags.writeable
    with pytest.raises(ValueError):
        restored.V.setflags(write=True)


def test_prediction_round_trip_is_byte_deterministic_and_float64_exact(
    tmp_path: Path,
) -> None:
    predictions = PredictionVector(
        value
        for value in (
            np.float64(-0.0),
            np.nextafter(np.float64(1.0), np.float64(2.0)),
            np.float32(0.125),
        )
    )

    first = save_predictions(tmp_path / "first.npy", predictions)
    second = save_predictions(tmp_path / "second.npy", predictions)
    restored = load_predictions(
        first.path,
        expected_file_sha256=first.file_sha256,
        expected_prediction_digest=first.prediction_digest,
        expected_row_count=predictions.row_count,
    )

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.file_sha256 == second.file_sha256
    assert restored.digest == predictions.digest
    assert restored.scores.tobytes() == predictions.scores.tobytes()
    assert not restored.scores.flags.writeable


def test_artifact_writes_are_no_overwrite_and_preserve_existing_bytes(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    checkpoint_path = tmp_path / "checkpoint.npz"
    prediction_path = tmp_path / "predictions.npy"
    save_checkpoint(checkpoint_path, checkpoint)
    save_predictions(prediction_path, PredictionVector([0.1, 0.2]))
    checkpoint_bytes = checkpoint_path.read_bytes()
    prediction_bytes = prediction_path.read_bytes()

    with pytest.raises(BaselineArtifactError, match="cannot be overwritten"):
        save_checkpoint(checkpoint_path, checkpoint)
    with pytest.raises(BaselineArtifactError, match="cannot be overwritten"):
        save_predictions(prediction_path, PredictionVector([0.3, 0.4]))

    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert prediction_path.read_bytes() == prediction_bytes


def test_checkpoint_load_rejects_corruption_and_identity_mismatches(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    artifact = save_checkpoint(tmp_path / "checkpoint.npz", checkpoint)

    with pytest.raises(BaselineArtifactError, match="file SHA-256 mismatch"):
        load_checkpoint(artifact.path, expected_file_sha256="0" * 64)
    with pytest.raises(BaselineArtifactError, match="logical digest mismatch"):
        load_checkpoint(artifact.path, expected_checkpoint_digest="0" * 64)
    with pytest.raises(BaselineArtifactError, match="encoding digest mismatch"):
        load_checkpoint(artifact.path, expected_encoding_digest="0" * 64)
    with pytest.raises(BaselineArtifactError, match="starter manifest digest mismatch"):
        load_checkpoint(artifact.path, expected_starter_manifest_digest="0" * 64)
    with pytest.raises(BaselineArtifactError, match="config digest mismatch"):
        load_checkpoint(artifact.path, expected_config_digest="0" * 64)
    with pytest.raises(BaselineArtifactError, match="seed mismatch"):
        load_checkpoint(artifact.path, expected_seed=1)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a NumPy archive")
    with pytest.raises(BaselineArtifactError, match="cannot decode checkpoint"):
        load_checkpoint(corrupt)


def test_prediction_load_rejects_corruption_pickle_and_identity_mismatches(
    tmp_path: Path,
) -> None:
    predictions = PredictionVector([0.1, 0.2])
    artifact = save_predictions(tmp_path / "predictions.npy", predictions)

    with pytest.raises(BaselineArtifactError, match="file SHA-256 mismatch"):
        load_predictions(artifact.path, expected_file_sha256="0" * 64)
    with pytest.raises(BaselineArtifactError, match="logical digest mismatch"):
        load_predictions(artifact.path, expected_prediction_digest="0" * 64)
    with pytest.raises(BaselineArtifactError, match="row count mismatch"):
        load_predictions(artifact.path, expected_row_count=3)

    object_path = tmp_path / "object.npy"
    with object_path.open("wb") as handle:
        np.save(handle, np.asarray([{"unsafe": True}], dtype=object), allow_pickle=True)
    with pytest.raises(BaselineArtifactError, match="cannot decode prediction"):
        load_predictions(object_path)


def test_load_decodes_the_same_file_snapshot_that_was_hash_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _checkpoint()
    replacement = StarterFMCheckpoint(
        V=np.ones((2, 16), dtype=np.float32),
        W=np.ones(2, dtype=np.float32),
        b=np.float32(1.0),
        encoding_digest="e" * 64,
        config_digest="c" * 64,
        starter_manifest_digest="a" * 64,
        seed=0,
        best_epoch=1,
        epochs_completed=1,
        optimizer_steps=1,
    )
    original_artifact = save_checkpoint(tmp_path / "original.npz", original)
    replacement_artifact = save_checkpoint(tmp_path / "replacement.npz", replacement)
    real_fdopen = os.fdopen
    replaced = False

    def replace_path_after_open(descriptor: int, mode: str) -> IO[bytes]:
        nonlocal replaced
        if not replaced:
            os.replace(replacement_artifact.path, original_artifact.path)
            replaced = True
        return cast(IO[bytes], real_fdopen(descriptor, mode))

    monkeypatch.setattr(os, "fdopen", replace_path_after_open)

    restored = load_checkpoint(
        original_artifact.path,
        expected_file_sha256=original_artifact.file_sha256,
    )

    assert restored.digest == original.digest
    assert restored.digest != replacement.digest


def test_atomic_install_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync
    observed_directory_flags: list[bool] = []

    def recording_fsync(descriptor: int) -> None:
        observed_directory_flags.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    save_checkpoint(tmp_path / "checkpoint.npz", _checkpoint())

    assert observed_directory_flags == [False, True]


def test_failed_directory_durability_check_removes_the_new_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.npz"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(BaselineArtifactError, match="directory durable"):
        save_checkpoint(destination, _checkpoint())

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".checkpoint.npz.*.tmp"))


def test_artifact_load_rejects_symlinks(tmp_path: Path) -> None:
    artifact = save_checkpoint(tmp_path / "checkpoint.npz", _checkpoint())
    link = tmp_path / "checkpoint-link.npz"
    link.symlink_to(artifact.path)

    with pytest.raises(BaselineArtifactError, match="non-symlink"):
        load_checkpoint(link)


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [[0.1]],
        [True, False],
        ["0.1"],
        [complex(1.0, 0.0)],
        [np.nan],
        [np.inf],
    ],
)
def test_prediction_vector_rejects_unsafe_or_nonfinite_values(scores: object) -> None:
    with pytest.raises(BaselineArtifactError):
        PredictionVector(scores)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"V": np.zeros((2, 16), dtype=np.bool_)}, "checkpoint V must contain real numeric"),
        ({"W": ["0", "1"]}, "checkpoint W must contain real numeric"),
        ({"b": "0"}, "checkpoint b must be a finite float32 scalar"),
        ({"V": np.full((2, 16), np.nan)}, "only finite"),
        ({"V": np.zeros((2, 15))}, "official factor dimension"),
        ({"W": [0.0]}, "dimensions must agree"),
        ({"best_epoch": 6}, "cannot exceed"),
        ({"epochs_completed": 41, "optimizer_steps": 41}, "official maximum"),
        ({"optimizer_steps": 11}, "fixed positive batch count"),
        ({"seed": True}, "uint32-compatible"),
        ({"encoding_digest": "E" * 64}, "lowercase SHA-256"),
    ],
)
def test_checkpoint_rejects_invalid_types_shapes_and_identity(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "V": np.zeros((2, 16), dtype=np.float32),
        "W": [0.0, 0.25],
        "b": 0.0,
        "encoding_digest": "e" * 64,
        "config_digest": "c" * 64,
        "starter_manifest_digest": "a" * 64,
        "seed": 0,
        "best_epoch": 2,
        "epochs_completed": 5,
        "optimizer_steps": 10,
    }
    values.update(changes)
    with pytest.raises(BaselineArtifactError, match=message):
        StarterFMCheckpoint(**values)  # type: ignore[arg-type]

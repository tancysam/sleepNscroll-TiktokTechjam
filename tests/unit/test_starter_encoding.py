from __future__ import annotations

import json
import os
import stat
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.baselines.encoding import (
    MAX_ENCODING_ARCHIVE_BYTES,
    StarterEncoding,
    StarterEncodingError,
)
from kuairand_agent.data.canonical import CanonicalInputs


def _inputs(
    *,
    users: tuple[str, ...] = ("u2", "u1", "u2", "u3"),
    videos: tuple[str, ...] = ("v2", "v1", "v3", "v1"),
    authors: tuple[str, ...] = ("a2", "a1", "a2", "UNK"),
    tabs: tuple[str, ...] = ("1", "0", "1", "2"),
    durations: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0),
) -> CanonicalInputs:
    row_count = len(users)
    return CanonicalInputs(
        user_id=users,
        video_id=videos,
        date=(20220408,) * row_count,
        duration_ms=durations,
        tab=tabs,
        author_id=authors,
        time_ms=tuple(range(row_count)),
    )


def test_fit_matches_organizer_first_seen_vocab_unknown_offsets_and_dtype() -> None:
    inputs = _inputs()
    encoding = StarterEncoding.fit(inputs)

    expected_edges = np.quantile(np.asarray(inputs.duration_ms), np.linspace(0, 1, 11)[1:-1])
    assert encoding.edges == tuple(float(value) for value in expected_edges)
    assert encoding.field_names == (
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "dur_bucket",
    )
    assert encoding.vocabs[:4] == (
        ("u2", "u1", "u3"),
        ("v2", "v1", "v3"),
        ("a2", "a1", "UNK"),
        ("1", "0", "2"),
    )
    assert encoding.field_dims == tuple(len(vocab) + 1 for vocab in encoding.vocabs)
    assert encoding.unknown_ids == tuple(len(vocab) for vocab in encoding.vocabs)
    assert encoding.offsets == tuple(
        int(value) for value in np.cumsum((0, *encoding.field_dims[:-1]))
    )
    assert encoding.total_dim == sum(encoding.field_dims)

    transformed = encoding.transform(inputs)
    assert transformed.dtype == np.int32
    assert transformed.shape == (4, 5)
    assert transformed.flags.c_contiguous
    assert not transformed.flags.writeable


def test_transform_maps_unseen_values_to_each_fields_trailing_unknown_slot() -> None:
    encoding = StarterEncoding.fit(_inputs())
    unseen = _inputs(
        users=("new-user",),
        videos=("new-video",),
        authors=("new-author",),
        tabs=("14",),
        durations=(100.0,),
    )

    transformed = encoding.transform(unseen)
    expected = np.asarray(
        [
            [
                unknown_id + offset
                for unknown_id, offset in zip(encoding.unknown_ids, encoding.offsets, strict=True)
            ]
        ],
        dtype=np.int32,
    )
    # Duration 100 maps to the known bucket "9"; the other four fields are unknown.
    expected[0, 4] = encoding.vocabs[4].index("9") + encoding.offsets[4]
    np.testing.assert_array_equal(transformed, expected)


def test_save_load_is_exact_object_free_and_manifest_is_deterministic(tmp_path: Path) -> None:
    encoding = StarterEncoding.fit(_inputs())
    artifact = encoding.save(tmp_path / "encoding.npz")
    second_artifact = encoding.save(tmp_path / "encoding-copy.npz")
    restored = StarterEncoding.load(
        artifact.path,
        expected_file_sha256=artifact.file_sha256,
    )

    assert restored == encoding
    assert restored.digest == encoding.digest == artifact.digest
    assert second_artifact.file_sha256 == artifact.file_sha256
    assert restored.manifest() == encoding.manifest()
    assert json.dumps(restored.manifest(), sort_keys=True, separators=(",", ":")) == json.dumps(
        encoding.manifest(), sort_keys=True, separators=(",", ":")
    )
    with np.load(artifact.path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_load_rejects_tampered_derived_metadata(tmp_path: Path) -> None:
    path = tmp_path / "encoding.npz"
    StarterEncoding.fit(_inputs()).save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["total_dim"] = arrays["total_dim"] + 1
    with path.open("wb") as handle:
        np.savez(handle, **arrays)

    with pytest.raises(StarterEncodingError, match="total_dim metadata is corrupt"):
        StarterEncoding.load(path)


def test_save_validates_then_atomically_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "encoding.npz"
    path.write_bytes(b"user-owned-existing-artifact")

    with pytest.raises(StarterEncodingError, match="refusing to overwrite"):
        StarterEncoding.fit(_inputs()).save(path)

    assert path.read_bytes() == b"user-owned-existing-artifact"
    assert not tuple(tmp_path.glob(".encoding.npz.*.tmp"))


def test_save_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        observed_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    StarterEncoding.fit(_inputs()).save(tmp_path / "encoding.npz")

    assert len(observed_modes) >= 2
    assert any(stat.S_ISREG(mode) for mode in observed_modes[:-1])
    assert stat.S_ISDIR(observed_modes[-1])


def test_load_checks_retained_file_sha256(tmp_path: Path) -> None:
    artifact = StarterEncoding.fit(_inputs()).save(tmp_path / "encoding.npz")

    assert (
        StarterEncoding.load(artifact.path, expected_file_sha256=artifact.file_sha256).digest
        == artifact.digest
    )
    with pytest.raises(StarterEncodingError, match="SHA-256 mismatch"):
        StarterEncoding.load(artifact.path, expected_file_sha256="0" * 64)
    with pytest.raises(StarterEncodingError, match="must be lowercase SHA-256"):
        StarterEncoding.load(artifact.path, expected_file_sha256="not-a-digest")


@pytest.mark.parametrize("kind", ["symlink", "directory", "empty", "oversize"])
def test_load_rejects_unsafe_file_types_and_sizes(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "unsafe.npz"
    if kind == "symlink":
        target = tmp_path / "target.npz"
        StarterEncoding.fit(_inputs()).save(target)
        path.symlink_to(target)
        message = "must not be a symlink"
    elif kind == "directory":
        path.mkdir()
        message = "must be a regular file"
    elif kind == "empty":
        path.touch()
        message = "must not be empty"
    else:
        with path.open("wb") as handle:
            handle.truncate(MAX_ENCODING_ARCHIVE_BYTES + 1)
        message = "exceeds"

    with pytest.raises(StarterEncodingError, match=message):
        StarterEncoding.load(path)


def _rewrite_zip(
    source: Path,
    destination: Path,
    *,
    reverse: bool = False,
    duplicate_first: bool = False,
) -> None:
    with zipfile.ZipFile(source, "r") as existing:
        entries = [(member.filename, existing.read(member)) for member in existing.infolist()]
    if reverse:
        entries.reverse()
    if duplicate_first:
        entries.append(entries[0])
    with (
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as changed,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        for name, contents in entries:
            changed.writestr(name, contents)


@pytest.mark.parametrize("reverse,duplicate", [(True, False), (False, True)])
def test_load_rejects_reordered_or_duplicate_zip_members(
    tmp_path: Path,
    reverse: bool,
    duplicate: bool,
) -> None:
    original = StarterEncoding.fit(_inputs()).save(tmp_path / "original.npz")
    changed = tmp_path / "changed.npz"
    _rewrite_zip(original.path, changed, reverse=reverse, duplicate_first=duplicate)

    with pytest.raises(StarterEncodingError, match="duplicated, reordered, or unexpected"):
        StarterEncoding.load(changed)


def test_fit_rejects_empty_training_inputs() -> None:
    empty = CanonicalInputs(
        user_id=(),
        video_id=(),
        date=(),
        duration_ms=(),
        tab=(),
        author_id=(),
        time_ms=(),
    )
    with pytest.raises(StarterEncodingError, match="empty training"):
        StarterEncoding.fit(empty)

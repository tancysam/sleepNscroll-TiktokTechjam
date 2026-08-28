from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from kuairand_agent.execution.artifacts import (
    ArtifactCollisionError,
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactPolicyError,
    ArtifactStore,
    ArtifactTooLargeError,
)


def test_put_bytes_is_content_addressed_and_deduplicated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store", max_object_bytes=1024)

    first = store.put_bytes(b"same bytes", kind=ArtifactKind.SOURCE)
    second = store.put_bytes(b"same bytes", kind=ArtifactKind.INPUT)

    assert first.sha256 == hashlib.sha256(b"same bytes").hexdigest()
    assert second.sha256 == first.sha256
    assert second.kind is ArtifactKind.INPUT
    assert store.object_path(first) == store.object_path(second)
    assert store.verify(first).read_bytes() == b"same bytes"
    assert not tuple(store.staging_root.iterdir())


def test_different_bytes_never_overwrite_an_existing_object(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    first = store.put_bytes(b"incumbent", kind=ArtifactKind.CHECKPOINT)
    second = store.put_bytes(b"challenger", kind=ArtifactKind.CHECKPOINT)

    assert first.sha256 != second.sha256
    assert store.verify(first).read_bytes() == b"incumbent"
    assert store.verify(second).read_bytes() == b"challenger"


def test_put_file_rejects_symlink_and_special_file(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"content")
    link = tmp_path / "link.bin"
    link.symlink_to(regular)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)

    with pytest.raises(ArtifactPolicyError, match="regular non-symlink"):
        store.put_file(link, kind=ArtifactKind.OTHER)
    with pytest.raises(ArtifactPolicyError, match="regular non-symlink"):
        store.put_file(fifo, kind=ArtifactKind.OTHER)


def test_size_ceiling_is_enforced_and_staging_is_cleaned(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store", max_object_bytes=4, chunk_bytes=2)
    source = tmp_path / "large.bin"
    source.write_bytes(b"12345")

    with pytest.raises(ArtifactTooLargeError):
        store.put_bytes(b"12345", kind=ArtifactKind.OTHER)
    with pytest.raises(ArtifactTooLargeError):
        store.put_file(source, kind=ArtifactKind.OTHER)

    assert not tuple(store.staging_root.iterdir())
    assert not tuple(store.objects_root.rglob("[0-9a-f]" * 64))


def test_corrupt_object_fails_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"original", kind=ArtifactKind.OUTPUT)
    object_path = store.object_path(ref)
    object_path.chmod(0o600)
    object_path.write_bytes(b"tampered")
    object_path.chmod(0o444)

    with pytest.raises(ArtifactIntegrityError, match="digest"):
        store.verify(ref)


def test_existing_corrupt_digest_path_is_never_replaced(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"protected", kind=ArtifactKind.CHECKPOINT)
    object_path = store.object_path(ref)
    object_path.chmod(0o600)
    object_path.write_bytes(b"corrupted")
    object_path.chmod(0o444)

    with pytest.raises(ArtifactCollisionError):
        store.put_bytes(b"protected", kind=ArtifactKind.CHECKPOINT)

    assert object_path.read_bytes() == b"corrupted"
    assert not tuple(store.staging_root.iterdir())


def test_verify_rejects_external_hardlink(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"immutable", kind=ArtifactKind.SOURCE)
    external_link = tmp_path / "external-link"
    os.link(store.object_path(ref), external_link)

    with pytest.raises(ArtifactIntegrityError, match="hardlinks"):
        store.verify(ref)


def test_directory_artifact_is_a_deterministic_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "lib").mkdir(parents=True)
        (root / "candidate.py").write_text("print('ok')\n", encoding="utf-8")
        (root / "lib" / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")

    left_ref = store.put_directory(left, kind=ArtifactKind.SOURCE)
    right_ref = store.put_directory(right, kind=ArtifactKind.SOURCE)

    assert left_ref.sha256 == right_ref.sha256
    assert tuple(entry.path for entry in left_ref.entries) == (
        "candidate.py",
        "lib/feature.py",
    )
    assert store.verify_directory(left_ref) is left_ref
    assert store.load_directory(left_ref.manifest_artifact) == left_ref


def test_directory_artifact_rejects_links_and_special_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    first = source / "candidate.py"
    first.write_text("pass\n", encoding="utf-8")
    hardlink = source / "copy.py"
    os.link(first, hardlink)

    with pytest.raises(ArtifactPolicyError, match="hardlinked"):
        store.put_directory(source, kind=ArtifactKind.SOURCE)

    hardlink.unlink()
    link = source / "linked.py"
    link.symlink_to(first)
    with pytest.raises(ArtifactPolicyError, match="symlink"):
        store.put_directory(source, kind=ArtifactKind.SOURCE)


def test_orphan_staging_file_is_harmless(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    orphan = store.staging_root / "orphan-from-crash"
    orphan.write_bytes(b"unreferenced")

    ref = store.put_bytes(b"committed", kind=ArtifactKind.OTHER)

    assert store.verify(ref).read_bytes() == b"committed"
    assert orphan.read_bytes() == b"unreferenced"


def test_read_bytes_honors_a_narrower_bound(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store", max_object_bytes=100)
    ref = store.put_bytes(b"12345", kind=ArtifactKind.LOG)

    with pytest.raises(ArtifactTooLargeError):
        store.read_bytes(ref, max_bytes=4)
    assert store.read_bytes(ref, max_bytes=5) == b"12345"


def test_store_root_must_be_private_and_cannot_traverse_symlinked_layout(
    tmp_path: Path,
) -> None:
    public_root = tmp_path / "public-store"
    public_root.mkdir(mode=0o755)
    public_root.chmod(0o755)
    with pytest.raises(ArtifactPolicyError, match="private"):
        ArtifactStore(public_root)

    private_root = tmp_path / "private-store"
    private_root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (private_root / "objects").symlink_to(outside)
    with pytest.raises(ArtifactPolicyError, match="real directory"):
        ArtifactStore(private_root)


def test_store_detects_layout_replacement_after_initialization(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"committed", kind=ArtifactKind.OTHER)
    objects_root = store.root / "objects"
    moved = store.root / "objects-original"
    objects_root.rename(moved)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    objects_root.symlink_to(outside)

    with pytest.raises(ArtifactPolicyError, match="real directory"):
        store.verify(ref)

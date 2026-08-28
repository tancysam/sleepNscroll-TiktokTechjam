from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from kuairand_agent.data.acquire import (
    ARCHIVE_SIZE_BYTES,
    ARCHIVE_TAR_SIZE_BYTES,
    OFFICIAL_ARCHIVE_SPEC,
    OFFICIAL_ARCHIVE_URL,
    OFFICIAL_MEMBER_MANIFEST,
    ArchiveIntegrityError,
    ArchiveMember,
    ArchiveSpec,
    MemberType,
    download_and_prepare,
    prepare_archive,
    verify_archive,
)


def _write_tar(
    path: Path,
    entries: list[tuple[str, bytes | None]],
    *,
    type_overrides: dict[str, bytes] | None = None,
    link_overrides: dict[str, str] | None = None,
    mode_overrides: dict[str, int] | None = None,
) -> None:
    type_overrides = type_overrides or {}
    link_overrides = link_overrides or {}
    mode_overrides = mode_overrides or {}
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            is_directory = content is None
            info.type = type_overrides.get(
                name, tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
            )
            info.mode = mode_overrides.get(name, 0o755 if is_directory else 0o644)
            info.linkname = link_overrides.get(name, "")
            payload = b"" if content is None else content
            info.size = 0 if info.type != tarfile.REGTYPE else len(payload)
            archive.addfile(info, None if info.type != tarfile.REGTYPE else io.BytesIO(payload))


def _member(
    name: str,
    content: bytes | None,
    *,
    header: tuple[str, ...] | None = None,
) -> ArchiveMember:
    if content is None:
        return ArchiveMember(name=name, type=MemberType.DIRECTORY, size=0)
    return ArchiveMember(
        name=name,
        type=MemberType.REGULAR,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        header=header,
    )


def _synthetic_spec(
    archive: Path,
    members: tuple[ArchiveMember, ...],
    *,
    source: str = "synthetic-test-fixture",
) -> ArchiveSpec:
    compressed = archive.read_bytes()
    with gzip.open(archive, "rb") as stream:
        tar_size = sum(len(chunk) for chunk in iter(lambda: stream.read(64 * 1024), b""))
    return ArchiveSpec(
        source=source,
        filename=archive.name,
        size=len(compressed),
        md5=hashlib.md5(compressed, usedforsecurity=False).hexdigest(),
        sha256=hashlib.sha256(compressed).hexdigest(),
        tar_size=tar_size,
        members=members,
    )


def _safe_fixture(tmp_path: Path) -> tuple[Path, ArchiveSpec]:
    archive = tmp_path / "fixture.tar.gz"
    header = ("id", "value")
    csv = b"id,value\n1,alpha\n"
    entries: list[tuple[str, bytes | None]] = [
        ("Fixture/", None),
        ("Fixture/data/", None),
        ("Fixture/data/rows.csv", csv),
        ("Fixture/LICENSE", b"fixture license\n"),
    ]
    _write_tar(archive, entries)
    members = (
        _member("Fixture/", None),
        _member("Fixture/data/", None),
        _member("Fixture/data/rows.csv", csv, header=header),
        _member("Fixture/LICENSE", b"fixture license\n"),
    )
    return archive, _synthetic_spec(archive, members)


def _simple_expected_members(*names: str) -> tuple[ArchiveMember, ...]:
    return (_member("Fixture/", None), *(_member(name, b"payload\n") for name in names))


def _rewrite_first_name_with_embedded_nul(archive: Path) -> None:
    tar_payload = bytearray(gzip.decompress(archive.read_bytes()))
    malicious_name = b"Fixture/\x00hidden"
    tar_payload[:100] = malicious_name + b"\x00" * (100 - len(malicious_name))
    tar_payload[148:156] = b" " * 8
    checksum = sum(tar_payload[:512])
    tar_payload[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    archive.write_bytes(gzip.compress(bytes(tar_payload), mtime=0))


def test_synthetic_manifest_seam_verifies_and_installs_without_official_digest(
    tmp_path: Path,
) -> None:
    archive, spec = _safe_fixture(tmp_path)

    verification = verify_archive(archive, spec=spec)
    result = prepare_archive(archive, tmp_path / "prepared", spec=spec)

    assert verification.archive_sha256 == spec.sha256
    assert verification.member_sha256 == {
        member.name: member.sha256 for member in spec.members if member.sha256 is not None
    }
    assert (result.destination / "Fixture/data/rows.csv").read_bytes() == b"id,value\n1,alpha\n"
    assert (result.destination / "Fixture/LICENSE").read_bytes() == b"fixture license\n"
    manifest = json.loads(result.integrity_manifest.read_text(encoding="ascii"))
    assert manifest["archive"]["sha256"] == spec.sha256
    assert [member["name"] for member in manifest["members"]] == [
        member.name for member in spec.members
    ]
    assert (
        result.manifest_sha256 == hashlib.sha256(result.integrity_manifest.read_bytes()).hexdigest()
    )
    assert result.dataset_root == result.destination / "Fixture"


def test_official_spec_pins_complete_exact_archive_and_ordered_member_manifest() -> None:
    assert ARCHIVE_SIZE_BYTES == 47_432_272
    assert ARCHIVE_TAR_SIZE_BYTES == 203_547_136
    assert OFFICIAL_ARCHIVE_SPEC.md5 == "0820331067a3784d9691136f772b35a7"
    assert (
        OFFICIAL_ARCHIVE_SPEC.sha256
        == "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"
    )
    assert sum(member.size for member in OFFICIAL_MEMBER_MANIFEST) == 203_539_133
    observed_members = [
        (member.name, member.type.value, member.size) for member in OFFICIAL_MEMBER_MANIFEST
    ]
    assert observed_members == [
        ("KuaiRand-Pure/", "directory", 0),
        ("KuaiRand-Pure/LICENSE", "regular", 20_138),
        ("KuaiRand-Pure/data/", "directory", 0),
        ("KuaiRand-Pure/load_data_pure.py", "regular", 1_608),
        ("KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv", "regular", 83_961_282),
        ("KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv", "regular", 87_086_116),
        ("KuaiRand-Pure/data/user_features_pure.csv", "regular", 3_519_028),
        ("KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv", "regular", 21_765_075),
        (
            "KuaiRand-Pure/data/video_features_statistic_pure.csv",
            "regular",
            6_559_217,
        ),
        ("KuaiRand-Pure/data/video_features_basic_pure.csv", "regular", 626_669),
    ]
    assert [member.sha256 for member in OFFICIAL_MEMBER_MANIFEST] == [
        None,
        "187442db4df3afd21f2f0525739fd4beac28a62daaba3ee8d3533f60e7c33ec7",
        None,
        "19b6117c9c82a6480af72603e66579f1e0e824e16ce826eb9e6ac98fbf1ce6af",
        "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
        "60b80994da969cd53da4d50c37ba3dafd6fb185df804c92c8410df34845a9d2c",
        "dc729a656301b4c6d07f713fe41d05ec9bfaab670b90e531c70037caf033c011",
        "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
        "d5c9e237ef2c6c1fc0e7f27e952f215d6626ecd934b01a6c53ecfcc72540f6b6",
        "a6f7ee02684c5777422306cdc416e170302288aa89aca9dfea995edbd625bcc2",
    ]
    assert OFFICIAL_MEMBER_MANIFEST[4].header == (
        "user_id",
        "video_id",
        "date",
        "hourmin",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
        "is_rand",
        "tab",
    )
    assert OFFICIAL_MEMBER_MANIFEST[4].header == OFFICIAL_MEMBER_MANIFEST[5].header
    assert OFFICIAL_MEMBER_MANIFEST[4].header == OFFICIAL_MEMBER_MANIFEST[7].header
    assert len(OFFICIAL_MEMBER_MANIFEST[6].header or ()) == 31
    assert len(OFFICIAL_MEMBER_MANIFEST[8].header or ()) == 52
    assert OFFICIAL_MEMBER_MANIFEST[9].header == (
        "video_id",
        "author_id",
        "video_type",
        "upload_dt",
        "upload_type",
        "visible_status",
        "video_duration",
        "server_width",
        "server_height",
        "music_id",
        "music_type",
        "tag",
    )


@pytest.mark.parametrize(
    "malicious_name",
    [
        "/absolute",
        "//network/absolute",
        "C:/windows/absolute",
        "Fixture/../escape",
        "Fixture/./dot",
        "Fixture//empty",
        "Fixture\\backslash",
    ],
)
def test_unsafe_member_paths_are_rejected(
    tmp_path: Path,
    malicious_name: str,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, [("Fixture/", None), (malicious_name, b"payload\n")])
    spec = _synthetic_spec(archive, _simple_expected_members("Fixture/safe"))

    with pytest.raises(
        ArchiveIntegrityError,
        match=r"path|absolute|dot|traversal|empty segment",
    ):
        verify_archive(archive, spec=spec)


def test_embedded_nul_data_in_raw_tar_name_field_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "nul.tar.gz"
    _write_tar(archive, [("Fixture/", None)])
    _rewrite_first_name_with_embedded_nul(archive)
    spec = _synthetic_spec(archive, (_member("Fixture/", None),))

    with pytest.raises(ArchiveIntegrityError, match="embedded NUL"):
        verify_archive(archive, spec=spec)


def test_duplicate_members_are_rejected_before_order_matching(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.tar.gz"
    _write_tar(
        archive,
        [
            ("Fixture/", None),
            ("Fixture/first", b"payload\n"),
            ("Fixture/first", b"payload\n"),
        ],
    )
    spec = _synthetic_spec(
        archive,
        _simple_expected_members("Fixture/first", "Fixture/second"),
    )

    with pytest.raises(ArchiveIntegrityError, match="duplicate"):
        verify_archive(archive, spec=spec)


def test_case_colliding_members_are_rejected_on_every_filesystem(tmp_path: Path) -> None:
    archive = tmp_path / "collision.tar.gz"
    _write_tar(
        archive,
        [
            ("Fixture/", None),
            ("Fixture/Member", b"payload\n"),
            ("Fixture/member", b"payload\n"),
        ],
    )
    spec = _synthetic_spec(
        archive,
        _simple_expected_members("Fixture/Member", "Fixture/other"),
    )

    with pytest.raises(ArchiveIntegrityError, match="case-colliding"):
        verify_archive(archive, spec=spec)


@pytest.mark.parametrize(
    ("member_type", "link_name"),
    [
        (tarfile.SYMTYPE, "Fixture/target"),
        (tarfile.LNKTYPE, "Fixture/target"),
        (tarfile.CHRTYPE, ""),
        (tarfile.BLKTYPE, ""),
        (tarfile.FIFOTYPE, ""),
        (tarfile.GNUTYPE_SPARSE, ""),
    ],
)
def test_links_and_special_members_are_rejected(
    tmp_path: Path,
    member_type: bytes,
    link_name: str,
) -> None:
    archive = tmp_path / "special.tar.gz"
    name = "Fixture/member"
    _write_tar(
        archive,
        [("Fixture/", None), (name, b"payload\n")],
        type_overrides={name: member_type},
        link_overrides={name: link_name},
    )
    spec = _synthetic_spec(archive, _simple_expected_members(name))

    with pytest.raises(
        ArchiveIntegrityError,
        match=r"links|sparse|devices|FIFOs|extension",
    ):
        verify_archive(archive, spec=spec)


def test_executable_regular_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "executable.tar.gz"
    name = "Fixture/program"
    _write_tar(
        archive,
        [("Fixture/", None), (name, b"payload\n")],
        mode_overrides={name: 0o755},
    )
    spec = _synthetic_spec(archive, _simple_expected_members(name))

    with pytest.raises(ArchiveIntegrityError, match="executable"):
        verify_archive(archive, spec=spec)


def test_unexpected_or_out_of_order_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unexpected.tar.gz"
    _write_tar(
        archive,
        [("Fixture/", None), ("Fixture/unexpected", b"payload\n")],
    )
    spec = _synthetic_spec(archive, _simple_expected_members("Fixture/expected"))

    with pytest.raises(ArchiveIntegrityError, match="unexpected or out-of-order"):
        verify_archive(archive, spec=spec)


@pytest.mark.parametrize("identity_field", ["size", "md5", "sha256"])
def test_whole_archive_identity_mismatch_is_rejected_before_tar_use(
    tmp_path: Path,
    identity_field: str,
) -> None:
    archive, spec = _safe_fixture(tmp_path)
    if identity_field == "size":
        mismatched = replace(spec, size=spec.size + 1)
    elif identity_field == "md5":
        mismatched = replace(spec, md5="0" * 32)
    else:
        mismatched = replace(spec, sha256="0" * 64)

    with pytest.raises(ArchiveIntegrityError, match=r"identity|size mismatch"):
        verify_archive(archive, spec=mismatched)


def test_member_size_digest_and_header_mismatches_are_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "member-mismatch.tar.gz"
    content = b"id,wrong\n1,value\n"
    _write_tar(archive, [("Fixture/", None), ("Fixture/rows.csv", content)])
    actual = _member("Fixture/rows.csv", content, header=("id", "value"))
    base = _synthetic_spec(archive, (_member("Fixture/", None), actual))

    with pytest.raises(ArchiveIntegrityError, match="size mismatch"):
        verify_archive(
            archive,
            spec=replace(base, members=(base.members[0], replace(actual, size=actual.size + 1))),
        )
    with pytest.raises(ArchiveIntegrityError, match="digest mismatch"):
        verify_archive(
            archive,
            spec=replace(
                base,
                members=(base.members[0], replace(actual, sha256="0" * 64, header=None)),
            ),
        )
    with pytest.raises(ArchiveIntegrityError, match="CSV header mismatch"):
        verify_archive(archive, spec=base)


def test_payload_bound_is_enforced_by_synthetic_manifest_seam(tmp_path: Path) -> None:
    _archive, spec = _safe_fixture(tmp_path)
    oversized = ArchiveMember(
        name="Fixture/oversized",
        type=MemberType.REGULAR,
        size=203_539_134,
        sha256="0" * 64,
    )
    oversized_spec = replace(
        spec,
        tar_size=203_541_504,
        members=(spec.members[0], oversized),
    )

    with pytest.raises(ArchiveIntegrityError, match="payload exceeds"):
        oversized_spec.validate()


def test_failed_extraction_removes_private_staging_and_destination(tmp_path: Path) -> None:
    archive = tmp_path / "failure.tar.gz"
    _write_tar(
        archive,
        [
            ("Fixture/", None),
            ("Fixture/good", b"payload\n"),
            ("Fixture/bad", b"payload\n"),
        ],
    )
    good = _member("Fixture/good", b"payload\n")
    bad = replace(_member("Fixture/bad", b"payload\n"), sha256="0" * 64)
    spec = _synthetic_spec(archive, (_member("Fixture/", None), good, bad))
    destination = tmp_path / "prepared"

    with pytest.raises(ArchiveIntegrityError, match="digest mismatch"):
        prepare_archive(archive, destination, spec=spec)

    assert not destination.exists()
    assert list(tmp_path.glob(".prepared.staging-*")) == []


def test_double_build_refuses_overwrite_and_preserves_first_install(tmp_path: Path) -> None:
    archive, spec = _safe_fixture(tmp_path)
    destination = tmp_path / "prepared"
    first = prepare_archive(archive, destination, spec=spec)
    before = first.integrity_manifest.read_bytes()

    with pytest.raises(ArchiveIntegrityError, match="destination already exists"):
        prepare_archive(archive, destination, spec=spec)

    assert first.integrity_manifest.read_bytes() == before
    assert (destination / "Fixture/data/rows.csv").read_bytes() == b"id,value\n1,alpha\n"
    assert list(tmp_path.glob(".prepared.staging-*")) == []


def test_two_clean_builds_have_identical_manifests_and_controlled_modes(tmp_path: Path) -> None:
    archive, spec = _safe_fixture(tmp_path)
    first = prepare_archive(archive, tmp_path / "first", spec=spec)
    second = prepare_archive(archive, tmp_path / "second", spec=spec)

    assert first.integrity_manifest.read_bytes() == second.integrity_manifest.read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256
    assert stat.S_IMODE((first.destination / "Fixture").stat().st_mode) == 0o700
    assert stat.S_IMODE((first.destination / "Fixture/data/rows.csv").stat().st_mode) == 0o600


def test_restrictive_umask_cannot_change_modes_or_defeat_failure_cleanup(tmp_path: Path) -> None:
    archive, spec = _safe_fixture(tmp_path)
    bad_member = replace(spec.members[2], sha256="0" * 64)
    bad_spec = replace(spec, members=(*spec.members[:2], bad_member, *spec.members[3:]))
    previous_umask = os.umask(0o700)
    try:
        result = prepare_archive(archive, tmp_path / "prepared", spec=spec)
        with pytest.raises(ArchiveIntegrityError, match="digest mismatch"):
            prepare_archive(archive, tmp_path / "failed", spec=bad_spec)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((result.destination / "Fixture").stat().st_mode) == 0o700
    assert stat.S_IMODE((result.destination / "Fixture/data/rows.csv").stat().st_mode) == 0o600
    assert stat.S_IMODE(result.integrity_manifest.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".prepared.staging-*")) == []
    assert list(tmp_path.glob(".failed.staging-*")) == []


def test_csv_body_is_copied_as_opaque_bytes_without_parsing_final_tail(tmp_path: Path) -> None:
    archive = tmp_path / "opaque.tar.gz"
    content = b"id\n\xff\x00long_view,1,unparsed-final-byte:\xfe"
    _write_tar(archive, [("Fixture/", None), ("Fixture/opaque.csv", content)])
    spec = _synthetic_spec(
        archive,
        (
            _member("Fixture/", None),
            _member("Fixture/opaque.csv", content, header=("id",)),
        ),
    )

    result = prepare_archive(archive, tmp_path / "prepared", spec=spec)

    assert (result.destination / "Fixture/opaque.csv").read_bytes() == content


def test_explicit_download_streams_only_pinned_url_then_uses_secure_prepare(
    tmp_path: Path,
) -> None:
    archive, spec = _safe_fixture(tmp_path)
    downloaded = archive.read_bytes()
    calls: list[tuple[str, float]] = []

    def opener(url: str, timeout_seconds: float) -> io.BytesIO:
        calls.append((url, timeout_seconds))
        return io.BytesIO(downloaded)

    result = download_and_prepare(
        tmp_path / "downloaded",
        timeout_seconds=12.5,
        opener=opener,
        spec=spec,
    )

    assert calls == [(OFFICIAL_ARCHIVE_URL, 12.5)]
    assert (result.destination / "Fixture/data/rows.csv").read_bytes() == b"id,value\n1,alpha\n"
    assert list(tmp_path.glob(".downloaded.download-*")) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("truncated", "ended early"),
        ("oversized", "exceeds"),
        ("digest", "digest differs"),
    ],
)
def test_explicit_download_rejects_bounded_stream_identity_failures_and_cleans_up(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    archive, spec = _safe_fixture(tmp_path)
    original = archive.read_bytes()
    if mutation == "truncated":
        payload = original[:-1]
    elif mutation == "oversized":
        payload = original + b"x"
    else:
        payload = bytes([original[0] ^ 1]) + original[1:]

    def opener(_url: str, _timeout_seconds: float) -> io.BytesIO:
        return io.BytesIO(payload)

    destination = tmp_path / "downloaded"
    with pytest.raises(ArchiveIntegrityError, match=message):
        download_and_prepare(destination, opener=opener, spec=spec)

    assert not destination.exists()
    assert list(tmp_path.glob(".downloaded.download-*")) == []


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, float("inf"), 301.0])
def test_explicit_download_rejects_unbounded_timeout_before_opening(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    archive, spec = _safe_fixture(tmp_path)
    opened = False

    def opener(_url: str, _timeout_seconds: float) -> io.BytesIO:
        nonlocal opened
        opened = True
        return io.BytesIO(archive.read_bytes())

    with pytest.raises(ArchiveIntegrityError, match="timeout"):
        download_and_prepare(
            tmp_path / "downloaded",
            timeout_seconds=timeout_seconds,
            opener=opener,
            spec=spec,
        )
    assert opened is False


def test_archive_and_existing_destination_symlinks_are_never_followed(tmp_path: Path) -> None:
    archive, spec = _safe_fixture(tmp_path)
    source_directory = tmp_path / "source-link"
    source_directory.mkdir()
    linked_archive = source_directory / archive.name
    linked_archive.symlink_to(archive)

    with pytest.raises(ArchiveIntegrityError, match="securely open"):
        verify_archive(linked_archive, spec=spec)

    other = tmp_path / "other"
    other.mkdir()
    destination = tmp_path / "prepared"
    destination.symlink_to(other, target_is_directory=True)
    with pytest.raises(ArchiveIntegrityError, match="destination already exists"):
        prepare_archive(archive, destination, spec=spec)


def test_non_regular_archive_and_destination_parent_loop_fail_without_blocking(
    tmp_path: Path,
) -> None:
    archive, spec = _safe_fixture(tmp_path)
    fifo_directory = tmp_path / "fifo"
    fifo_directory.mkdir()
    fifo = fifo_directory / archive.name
    os.mkfifo(fifo)
    with pytest.raises(ArchiveIntegrityError, match="not a regular file"):
        verify_archive(fifo, spec=spec)

    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    with pytest.raises(ArchiveIntegrityError, match="destination parent"):
        prepare_archive(archive, loop / "prepared", spec=spec)


@pytest.mark.skipif(
    "KUAIRAND_PURE_ARCHIVE" not in os.environ,
    reason="set KUAIRAND_PURE_ARCHIVE to run the verified official-archive gate",
)
def test_official_archive_matches_every_pinned_byte_identity() -> None:
    archive = Path(os.environ["KUAIRAND_PURE_ARCHIVE"])
    result = verify_archive(archive)
    assert result.archive_size == ARCHIVE_SIZE_BYTES
    assert result.tar_size == ARCHIVE_TAR_SIZE_BYTES
    assert result.payload_size == 203_539_133
    assert tuple(result.members) == OFFICIAL_MEMBER_MANIFEST


@pytest.mark.skipif(
    "KUAIRAND_PURE_ARCHIVE" not in os.environ,
    reason="set KUAIRAND_PURE_ARCHIVE to run the verified official-extraction gate",
)
def test_official_archive_securely_installs_exact_payload(tmp_path: Path) -> None:
    archive = Path(os.environ["KUAIRAND_PURE_ARCHIVE"])
    result = prepare_archive(archive, tmp_path / "prepared")

    assert result.dataset_root.name == "KuaiRand-Pure"
    assert result.verification.member_sha256 == {
        member.name: member.sha256
        for member in OFFICIAL_MEMBER_MANIFEST
        if member.sha256 is not None
    }
    for member in OFFICIAL_MEMBER_MANIFEST:
        extracted = result.destination / member.name
        if member.type is MemberType.DIRECTORY:
            assert extracted.is_dir()
            assert stat.S_IMODE(extracted.stat().st_mode) == 0o700
        else:
            assert extracted.is_file()
            assert extracted.stat().st_size == member.size
            assert stat.S_IMODE(extracted.stat().st_mode) == 0o600
    assert result.integrity_manifest.stat().st_size > 0

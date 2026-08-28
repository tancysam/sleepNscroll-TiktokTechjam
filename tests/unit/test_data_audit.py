from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.contract import SplitName
from kuairand_agent.data.audit import (
    OFFICIAL_AUDIT_CONTRACT,
    OFFICIAL_SOURCE_IDENTITIES,
    OFFICIAL_SPLIT_IDENTITIES,
    AuditContract,
    DataAuditError,
    ExpectedSourceIdentity,
    ExpectedSplitIdentity,
    audit_dataset,
    write_audit_report,
)
from kuairand_agent.data.fields import (
    CSV_HEADERS,
    RANDOMIZED_MEMBER,
    STANDARD_LATE_MEMBER,
    STANDARD_LOG_HEADER,
    STANDARD_TRAIN_MEMBER,
    USER_SNAPSHOT_MEMBER,
    VIDEO_BASIC_HEADER,
    VIDEO_BASIC_MEMBER,
    VIDEO_STATISTIC_MEMBER,
)

_OUTCOMES = {
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
}
_MEMBER_ORDER = (
    STANDARD_TRAIN_MEMBER,
    RANDOMIZED_MEMBER,
    USER_SNAPSHOT_MEMBER,
    STANDARD_LATE_MEMBER,
    VIDEO_STATISTIC_MEMBER,
    VIDEO_BASIC_MEMBER,
)


def _physical(member: str) -> str:
    return member.removeprefix("data/")


def _log_row(
    *,
    user_id: int,
    video_id: int,
    date: int,
    time_ms: int,
    target: int,
    outcome_poison: bytes | None = None,
    safe_overrides: Mapping[str, bytes] | None = None,
) -> bytes:
    values: dict[str, bytes] = {
        "user_id": str(user_id).encode("ascii"),
        "video_id": str(video_id).encode("ascii"),
        "date": str(date).encode("ascii"),
        "hourmin": b"1200",
        "time_ms": str(time_ms).encode("ascii"),
        "is_click": str(target).encode("ascii"),
        "is_like": b"0",
        "is_follow": b"0",
        "is_comment": b"0",
        "is_forward": b"0",
        "is_hate": b"0",
        "long_view": str(target).encode("ascii"),
        "play_time_ms": b"12.5",
        "duration_ms": b"30.0",
        "profile_stay_time": b"0.0",
        "comment_stay_time": b"0.0",
        "is_profile_enter": b"0",
        "is_rand": b"0",
        "tab": b"1",
    }
    if outcome_poison is not None:
        for field_name in _OUTCOMES:
            values[field_name] = outcome_poison
    if safe_overrides is not None:
        values.update(safe_overrides)
    return b",".join(values[field_name] for field_name in STANDARD_LOG_HEADER) + b"\n"


def _header(member: str) -> bytes:
    return ",".join(CSV_HEADERS[member]).encode("ascii") + b"\n"


def _basic_row(video_id: int, author_id: int) -> bytes:
    values = {name: b"0" for name in VIDEO_BASIC_HEADER}
    values["video_id"] = str(video_id).encode("ascii")
    values["author_id"] = str(author_id).encode("ascii")
    return b",".join(values[name] for name in VIDEO_BASIC_HEADER) + b"\n"


def _opaque_row(member: str) -> bytes:
    return b",".join(b"opaque" for _ in CSV_HEADERS[member]) + b"\n"


def _write_fixture(
    root: Path,
    *,
    final_poison: bytes = b"final-not-a-number",
    train_target: bytes | None = None,
    valid_aux_poison: bytes | None = None,
    final_safe_overrides: Mapping[str, bytes] | None = None,
) -> Path:
    data = root / "KuaiRand-Pure" / "data"
    data.mkdir(parents=True)
    train_rows = [
        _log_row(user_id=1, video_id=10, date=20220408, time_ms=1, target=1),
        _log_row(user_id=1, video_id=10, date=20220408, time_ms=2, target=0),
        _log_row(user_id=2, video_id=11, date=20220409, time_ms=3, target=1),
        _log_row(user_id=3, video_id=12, date=20220409, time_ms=4, target=0),
    ]
    if train_target is not None:
        train_rows[0] = _log_row(
            user_id=1,
            video_id=10,
            date=20220408,
            time_ms=1,
            target=1,
            safe_overrides={"long_view": train_target},
        )
    valid_poison = valid_aux_poison
    valid_rows = [
        _log_row(
            user_id=1,
            video_id=10,
            date=20220422,
            time_ms=5,
            target=1,
            outcome_poison=valid_poison,
            safe_overrides={"long_view": b"1"} if valid_poison is not None else None,
        ),
        _log_row(user_id=4, video_id=13, date=20220428, time_ms=6, target=0),
    ]
    final_rows = [
        _log_row(
            user_id=4,
            video_id=13,
            date=20220429,
            time_ms=7,
            target=0,
            outcome_poison=final_poison,
            safe_overrides=final_safe_overrides,
        ),
        _log_row(
            user_id=5,
            video_id=14,
            date=20220508,
            time_ms=8,
            target=1,
            outcome_poison=final_poison,
        ),
    ]
    (data / _physical(STANDARD_TRAIN_MEMBER)).write_bytes(
        _header(STANDARD_TRAIN_MEMBER) + b"".join(train_rows)
    )
    (data / _physical(STANDARD_LATE_MEMBER)).write_bytes(
        _header(STANDARD_LATE_MEMBER) + b"".join((*valid_rows, *final_rows))
    )
    for member in (RANDOMIZED_MEMBER, USER_SNAPSHOT_MEMBER, VIDEO_STATISTIC_MEMBER):
        (data / _physical(member)).write_bytes(_header(member) + _opaque_row(member))
    (data / _physical(VIDEO_BASIC_MEMBER)).write_bytes(
        _header(VIDEO_BASIC_MEMBER)
        + b"".join(
            (
                _basic_row(10, 100),
                _basic_row(11, 101),
                _basic_row(13, 103),
                _basic_row(14, 104),
            )
        )
    )
    return data


def _contract(data: Path, *, exact_splits: bool = False) -> AuditContract:
    identities: list[ExpectedSourceIdentity] = []
    for member in _MEMBER_ORDER:
        payload = (data / _physical(member)).read_bytes()
        identities.append(
            ExpectedSourceIdentity(
                member=member,
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                row_count=len(payload.splitlines()) - 1,
            )
        )
    expected_splits = (
        (
            ExpectedSplitIdentity(SplitName.TRAIN, 4, 20220408, 20220409),
            ExpectedSplitIdentity(SplitName.VALID, 2, 20220422, 20220428),
            ExpectedSplitIdentity(SplitName.TEST, 2, 20220429, 20220508),
        )
        if exact_splits
        else ()
    )
    return AuditContract(tuple(identities), expected_splits)


def _split(report_manifest: Mapping[str, object], name: str) -> Mapping[str, object]:
    splits = cast(list[Mapping[str, object]], report_manifest["splits"])
    return next(item for item in splits if item["split"] == name)


def test_audit_is_source_qualified_complete_and_deterministic(tmp_path: Path) -> None:
    data = _write_fixture(tmp_path)
    contract = _contract(data, exact_splits=True)

    first = audit_dataset(data.parent, contract=contract)
    second = audit_dataset(data, contract=contract)
    manifest = first.manifest()

    assert first.digest == second.digest
    assert first.semantic_digest == second.semantic_digest
    assert first.to_json(indent=None) == second.to_json(indent=None)
    assert manifest["source_root"] == "KuaiRand-Pure/data"
    assert [source.member for source in first.sources] == list(_MEMBER_ORDER)
    assert all(source.manifest()["identity_verified"] for source in first.sources)
    assert (
        cast(Mapping[str, object], manifest["archive_identity"])[
            "extracted_member_bytes_reverified"
        ]
        is True
    )

    train = _split(manifest, "train")
    valid = _split(manifest, "valid")
    final = _split(manifest, "test")
    assert train["row_count"] == 4
    assert train["user_cardinality"] == 3
    assert train["video_cardinality"] == 3
    assert train["author_cardinality"] == 3  # includes the explicit UNK category
    assert cast(Mapping[str, object], train["primary_target"])["positives"] == 2
    assert cast(Mapping[str, object], train["user_label_support"]) == {
        "all_negative": 1,
        "mixed": 1,
        "all_positive": 1,
    }
    repeated = cast(Mapping[str, object], train["repeated_user_video_pairs"])
    assert repeated == {
        "distinct_repeated_pairs": 1,
        "rows_in_repeated_pairs": 2,
        "duplicate_rows_beyond_first": 1,
        "maximum_multiplicity": 2,
    }
    assert cast(Mapping[str, object], valid["primary_target"])["positive_rate"] == 0.5
    assert final["primary_target"] is None
    assert final["target_access"] == "forbidden_during_development"
    assert final["user_label_support"] is None

    trace = cast(Mapping[str, object], manifest["final_outcome_trace"])
    assert trace["row_count"] == 2
    assert trace["skipped_cell_count"] == 22
    assert trace["outcome_cells_materialized"] == 0
    assert trace["outcome_cells_decoded"] == 0
    assert trace["outcome_cells_converted"] == 0
    assert trace["outcome_cells_validated"] == 0
    assert trace["outcome_cells_aggregated"] == 0
    assert trace["outcome_cells_logged"] == 0
    assert trace["outcome_cells_scored"] == 0
    assert trace["skipped_values_recorded"] is False

    associations = cast(Mapping[str, object], manifest["train_associations"])
    assert associations["scope"] == "training rows only"
    binary = cast(Mapping[str, object], associations["binary"])
    assert set(binary) == {
        f"{STANDARD_TRAIN_MEMBER}:{field_name}"
        for field_name in (
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "is_profile_enter",
        )
    }


def test_invalid_utf8_and_malformed_final_outcomes_are_byte_skipped(tmp_path: Path) -> None:
    left_data = _write_fixture(tmp_path / "left", final_poison=b"not-a-number")
    right_data = _write_fixture(tmp_path / "right", final_poison=b"\xff\xfePOISON")

    left = audit_dataset(left_data, contract=_contract(left_data))
    right = audit_dataset(right_data, contract=_contract(right_data))

    assert left.semantic_digest == right.semantic_digest
    assert left.digest != right.digest  # source integrity still detects changed bytes
    assert left.splits == right.splits
    assert left.final_outcome_trace.manifest() == right.final_outcome_trace.manifest()
    assert left.sources[3].sha256 != right.sources[3].sha256
    assert "POISON" not in right.to_json()
    assert "not-a-number" not in left.to_json()
    assert "POISON" not in right.readable_report()


def test_validation_auxiliary_outcomes_are_also_skipped_except_long_view(tmp_path: Path) -> None:
    data = _write_fixture(tmp_path, valid_aux_poison=b"\xff\xfeNOT-AN-AUX-TARGET")

    report = audit_dataset(data, contract=_contract(data))

    valid = _split(report.manifest(), "valid")
    assert cast(Mapping[str, object], valid["primary_target"])["positives"] == 1
    assert "NOT-AN-AUX-TARGET" not in report.to_json()


def test_permitted_train_outcome_is_validated(tmp_path: Path) -> None:
    data = _write_fixture(tmp_path, train_target=b"not-binary")

    with pytest.raises(DataAuditError, match="non-binary field long_view"):
        audit_dataset(data, contract=_contract(data))


def test_final_safe_fields_are_still_validated_without_touching_outcomes(tmp_path: Path) -> None:
    data = _write_fixture(
        tmp_path,
        final_poison=b"\xff\xfeOUTCOME",
        final_safe_overrides={"time_ms": b"not-a-timestamp"},
    )

    with pytest.raises(DataAuditError, match="invalid integer field time_ms") as caught:
        audit_dataset(data, contract=_contract(data))
    assert "OUTCOME" not in str(caught.value)


def test_header_member_set_and_hash_mismatches_fail_closed(tmp_path: Path) -> None:
    data = _write_fixture(tmp_path)
    contract = _contract(data)
    (data / "unexpected.csv").write_bytes(b"unexpected\n")
    with pytest.raises(DataAuditError, match="member set differs"):
        audit_dataset(data, contract=contract)

    (data / "unexpected.csv").unlink()
    late_path = data / _physical(STANDARD_LATE_MEMBER)
    late_path.write_bytes(late_path.read_bytes() + b"\n")
    with pytest.raises(DataAuditError, match="size mismatch"):
        audit_dataset(data, contract=contract)

    replacement_contract = _contract(data)
    train_path = data / _physical(STANDARD_TRAIN_MEMBER)
    payload = train_path.read_bytes()
    train_path.write_bytes(b"wrong,header\n" + payload.split(b"\n", 1)[1])
    with pytest.raises(DataAuditError, match="header differs"):
        audit_dataset(data, contract=_contract(data))
    assert replacement_contract.sources[0].member == STANDARD_TRAIN_MEMBER


def test_split_identity_is_checked_after_safe_streaming(tmp_path: Path) -> None:
    data = _write_fixture(tmp_path)
    base = _contract(data)
    wrong = AuditContract(
        sources=base.sources,
        expected_splits=(ExpectedSplitIdentity(SplitName.TEST, 3, 20220429, 20220508),),
    )

    with pytest.raises(DataAuditError, match="test split identity mismatch"):
        audit_dataset(data, contract=wrong)


def test_atomic_json_and_markdown_reports_contain_skip_evidence(tmp_path: Path) -> None:
    data = _write_fixture(tmp_path / "source")
    report = audit_dataset(data, contract=_contract(data))

    json_path, markdown_path = write_audit_report(
        report,
        json_path=tmp_path / "evidence" / "audit.json",
        markdown_path=tmp_path / "evidence" / "audit.md",
    )

    assert f'"digest": "{report.digest}"' in json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "byte-skipped" in markdown
    assert "Materialized outcome cells: 0" in markdown
    assert "| `data/log_standard_4_22_to_5_08_pure.csv` | `long_view` |" in markdown


def test_official_source_manifest_pins_all_six_extracted_members() -> None:
    assert [(item.member, item.size, item.row_count) for item in OFFICIAL_SOURCE_IDENTITIES] == [
        (STANDARD_TRAIN_MEMBER, 83_961_282, 1_141_112),
        (RANDOMIZED_MEMBER, 87_086_116, 1_186_059),
        (USER_SNAPSHOT_MEMBER, 3_519_028, 27_285),
        (STANDARD_LATE_MEMBER, 21_765_075, 295_497),
        (VIDEO_STATISTIC_MEMBER, 6_559_217, 7_583),
        (VIDEO_BASIC_MEMBER, 626_669, 7_583),
    ]
    assert [
        (item.split, item.row_count, item.first_date, item.last_date)
        for item in OFFICIAL_SPLIT_IDENTITIES
    ] == [
        (SplitName.TRAIN, 1_141_112, 20220409, 20220421),
        (SplitName.VALID, 124_909, 20220422, 20220428),
        (SplitName.TEST, 170_588, 20220429, 20220508),
    ]
    assert OFFICIAL_AUDIT_CONTRACT.expected_splits == OFFICIAL_SPLIT_IDENTITIES


@pytest.mark.skipif(
    "KUAIRAND_PURE_DATA_DIR" not in os.environ,
    reason="set KUAIRAND_PURE_DATA_DIR to run the verified full-data audit gate",
)
def test_verified_official_full_data_audit() -> None:
    report = audit_dataset(Path(os.environ["KUAIRAND_PURE_DATA_DIR"]))
    manifest = report.manifest()
    assert _split(manifest, "train")["first_date"] == 20220409
    assert _split(manifest, "train")["row_count"] == 1_141_112
    assert _split(manifest, "valid")["last_date"] == 20220428
    assert _split(manifest, "valid")["row_count"] == 124_909
    assert _split(manifest, "test")["last_date"] == 20220508
    assert _split(manifest, "test")["row_count"] == 170_588
    trace = cast(Mapping[str, object], manifest["final_outcome_trace"])
    assert trace["outcome_cells_materialized"] == 0
    assert trace["skipped_cell_count"] == 1_876_468

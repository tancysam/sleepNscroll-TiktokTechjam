from __future__ import annotations

from collections import Counter

import pytest

from kuairand_agent.data.fields import (
    CSV_HEADERS,
    EXPECTED_ROLE_COUNTS,
    FIELD_POLICY_DIGEST,
    FIELD_REGISTRY,
    RANDOMIZED_MEMBER,
    STANDARD_LATE_MEMBER,
    STANDARD_LOG_HEADER,
    STANDARD_TRAIN_MEMBER,
    USER_SNAPSHOT_HEADER,
    USER_SNAPSHOT_MEMBER,
    VIDEO_BASIC_HEADER,
    VIDEO_BASIC_MEMBER,
    VIDEO_STATISTIC_HEADER,
    VIDEO_STATISTIC_MEMBER,
    FieldKey,
    FieldPolicyError,
    FieldRole,
    field_policy_digest,
    field_policy_manifest,
    field_spec,
    fields_for_member,
    role_counts,
    validate_header,
)


def test_exact_six_archive_headers_and_orders_are_frozen() -> None:
    assert tuple(CSV_HEADERS) == (
        STANDARD_TRAIN_MEMBER,
        STANDARD_LATE_MEMBER,
        RANDOMIZED_MEMBER,
        USER_SNAPSHOT_MEMBER,
        VIDEO_STATISTIC_MEMBER,
        VIDEO_BASIC_MEMBER,
    )
    assert CSV_HEADERS[STANDARD_TRAIN_MEMBER] == STANDARD_LOG_HEADER
    assert CSV_HEADERS[STANDARD_LATE_MEMBER] == STANDARD_LOG_HEADER
    assert CSV_HEADERS[RANDOMIZED_MEMBER] == STANDARD_LOG_HEADER
    assert len(STANDARD_LOG_HEADER) == 19
    assert len(USER_SNAPSHOT_HEADER) == 31
    assert len(VIDEO_STATISTIC_HEADER) == 52
    assert VIDEO_BASIC_HEADER == (
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


def test_registry_is_exhaustive_with_exact_reviewed_role_counts() -> None:
    assert len(FIELD_REGISTRY) == 152
    assert sum(len(header) for header in CSV_HEADERS.values()) == 152
    assert dict(role_counts()) == dict(EXPECTED_ROLE_COUNTS)
    assert Counter(spec.role for spec in FIELD_REGISTRY.values()) == Counter(
        {
            FieldRole.INFERENCE_INPUT: 20,
            FieldRole.TRAINING_PRIMARY_TARGET: 2,
            FieldRole.TRAINING_AUXILIARY_TARGET: 20,
            FieldRole.STRICT_PAST_HISTORY_SOURCE: 2,
            FieldRole.TRUSTED_ALIGNMENT_ONLY: 1,
            FieldRole.BLOCKED: 107,
        }
    )
    assert all(spec.role in FieldRole for spec in FIELD_REGISTRY.values())


def test_standard_log_roles_and_secondary_history_metadata_are_exact() -> None:
    for member in (STANDARD_TRAIN_MEMBER, STANDARD_LATE_MEMBER):
        for name in ("user_id", "video_id", "date", "duration_ms", "tab"):
            spec = field_spec(FieldKey(member, name))
            assert spec.role is FieldRole.INFERENCE_INPUT
            assert spec.enabled
            assert not spec.history_eligible

        primary = field_spec(FieldKey(member, "long_view"))
        assert primary.role is FieldRole.TRAINING_PRIMARY_TARGET
        assert primary.enabled and primary.history_eligible

        time = field_spec(FieldKey(member, "time_ms"))
        assert time.role is FieldRole.STRICT_PAST_HISTORY_SOURCE
        assert time.enabled and time.history_eligible

        for name in ("hourmin", "is_rand"):
            spec = field_spec(FieldKey(member, name))
            assert spec.role is FieldRole.BLOCKED
            assert not spec.enabled

        auxiliary_names = {
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "play_time_ms",
            "profile_stay_time",
            "comment_stay_time",
            "is_profile_enter",
        }
        for name in auxiliary_names:
            spec = field_spec(FieldKey(member, name))
            assert spec.role is FieldRole.TRAINING_AUXILIARY_TARGET
            assert not spec.enabled
            assert spec.history_eligible


def test_source_qualification_overrides_same_column_name() -> None:
    standard = field_spec(FieldKey(STANDARD_TRAIN_MEMBER, "user_id"))
    randomized = field_spec(FieldKey(RANDOMIZED_MEMBER, "user_id"))
    snapshot = field_spec(FieldKey(USER_SNAPSHOT_MEMBER, "user_id"))
    assert standard.role is FieldRole.INFERENCE_INPUT
    assert standard.enabled
    assert randomized.role is FieldRole.BLOCKED and not randomized.enabled
    assert snapshot.role is FieldRole.BLOCKED and not snapshot.enabled


@pytest.mark.parametrize(
    "member",
    [RANDOMIZED_MEMBER, USER_SNAPSHOT_MEMBER, VIDEO_STATISTIC_MEMBER],
)
def test_quarantined_members_are_wholly_blocked(member: str) -> None:
    assert fields_for_member(member)
    assert all(spec.role is FieldRole.BLOCKED for _, spec in fields_for_member(member))
    assert all(not spec.enabled for _, spec in fields_for_member(member))


def test_basic_video_alignment_enabled_author_and_conditional_static_policy() -> None:
    video_id = field_spec(FieldKey(VIDEO_BASIC_MEMBER, "video_id"))
    author_id = field_spec(FieldKey(VIDEO_BASIC_MEMBER, "author_id"))
    visible = field_spec(FieldKey(VIDEO_BASIC_MEMBER, "visible_status"))
    assert video_id.role is FieldRole.TRUSTED_ALIGNMENT_ONLY and video_id.enabled
    assert author_id.role is FieldRole.INFERENCE_INPUT and author_id.enabled
    assert visible.role is FieldRole.BLOCKED and not visible.enabled

    conditional = set(VIDEO_BASIC_HEADER) - {"video_id", "author_id", "visible_status"}
    for name in conditional:
        spec = field_spec(FieldKey(VIDEO_BASIC_MEMBER, name))
        assert spec.role is FieldRole.INFERENCE_INPUT
        assert not spec.enabled
        assert spec.conditions


def test_unknown_or_nonexact_headers_fail_closed() -> None:
    assert validate_header(STANDARD_TRAIN_MEMBER, STANDARD_LOG_HEADER) == STANDARD_LOG_HEADER
    with pytest.raises(FieldPolicyError, match="unregistered archive member"):
        validate_header("data/not-real.csv", STANDARD_LOG_HEADER)
    with pytest.raises(FieldPolicyError, match="header mismatch"):
        validate_header(STANDARD_TRAIN_MEMBER, STANDARD_LOG_HEADER[:-1])
    with pytest.raises(FieldPolicyError, match="header mismatch"):
        validate_header(
            STANDARD_TRAIN_MEMBER,
            (STANDARD_LOG_HEADER[1], STANDARD_LOG_HEADER[0], *STANDARD_LOG_HEADER[2:]),
        )
    with pytest.raises(FieldPolicyError, match="duplicate header"):
        validate_header(STANDARD_TRAIN_MEMBER, (*STANDARD_LOG_HEADER[:-1], "user_id"))
    with pytest.raises(FieldPolicyError, match="unregistered field"):
        field_spec(FieldKey(STANDARD_TRAIN_MEMBER, "surprise"))


def test_policy_manifest_and_digest_are_deterministic_and_value_free() -> None:
    first = field_policy_manifest()
    second = field_policy_manifest()
    assert first == second
    assert field_policy_digest() == FIELD_POLICY_DIGEST
    assert len(FIELD_POLICY_DIGEST) == 64
    assert len(first["fields"]) == 152  # type: ignore[arg-type]
    assert "values" not in first


def test_registry_and_headers_are_read_only() -> None:
    with pytest.raises(TypeError):
        FIELD_REGISTRY[FieldKey(STANDARD_TRAIN_MEMBER, "user_id")] = field_spec(  # type: ignore[index]
            FieldKey(STANDARD_TRAIN_MEMBER, "user_id")
        )
    with pytest.raises(TypeError):
        CSV_HEADERS["data/extra.csv"] = ("field",)  # type: ignore[index]

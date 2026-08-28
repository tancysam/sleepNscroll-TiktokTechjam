"""Exhaustive, source-qualified field policy for the KuaiRand-Pure archive.

Column names are not authorities by themselves.  In particular, ``user_id`` from a
standard-policy log is an approved input while the identically named column from the
randomized log or user snapshot is blocked.  Every decision therefore uses :class:`FieldKey`
and the complete archive member path.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class FieldPolicyError(ValueError):
    """Raised when an archive schema or field request is not exactly registered."""


class FieldRole(StrEnum):
    """The one and only legal role assigned to an archive field instance."""

    INFERENCE_INPUT = "inference_input"
    TRAINING_PRIMARY_TARGET = "training_primary_target"
    TRAINING_AUXILIARY_TARGET = "training_auxiliary_target"
    STRICT_PAST_HISTORY_SOURCE = "strict_past_history_source"
    TRUSTED_ALIGNMENT_ONLY = "trusted_alignment_only"
    BLOCKED = "blocked"


class LogicalDType(StrEnum):
    """Stable logical types used by capability schema validation."""

    INTEGER = "integer"
    NUMBER = "number"
    BINARY = "binary"
    STRING = "string"


@dataclass(frozen=True, slots=True, order=True)
class FieldKey:
    """A field identity qualified by its exact archive member."""

    member: str
    column: str

    def __post_init__(self) -> None:
        if not self.member or not self.column or "\x00" in self.member or "\x00" in self.column:
            raise FieldPolicyError("field member and column must be non-empty and contain no NUL")

    @property
    def wire_name(self) -> str:
        """Return the unambiguous stable representation used in manifests."""

        return f"{self.member}:{self.column}"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Maximum legal route plus the current grant for one source-qualified field."""

    role: FieldRole
    logical_dtype: LogicalDType
    enabled: bool
    history_eligible: bool = False
    conditions: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.role is FieldRole.BLOCKED and self.enabled:
            raise FieldPolicyError("blocked fields cannot be enabled")
        if self.history_eligible and self.role not in {
            FieldRole.TRAINING_PRIMARY_TARGET,
            FieldRole.TRAINING_AUXILIARY_TARGET,
            FieldRole.STRICT_PAST_HISTORY_SOURCE,
        }:
            raise FieldPolicyError("history eligibility is limited to trusted temporal sources")

    @property
    def candidate_input_enabled(self) -> bool:
        """Whether this field may be copied into a candidate input capability."""

        return self.enabled and self.role is FieldRole.INFERENCE_INPUT


STANDARD_TRAIN_MEMBER: Final = "data/log_standard_4_08_to_4_21_pure.csv"
STANDARD_LATE_MEMBER: Final = "data/log_standard_4_22_to_5_08_pure.csv"
RANDOMIZED_MEMBER: Final = "data/log_random_4_22_to_5_08_pure.csv"
USER_SNAPSHOT_MEMBER: Final = "data/user_features_pure.csv"
VIDEO_STATISTIC_MEMBER: Final = "data/video_features_statistic_pure.csv"
VIDEO_BASIC_MEMBER: Final = "data/video_features_basic_pure.csv"

STANDARD_LOG_HEADER: Final = (
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

USER_SNAPSHOT_HEADER: Final = (
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "follow_user_num_range",
    "fans_user_num",
    "fans_user_num_range",
    "friend_user_num",
    "friend_user_num_range",
    "register_days",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
)

VIDEO_STATISTIC_HEADER: Final = (
    "video_id",
    "counts",
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
    "direct_comment_user_num",
    "reply_comment_user_num",
    "share_all_cnt",
    "share_all_user_num",
    "outsite_share_all_cnt",
)

VIDEO_BASIC_HEADER: Final = (
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

CSV_HEADERS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        STANDARD_TRAIN_MEMBER: STANDARD_LOG_HEADER,
        STANDARD_LATE_MEMBER: STANDARD_LOG_HEADER,
        RANDOMIZED_MEMBER: STANDARD_LOG_HEADER,
        USER_SNAPSHOT_MEMBER: USER_SNAPSHOT_HEADER,
        VIDEO_STATISTIC_MEMBER: VIDEO_STATISTIC_HEADER,
        VIDEO_BASIC_MEMBER: VIDEO_BASIC_HEADER,
    }
)

_BINARY_LOG_FIELDS: Final = frozenset(
    {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
    }
)
_NUMBER_LOG_FIELDS: Final = frozenset(
    {"play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"}
)
_AUXILIARY_FIELDS: Final = frozenset(
    {
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
)


def _log_dtype(column: str) -> LogicalDType:
    if column in {"user_id", "video_id", "tab"}:
        # Preserve organizer CSV tokens exactly.  These are categorical identifiers, not
        # quantities; numeric normalization can change identity (for example leading zeros).
        return LogicalDType.STRING
    if column in _BINARY_LOG_FIELDS:
        return LogicalDType.BINARY
    if column in _NUMBER_LOG_FIELDS:
        return LogicalDType.NUMBER
    return LogicalDType.INTEGER


def _standard_spec(column: str) -> FieldSpec:
    dtype = _log_dtype(column)
    if column in {"user_id", "video_id", "date", "duration_ms", "tab"}:
        return FieldSpec(
            FieldRole.INFERENCE_INPUT,
            dtype,
            enabled=True,
            rationale="reviewed pre-impression standard-log input",
        )
    if column == "long_view":
        return FieldSpec(
            FieldRole.TRAINING_PRIMARY_TARGET,
            dtype,
            enabled=True,
            history_eligible=True,
            rationale="native ranking label; same-row use is target-only",
        )
    if column in _AUXILIARY_FIELDS:
        return FieldSpec(
            FieldRole.TRAINING_AUXILIARY_TARGET,
            dtype,
            enabled=False,
            history_eligible=True,
            conditions=("explicit auxiliary loss, observability mask, and weight",),
            rationale="same-impression outcome withheld from the initial FM slice",
        )
    if column == "time_ms":
        return FieldSpec(
            FieldRole.STRICT_PAST_HISTORY_SOURCE,
            dtype,
            enabled=True,
            history_eligible=True,
            rationale="trusted chronological key; never a direct candidate feature",
        )
    if column == "hourmin":
        return FieldSpec(
            FieldRole.BLOCKED,
            dtype,
            enabled=False,
            rationale="published HHSS semantics conflict with the apparent hour/minute name",
        )
    if column == "is_rand":
        return FieldSpec(
            FieldRole.BLOCKED,
            dtype,
            enabled=False,
            rationale="exposure-policy marker is outside the standard-log benchmark inputs",
        )
    raise AssertionError(f"unclassified standard-log column: {column}")


def _blocked(dtype: LogicalDType, rationale: str) -> FieldSpec:
    return FieldSpec(FieldRole.BLOCKED, dtype, enabled=False, rationale=rationale)


def _basic_spec(column: str) -> FieldSpec:
    if column == "video_id":
        return FieldSpec(
            FieldRole.TRUSTED_ALIGNMENT_ONLY,
            LogicalDType.STRING,
            enabled=True,
            conditions=("unique many-to-one join without row expansion or reordering",),
            rationale="trusted side-table join key; duplicate candidate ID is not exposed",
        )
    if column == "author_id":
        return FieldSpec(
            FieldRole.INFERENCE_INPUT,
            LogicalDType.STRING,
            enabled=True,
            conditions=("unique many-to-one join without row expansion or reordering",),
            rationale="organizer FM baseline input with an explicit missing-value token",
        )
    if column == "visible_status":
        return _blocked(
            LogicalDType.INTEGER,
            "current mutable video status has no historical as-of timestamp",
        )

    dtype = {
        "music_id": LogicalDType.STRING,
        "video_type": LogicalDType.STRING,
        "upload_type": LogicalDType.STRING,
        "upload_dt": LogicalDType.STRING,
        "video_duration": LogicalDType.NUMBER,
        "server_width": LogicalDType.NUMBER,
        "server_height": LogicalDType.NUMBER,
        "music_type": LogicalDType.STRING,
        "tag": LogicalDType.STRING,
    }[column]
    conditions = ["explicit one-field-at-a-time branch", "lossless trusted join"]
    if column == "upload_dt":
        conditions.append("unambiguous timestamp no later than the impression")
    if column == "video_duration":
        conditions.append("units verified against duration_ms")
    if column in {"server_width", "server_height"}:
        conditions.append("nonnegative values and stable missing-value semantics")
    if column == "tag":
        conditions.append("deterministic bounded parsing")
    return FieldSpec(
        FieldRole.INFERENCE_INPUT,
        dtype,
        enabled=False,
        conditions=tuple(conditions),
        rationale="plausibly static metadata withheld until its field-specific gates pass",
    )


def _build_registry() -> dict[FieldKey, FieldSpec]:
    registry: dict[FieldKey, FieldSpec] = {}
    for member in (STANDARD_TRAIN_MEMBER, STANDARD_LATE_MEMBER):
        for column in STANDARD_LOG_HEADER:
            registry[FieldKey(member, column)] = _standard_spec(column)

    for column in STANDARD_LOG_HEADER:
        registry[FieldKey(RANDOMIZED_MEMBER, column)] = _blocked(
            _log_dtype(column),
            "randomized-exposure member is quarantined as a whole",
        )

    for column in USER_SNAPSHOT_HEADER:
        dtype = LogicalDType.STRING if column == "user_id" else LogicalDType.INTEGER
        registry[FieldKey(USER_SNAPSHOT_MEMBER, column)] = _blocked(
            dtype,
            "mutable user snapshot has no verified historical as-of timestamp",
        )

    for column in VIDEO_STATISTIC_HEADER:
        dtype = LogicalDType.STRING if column == "video_id" else LogicalDType.NUMBER
        registry[FieldKey(VIDEO_STATISTIC_MEMBER, column)] = _blocked(
            dtype,
            "period aggregate overlaps evaluation and has no per-row as-of cutoff",
        )

    for column in VIDEO_BASIC_HEADER:
        registry[FieldKey(VIDEO_BASIC_MEMBER, column)] = _basic_spec(column)
    return registry


_REGISTRY: Final = _build_registry()
FIELD_REGISTRY: Final[Mapping[FieldKey, FieldSpec]] = MappingProxyType(_REGISTRY)
EXPECTED_ROLE_COUNTS: Final[Mapping[FieldRole, int]] = MappingProxyType(
    {
        FieldRole.INFERENCE_INPUT: 20,
        FieldRole.TRAINING_PRIMARY_TARGET: 2,
        FieldRole.TRAINING_AUXILIARY_TARGET: 20,
        FieldRole.STRICT_PAST_HISTORY_SOURCE: 2,
        FieldRole.TRUSTED_ALIGNMENT_ONLY: 1,
        FieldRole.BLOCKED: 107,
    }
)


def validate_header(member: str, columns: Iterable[str]) -> tuple[str, ...]:
    """Require the exact registered ordered header for one archive member."""

    try:
        expected = CSV_HEADERS[member]
    except KeyError as exc:
        raise FieldPolicyError(f"unregistered archive member: {member!r}") from exc
    actual = tuple(columns)
    duplicates = sorted(name for name, count in Counter(actual).items() if count > 1)
    if duplicates:
        raise FieldPolicyError(f"duplicate header field(s) for {member}: {duplicates!r}")
    if actual != expected:
        raise FieldPolicyError(
            f"header mismatch for {member}: expected {expected!r}, got {actual!r}"
        )
    return actual


def field_spec(key: FieldKey) -> FieldSpec:
    """Resolve a registered field or fail closed."""

    try:
        return FIELD_REGISTRY[key]
    except KeyError as exc:
        raise FieldPolicyError(f"unregistered field: {key.wire_name}") from exc


def fields_for_member(member: str) -> tuple[tuple[FieldKey, FieldSpec], ...]:
    """Return one member's registry entries in exact CSV order."""

    try:
        header = CSV_HEADERS[member]
    except KeyError as exc:
        raise FieldPolicyError(f"unregistered archive member: {member!r}") from exc
    return tuple((key := FieldKey(member, column), FIELD_REGISTRY[key]) for column in header)


def role_counts() -> Mapping[FieldRole, int]:
    """Return a fresh immutable count of exhaustive registry roles."""

    counts = Counter(spec.role for spec in FIELD_REGISTRY.values())
    return MappingProxyType({role: counts[role] for role in FieldRole})


def field_policy_manifest() -> dict[str, object]:
    """Return the deterministic policy manifest without reading any data values."""

    entries: list[dict[str, object]] = []
    for member, header in CSV_HEADERS.items():
        for column in header:
            key = FieldKey(member, column)
            spec = FIELD_REGISTRY[key]
            entries.append(
                {
                    "member": member,
                    "column": column,
                    "role": spec.role.value,
                    "logical_dtype": spec.logical_dtype.value,
                    "enabled": spec.enabled,
                    "history_eligible": spec.history_eligible,
                    "conditions": list(spec.conditions),
                    "rationale": spec.rationale,
                }
            )
    return {
        "schema_version": 1,
        "headers": {member: list(header) for member, header in CSV_HEADERS.items()},
        "fields": entries,
        "role_counts": {role.value: count for role, count in role_counts().items()},
    }


def field_policy_digest() -> str:
    """Return the SHA-256 identity of headers, roles, grants, and conditions."""

    payload = json.dumps(
        field_policy_manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


FIELD_POLICY_DIGEST: Final = field_policy_digest()


def _validate_registry_invariants() -> None:
    expected_total = sum(len(header) for header in CSV_HEADERS.values())
    if expected_total != 152 or len(FIELD_REGISTRY) != expected_total:
        raise RuntimeError("field registry must exhaustively contain 152 source-qualified fields")
    if dict(role_counts()) != dict(EXPECTED_ROLE_COUNTS):
        raise RuntimeError("field registry role counts differ from the reviewed policy")


_validate_registry_invariants()

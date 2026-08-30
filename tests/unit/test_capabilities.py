from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pytest

import kuairand_agent.data.capabilities as capabilities
from kuairand_agent.data.capabilities import (
    CandidateCapabilityName,
    CapabilityError,
    DataPhase,
    build_candidate_inputs,
    build_train_targets,
    default_candidate_fields,
    validate_candidate_capability_request,
)
from kuairand_agent.data.fields import (
    RANDOMIZED_MEMBER,
    STANDARD_LATE_MEMBER,
    STANDARD_TRAIN_MEMBER,
    VIDEO_BASIC_MEMBER,
    FieldKey,
)


def _candidate_columns(
    phase: DataPhase,
    *,
    target_values: np.ndarray | None = None,
) -> dict[FieldKey, object]:
    log_member = (
        STANDARD_LATE_MEMBER
        if phase in {DataPhase.OUTER_VALID, DataPhase.FINAL}
        else STANDARD_TRAIN_MEMBER
    )
    columns: dict[FieldKey, object] = {
        FieldKey(log_member, "user_id"): ["0011", "12", "13"],
        FieldKey(log_member, "video_id"): ["0021", "22", "23"],
        FieldKey(VIDEO_BASIC_MEMBER, "author_id"): ["0031", "UNK", "33"],
        FieldKey(log_member, "tab"): ["0", "1", "0"],
        FieldKey(log_member, "duration_ms"): np.array([1000, 2500, 3000], dtype=np.float64),
        FieldKey(log_member, "date"): np.array([20220408, 20220409, 20220410]),
        FieldKey(log_member, "time_ms"): np.array([1, 2, 3]),
        FieldKey(log_member, "long_view"): (
            np.array([0, 1, 0], dtype=np.int8) if target_values is None else target_values
        ),
        FieldKey(log_member, "is_click"): np.array([1, 0, 1], dtype=np.int8),
    }
    return columns


def test_default_candidate_input_schema_is_exact_ordered_and_disjoint_from_targets() -> None:
    inputs = build_candidate_inputs(DataPhase.TRAIN, _candidate_columns(DataPhase.TRAIN))
    targets = build_train_targets(DataPhase.TRAIN, _candidate_columns(DataPhase.TRAIN))

    assert inputs.capability_name is CandidateCapabilityName.TRAIN_INPUTS
    assert tuple(inputs.columns) == ("user_id", "video_id", "author_id", "tab", "duration_ms")
    assert [column.logical_dtype.value for column in inputs.schema] == [
        "string",
        "string",
        "string",
        "string",
        "number",
    ]
    assert [column.storage_dtype for column in inputs.schema] == [
        "<U64",
        "<U64",
        "<U64",
        "<U64",
        "<f8",
    ]
    assert inputs.row_count == 3
    assert tuple(targets.columns) == ("long_view",)
    assert targets.primary.dtype == np.dtype("i1")
    assert set(inputs.columns).isdisjoint(targets.columns)


@pytest.mark.parametrize(
    ("phase", "expected_name"),
    [
        (DataPhase.TRAIN, CandidateCapabilityName.TRAIN_INPUTS),
        (DataPhase.INNER_TRAIN, CandidateCapabilityName.TRAIN_INPUTS),
        (DataPhase.INNER_VALID, CandidateCapabilityName.INNER_VALID_INPUTS),
        (DataPhase.OUTER_VALID, CandidateCapabilityName.OUTER_VALID_INPUTS),
        (DataPhase.FINAL, CandidateCapabilityName.FINAL_INPUTS),
    ],
)
def test_input_phase_rules_select_only_the_correct_standard_member(
    phase: DataPhase, expected_name: CandidateCapabilityName
) -> None:
    inputs = build_candidate_inputs(phase, _candidate_columns(phase))
    assert inputs.phase is phase
    assert inputs.capability_name is expected_name
    assert tuple(column.field_key for column in inputs.schema) == default_candidate_fields(phase)

    wrong_member = (
        STANDARD_TRAIN_MEMBER
        if phase in {DataPhase.OUTER_VALID, DataPhase.FINAL}
        else STANDARD_LATE_MEMBER
    )
    wrong = [FieldKey(wrong_member, "user_id")]
    wrong_columns = _candidate_columns(phase)
    wrong_columns[wrong[0]] = ["1", "2", "3"]
    with pytest.raises(CapabilityError, match="not legal"):
        build_candidate_inputs(phase, wrong_columns, requested_fields=wrong)


def test_arrays_are_deep_copied_bytes_backed_and_read_only() -> None:
    source = _candidate_columns(DataPhase.TRAIN)
    original = source[FieldKey(STANDARD_TRAIN_MEMBER, "duration_ms")]
    assert isinstance(original, np.ndarray)
    inputs = build_candidate_inputs(DataPhase.TRAIN, source)
    before = inputs.column("duration_ms").copy()
    original[0] = 999
    assert np.array_equal(inputs.column("duration_ms"), before)
    assert not inputs.column("duration_ms").flags.writeable
    with pytest.raises(ValueError):
        inputs.column("duration_ms")[0] = 999
    with pytest.raises(ValueError):
        inputs.column("duration_ms").setflags(write=True)
    with pytest.raises(TypeError):
        inputs.columns["extra"] = np.array([1])  # type: ignore[index]


@pytest.mark.parametrize("phase", [DataPhase.TRAIN, DataPhase.OUTER_VALID, DataPhase.FINAL])
def test_current_or_protected_outcome_sidecar_mutation_cannot_change_inputs(
    phase: DataPhase,
) -> None:
    target = np.array([0, 1, 0], dtype=np.int8)
    available = _candidate_columns(phase, target_values=target)
    before = build_candidate_inputs(phase, available)
    target[:] = 1 - target
    after = build_candidate_inputs(phase, available)
    assert before.digest == after.digest
    assert before.logical_content_digest == after.logical_content_digest
    assert before.manifest() == after.manifest()
    for name in before.columns:
        assert np.array_equal(before.column(name), after.column(name))


def test_manifest_contains_schema_and_content_identity_but_not_values() -> None:
    inputs = build_candidate_inputs(DataPhase.FINAL, _candidate_columns(DataPhase.FINAL))
    manifest = inputs.manifest()
    assert manifest["row_count"] == 3
    assert manifest["logical_content_digest"] == inputs.logical_content_digest
    assert manifest["columns"] == [column.manifest() for column in inputs.schema]
    assert "values" not in manifest
    assert "row_id" not in json.dumps(manifest, sort_keys=True)
    mutated = manifest["columns"]
    assert isinstance(mutated, list)
    mutated.clear()
    assert len(inputs.manifest()["columns"]) == 5  # type: ignore[arg-type]


def test_candidate_requests_for_raw_history_blocked_disabled_and_protected_data_reject() -> None:
    available = _candidate_columns(DataPhase.TRAIN)
    for key, message in (
        (FieldKey(STANDARD_TRAIN_MEMBER, "time_ms"), "strict_past_history_source"),
        (FieldKey(STANDARD_TRAIN_MEMBER, "long_view"), "training_primary_target"),
        (FieldKey(STANDARD_TRAIN_MEMBER, "hourmin"), "blocked"),
        (FieldKey(VIDEO_BASIC_MEMBER, "music_id"), "disabled"),
    ):
        available[key] = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(CapabilityError, match=message):
            build_candidate_inputs(DataPhase.TRAIN, available, requested_fields=[key])

    randomized = FieldKey(RANDOMIZED_MEMBER, "user_id")
    available[randomized] = ["1", "2", "3"]
    with pytest.raises(CapabilityError, match="not legal"):
        build_candidate_inputs(DataPhase.TRAIN, available, requested_fields=[randomized])

    for name in (
        "train_history_sources",
        "alignment",
        "inner_valid_targets",
        "outer_valid_targets",
        "raw_archive",
    ):
        with pytest.raises(CapabilityError, match="not allowed"):
            validate_candidate_capability_request(name)


def test_candidate_capability_allowlist_is_complete_and_has_no_final_target_factory() -> None:
    for capability_name in CandidateCapabilityName:
        assert validate_candidate_capability_request(capability_name.value) is capability_name
    forbidden_symbol = "final" + "_targets"
    assert not hasattr(capabilities, forbidden_symbol)
    assert forbidden_symbol not in {capability.value for capability in CandidateCapabilityName}


def test_training_targets_reject_nontraining_phases_and_disabled_auxiliary() -> None:
    available = _candidate_columns(DataPhase.TRAIN)
    for phase in (DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL):
        with pytest.raises(CapabilityError, match="only for train or inner_train"):
            build_train_targets(phase, available)

    # is_follow remains an ungranted auxiliary target, so requesting it must still fail closed.
    auxiliary = FieldKey(STANDARD_TRAIN_MEMBER, "is_follow")
    with pytest.raises(CapabilityError, match="disabled"):
        build_train_targets(
            DataPhase.TRAIN,
            available,
            requested_fields=[FieldKey(STANDARD_TRAIN_MEMBER, "long_view"), auxiliary],
        )


def test_training_target_values_are_copied_read_only_and_never_embedded_in_manifest() -> None:
    source = _candidate_columns(DataPhase.TRAIN)
    key = FieldKey(STANDARD_TRAIN_MEMBER, "long_view")
    raw = source[key]
    assert isinstance(raw, np.ndarray)
    targets = build_train_targets(DataPhase.TRAIN, source)
    raw[:] = 1
    assert targets.primary.tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        targets.primary[0] = 1
    assert "values" not in targets.manifest()


def test_unknown_schema_shape_length_and_dtype_fail_closed() -> None:
    available = _candidate_columns(DataPhase.TRAIN)
    unknown = FieldKey(STANDARD_TRAIN_MEMBER, "surprise")
    available[unknown] = np.array([1, 2, 3])
    with pytest.raises(CapabilityError, match="unregistered field"):
        build_candidate_inputs(DataPhase.TRAIN, available)

    available = _candidate_columns(DataPhase.TRAIN)
    available[FieldKey(STANDARD_TRAIN_MEMBER, "tab")] = np.array([0, 1])
    with pytest.raises(CapabilityError, match="length 2, expected 3"):
        build_candidate_inputs(DataPhase.TRAIN, available)

    available = _candidate_columns(DataPhase.TRAIN)
    available[FieldKey(STANDARD_TRAIN_MEMBER, "user_id")] = np.array([1.5, 2.5, 3.5])
    with pytest.raises(CapabilityError, match="strings or integers"):
        build_candidate_inputs(DataPhase.TRAIN, available)

    available = _candidate_columns(DataPhase.TRAIN)
    available[FieldKey(STANDARD_TRAIN_MEMBER, "video_id")] = np.array([[1], [2], [3]])
    with pytest.raises(CapabilityError, match="one-dimensional"):
        build_candidate_inputs(DataPhase.TRAIN, available)


def test_mapping_values_are_not_fetched_for_unrequested_outcome_sidecars() -> None:
    class GuardedMapping(Mapping[FieldKey, object]):
        def __init__(self, raw: dict[FieldKey, object], guarded: FieldKey) -> None:
            self._raw = raw
            self._guarded = guarded

        def __getitem__(self, key: FieldKey) -> object:
            if key == self._guarded:
                raise AssertionError("protected sidecar value was fetched")
            return self._raw[key]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._raw)

        def __len__(self) -> int:
            return len(self._raw)

    raw = _candidate_columns(DataPhase.FINAL)
    guarded_key = FieldKey(STANDARD_LATE_MEMBER, "long_view")
    guarded = GuardedMapping(raw, guarded_key)
    result = build_candidate_inputs(DataPhase.FINAL, guarded)
    assert result.row_count == 3

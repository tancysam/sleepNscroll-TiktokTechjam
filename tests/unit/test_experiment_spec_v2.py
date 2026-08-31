from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kuairand_agent.domain.experiment import (
    ExperimentSpec,
    ExperimentSpecError,
    FeatureViewId,
    MechanismMetadata,
    Objective,
    parse_experiment_spec,
)
from kuairand_agent.domain.identity import ExperimentId
from kuairand_agent.proposal.deterministic import deterministic_proposals

_DIGEST = "a" * 64


def test_prose_is_outside_semantic_experiment_identity() -> None:
    original = deterministic_proposals()[1]
    rewritten = original.with_metadata(
        MechanismMetadata(
            mechanism="Different explanation of the same bounded mechanism.",
            falsifiable_hypothesis="Reworded hypothesis without changing executable science.",
            expected_metric_effect="The same metrics are expected to move.",
            leakage_argument="The same strict capability policy remains in force.",
            rejection_criterion="The same frozen evidence rejects it.",
            attributions=("a different advisory citation",),
        )
    )

    assert original.canonical_bytes != rewritten.canonical_bytes
    assert original.semantic_digest == rewritten.semantic_digest
    assert original.family_id == rewritten.family_id
    assert original.derive_experiment_id(
        data_identities={"train": _DIGEST},
        fold_identities={"fold-b": _DIGEST},
        code_artifact_sha256=_DIGEST,
    ) == rewritten.derive_experiment_id(
        data_identities={"train": _DIGEST},
        fold_identities={"fold-b": _DIGEST},
        code_artifact_sha256=_DIGEST,
    )


def test_semantic_feature_or_objective_change_changes_identity() -> None:
    base = deterministic_proposals()[1]
    changed_feature = replace(
        base,
        feature_view_ids=(FeatureViewId.LEAK_SAFE_BASE, FeatureViewId.OFFICIAL_FM_SPARSE),
    )
    assert base.semantic_digest != changed_feature.semantic_digest

    pointwise = deterministic_proposals()[2]
    assert base.objective is Objective.LAMBDARANK_NDCG5
    assert base.semantic_digest != pointwise.semantic_digest

    base_id = base.derive_experiment_id(
        data_identities={"train": _DIGEST},
        fold_identities={"fold-b": _DIGEST},
        code_artifact_sha256=_DIGEST,
    )
    assert isinstance(base_id, ExperimentId)


def test_canonical_round_trip_is_schema_versioned_and_exact() -> None:
    spec = deterministic_proposals()[-1]
    restored = parse_experiment_spec(spec.canonical_bytes)
    assert restored == spec
    assert restored.semantic_digest == spec.semantic_digest

    noncanonical = b" " + spec.canonical_bytes
    with pytest.raises(ExperimentSpecError, match="not canonical"):
        parse_experiment_spec(noncanonical)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_code", "def train(): pass"),
        ("external_dataset_url", "https://example.invalid/interactions.csv"),
        ("hidden_test_target", "long_view"),
        ("protected_score", 0.99),
    ],
)
def test_free_form_source_external_hidden_and_protected_fields_are_rejected(
    field: str, value: object
) -> None:
    raw = json.loads(deterministic_proposals()[1].canonical_bytes)
    raw["semantic"][field] = value
    with pytest.raises(ExperimentSpecError, match=r"unknown ExperimentSpec\.semantic field"):
        ExperimentSpec.from_mapping(raw)


def test_unknown_target_and_feature_view_are_rejected_before_training() -> None:
    raw = json.loads(deterministic_proposals()[1].canonical_bytes)
    raw["semantic"]["training_target"] = "final.long_view"
    with pytest.raises(ExperimentSpecError, match="not allowlisted"):
        ExperimentSpec.from_mapping(raw)

    raw = json.loads(deterministic_proposals()[1].canonical_bytes)
    raw["semantic"]["feature_view_ids"] = ["external_embeddings_v1"]
    with pytest.raises(ExperimentSpecError, match="not allowlisted"):
        ExperimentSpec.from_mapping(raw)

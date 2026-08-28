from __future__ import annotations

import json

import pytest

from kuairand_agent.research.schemas import (
    GeneratedPackage,
    Proposal,
    ResearchOperation,
    SchemaValidationError,
    parse_json_object,
    response_json_schema,
)


def proposal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "proposal_id": "tab-bias-v1",
        "hypothesis": "A learned tab offset improves within-user ordering.",
        "mechanism": "Add a train-fitted tab bias to the candidate score.",
        "expected_metric_effects": ["GAUC", "nDCG@5"],
        "parent_candidate_id": "fm-seed",
        "principal_change": "One tab-bias scoring term.",
        "files_expected": ["candidate.py"],
        "required_fields": [
            {
                "source_field": "data/log_standard_4_08_to_4_21_pure.csv:tab",
                "role": "inference_input",
                "purpose": "Fit and apply a categorical tab bias.",
            }
        ],
        "objective": "binary cross entropy",
        "sampling": "all logged impressions",
        "grouping": "user impression groups",
        "weighting": "uniform rows",
        "causal_cutoff": "No response-derived feature is used.",
        "estimated_runtime_seconds": 30,
        "estimated_memory_mb": 256,
        "smoke_plan": "Fit two tabs and predict four rows.",
        "inner_fold_plan": "Screen seed 0 on Fold B.",
        "falsification_criteria": "Reject if Fold B primary does not improve.",
        "promotion_criteria": "Require positive mean across A and B.",
        "maximum_repairs": 1,
        "rollback_parent_id": "fm-seed",
        "attributions": ["local scripted vertical-slice fixture"],
    }


def test_proposal_strict_json_round_trip_has_stable_identity() -> None:
    raw = proposal_payload()
    proposal = Proposal.from_mapping(raw)

    assert proposal.to_wire() == raw
    assert Proposal.from_json(proposal.to_json()) == proposal
    assert proposal.digest == "021fecb57ed339dcb3c2431e610adaf8b864a5c4ac64e5cb20fe56cae3b86c24"


def test_proposal_rejects_unknown_fields_and_duplicate_json_keys() -> None:
    raw = proposal_payload()
    raw["unexpected"] = True
    with pytest.raises(SchemaValidationError, match="unknown proposal field"):
        Proposal.from_mapping(raw)

    duplicated = '{"schema_version":1,"schema_version":1}'
    with pytest.raises(SchemaValidationError, match="duplicate JSON key"):
        parse_json_object(duplicated)

    malformed_types = proposal_payload()
    malformed_types["files_expected"] = [1]
    with pytest.raises(SchemaValidationError, match="files_expected"):
        Proposal.from_mapping(malformed_types)


def test_proposal_json_rejects_non_object() -> None:
    with pytest.raises(SchemaValidationError, match="JSON object"):
        Proposal.from_json(json.dumps([proposal_payload()]))


@pytest.mark.parametrize(
    ("operation", "title"),
    [
        (ResearchOperation.PROPOSE, "Proposal"),
        (ResearchOperation.IMPLEMENT, "GeneratedPackage"),
        (ResearchOperation.REPAIR, "GeneratedPackage"),
        (ResearchOperation.REFLECT, "Reflection"),
    ],
)
def test_provider_response_json_schemas_are_strict_and_operation_specific(
    operation: ResearchOperation, title: str
) -> None:
    schema = response_json_schema(operation)

    assert schema["title"] == title
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    if operation in {ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}:
        files = schema["properties"]["files"]
        assert files["maxItems"] == 12
        assert files["items"]["additionalProperties"] is False


def test_generated_package_json_round_trip_rejects_schema_drift() -> None:
    payload = {
        "schema_version": 1,
        "request_id": "implement-1",
        "response_id": "response-1",
        "files": [{"path": "candidate.py", "content": "def score(x):\n    return x\n"}],
        "material_change_summary": "Change score.",
        "material_symbols": ["score"],
    }
    package = GeneratedPackage.from_mapping(payload)
    assert GeneratedPackage.from_json(package.to_json()) == package

    payload["patch"] = "forbidden"
    with pytest.raises(SchemaValidationError, match="unknown generated_package field"):
        GeneratedPackage.from_mapping(payload)

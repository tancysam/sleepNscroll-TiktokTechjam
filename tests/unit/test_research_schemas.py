from __future__ import annotations

import json

import pytest

from kuairand_agent.research.schemas import (
    FailureCategory,
    GeneratedFile,
    GeneratedPackage,
    ImplementationRequest,
    ParentSnapshot,
    ParentSourceFile,
    Proposal,
    ProposalRequest,
    RejectedPackageSnapshot,
    RepairRequest,
    ResearchOperation,
    SchemaValidationError,
    parse_json_object,
    response_json_schema,
)
from kuairand_agent.research.source_policy import DEFAULT_CANDIDATE_SOURCE_POLICY


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


@pytest.mark.parametrize("operation", tuple(ResearchOperation))
def test_provider_response_json_schemas_use_only_api_supported_uniqueness_constraints(
    operation: ResearchOperation,
) -> None:
    """Responses Structured Outputs rejects ``uniqueItems`` even for strict schemas.

    Duplicate rejection remains enforced by the typed local response parsers after generation.
    """

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert "uniqueItems" not in value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(response_json_schema(operation))


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


@pytest.mark.parametrize(
    "material_symbol",
    ["README.md", "candidate.py", "candidate.py.train_model", "train-model"],
)
def test_generated_package_rejects_non_python_material_symbol_names(
    material_symbol: str,
) -> None:
    payload = {
        "schema_version": 1,
        "request_id": "repair-1",
        "response_id": "response-1",
        "files": [{"path": "candidate.py", "content": "def train_model():\n    pass\n"}],
        "material_change_summary": "Change the training implementation.",
        "material_symbols": [material_symbol],
    }

    with pytest.raises(SchemaValidationError, match="material_symbols"):
        GeneratedPackage.from_mapping(payload)


def _parent_snapshot() -> ParentSnapshot:
    return ParentSnapshot(
        candidate_id="fm-seed",
        files=(
            ParentSourceFile.create(
                "candidate.py", "def score(value: float) -> float:\n    return value\n"
            ),
        ),
    )


def test_agent_requests_round_trip_one_exact_candidate_source_policy() -> None:
    proposal = Proposal.from_mapping(proposal_payload())
    parent = _parent_snapshot()
    proposal_request = ProposalRequest.create(
        request_id="propose-1",
        campaign_id="campaign-1",
        scientific_iteration=1,
        parent_candidate_id=parent.candidate_id,
        safe_context={"evidence_cursor": "initial"},
    )
    implementation_request = ImplementationRequest.create(
        request_id="implement-1",
        proposal=proposal,
        parent=parent,
        safe_context={"evidence_cursor": "initial"},
    )

    for request, parser in (
        (proposal_request, ProposalRequest.from_mapping),
        (implementation_request, ImplementationRequest.from_mapping),
    ):
        wire = request.to_wire()
        assert wire["source_policy"] == DEFAULT_CANDIDATE_SOURCE_POLICY.to_wire()
        assert wire["source_policy_digest"] == DEFAULT_CANDIDATE_SOURCE_POLICY.digest
        assert parser(wire) == request

        tampered = dict(wire)
        tampered["source_policy_digest"] = "0" * 64
        with pytest.raises(SchemaValidationError, match="source policy digest mismatch"):
            parser(tampered)


def test_rejected_package_snapshot_preserves_forbidden_source_inertly_and_detects_tampering() -> (
    None
):
    package = GeneratedPackage(
        request_id="implement-1",
        response_id="response-with-forbidden-source",
        files=(
            GeneratedFile(
                "baseline.py",
                "def proposed_score(value: float) -> float:\n    return value + 0.1\n",
            ),
        ),
        material_change_summary="Implement a scientifically material score adjustment.",
        material_symbols=("proposed_score",),
    )

    snapshot = RejectedPackageSnapshot.from_generated_package(package)
    assert snapshot.package_digest == package.digest
    assert snapshot.files[0].path == "baseline.py"
    assert RejectedPackageSnapshot.from_mapping(snapshot.to_wire()) == snapshot

    tampered = snapshot.to_wire()
    tampered["files"][0]["content"] += "# changed\n"  # type: ignore[index]
    with pytest.raises(SchemaValidationError, match="digest mismatch"):
        RejectedPackageSnapshot.from_mapping(tampered)


def test_repair_request_round_trip_keeps_trusted_parent_and_rejected_package_separate() -> None:
    package = GeneratedPackage(
        request_id="implement-1",
        response_id="response-with-forbidden-source",
        files=(GeneratedFile("baseline.py", "def score(value):\n    return value + 1\n"),),
        material_change_summary="Implement a new score.",
        material_symbols=("score",),
    )
    rejected = RejectedPackageSnapshot.from_generated_package(package)
    request = RepairRequest.create(
        request_id="repair-1",
        proposal_id="tab-bias-v1",
        failed_candidate_id="candidate-rejected-1",
        failed_child=_parent_snapshot(),
        rejected_package=rejected,
        failure_category=FailureCategory.STATIC_POLICY,
        diagnostics="reserved candidate filename is forbidden: 'baseline.py'",
        remaining_repairs=1,
        safe_context={"evidence_cursor": "initial"},
    )

    wire = request.to_wire()
    assert wire["failed_child"]["candidate_id"] == "fm-seed"  # type: ignore[index]
    assert wire["rejected_package"]["files"][0]["path"] == "baseline.py"  # type: ignore[index]
    assert wire["source_policy_digest"] == DEFAULT_CANDIDATE_SOURCE_POLICY.digest
    assert RepairRequest.from_mapping(wire) == request

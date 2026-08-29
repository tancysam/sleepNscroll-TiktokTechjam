from __future__ import annotations

from typing import cast

import pytest

from kuairand_agent.data.fields import field_policy_digest, field_policy_manifest
from kuairand_agent.research.context import (
    AggregateRecord,
    MetricSummary,
    ResearchBudgetContext,
    SafeContextError,
    build_safe_research_context,
    redact_text,
)


def input_capability_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "capability_name": "inner_valid_inputs",
        "phase": "inner_valid",
        "row_count": 4,
        "columns": [
            {
                "name": "tab",
                "source_field": "data/log_standard_4_08_to_4_21_pure.csv:tab",
                "logical_dtype": "string",
                "storage_dtype": "<U64",
            }
        ],
        "field_policy_digest": field_policy_digest(),
        "capability_schema_digest": "b" * 64,
        "logical_content_digest": "c" * 64,
        "capability_digest": "d" * 64,
    }


def test_safe_context_contains_only_metadata_and_aggregate_results() -> None:
    context = build_safe_research_context(
        starter_manifest_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        capability_manifests=(input_capability_manifest(),),
        train_eda=(AggregateRecord("train_shape", {"row_count": 100, "tab_cardinality": 3}),),
        validation_input_eda=(AggregateRecord("valid_shape", {"row_count": 4}),),
        method_cards=(AggregateRecord("tab_bias", {"source": "scripted fixture"}),),
        inner_metrics=(MetricSummary("fold_b", 0.61, 0.52, 0.565, exact=True),),
        outer_metrics=(MetricSummary("public_valid", 0.612345, 0.523456, 0.5679005),),
        campaign_records=(AggregateRecord("incumbent", {"candidate_id": "fm-seed"}),),
        budgets=ResearchBudgetContext(
            remaining_attempts=44,
            remaining_wall_seconds=18_000,
            remaining_outer_promotions=6,
            intervention_count=0,
        ),
    )
    wire = context.to_wire()
    outer_metrics = cast(list[object], wire["outer_metrics"])
    inner_metrics = cast(list[dict[str, object]], wire["inner_metrics"])
    benchmark = cast(dict[str, object], wire["benchmark"])
    task = cast(dict[str, object], benchmark["task"])
    field_policy = cast(dict[str, object], wire["field_policy"])
    assert outer_metrics == [
        {
            "name": "public_valid",
            "GAUC": 0.6123,
            "nDCG@5": 0.5235,
            "primary": 0.5679,
            "precision": "rounded_4dp",
        }
    ]
    assert inner_metrics[0]["precision"] == "exact"
    assert task["target"] == "long_view"
    enabled_fields = cast(list[dict[str, object]], field_policy["fields"])
    disabled_by_member = cast(dict[str, list[str]], field_policy["disabled_columns_by_member"])
    assert enabled_fields
    assert all(field["enabled"] is True for field in enabled_fields)
    represented_field_count = sum(len(columns) for columns in disabled_by_member.values()) + len(
        enabled_fields
    )
    assert represented_field_count == len(cast(list[object], field_policy_manifest()["fields"]))
    assert "every unspecified field is forbidden" in cast(str, field_policy["policy_semantics"])
    assert (
        context.digest
        == build_safe_research_context(
            starter_manifest_sha256="1" * 64,
            dataset_manifest_sha256="2" * 64,
            capability_manifests=(input_capability_manifest(),),
            train_eda=(AggregateRecord("train_shape", {"row_count": 100, "tab_cardinality": 3}),),
            validation_input_eda=(AggregateRecord("valid_shape", {"row_count": 4}),),
            method_cards=(AggregateRecord("tab_bias", {"source": "scripted fixture"}),),
            inner_metrics=(MetricSummary("fold_b", 0.61, 0.52, 0.565, exact=True),),
            outer_metrics=(MetricSummary("public_valid", 0.612345, 0.523456, 0.5679005),),
            campaign_records=(AggregateRecord("incumbent", {"candidate_id": "fm-seed"}),),
            budgets=ResearchBudgetContext(44, 18_000, 6, 0),
        ).digest
    )


def test_metric_summary_accepts_exact_organizer_float32_primary_semantics() -> None:
    summary = MetricSummary(
        "official_fm_fold_B",
        0.6583324670791626,
        0.49251559376716614,
        0.5754240155220032,
        exact=True,
    )

    assert summary.to_wire(outer=False) == {
        "name": "official_fm_fold_B",
        "GAUC": 0.6583324670791626,
        "nDCG@5": 0.49251559376716614,
        "primary": 0.5754240155220032,
        "precision": "exact",
    }


def test_metric_summary_still_rejects_non_organizer_primary() -> None:
    with pytest.raises(SafeContextError, match="mean of GAUC"):
        MetricSummary("invalid", 0.6, 0.5, 0.550001, exact=True)


def test_safe_context_rejects_target_capabilities_and_row_level_data() -> None:
    target_manifest = input_capability_manifest()
    target_manifest["capability_name"] = "train_targets"
    with pytest.raises(SafeContextError, match="target capability"):
        build_safe_research_context(
            starter_manifest_sha256="1" * 64,
            dataset_manifest_sha256="2" * 64,
            capability_manifests=(target_manifest,),
            budgets=ResearchBudgetContext(1, 60, 1, 0),
        )

    with pytest.raises(SafeContextError, match="row-level or protected"):
        AggregateRecord("unsafe", {"public_validation_labels": 1})
    with pytest.raises(SafeContextError, match="aggregate scalar"):
        AggregateRecord("unsafe", {"values": [0, 1]})  # type: ignore[dict-item]

    # Scalar diversity diagnostics are expressly safe even though their names mention
    # predictions; only row-level prediction vectors are prohibited.
    diversity = AggregateRecord("diversity", {"prediction_rank_correlation": 0.42})
    assert diversity.to_wire()["values"] == {"prediction_rank_correlation": 0.42}


def test_safe_context_rejects_a_protected_field_disguised_as_an_input() -> None:
    forged = input_capability_manifest()
    forged["columns"] = [
        {
            "name": "long_view",
            "source_field": "data/log_standard_4_08_to_4_21_pure.csv:long_view",
            "logical_dtype": "binary",
            "storage_dtype": "i1",
        }
    ]

    with pytest.raises(SafeContextError, match="not an enabled inference input"):
        build_safe_research_context(
            starter_manifest_sha256="1" * 64,
            dataset_manifest_sha256="2" * 64,
            capability_manifests=(forged,),
            budgets=ResearchBudgetContext(1, 60, 1, 0),
        )


def test_secret_redaction_covers_explicit_and_structural_credentials() -> None:
    secret = "sk-test-do-not-store"
    raw = f"Authorization: Bearer {secret}\nOPENAI_API_KEY={secret}\nvalue={secret}"
    redacted = redact_text(raw, secrets=(secret,))

    assert secret not in redacted
    assert redacted.count("[REDACTED]") >= 3

    context = build_safe_research_context(
        starter_manifest_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        capability_manifests=(),
        method_cards=(AggregateRecord("provider_note", {"note": f"token={secret}"}),),
        budgets=ResearchBudgetContext(1, 60, 1, 0),
        secrets=(secret,),
    )
    assert secret not in str(context.to_wire())

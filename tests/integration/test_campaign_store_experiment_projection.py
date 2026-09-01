from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kuairand_agent.campaign.store import (
    CampaignStore,
    ExperimentRecord,
    ProposalRecord,
    SourceSnapshotRecord,
    StoreVersionError,
)


def _create_campaign(path: Path) -> CampaignStore:
    return CampaignStore.create(
        path,
        campaign_id="projection-campaign",
        config_digest="1" * 64,
        benchmark_digest="2" * 64,
        starter_digest="3" * 64,
        dataset_digest="4" * 64,
        environment_digest="5" * 64,
        source_digest="6" * 64,
        hard_deadline_utc="2030-01-01T00:00:00Z",
        initial_convergence={
            "schema_version": 2,
            "best_primary": 0.6016,
            "non_material_streak": 0,
            "unmeasured_streak": 0,
            "completed_iterations": 0,
            "required_completion_pending": False,
        },
    )


def test_experiment_projection_is_exact_immutable_and_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path) as store:
        assert store.experiment("missing-experiment") is None
        store.create_experiment(
            experiment_id="generated-tree-001",
            iteration_number=1,
            parent_experiment_id=None,
            hypothesis="Causal pure features improve long-view ranking.",
            mechanism="Grouped LambdaRank aligns training with nDCG@5.",
            method_attribution="scripted generated-source research loop",
            status="PLANNED",
            metadata={"lineage": {"symbols": ["train_model", "predict_scores"]}},
            expected_revision=0,
        )
        expected = store.experiment("generated-tree-001")

        assert expected is not None
        assert isinstance(expected, ExperimentRecord)
        assert expected.experiment_id == "generated-tree-001"
        assert expected.iteration_number == 1
        assert expected.parent_experiment_id is None
        assert expected.hypothesis == "Causal pure features improve long-view ranking."
        assert expected.mechanism == "Grouped LambdaRank aligns training with nDCG@5."
        assert expected.method_attribution == "scripted generated-source research loop"
        assert expected.status == "PLANNED"
        assert expected.metadata["lineage"]["symbols"] == ("train_model", "predict_scores")  # type: ignore[index]
        assert expected.created_at.endswith("Z")

        with pytest.raises(TypeError):
            expected.metadata["new"] = True  # type: ignore[index]
        with pytest.raises(TypeError):
            expected.metadata["lineage"]["symbols"] = ()  # type: ignore[index]

    with CampaignStore.open(path, read_only=True) as reopened:
        assert reopened.experiment("generated-tree-001") == expected


def test_experiment_projection_rejects_noncanonical_stored_identity(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path) as store:
        store.create_experiment(
            experiment_id="generated-tree-001",
            iteration_number=1,
            hypothesis="A hypothesis",
            mechanism="A mechanism",
            method_attribution="generated source",
            metadata={"a": 1, "z": 2},
            expected_revision=0,
        )

    with sqlite3.connect(path) as raw:
        raw.execute(
            "UPDATE experiments SET metadata_json = ? WHERE experiment_id = ?",
            ('{"z":2, "a":1}', "generated-tree-001"),
        )

    with (
        CampaignStore.open(path, read_only=True) as reopened,
        pytest.raises(StoreVersionError, match="canonical JSON"),
    ):
        reopened.experiment("generated-tree-001")


def test_experiment_projection_rejects_empty_stored_scientific_text(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path) as store:
        store.create_experiment(
            experiment_id="generated-tree-001",
            iteration_number=1,
            hypothesis="A hypothesis",
            mechanism="A mechanism",
            method_attribution="generated source",
            expected_revision=0,
        )

    # SQLite's immutable create path rejects this value, but a projection must also fail closed
    # when storage is externally damaged before a resume.
    with sqlite3.connect(path) as raw:
        raw.execute(
            "UPDATE experiments SET hypothesis = '' WHERE experiment_id = ?",
            ("generated-tree-001",),
        )

    with (
        CampaignStore.open(path, read_only=True) as reopened,
        pytest.raises(StoreVersionError, match="hypothesis must be a non-empty string"),
    ):
        reopened.experiment("generated-tree-001")


def test_generated_lineage_projections_are_exact_and_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite"
    with _create_campaign(path) as store:
        store.create_experiment(
            experiment_id="generated-tree-001",
            iteration_number=1,
            hypothesis="A hypothesis",
            mechanism="A mechanism",
            method_attribution="generated source",
            expected_revision=0,
        )
        store.record_proposal(
            proposal_id="proposal-001",
            experiment_id="generated-tree-001",
            request_digest="7" * 64,
            response_digest="8" * 64,
            provider="scripted",
            metadata={"calls": ["propose"]},
            expected_revision=1,
        )
        store.record_source_snapshot(
            snapshot_id="source-001",
            experiment_id="generated-tree-001",
            source_digest="9" * 64,
            parent_source_digest="a" * 64,
            diff_digest="b" * 64,
            metadata={"changed_paths": ["candidate.py"]},
            expected_revision=2,
        )

        assert store.proposal("missing-proposal") is None
        proposal = store.proposal("proposal-001")
        assert proposal is not None
        assert isinstance(proposal, ProposalRecord)
        assert proposal.proposal_id == "proposal-001"
        assert proposal.experiment_id == "generated-tree-001"
        assert proposal.request_digest == "7" * 64
        assert proposal.response_digest == "8" * 64
        assert proposal.provider == "scripted"
        assert proposal.metadata["calls"] == ("propose",)
        assert proposal.created_at.endswith("Z")

        assert store.source_snapshot("missing-source") is None
        source = store.source_snapshot("source-001")
        assert source is not None
        assert isinstance(source, SourceSnapshotRecord)
        assert source.snapshot_id == "source-001"
        assert source.experiment_id == "generated-tree-001"
        assert source.source_digest == "9" * 64
        assert source.parent_source_digest == "a" * 64
        assert source.diff_digest == "b" * 64
        assert source.metadata["changed_paths"] == ("candidate.py",)
        assert source.created_at.endswith("Z")

    with CampaignStore.open(path, read_only=True) as reopened:
        assert reopened.proposal("proposal-001") == proposal
        assert reopened.source_snapshot("source-001") == source

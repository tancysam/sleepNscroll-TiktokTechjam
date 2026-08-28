from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import kuairand_agent.campaign.full_campaign as full_campaign
import kuairand_agent.campaign.full_campaign_runtime as full_campaign_runtime
from kuairand_agent.campaign.full_campaign import (
    FinalizationSelectionPlan,
    FullCampaignError,
    FullCampaignOutcome,
    FullCampaignOutcomeRepository,
    FullCampaignProgressLedger,
    FullCampaignStage,
    InnerFoldSelectionEvidence,
    MatchedSeedSelectionEvidence,
    QualifiedFMMemberPlan,
    build_production_feature_bundle,
    encode_numeric_user_groups,
    prepare_campaign_data_plane,
    select_fold_b_fusion,
)
from kuairand_agent.campaign.selector import OrganizerMetrics
from kuairand_agent.candidates.fusion import FUSION_WEIGHT_GRID
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import (
    APPROVED_AUXILIARY_TARGETS,
    OUTCOME_FIELDS,
    CanonicalAlignment,
    CanonicalDataset,
    CanonicalInputs,
    CanonicalSplit,
    OutcomeAccessTrace,
    ProtectedTargets,
    TrainingTargets,
)
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.scoring.protected import ScoreResult
from kuairand_agent.scoring.submission import prediction_digest


def _inputs(dates: Sequence[int]) -> CanonicalInputs:
    count = len(dates)
    return CanonicalInputs(
        user_id=tuple(f"u-{index // 2}" for index in range(count)),
        video_id=tuple(f"v-{index}" for index in range(count)),
        date=tuple(dates),
        duration_ms=tuple(float(1_000 + 100 * index) for index in range(count)),
        tab=tuple("1" for _ in range(count)),
        author_id=tuple(f"a-{index % 3}" for index in range(count)),
        time_ms=tuple(10_000 + index for index in range(count)),
    )


def _alignment(name: SplitName, inputs: CanonicalInputs) -> CanonicalAlignment:
    return CanonicalAlignment(
        split=name,
        row_id=tuple(range(len(inputs))),
        user_id=inputs.user_id,
        video_id=inputs.video_id,
    )


def _dataset() -> CanonicalDataset:
    train_dates = tuple(date for date in range(20220408, 20220422) for _ in range(2))
    train_inputs = _inputs(train_dates)
    train_labels = tuple(index % 2 for index in range(len(train_inputs)))
    train_targets = TrainingTargets(
        {
            "long_view": train_labels,
            **{name: tuple(0 for _ in train_labels) for name in APPROVED_AUXILIARY_TARGETS},
        }
    )
    train = CanonicalSplit(
        name=SplitName.TRAIN,
        inputs=train_inputs,
        alignment=_alignment(SplitName.TRAIN, train_inputs),
        targets=train_targets,
        outcome_trace=OutcomeAccessTrace(
            SplitName.TRAIN,
            len(train_inputs),
            OUTCOME_FIELDS,
            (),
        ),
    )

    valid_inputs = _inputs((20220422, 20220422, 20220422, 20220422))
    valid = CanonicalSplit(
        name=SplitName.VALID,
        inputs=valid_inputs,
        alignment=_alignment(SplitName.VALID, valid_inputs),
        targets=ProtectedTargets((0, 1, 0, 1)),
        outcome_trace=OutcomeAccessTrace(
            SplitName.VALID,
            len(valid_inputs),
            ("long_view",),
            APPROVED_AUXILIARY_TARGETS,
        ),
    )

    final_inputs = _inputs((20220423, 20220423, 20220423))
    final = CanonicalSplit(
        name=SplitName.TEST,
        inputs=final_inputs,
        alignment=_alignment(SplitName.TEST, final_inputs),
        targets=None,
        outcome_trace=OutcomeAccessTrace(
            SplitName.TEST,
            len(final_inputs),
            (),
            OUTCOME_FIELDS,
        ),
    )
    return CanonicalDataset(train, valid, final, hashlib.sha256(b"authors").hexdigest())


def test_data_plane_and_feature_bundle_never_create_a_final_target_capability() -> None:
    dataset = _dataset()
    prepared = prepare_campaign_data_plane(dataset, expected_dataset_digest=dataset.digest)
    features = build_production_feature_bundle(
        prepared,
        builder_source_digest=hashlib.sha256(b"builder").hexdigest(),
    )

    assert tuple(fold.name for fold in prepared.folds.folds) == ("A", "B")
    assert prepared.final_inputs is dataset.final.inputs
    assert dataset.final.targets is None
    assert dataset.final.outcome_trace.parsed_cell_count == 0
    assert features.fold_a.prefix.row_count == len(prepared.fold_a.prefix_inputs)
    assert features.fold_b.query.row_count == len(prepared.fold_b.query_inputs)
    assert features.outer_and_final.query.row_count == (
        dataset.valid.row_count + dataset.final.row_count
    )
    assert features.outer_validation.row_count == dataset.valid.row_count
    assert features.final.row_count == dataset.final.row_count
    assert features.outer_validation.feature_names == features.final.feature_names


def test_research_context_evidence_contains_train_eda_and_input_only_late_periods() -> None:
    dataset = _dataset()
    prepared = prepare_campaign_data_plane(dataset, expected_dataset_digest=dataset.digest)
    features = build_production_feature_bundle(
        prepared,
        builder_source_digest=hashlib.sha256(b"context-builder").hexdigest(),
    )

    evidence = full_campaign_runtime._research_context_evidence(prepared, features)
    capability_phases = {
        (item["capability_name"], item["phase"]) for item in evidence.capability_manifests
    }
    assert capability_phases == {
        ("train_inputs", "train"),
        ("inner_valid_inputs", "inner_valid"),
        ("outer_valid_inputs", "outer_valid"),
        ("final_inputs", "final"),
    }
    assert evidence.train_eda[1].values["long_view_positive_rate"] == 0.5
    assert {item.name for item in evidence.validation_input_eda} == {
        "public_validation_inputs_only",
        "prediction_period_inputs_only",
    }
    rendered = repr(evidence.validation_input_eda).casefold()
    assert "long_view" not in rendered
    assert "label" not in rendered
    assert "outcome" not in rendered
    assert evidence.method_cards[0].values["uses_public_labels_for_features"] is False


def test_research_and_finalization_share_candidate_capability_identity_not_raw_input_identity() -> (
    None
):
    dataset = _dataset()
    prepared = prepare_campaign_data_plane(dataset, expected_dataset_digest=dataset.digest)

    research_validation = full_campaign.build_finalization_candidate_inputs(
        DataPhase.OUTER_VALID,
        prepared.outer_validation_inputs,
    )
    research_final = full_campaign.build_finalization_candidate_inputs(
        DataPhase.FINAL,
        prepared.final_inputs,
    )
    finalization_validation = full_campaign.build_finalization_candidate_inputs(
        DataPhase.OUTER_VALID,
        dataset.valid.inputs,
    )
    finalization_final = full_campaign.build_finalization_candidate_inputs(
        DataPhase.FINAL,
        dataset.final.inputs,
    )

    assert prepared.outer_validation_inputs.digest != research_validation.digest
    assert prepared.final_inputs.digest != research_final.digest
    assert research_validation.digest == finalization_validation.digest
    assert research_final.digest == finalization_final.digest

    original = dataset.final.inputs
    tampered = CanonicalInputs(
        user_id=original.user_id,
        video_id=original.video_id,
        date=original.date,
        duration_ms=(original.duration_ms[0] + 1.0, *original.duration_ms[1:]),
        tab=original.tab,
        author_id=original.author_id,
        time_ms=original.time_ms,
    )
    assert (
        full_campaign.build_finalization_candidate_inputs(DataPhase.FINAL, tampered).digest
        != research_final.digest
    )


def test_feature_bundle_uses_a_verified_content_addressed_run_cache(tmp_path: Path) -> None:
    dataset = _dataset()
    prepared = prepare_campaign_data_plane(dataset, expected_dataset_digest=dataset.digest)
    cache_dir = tmp_path / "run" / "production" / "feature-cache"

    cold = build_production_feature_bundle(
        prepared,
        builder_source_digest=hashlib.sha256(b"builder").hexdigest(),
        cache_dir=cache_dir,
    )
    retained_files = tuple(sorted(path.name for path in cache_dir.iterdir()))
    warm = build_production_feature_bundle(
        prepared,
        builder_source_digest=hashlib.sha256(b"builder").hexdigest(),
        cache_dir=cache_dir,
    )

    assert cold.digest == warm.digest
    np.testing.assert_array_equal(cold.fold_a.prefix.values, warm.fold_a.prefix.values)
    np.testing.assert_array_equal(cold.fold_b.query.values, warm.fold_b.query.values)
    np.testing.assert_array_equal(cold.final.values, warm.final.values)
    assert retained_files
    assert retained_files == tuple(sorted(path.name for path in cache_dir.iterdir()))


def test_data_plane_fails_closed_before_fold_work_on_identity_mismatch() -> None:
    dataset = _dataset()
    with pytest.raises(FullCampaignError, match="dataset identity"):
        prepare_campaign_data_plane(dataset, expected_dataset_digest="0" * 64)


def test_numeric_user_groups_are_first_seen_stable_and_nonidentifying() -> None:
    encoded = encode_numeric_user_groups(("u9", "u2", "u9", "u5", "u2"))
    np.testing.assert_array_equal(encoded, np.asarray((0, 1, 0, 2, 1), dtype=np.int64))
    assert encoded.dtype == np.dtype("int64")
    assert not encoded.flags.writeable


def test_fold_b_fusion_uses_only_predeclared_grid_and_exact_protected_receipts() -> None:
    users = ("u1", "u1", "u1", "u2", "u2", "u2")
    videos = tuple(f"v{index}" for index in range(6))
    tree = np.asarray((0.9, 0.2, 0.1, 0.8, 0.3, 0.1), dtype=np.float64)
    fm = np.asarray((0.1, 0.2, 0.9, 0.1, 0.3, 0.8), dtype=np.float64)
    scorer_digest = hashlib.sha256(b"scorer").hexdigest()

    def score(values: npt.NDArray[np.float64]) -> ScoreResult:
        # The selected vector is the equal rank blend. The callback represents the trusted
        # organizer scorer boundary; the controller sees aggregate receipts only.
        target = np.asarray((0.5, 0.5, 0.5, 0.5, 0.5, 0.5), dtype=np.float64)
        distance = float(np.mean(np.abs(values - target)))
        primary = 0.8 - distance
        return ScoreResult(
            gauc=primary,
            ndcg_at_5=primary,
            primary=primary,
            users=2,
            rows=6,
            scorer_digest=scorer_digest,
            prediction_digest=prediction_digest(values),
            runtime_seconds=0.01,
        )

    selection = select_fold_b_fusion(
        user_ids=users,
        video_ids=videos,
        tree_scores=tree,
        fm_scores=fm,
        scorer_digest=scorer_digest,
        score=score,
    )

    assert tuple(item.weights for item in selection.grid) == FUSION_WEIGHT_GRID
    assert selection.selected.weights == (0.5, 0.5)
    assert selection.selected.metrics.primary == max(
        item.metrics.primary for item in selection.grid
    )
    assert (
        selection.digest
        == select_fold_b_fusion(
            user_ids=users,
            video_ids=videos,
            tree_scores=tree,
            fm_scores=fm,
            scorer_digest=scorer_digest,
            score=score,
        ).digest
    )


def test_fold_b_fusion_rejects_a_receipt_for_different_prediction_bytes() -> None:
    scorer_digest = hashlib.sha256(b"scorer").hexdigest()

    def dishonest_score(values: npt.NDArray[np.float64]) -> ScoreResult:
        return ScoreResult(
            gauc=0.5,
            ndcg_at_5=0.5,
            primary=0.5,
            users=1,
            rows=len(values),
            scorer_digest=scorer_digest,
            prediction_digest="f" * 64,
            runtime_seconds=0.0,
        )

    with pytest.raises(FullCampaignError, match="prediction identity"):
        select_fold_b_fusion(
            user_ids=("u", "u"),
            video_ids=("v1", "v2"),
            tree_scores=(0.0, 1.0),
            fm_scores=(1.0, 0.0),
            scorer_digest=scorer_digest,
            score=dishonest_score,
        )


def test_progress_ledger_is_append_only_and_exact_retries_are_free(tmp_path: Path) -> None:
    request_digest = hashlib.sha256(b"request").hexdigest()
    data_digest = hashlib.sha256(b"data-plane").hexdigest()
    ledger = FullCampaignProgressLedger(tmp_path / "progress")

    first = ledger.append(
        request_digest=request_digest,
        stage=FullCampaignStage.DATA_PREPARED,
        evidence={"data_plane_digest": data_digest, "final_targets": None},
    )
    retry = ledger.append(
        request_digest=request_digest,
        stage=FullCampaignStage.DATA_PREPARED,
        evidence={"data_plane_digest": data_digest, "final_targets": None},
    )
    second = ledger.append(
        request_digest=request_digest,
        stage=FullCampaignStage.QUALIFICATION_VERIFIED,
        evidence={"qualification_digest": hashlib.sha256(b"qualification").hexdigest()},
    )

    reopened = FullCampaignProgressLedger(tmp_path / "progress", create=False)
    assert retry == first
    assert second.sequence == 2
    assert second.previous_digest == first.digest
    assert reopened.checkpoints() == (first, second)
    assert tuple(path.name for path in (tmp_path / "progress").iterdir()) == (
        "checkpoint-00000001.json",
        "checkpoint-00000002.json",
    )


def test_progress_ledger_rejects_stage_skips_and_tampering(tmp_path: Path) -> None:
    request_digest = hashlib.sha256(b"request").hexdigest()
    ledger = FullCampaignProgressLedger(tmp_path / "progress")
    ledger.append(
        request_digest=request_digest,
        stage=FullCampaignStage.DATA_PREPARED,
        evidence={"data_plane_digest": hashlib.sha256(b"data").hexdigest()},
    )

    with pytest.raises(FullCampaignError, match="next campaign stage"):
        ledger.append(
            request_digest=request_digest,
            stage=FullCampaignStage.FEATURES_READY,
            evidence={"feature_digest": hashlib.sha256(b"features").hexdigest()},
        )

    checkpoint = tmp_path / "progress" / "checkpoint-00000001.json"
    checkpoint.chmod(0o600)
    checkpoint.write_bytes(checkpoint.read_bytes().replace(b"data_plane", b"evil_plane"))
    with pytest.raises(FullCampaignError, match=r"canonical|digest"):
        ledger.checkpoints()


def test_terminal_outcome_reopens_with_selected_artifact_closure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = ArtifactStore(run_dir / "artifacts")
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidate.py").write_text("def predict_scores():\n    return 1\n")
    (source / "config.json").write_text("{}")
    source_ref = artifacts.put_directory(source, kind=ArtifactKind.SOURCE)

    input_refs = tuple(
        artifacts.put_bytes(name.encode(), kind=ArtifactKind.INPUT)
        for name in (
            "train-features",
            "train-targets",
            "train-groups",
            "validation-features",
            "final-features",
        )
    )
    checkpoint_ref = artifacts.put_bytes(b"tree", kind=ArtifactKind.CHECKPOINT)
    prediction_ref = artifacts.put_bytes(b"validation", kind=ArtifactKind.PREDICTION)
    digests = tuple(hashlib.sha256(f"outcome-{index}".encode()).hexdigest() for index in range(20))
    member = QualifiedFMMemberPlan(
        seed=0,
        checkpoint_sha256=digests[0],
        checkpoint_digest=digests[1],
        encoding_sha256=digests[2],
        encoding_digest=digests[3],
        config_digest=digests[4],
        starter_manifest_digest=digests[5],
        validation_prediction_digest=digests[6],
    )
    matched_seeds = tuple(
        MatchedSeedSelectionEvidence(
            seed=seed,
            scientific_request_digest=digests[17] if seed == 0 else digests[seed],
            scientific_record_digest=digests[18] if seed == 0 else digests[seed + 3],
            checkpoint=checkpoint_ref,
            candidate_validation_prediction=prediction_ref,
            fm_validation_prediction=artifacts.put_bytes(
                f"fm-validation-{seed}".encode(), kind=ArtifactKind.PREDICTION
            ),
            candidate_metrics=OrganizerMetrics(0.61 + seed / 1000, 0.62),
            fm_metrics=OrganizerMetrics(0.60, 0.61),
            candidate_wall_seconds=1.5 + seed,
            candidate_peak_rss_bytes=1000 + seed,
            candidate_disk_bytes=2000 + seed,
            fm_wall_seconds=2.5 + seed,
            fm_peak_rss_bytes=3000 + seed,
            fm_disk_bytes=4000 + seed,
            fm_member=(
                member
                if seed == 0
                else QualifiedFMMemberPlan(
                    seed=seed,
                    checkpoint_sha256=digests[seed],
                    checkpoint_digest=digests[seed + 1],
                    encoding_sha256=digests[seed + 2],
                    encoding_digest=digests[seed + 3],
                    config_digest=digests[seed + 4],
                    starter_manifest_digest=digests[5],
                    validation_prediction_digest=digests[seed + 5],
                )
            ),
        )
        for seed in (0, 1, 2)
    )
    selection = FinalizationSelectionPlan(
        experiment_id="iteration-01",
        candidate_id="generated-causal-lambdarank-v1",
        candidate_fingerprint=digests[7],
        source_digest=digests[8],
        parent_source_digest=digests[9],
        executable_change_digest=digests[10],
        config_digest=hashlib.sha256(b"{}").hexdigest(),
        training_policy_digest=digests[11],
        evidence_receipt_digest=digests[12],
        source_snapshot=source_ref,
        training_features=input_refs[0],
        training_targets=input_refs[1],
        training_user_groups=input_refs[2],
        validation_features=input_refs[3],
        final_features=input_refs[4],
        feature_bundle_digest=digests[13],
        feature_count=4,
        dataset_digest=digests[14],
        validation_inputs_digest=digests[15],
        final_inputs_digest=digests[16],
        frozen_fusion_weights=(0.5, 0.5),
        representative_seed=0,
        selected_outer_request_digest=digests[17],
        scientific_record_digest=digests[18],
        tree_checkpoint=checkpoint_ref,
        validation_prediction=prediction_ref,
        fm_member=member,
        inner_folds=(
            InnerFoldSelectionEvidence(
                fold_id="A",
                candidate=OrganizerMetrics(0.61, 0.62),
                parent=OrganizerMetrics(0.60, 0.61),
                reference=OrganizerMetrics(0.60, 0.61),
                candidate_wall_seconds=1.5,
                candidate_peak_rss_bytes=2_000,
                candidate_disk_bytes=3_000,
                parent_wall_seconds=2.5,
                parent_peak_rss_bytes=4_000,
                parent_disk_bytes=5_000,
            ),
            InnerFoldSelectionEvidence(
                fold_id="B",
                candidate=OrganizerMetrics(0.615, 0.625),
                parent=OrganizerMetrics(0.60, 0.61),
                reference=OrganizerMetrics(0.60, 0.61),
                candidate_wall_seconds=1.6,
                candidate_peak_rss_bytes=2_100,
                candidate_disk_bytes=3_100,
                parent_wall_seconds=2.6,
                parent_peak_rss_bytes=4_100,
                parent_disk_bytes=5_100,
            ),
        ),
        matched_seeds=matched_seeds,
        timeout_seconds=60,
        memory_limit_bytes=1024**3,
        threads=2,
    )
    request_digest = hashlib.sha256(b"request").hexdigest()
    progress = FullCampaignProgressLedger(run_dir / "production" / "progress")
    for stage in tuple(FullCampaignStage)[:-1]:
        progress.append(
            request_digest=request_digest,
            stage=stage,
            evidence={"stage_receipt": hashlib.sha256(stage.value.encode()).hexdigest()},
        )
    predecessor = progress.checkpoints()[-1]
    outcome = FullCampaignOutcome(
        run_dir=run_dir,
        campaign_id="campaign-1",
        request_digest=request_digest,
        progress_predecessor_digest=predecessor.digest,
        fallback_candidate_id="official-fm-fallback-seed-4",
        fallback_receipt_digest=digests[19],
        qualification_manifest_digest=digests[0],
        dataset_digest=digests[14],
        scorer_digest=digests[5],
        validation_row_count=5,
        final_row_count=3,
        scientific_result_digest=digests[1],
        reflection_request_digest=digests[2],
        reflection_response_digest=digests[3],
        reflection_transcript=artifacts.put_bytes(b"reflection", kind=ArtifactKind.LOG),
        selection=selection,
        launches_used=11,
        outer_queries_used=1,
    )
    repository = FullCampaignOutcomeRepository(
        run_dir=run_dir,
        artifact_store=artifacts,
        progress=progress,
    )

    committed = repository.commit(outcome)
    reopened = FullCampaignOutcomeRepository(
        run_dir=run_dir,
        artifact_store=ArtifactStore(run_dir / "artifacts"),
        progress=FullCampaignProgressLedger(run_dir / "production" / "progress", create=False),
    ).load(request_digest=request_digest)

    assert reopened == committed == outcome
    assert reopened.finalization_required
    assert reopened.fallback_preserved
    assert reopened.has_generated_selection
    assert reopened.selection is not None
    assert reopened.selection.experiment_id == "iteration-01"
    assert reopened.selection.manifest()["experiment_id"] == "iteration-01"
    assert reopened.selection.tree_checkpoint == checkpoint_ref
    assert repository.commit(outcome) == outcome

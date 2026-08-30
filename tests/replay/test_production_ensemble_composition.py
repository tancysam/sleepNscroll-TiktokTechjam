"""Public clean-replay coverage for the production ensemble-composition seam."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.candidates.fusion import fuse_ranked_members
from kuairand_agent.data.capabilities import CandidateInputs, DataPhase, build_candidate_inputs
from kuairand_agent.data.fields import STANDARD_LATE_MEMBER, VIDEO_BASIC_MEMBER, FieldKey
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactRef, ArtifactStore
from kuairand_agent.finalization.backends import (
    GeneratedLambdaRankReplayBackend,
    build_replay_backend,
)
from kuairand_agent.finalization.production import (
    GeneratedLambdaRankReplayMemberBundle,
    ProductionFinalizationError,
    build_generated_lambdarank_ensemble_replay,
)
from kuairand_agent.finalization.recipe import GeneratedLambdaRankReplayRecipe
from kuairand_agent.finalization.replay import (
    CleanReplayRequest,
    FrozenCandidateWorkspace,
    ReplayCapabilities,
    ReplayEquality,
    run_clean_replay,
)
from kuairand_agent.scoring.submission import AlignmentRow, prediction_digest


def _inputs(phase: DataPhase) -> CandidateInputs:
    return build_candidate_inputs(
        phase,
        {
            FieldKey(STANDARD_LATE_MEMBER, "user_id"): ("u1", "u1", "u1"),
            FieldKey(STANDARD_LATE_MEMBER, "video_id"): ("v1", "v2", "v3"),
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): ("a1", "a2", "a3"),
            FieldKey(STANDARD_LATE_MEMBER, "tab"): ("0", "1", "0"),
            FieldKey(STANDARD_LATE_MEMBER, "duration_ms"): (5.0, 25.0, 45.0),
        },
    )


def _npy_artifact(store: ArtifactStore, values: np.ndarray) -> ArtifactRef:
    payload = BytesIO()
    np.save(payload, np.asarray(values, dtype=np.float64), allow_pickle=False)
    return store.put_bytes(payload.getvalue(), kind=ArtifactKind.PREDICTION)


def _member(
    tmp_path: Path,
    store: ArtifactStore,
    *,
    marker: str,
    validation_inputs: CandidateInputs,
    final_inputs: CandidateInputs,
    validation_scores: np.ndarray,
) -> GeneratedLambdaRankReplayMemberBundle:
    root = tmp_path / f"member-{marker}"
    source = root / "source"
    features = root / "features"
    checkpoint = root / "checkpoint"
    for directory in (source, features, checkpoint):
        directory.mkdir(parents=True)
    (source / "candidate.py").write_text(f"# member {marker}\n", encoding="ascii")
    (source / "config.json").write_text(f'{{"member":"{marker}"}}\n', encoding="ascii")
    with (features / "validation.npy").open("xb") as handle:
        np.save(handle, np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]), allow_pickle=False)
    with (features / "final.npy").open("xb") as handle:
        np.save(handle, np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]), allow_pickle=False)
    (checkpoint / "tree-model.txt").write_text(f"tree {marker}\n", encoding="ascii")
    source_ref = store.put_directory(source, kind=ArtifactKind.SOURCE)
    features_ref = store.put_directory(features, kind=ArtifactKind.INPUT)
    checkpoint_ref = store.put_directory(checkpoint, kind=ArtifactKind.CHECKPOINT)
    entries = {entry.path: entry.artifact for entry in source_ref.entries}
    feature_entries = {entry.path: entry.artifact for entry in features_ref.entries}
    checkpoint_entries = {entry.path: entry.artifact for entry in checkpoint_ref.entries}
    recipe = GeneratedLambdaRankReplayRecipe(
        source_artifact_sha256=source_ref.sha256,
        candidate_source_sha256=entries["candidate.py"].sha256,
        candidate_config_sha256=entries["config.json"].sha256,
        feature_artifact_sha256=features_ref.sha256,
        validation_features_sha256=feature_entries["validation.npy"].sha256,
        final_features_sha256=feature_entries["final.npy"].sha256,
        checkpoint_artifact_sha256=checkpoint_ref.sha256,
        tree_checkpoint_sha256=checkpoint_entries["tree-model.txt"].sha256,
        data_sha256="1" * 64,
        validation_inputs_digest=validation_inputs.digest,
        final_inputs_digest=final_inputs.digest,
        feature_count=2,
        timeout_seconds=60,
        memory_limit_bytes=1024 * 1024,
        threads=1,
    )
    return GeneratedLambdaRankReplayMemberBundle(
        recipe=recipe,
        source=source_ref,
        features=features_ref,
        checkpoint=checkpoint_ref,
        validation_predictions=_npy_artifact(store, validation_scores),
    )


def _environment() -> tuple[dict[str, object], str]:
    body: dict[str, object] = {
        "schema_version": 1,
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "fixture", "machine": "test"},
        "packages": {"lightgbm": "fixture", "numpy": "fixture"},
        "uv_lock_sha256": "2" * 64,
    }
    canonical = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(b"kuairand-environment-v1\0" + canonical).hexdigest()
    return body | {"digest": digest}, digest


def test_production_composite_replays_exact_ordered_v1_members_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    validation_inputs = _inputs(DataPhase.OUTER_VALID)
    final_inputs = _inputs(DataPhase.FINAL)
    members = (
        _member(
            tmp_path,
            store,
            marker="a",
            validation_inputs=validation_inputs,
            final_inputs=final_inputs,
            validation_scores=np.asarray([3.0, 2.0, 1.0]),
        ),
        _member(
            tmp_path,
            store,
            marker="b",
            validation_inputs=validation_inputs,
            final_inputs=final_inputs,
            validation_scores=np.asarray([1.0, 3.0, 2.0]),
        ),
    )
    environment, environment_digest = _environment()
    composite = build_generated_lambdarank_ensemble_replay(
        members=members,
        validation_user_ids=("u1", "u1", "u1"),
        validation_video_ids=("v1", "v2", "v3"),
        fusion_weights=(0.75, 0.25),
        environment_sha256=environment_digest,
        artifact_store=store,
    )
    expected_validation = fuse_ranked_members(
        ("u1", "u1", "u1"),
        ("v1", "v2", "v3"),
        (np.asarray([3.0, 2.0, 1.0]), np.asarray([1.0, 3.0, 2.0])),
        weights=(0.75, 0.25),
        phase=DataPhase.OUTER_VALID,
    ).scores
    final_scores = {
        members[0].recipe.digest: np.asarray([2.0, 3.0, 1.0]),
        members[1].recipe.digest: np.asarray([1.0, 2.0, 3.0]),
    }
    calls: list[tuple[str, DataPhase, str]] = []

    def replay_member(
        self: GeneratedLambdaRankReplayBackend,
        *,
        workspace: FrozenCandidateWorkspace,
        inputs: CandidateInputs,
        phase: DataPhase,
        expected_inputs_digest: str,
    ) -> np.ndarray:
        assert inputs.digest == expected_inputs_digest
        calls.append((self.recipe.digest, phase, str(workspace.config_path)))
        if phase is DataPhase.OUTER_VALID:
            return np.asarray(
                {
                    members[0].recipe.digest: [3.0, 2.0, 1.0],
                    members[1].recipe.digest: [1.0, 3.0, 2.0],
                }[self.recipe.digest],
                dtype=np.float64,
            )
        return final_scores[self.recipe.digest]

    monkeypatch.setattr(GeneratedLambdaRankReplayBackend, "_predict", replay_member)
    alignment = tuple(
        AlignmentRow(index, user, video)
        for index, (user, video) in enumerate(
            zip(("u1", "u1", "u1"), ("v1", "v2", "v3"), strict=True)
        )
    )
    result = run_clean_replay(
        CleanReplayRequest(
            candidate_id="ordered-v1-ensemble",
            output_dir=tmp_path / "replayed",
            identity=composite.identity,
            artifacts=composite.replay_artifacts(),
            environment=environment,
            equality=ReplayEquality.EXACT,
        ),
        artifact_store=store,
        capabilities=ReplayCapabilities(
            "1" * 64,
            validation_inputs,
            final_inputs,
            alignment,
            alignment,
        ),
        backend=build_replay_backend(composite.recipe),
        protected_metric_evaluator=lambda _scores: {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55},
    )

    expected_final = fuse_ranked_members(
        ("u1", "u1", "u1"),
        ("v1", "v2", "v3"),
        (final_scores[members[0].recipe.digest], final_scores[members[1].recipe.digest]),
        weights=(0.75, 0.25),
        phase=DataPhase.FINAL,
    ).scores
    assert composite.identity.config_sha256 == composite.recipe.digest
    assert composite.recipe.member_recipe_digests == tuple(
        member.recipe.digest for member in members
    )
    assert composite.recipe.validation_fusion_digest != prediction_digest(expected_validation)
    assert result.evidence.validation.exact_prediction_bytes is True
    assert result.evidence.final.prediction_digest == prediction_digest(expected_final)
    assert [(digest, phase) for digest, phase, _ in calls] == [
        (members[0].recipe.digest, DataPhase.OUTER_VALID),
        (members[1].recipe.digest, DataPhase.OUTER_VALID),
        (members[0].recipe.digest, DataPhase.FINAL),
        (members[1].recipe.digest, DataPhase.FINAL),
    ]
    assert [path.rsplit("/", 2)[-2:] for _, _, path in calls] == [
        ["0", "recipe.json"],
        ["1", "recipe.json"],
        ["0", "recipe.json"],
        ["1", "recipe.json"],
    ]


def test_production_composite_rejects_member_artifact_rebinding(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    validation_inputs = _inputs(DataPhase.OUTER_VALID)
    final_inputs = _inputs(DataPhase.FINAL)
    member = _member(
        tmp_path,
        store,
        marker="a",
        validation_inputs=validation_inputs,
        final_inputs=final_inputs,
        validation_scores=np.asarray([3.0, 2.0, 1.0]),
    )
    rebound_checkpoint = replace(
        member.checkpoint,
        manifest_artifact=member.features.manifest_artifact,
    )

    with pytest.raises(ProductionFinalizationError, match="do not match"):
        GeneratedLambdaRankReplayMemberBundle(
            recipe=member.recipe,
            source=member.source,
            features=member.features,
            checkpoint=rebound_checkpoint,
            validation_predictions=member.validation_predictions,
        )

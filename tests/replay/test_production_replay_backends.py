from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from kuairand_agent.baselines.artifacts import StarterFMCheckpoint, save_checkpoint
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import StarterFMAdapter, StarterFMConfig
from kuairand_agent.candidates.fusion import fuse_ranked_predictions
from kuairand_agent.contract import sha256_file, verify_starter_kit
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.capabilities import CandidateInputs, DataPhase, build_candidate_inputs
from kuairand_agent.data.fields import STANDARD_LATE_MEMBER, VIDEO_BASIC_MEMBER, FieldKey
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.finalization.backends import build_replay_backend
from kuairand_agent.finalization.recipe import (
    GeneratedLambdaRankReplayRecipe,
    OfficialFMMemberRecipe,
    OfficialFMReplayRecipe,
    write_replay_recipe,
)
from kuairand_agent.finalization.replay import (
    CleanReplayRequest,
    FrozenCandidateWorkspace,
    FrozenReplayIdentity,
    ReplayArtifacts,
    ReplayCancelledError,
    ReplayCapabilities,
    ReplayEquality,
    ReplayError,
    run_clean_replay,
)
from kuairand_agent.scoring.submission import AlignmentRow, prediction_digest

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
TRAIN_FIXTURE = ROOT / "tests" / "helpers" / "train_lambdarank_fixture.py"


class _InputValues(TypedDict):
    users: tuple[str, ...]
    videos: tuple[str, ...]
    authors: tuple[str, ...]
    tabs: tuple[str, ...]
    durations: tuple[float, ...]


def _inputs(
    phase: DataPhase,
    *,
    users: tuple[str, ...],
    videos: tuple[str, ...],
    authors: tuple[str, ...],
    tabs: tuple[str, ...],
    durations: tuple[float, ...],
) -> CandidateInputs:
    return build_candidate_inputs(
        phase,
        {
            FieldKey(STANDARD_LATE_MEMBER, "user_id"): users,
            FieldKey(STANDARD_LATE_MEMBER, "video_id"): videos,
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): authors,
            FieldKey(STANDARD_LATE_MEMBER, "tab"): tabs,
            FieldKey(STANDARD_LATE_MEMBER, "duration_ms"): durations,
        },
    )


def _canonical(
    *,
    users: tuple[str, ...],
    videos: tuple[str, ...],
    authors: tuple[str, ...],
    tabs: tuple[str, ...],
    durations: tuple[float, ...],
) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=users,
        video_id=videos,
        author_id=authors,
        tab=tabs,
        duration_ms=durations,
        date=(20220422,) * len(users),
        time_ms=tuple(range(len(users))),
    )


def test_official_fm_backend_replays_immutable_checkpoint_over_candidate_inputs(
    tmp_path: Path,
) -> None:
    validation_values: _InputValues = {
        "users": ("u1", "u2", "unknown-user"),
        "videos": ("v1", "v2", "unknown-video"),
        "authors": ("a1", "a2", "unknown-author"),
        "tabs": ("0", "1", "2"),
        "durations": (5.0, 25.0, 95.0),
    }
    final_values: _InputValues = {
        "users": ("u2", "u1"),
        "videos": ("v2", "v1"),
        "authors": ("a2", "a1"),
        "tabs": ("1", "0"),
        "durations": (15.0, 85.0),
    }
    validation = _inputs(DataPhase.OUTER_VALID, **validation_values)
    final = _inputs(DataPhase.FINAL, **final_values)
    encoding = StarterEncoding(
        edges=tuple(float(value) for value in range(10, 100, 10)),
        vocabs=(
            ("u1", "u2"),
            ("v1", "v2"),
            ("a1", "a2"),
            ("0", "1"),
            tuple(str(value) for value in range(10)),
        ),
        training_inputs_digest="1" * 64,
    )
    encoding_artifact = encoding.save(tmp_path / "encoding.npz")
    config = StarterFMConfig(seed=4)
    values = np.arange(encoding.total_dim * 16, dtype=np.float32).reshape(encoding.total_dim, 16)
    checkpoint = StarterFMCheckpoint(
        V=(values - values.mean()) * np.float32(1e-4),
        W=np.linspace(-0.02, 0.02, encoding.total_dim, dtype=np.float32),
        b=np.float32(0.03),
        encoding_digest=encoding.digest,
        config_digest=config.digest,
        starter_manifest_digest=verify_starter_kit(STARTER).manifest_sha256,
        seed=4,
        best_epoch=1,
        epochs_completed=1,
        optimizer_steps=1,
    )
    checkpoint_artifact = save_checkpoint(tmp_path / "checkpoint.npz", checkpoint)
    source_artifact_sha256 = "2" * 64
    data_sha256 = "3" * 64
    member = OfficialFMMemberRecipe(
        checkpoint_sha256=checkpoint_artifact.file_sha256,
        checkpoint_digest=checkpoint.digest,
        encoding_sha256=encoding_artifact.file_sha256,
        encoding_digest=encoding.digest,
        config_digest=config.digest,
        starter_manifest_sha256=checkpoint.starter_manifest_digest,
        seed=4,
    )
    recipe = OfficialFMReplayRecipe(
        source_artifact_sha256=source_artifact_sha256,
        feature_artifact_sha256=encoding_artifact.file_sha256,
        checkpoint_artifact_sha256=checkpoint_artifact.file_sha256,
        data_sha256=data_sha256,
        validation_inputs_digest=validation.digest,
        final_inputs_digest=final.digest,
        fm_member=member,
    )
    recipe_path = write_replay_recipe(tmp_path / "recipe.json", recipe)
    identity = FrozenReplayIdentity(
        source_sha256=source_artifact_sha256,
        config_sha256=recipe.digest,
        features_sha256=encoding_artifact.file_sha256,
        checkpoint_sha256=checkpoint_artifact.file_sha256,
        validation_prediction_artifact_sha256="4" * 64,
        validation_prediction_digest="5" * 64,
        data_sha256=data_sha256,
        environment_sha256="6" * 64,
    )
    workspace = FrozenCandidateWorkspace(
        root=tmp_path,
        source_dir=STARTER,
        config_path=recipe_path,
        features_path=encoding_artifact.path,
        checkpoint_path=checkpoint_artifact.path,
        identity=identity,
    )
    backend = build_replay_backend(recipe)
    expected_validation = StarterFMAdapter(starter_dir=STARTER, config=config).predict(
        checkpoint=checkpoint,
        encoding=encoding,
        inputs=_canonical(**validation_values),
    )
    expected_final = StarterFMAdapter(starter_dir=STARTER, config=config).predict(
        checkpoint=checkpoint,
        encoding=encoding,
        inputs=_canonical(**final_values),
    )

    replayed = np.asarray(backend.replay_validation(workspace=workspace, inputs=validation))
    predicted = np.asarray(backend.predict_final(workspace=workspace, inputs=final))

    assert replayed.tobytes() == expected_validation.scores.tobytes()
    assert predicted.tobytes() == expected_final.scores.tobytes()

    changed_validation_values: _InputValues = {
        **validation_values,
        "durations": (5.0, 25.0, 96.0),
    }
    changed_validation = _inputs(DataPhase.OUTER_VALID, **changed_validation_values)
    with pytest.raises(ReplayError, match="capability differs"):
        backend.replay_validation(workspace=workspace, inputs=changed_validation)

    changed_workspace = replace(
        workspace,
        identity=replace(identity, features_sha256="0" * 64),
    )
    with pytest.raises(ReplayError, match="artifact identity differs"):
        backend.replay_validation(workspace=changed_workspace, inputs=validation)


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="the optional pinned research-tree dependency is not installed",
)
def test_generated_lambdarank_backend_runs_fresh_subprocess_and_replays_frozen_fusion(
    tmp_path: Path,
) -> None:
    validation_values: _InputValues = {
        "users": ("u1", "u1", "u1", "u2", "u2", "u2"),
        "videos": ("v1", "v2", "v3", "v1", "v2", "v3"),
        "authors": ("a1", "a2", "a3", "a1", "a2", "a3"),
        "tabs": ("0", "1", "0", "1", "0", "1"),
        "durations": (5.0, 25.0, 45.0, 15.0, 35.0, 55.0),
    }
    final_values: _InputValues = {
        "users": ("u3", "u3", "u4", "u4"),
        "videos": ("v1", "v3", "v2", "v3"),
        "authors": ("a1", "a3", "a2", "a3"),
        "tabs": ("0", "1", "0", "1"),
        "durations": (65.0, 75.0, 85.0, 95.0),
    }
    validation_inputs = _inputs(DataPhase.OUTER_VALID, **validation_values)
    final_inputs = _inputs(DataPhase.FINAL, **final_values)
    validation_features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 0.5], [0.5, 2.0], [1.5, 1.0], [2.5, 0.0]],
        dtype=np.float64,
    )
    final_features = np.asarray(
        [[0.25, 1.5], [1.25, 0.25], [2.25, 0.75], [0.75, 2.25]], dtype=np.float64
    )
    source = tmp_path / "candidate-source"
    source.mkdir()
    for name in ("candidate.py", "config.json", "README.md"):
        shutil.copyfile(ROOT / "candidate_templates" / "lambdarank" / name, source / name)
    preprocessing = tmp_path / "preprocessing"
    preprocessing.mkdir()
    with (preprocessing / "validation.npy").open("xb") as handle:
        np.save(handle, validation_features, allow_pickle=False)
    with (preprocessing / "final.npy").open("xb") as handle:
        np.save(handle, final_features, allow_pickle=False)
    encoding = StarterEncoding(
        edges=tuple(float(value) for value in range(10, 100, 10)),
        vocabs=(
            ("u1", "u2", "u3", "u4"),
            ("v1", "v2", "v3"),
            ("a1", "a2", "a3"),
            ("0", "1"),
            tuple(str(value) for value in range(10)),
        ),
        training_inputs_digest="1" * 64,
    )
    encoding_artifact = encoding.save(preprocessing / "fm-encoding.npz")
    fm_config = StarterFMConfig(seed=4)
    fm_values = np.arange(encoding.total_dim * 16, dtype=np.float32).reshape(encoding.total_dim, 16)
    fm_checkpoint = StarterFMCheckpoint(
        V=(fm_values - fm_values.mean()) * np.float32(1e-4),
        W=np.linspace(-0.02, 0.02, encoding.total_dim, dtype=np.float32),
        b=np.float32(0.03),
        encoding_digest=encoding.digest,
        config_digest=fm_config.digest,
        starter_manifest_digest=verify_starter_kit(STARTER).manifest_sha256,
        seed=4,
        best_epoch=1,
        epochs_completed=1,
        optimizer_steps=1,
    )
    model = tmp_path / "model"
    model.mkdir()
    tree_path = model / "tree-model.txt"
    tree_validation_path = tmp_path / "tree-validation.npy"
    tree_final_path = tmp_path / "tree-final.npy"
    subprocess.run(
        (
            sys.executable,
            str(TRAIN_FIXTURE),
            "--validation-features",
            str(preprocessing / "validation.npy"),
            "--final-features",
            str(preprocessing / "final.npy"),
            "--model",
            str(tree_path),
            "--validation-predictions",
            str(tree_validation_path),
            "--final-predictions",
            str(tree_final_path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    fm_checkpoint_artifact = save_checkpoint(model / "fm-checkpoint.npz", fm_checkpoint)

    store = ArtifactStore(tmp_path / "artifacts")
    source_ref = store.put_directory(source, kind=ArtifactKind.SOURCE)
    preprocessing_ref = store.put_directory(preprocessing, kind=ArtifactKind.INPUT)
    model_ref = store.put_directory(model, kind=ArtifactKind.CHECKPOINT)
    data_sha256 = "2" * 64
    fm_member = OfficialFMMemberRecipe(
        checkpoint_sha256=fm_checkpoint_artifact.file_sha256,
        checkpoint_digest=fm_checkpoint.digest,
        encoding_sha256=encoding_artifact.file_sha256,
        encoding_digest=encoding.digest,
        config_digest=fm_config.digest,
        starter_manifest_sha256=fm_checkpoint.starter_manifest_digest,
        seed=4,
    )
    recipe = GeneratedLambdaRankReplayRecipe(
        source_artifact_sha256=source_ref.sha256,
        candidate_source_sha256=sha256_file(source / "candidate.py"),
        candidate_config_sha256=sha256_file(source / "config.json"),
        feature_artifact_sha256=preprocessing_ref.sha256,
        validation_features_sha256=sha256_file(preprocessing / "validation.npy"),
        final_features_sha256=sha256_file(preprocessing / "final.npy"),
        checkpoint_artifact_sha256=model_ref.sha256,
        tree_checkpoint_sha256=sha256_file(tree_path),
        data_sha256=data_sha256,
        validation_inputs_digest=validation_inputs.digest,
        final_inputs_digest=final_inputs.digest,
        feature_count=2,
        timeout_seconds=60,
        memory_limit_bytes=2 * 1024**3,
        threads=1,
        fm_member=fm_member,
        fusion_weights=(0.75, 0.25),
    )
    recipe_path = write_replay_recipe(tmp_path / "replay-recipe.json", recipe)
    recipe_ref = store.put_file(recipe_path, kind=ArtifactKind.INPUT)
    assert recipe_ref.sha256 == recipe.digest

    tree_validation = np.load(tree_validation_path, allow_pickle=False)
    tree_final = np.load(tree_final_path, allow_pickle=False)
    adapter = StarterFMAdapter(starter_dir=STARTER, config=fm_config)
    fm_validation = adapter.predict(
        checkpoint=fm_checkpoint,
        encoding=encoding,
        inputs=_canonical(**validation_values),
    ).scores
    fm_final = adapter.predict(
        checkpoint=fm_checkpoint,
        encoding=encoding,
        inputs=_canonical(**final_values),
    ).scores
    expected_validation = fuse_ranked_predictions(
        validation_values["users"],
        validation_values["videos"],
        tree_validation,
        fm_validation,
        weights=(0.75, 0.25),
        phase=DataPhase.OUTER_VALID,
    ).scores
    expected_final = fuse_ranked_predictions(
        final_values["users"],
        final_values["videos"],
        tree_final,
        fm_final,
        weights=(0.75, 0.25),
        phase=DataPhase.FINAL,
    ).scores
    reference_path = tmp_path / "reference.npy"
    with reference_path.open("xb") as handle:
        np.save(handle, expected_validation, allow_pickle=False)
    reference_ref = store.put_file(reference_path, kind=ArtifactKind.PREDICTION)
    environment_body: dict[str, object] = {
        "schema_version": 1,
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "fixture", "machine": "test"},
        "packages": {"lightgbm": "4.7.0", "numpy": "2.5.2"},
        "uv_lock_sha256": "3" * 64,
    }
    environment_digest = hashlib.sha256(
        b"kuairand-environment-v1\0"
        + json.dumps(
            environment_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    identity = FrozenReplayIdentity(
        source_sha256=source_ref.sha256,
        config_sha256=recipe_ref.sha256,
        features_sha256=preprocessing_ref.sha256,
        checkpoint_sha256=model_ref.sha256,
        validation_prediction_artifact_sha256=reference_ref.sha256,
        validation_prediction_digest=prediction_digest(expected_validation),
        data_sha256=data_sha256,
        environment_sha256=environment_digest,
    )
    validation_alignment = tuple(
        AlignmentRow(index, user, video)
        for index, (user, video) in enumerate(
            zip(validation_values["users"], validation_values["videos"], strict=True)
        )
    )
    final_alignment = tuple(
        AlignmentRow(index, user, video)
        for index, (user, video) in enumerate(
            zip(final_values["users"], final_values["videos"], strict=True)
        )
    )

    result = run_clean_replay(
        CleanReplayRequest(
            candidate_id="generated-fused",
            output_dir=tmp_path / "replayed",
            identity=identity,
            artifacts=ReplayArtifacts(
                source_ref,
                recipe_ref,
                preprocessing_ref,
                model_ref,
                reference_ref,
            ),
            environment=environment_body | {"digest": environment_digest},
            equality=ReplayEquality.EXACT,
        ),
        artifact_store=store,
        capabilities=ReplayCapabilities(
            data_sha256,
            validation_inputs,
            final_inputs,
            validation_alignment,
            final_alignment,
        ),
        backend=build_replay_backend(recipe),
        protected_metric_evaluator=lambda scores: {
            "GAUC": 0.6,
            "nDCG@5": 0.5,
            "primary": 0.55,
        },
    )

    assert result.evidence.validation.exact_prediction_bytes is True
    assert result.evidence.final.prediction_digest == prediction_digest(expected_final)
    assert result.evidence.final.final_outcomes_accessed is False

    cancellation = threading.Event()
    cancellation.set()
    cancelled_backend = build_replay_backend(recipe, cancel_event=cancellation)
    cancelled_workspace = FrozenCandidateWorkspace(
        root=tmp_path,
        source_dir=source,
        config_path=recipe_path,
        features_path=preprocessing,
        checkpoint_path=model,
        identity=identity,
    )
    with pytest.raises(ReplayCancelledError, match="cancelled"):
        cancelled_backend.replay_validation(
            workspace=cancelled_workspace,
            inputs=validation_inputs,
        )

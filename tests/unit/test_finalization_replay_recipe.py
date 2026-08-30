from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pytest

from kuairand_agent.finalization.recipe import (
    GeneratedLambdaRankReplayRecipe,
    OfficialFMMemberRecipe,
    ReplayRecipeError,
    load_replay_recipe,
    write_replay_recipe,
)


def _fm_member() -> OfficialFMMemberRecipe:
    return OfficialFMMemberRecipe(
        checkpoint_sha256="1" * 64,
        checkpoint_digest="2" * 64,
        encoding_sha256="3" * 64,
        encoding_digest="4" * 64,
        config_digest="5" * 64,
        starter_manifest_sha256="6" * 64,
        seed=4,
    )


class _RecipeKwargs(TypedDict):
    source_artifact_sha256: str
    candidate_source_sha256: str
    candidate_config_sha256: str
    feature_artifact_sha256: str
    validation_features_sha256: str
    final_features_sha256: str
    checkpoint_artifact_sha256: str
    tree_checkpoint_sha256: str
    data_sha256: str
    validation_inputs_digest: str
    final_inputs_digest: str
    feature_count: int
    timeout_seconds: int
    memory_limit_bytes: int
    threads: int


def _recipe_kwargs() -> _RecipeKwargs:
    return {
        "source_artifact_sha256": "7" * 64,
        "candidate_source_sha256": "8" * 64,
        "candidate_config_sha256": "9" * 64,
        "feature_artifact_sha256": "a" * 64,
        "validation_features_sha256": "b" * 64,
        "final_features_sha256": "c" * 64,
        "checkpoint_artifact_sha256": "d" * 64,
        "tree_checkpoint_sha256": "e" * 64,
        "data_sha256": "f" * 64,
        "validation_inputs_digest": "0" * 64,
        "final_inputs_digest": "1" * 64,
        "feature_count": 33,
        "timeout_seconds": 300,
        "memory_limit_bytes": 4 * 1024**3,
        "threads": 4,
    }


def _recipe() -> GeneratedLambdaRankReplayRecipe:
    return GeneratedLambdaRankReplayRecipe(
        **_recipe_kwargs(),
        fm_member=_fm_member(),
        fusion_weights=(0.75, 0.25),
    )


def test_generated_recipe_round_trips_as_one_canonical_allowlisted_artifact(
    tmp_path: Path,
) -> None:
    recipe = _recipe()

    path = write_replay_recipe(tmp_path / "replay-recipe.json", recipe)
    restored = load_replay_recipe(path, expected_sha256=recipe.digest)

    assert restored == recipe
    assert path.read_bytes() == recipe.canonical_bytes
    assert json.loads(path.read_text(encoding="ascii"))["backend"] == ("generated_lambdarank_v1")


def test_recipe_rejects_unknown_backends_and_non_grid_or_unpaired_fusion(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown.json"
    payload = _recipe().manifest() | {"backend": "python_import_path"}
    unknown.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    with pytest.raises(ReplayRecipeError, match="allowlisted"):
        load_replay_recipe(unknown)
    with pytest.raises(ReplayRecipeError, match="FUSION_WEIGHT_GRID"):
        GeneratedLambdaRankReplayRecipe(
            **_recipe_kwargs(),
            fm_member=_fm_member(),
            fusion_weights=(0.625, 0.375),
        )
    with pytest.raises(ReplayRecipeError, match="together"):
        GeneratedLambdaRankReplayRecipe(
            **_recipe_kwargs(),
            fm_member=None,
            fusion_weights=(0.75, 0.25),
        )

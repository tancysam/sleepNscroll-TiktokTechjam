from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kuairand_agent.finalization.recipe import (
    GeneratedLambdaRankEnsembleReplayRecipe,
    GeneratedLambdaRankReplayRecipe,
    ReplayRecipeError,
    load_replay_recipe,
    write_replay_recipe,
)


def _member(marker: str) -> GeneratedLambdaRankReplayRecipe:
    return GeneratedLambdaRankReplayRecipe(
        source_artifact_sha256="1" * 64,
        candidate_source_sha256=marker * 64,
        candidate_config_sha256="3" * 64,
        feature_artifact_sha256="4" * 64,
        validation_features_sha256="5" * 64,
        final_features_sha256="6" * 64,
        checkpoint_artifact_sha256="7" * 64,
        tree_checkpoint_sha256="8" * 64,
        data_sha256="9" * 64,
        validation_inputs_digest="a" * 64,
        final_inputs_digest="b" * 64,
        feature_count=2,
        timeout_seconds=60,
        memory_limit_bytes=1024 * 1024,
        threads=1,
    )


def _recipe() -> GeneratedLambdaRankEnsembleReplayRecipe:
    members = (_member("c"), _member("d"))
    return GeneratedLambdaRankEnsembleReplayRecipe(
        members=members,
        member_recipe_digests=tuple(member.digest for member in members),
        validation_member_prediction_digests=("e" * 64, "f" * 64),
        fusion_weights=(0.5, 0.5),
        validation_fusion_digest="0" * 64,
    )


def test_generated_ensemble_recipe_round_trips_with_ordered_complete_v1_members(
    tmp_path: Path,
) -> None:
    recipe = _recipe()

    path = write_replay_recipe(tmp_path / "ensemble-recipe.json", recipe)
    restored = load_replay_recipe(path, expected_sha256=recipe.digest)

    assert restored == recipe
    assert path.read_bytes() == recipe.canonical_bytes
    assert recipe.manifest()["backend"] == "generated_lambdarank_ensemble_v2"
    assert recipe.data_sha256 == recipe.members[0].data_sha256
    assert recipe.validation_inputs_digest == recipe.members[0].validation_inputs_digest
    assert recipe.final_inputs_digest == recipe.members[0].final_inputs_digest


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda recipe: replace(
                recipe, member_recipe_digests=tuple(reversed(recipe.member_recipe_digests))
            ),
            "exactly bind ordered members",
        ),
        (
            lambda recipe: replace(recipe, fusion_weights=(0.75, 0.5)),
            "summing to one",
        ),
        (
            lambda recipe: replace(recipe, members=(recipe.members[0],)),
            "between 2",
        ),
        (
            lambda recipe: replace(
                recipe,
                members=(
                    recipe.members[0],
                    replace(recipe.members[1], checkpoint_artifact_sha256="0" * 64),
                ),
            ),
            "exactly bind ordered members",
        ),
    ],
)
def test_generated_ensemble_recipe_rejects_partial_or_rebound_members(
    mutate: object, message: str
) -> None:
    with pytest.raises(ReplayRecipeError, match=message):
        mutate(_recipe())  # type: ignore[operator]

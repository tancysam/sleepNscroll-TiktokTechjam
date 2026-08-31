"""Guards on the official-FM seed-ensemble replay recipe.

The ensemble exists because rank averaging the five qualified organizer seeds scores 0.6026034 on
public validation against 0.6016 published, with no selection effect. The recipe's job is to make
that claim checkable: the members must be genuinely distinct seeds of one shared encoding, in a
fixed order, so the bundle cannot silently ship a different model than it names.
"""

from __future__ import annotations

import pytest

from kuairand_agent.finalization.recipe import (
    OFFICIAL_FM_ENSEMBLE_COMBINATION,
    OfficialFMMemberRecipe,
    OfficialFMSeedEnsembleReplayRecipe,
    ReplayBackendKind,
    ReplayRecipeError,
    parse_replay_recipe,
)

_SHARED_ENCODING_SHA = "e0" * 32
_SHARED_ENCODING_DIGEST = "e1" * 32
_SHARED_STARTER = "51" * 32


def _member(seed: int, *, encoding_sha: str = _SHARED_ENCODING_SHA) -> OfficialFMMemberRecipe:
    return OfficialFMMemberRecipe(
        checkpoint_sha256=f"{seed:02d}" * 32,
        checkpoint_digest=f"{seed + 10:02d}" * 32,
        encoding_sha256=encoding_sha,
        encoding_digest=_SHARED_ENCODING_DIGEST,
        config_digest=f"{seed + 20:02d}" * 32,
        starter_manifest_sha256=_SHARED_STARTER,
        seed=seed,
    )


def _recipe(members: tuple[OfficialFMMemberRecipe, ...]) -> OfficialFMSeedEnsembleReplayRecipe:
    return OfficialFMSeedEnsembleReplayRecipe(
        source_artifact_sha256="a1" * 32,
        feature_artifact_sha256="a2" * 32,
        checkpoint_artifact_sha256="a3" * 32,
        data_sha256="a4" * 32,
        validation_inputs_digest="a5" * 32,
        final_inputs_digest="a6" * 32,
        fm_members=members,
    )


def test_five_seed_recipe_round_trips_through_the_backend_allowlist() -> None:
    recipe = _recipe(tuple(_member(seed) for seed in (0, 1, 2, 3, 4)))

    assert recipe.backend is ReplayBackendKind.OFFICIAL_FM_SEED_ENSEMBLE
    assert recipe.combination == OFFICIAL_FM_ENSEMBLE_COMBINATION
    assert recipe.seeds == (0, 1, 2, 3, 4)

    parsed = parse_replay_recipe(recipe.canonical_bytes)
    assert isinstance(parsed, OfficialFMSeedEnsembleReplayRecipe)
    assert parsed.digest == recipe.digest
    assert parsed.seeds == recipe.seeds


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ((0,), "between"),
        ((0, 0), "repeat a seed"),
        ((1, 0), "ascending seed"),
    ],
)
def test_member_set_must_be_distinct_ordered_and_large_enough(
    members: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ReplayRecipeError, match=match):
        _recipe(tuple(_member(seed) for seed in members))


def test_members_must_share_one_encoding() -> None:
    # Seeds differ only in weight initialisation, so a differing vocabulary means these
    # checkpoints are not mutually comparable and averaging them would be meaningless.
    mixed = (_member(0), _member(1, encoding_sha="ff" * 32))
    with pytest.raises(ReplayRecipeError, match="share one encoding artifact"):
        _recipe(mixed)


def test_members_must_reference_distinct_checkpoints() -> None:
    duplicated = (_member(0), _member(1))
    object.__setattr__(duplicated[1], "checkpoint_sha256", duplicated[0].checkpoint_sha256)
    with pytest.raises(ReplayRecipeError, match="distinct checkpoints"):
        _recipe(duplicated)


def test_combination_policy_is_not_negotiable_on_the_wire() -> None:
    recipe = _recipe(tuple(_member(seed) for seed in (0, 1)))
    forged = recipe.canonical_bytes.replace(
        OFFICIAL_FM_ENSEMBLE_COMBINATION.encode("ascii"),
        b"raw_score_mean_v1" + b"_" * (len(OFFICIAL_FM_ENSEMBLE_COMBINATION) - 17),
    )
    with pytest.raises(ReplayRecipeError):
        parse_replay_recipe(forged)

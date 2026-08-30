"""Bounded train-only pilot for duration-conditioned observed-pair FM.

This diagnostic compares two byte-reproducible arms on the frozen official-training folds:

* ``uniform_control``: the existing positive-ticket GAUC-aligned logged-pair sampler;
* ``duration_conditioned``: an equal-budget 50/50 mixture whose intervention pairs share
  both user and duration bucket.

The default budget is intentionally smaller than the protected production budget.  It is a
pilot, not a promotion result: Fold A and Fold B are both derived from official training, and
the trusted fold scorer is used only on those train-derived query windows.  No public-validation
or final-period rows, outcomes, or scoring calls are reachable from this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "candidate_seed"))

from reference_observed_pair_fm import (  # noqa: E402
    _fit_reference_observed_pair_fm,
    reference_observed_pair_fm_diagnostics,
    reference_observed_pair_fm_scores,
)

from kuairand_agent.baselines.fold_controls import (  # noqa: E402
    build_fold_scoring_context,
    run_fold_fm_control,
)
from kuairand_agent.campaign.full_campaign import (  # noqa: E402
    encode_numeric_user_groups,
    prepare_campaign_data_plane,
)
from kuairand_agent.campaign.pure_features import build_pure_feature_pair  # noqa: E402
from kuairand_agent.candidates.fusion import (  # noqa: E402
    FUSION_WEIGHT_GRID,
    fuse_ranked_members,
)
from kuairand_agent.data.canonical import load_canonical_dataset  # noqa: E402
from kuairand_agent.data.capabilities import DataPhase  # noqa: E402
from kuairand_agent.scoring.submission import prediction_digest  # noqa: E402

PILOT_PAIRS_PER_EPOCH: Final = 8192
PILOT_EPOCHS: Final = 2
FULL_PAIRS_PER_EPOCH: Final = 250_000
FULL_EPOCHS: Final = 5
SEED: Final = 20260830
FULL_REPLICATE_SEEDS: Final = (0, 1, 2)
ARMS: Final = ("uniform_control", "duration_conditioned")


def _metrics(score: object) -> dict[str, float]:
    return {
        "GAUC": float(score.gauc),
        "nDCG@5": float(score.ndcg_at_5),
        "primary": float(score.primary),
    }


def _builder_digest() -> str:
    return hashlib.sha256(
        (ROOT / "src" / "kuairand_agent" / "campaign" / "pure_features.py").read_bytes()
    ).hexdigest()


def _fold_result(
    *,
    fold_name: str,
    data: object,
    dataset_digest: str,
    pairs_per_epoch: int,
    epochs: int,
    seed: int,
    official_relative: bool = False,
    frozen_fusion_weight: float | None = None,
) -> dict[str, object]:
    fold = data.fold_a if fold_name == "A" else data.fold_b
    pair = build_pure_feature_pair(
        prefix_inputs=fold.prefix_inputs,
        prefix_labels=fold.prefix_labels,
        prefix_click_labels=fold.prefix_click_labels,
        prefix_watch_progress=fold.prefix_watch_progress,
        query_inputs=fold.query_inputs,
        dataset_digest=dataset_digest,
        split_role=f"research_observed_pair_duration_fold_{fold_name}",
        builder_source_digest=_builder_digest(),
        cache_dir=ROOT / "runs" / "diagnostics" / "feature-cache-v8",
    )
    targets = np.asarray(fold.prefix_labels, dtype=np.float64)
    groups = encode_numeric_user_groups(fold.prefix_inputs.user_id)
    scorer = build_fold_scoring_context(
        ROOT / "kuairand-starter-kit",
        fold_name,
        fold.fold.digest,
        fold.query_inputs,
        fold.query_labels,
    )
    arms: dict[str, object] = {}
    arm_scores: dict[str, np.ndarray] = {}
    for arm in ARMS:
        started = time.monotonic()
        checkpoint = _fit_reference_observed_pair_fm(
            pair.prefix.values,
            targets,
            groups,
            arm=arm,
            pairs_per_epoch=pairs_per_epoch,
            epochs=epochs,
            seed=seed,
        )
        train_seconds = time.monotonic() - started
        scores = reference_observed_pair_fm_scores(pair.query.values, checkpoint)
        metrics = _metrics(scorer(scores))
        diagnostics = reference_observed_pair_fm_diagnostics(checkpoint)
        arm_scores[arm] = scores
        arms[arm] = {
            "metrics": metrics,
            "runtime_seconds": train_seconds,
            "diagnostics": diagnostics,
            "prediction_digest": prediction_digest(scores),
        }
    control = arms["uniform_control"]
    treatment = arms["duration_conditioned"]
    control_metrics = control["metrics"]
    treatment_metrics = treatment["metrics"]
    deltas = {
        name: treatment_metrics[name] - control_metrics[name]
        for name in ("GAUC", "nDCG@5", "primary")
    }
    result: dict[str, object] = {
        "fold": fold_name,
        "prefix_rows": pair.prefix.row_count,
        "query_rows": pair.query.row_count,
        "feature_count": pair.prefix.feature_count,
        "pairs_per_epoch": pairs_per_epoch,
        "epochs": epochs,
        "seed": seed,
        "arms": arms,
        "duration_minus_uniform": deltas,
    }
    if official_relative:
        official_started = time.monotonic()
        official_run = run_fold_fm_control(
            fold.prefix_inputs,
            fold.prefix_labels,
            fold.query_inputs,
            fold.query_labels,
            ROOT / "kuairand-starter-kit",
            seed=seed,
            fold_name=fold_name,
            fold_token=fold.fold.digest,
        )
        official_scores = np.asarray(official_run.predictions.scores, dtype=np.float64)
        replay_scores = official_run.replay_predictions(
            starter_dir=ROOT / "kuairand-starter-kit",
            query_inputs=fold.query_inputs,
        ).scores
        if not np.array_equal(official_scores, replay_scores):
            raise RuntimeError("official FM helper replay changed validation prediction bytes")
        official_metrics = _metrics(official_run.aggregate_metrics)
        result["official_fm"] = {
            "metrics": official_metrics,
            "runtime_seconds": time.monotonic() - official_started,
            "prediction_digest": official_run.predictions.digest,
            "checkpoint_digest": official_run.checkpoint.digest,
            "encoding_digest": official_run.encoding_digest,
            "fold_control_digest": official_run.digest,
            "replay_prediction_bytes_identical": True,
        }
        result["duration_minus_official_fm"] = {
            name: arms["duration_conditioned"]["metrics"][name] - official_metrics[name]
            for name in ("GAUC", "nDCG@5", "primary")
        }
        result["uniform_minus_official_fm"] = {
            name: arms["uniform_control"]["metrics"][name] - official_metrics[name]
            for name in ("GAUC", "nDCG@5", "primary")
        }

        result["_duration_scores"] = arm_scores["duration_conditioned"]
        result["_official_scores"] = official_scores
        if frozen_fusion_weight is not None:
            users = fold.query_inputs.user_id
            videos = fold.query_inputs.video_id
            phase = DataPhase.INNER_VALID
            per_seed = fuse_ranked_members(
                users,
                videos,
                (arm_scores["duration_conditioned"], official_scores),
                weights=(float(frozen_fusion_weight), 1.0 - float(frozen_fusion_weight)),
                phase=phase,
            )
            frozen_metrics = _metrics(scorer(per_seed.scores))
            result["frozen_duration_official_fusion"] = {
                "selected_on_fold": "B",
                "duration_weight": float(frozen_fusion_weight),
                "official_fm_weight": 1.0 - float(frozen_fusion_weight),
                "metrics": frozen_metrics,
                "primary_delta_to_official_fm": frozen_metrics["primary"]
                - official_metrics["primary"],
                "fusion_digest": per_seed.fusion_digest,
                "prediction_digest": per_seed.prediction_digest,
            }
    return result


def _deployable_seed_fusion(
    *,
    users: tuple[str, ...],
    videos: tuple[str, ...],
    duration_scores: list[np.ndarray],
    official_scores: list[np.ndarray],
    duration_weight: float,
) -> object:
    """Mirror v1 duration/FM fusion followed by equal-weight v2 seed fusion."""

    if len(duration_scores) != len(official_scores) or len(duration_scores) < 2:
        raise RuntimeError("seed fusion requires at least two matched duration/FM members")
    per_seed = tuple(
        fuse_ranked_members(
            users,
            videos,
            (duration, official),
            weights=(float(duration_weight), 1.0 - float(duration_weight)),
            phase=DataPhase.INNER_VALID,
        ).scores
        for duration, official in zip(duration_scores, official_scores, strict=True)
    )
    equal_weight = tuple(1.0 / len(per_seed) for _ in per_seed)
    return fuse_ranked_members(
        users,
        videos,
        per_seed,
        weights=equal_weight,
        phase=DataPhase.INNER_VALID,
    )


def _select_shared_fusion(
    *,
    scorer: object,
    users: tuple[str, ...],
    videos: tuple[str, ...],
    duration_scores: list[np.ndarray],
    official_scores: list[np.ndarray],
) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    """Select one legacy grid weight on the aggregate Fold-B seed portfolio."""

    grid_results: list[dict[str, object]] = []
    for grid_point in FUSION_WEIGHT_GRID:
        duration_weight = float(grid_point[0])
        fused = _deployable_seed_fusion(
            users=users,
            videos=videos,
            duration_scores=duration_scores,
            official_scores=official_scores,
            duration_weight=duration_weight,
        )
        metrics = _metrics(scorer(fused.scores))
        grid_results.append(
            {
                "duration_weight": duration_weight,
                "official_fm_weight": 1.0 - duration_weight,
                "metrics": metrics,
                "fusion_digest": fused.fusion_digest,
                "prediction_digest": fused.prediction_digest,
            }
        )
    selected = max(
        grid_results,
        key=lambda item: (
            float(item["metrics"]["primary"]),
            float(item["metrics"]["GAUC"]),
            -float(item["duration_weight"]),
        ),
    )
    return float(selected["duration_weight"]), selected, grid_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="Seed to run; repeat for a multi-seed replicate (default: pilot seed).",
    )
    parser.add_argument(
        "--full-replicate",
        action="store_true",
        help="Run the predeclared full-budget seeds 0, 1, and 2.",
    )
    parser.add_argument(
        "--official-relative",
        action="store_true",
        help="Also train exact official-FM fold controls and evaluate frozen rank fusion.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit a compact summary after the requested training run.",
    )
    args = parser.parse_args()
    if args.full_replicate and (
        args.pairs_per_epoch is not None or args.epochs is not None
    ):
        raise SystemExit("--full-replicate cannot be combined with --pairs-per-epoch/--epochs")
    pairs_per_epoch = (
        FULL_PAIRS_PER_EPOCH
        if args.full_replicate
        else (PILOT_PAIRS_PER_EPOCH if args.pairs_per_epoch is None else args.pairs_per_epoch)
    )
    epochs = (
        FULL_EPOCHS
        if args.full_replicate
        else (PILOT_EPOCHS if args.epochs is None else args.epochs)
    )
    seeds = tuple(
        FULL_REPLICATE_SEEDS
        if args.full_replicate and args.seeds is None
        else (args.seeds if args.seeds is not None else (SEED,))
    )
    if pairs_per_epoch < 2 or pairs_per_epoch % 2:
        raise SystemExit("--pairs-per-epoch must be an even integer >= 2")
    if epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    started = time.monotonic()
    dataset = load_canonical_dataset(ROOT / ".data" / "KuaiRand-Pure" / "data")
    data = prepare_campaign_data_plane(dataset, expected_dataset_digest=dataset.digest)
    folds: list[dict[str, object]] = []
    official_relative_result: dict[str, object] | None = None
    if args.official_relative:
        if len(seeds) < 2:
            raise SystemExit("--official-relative requires at least two seeds")
        fold_b_results: list[dict[str, object]] = []
        for seed in seeds:
            fold_b = _fold_result(
                fold_name="B",
                data=data,
                dataset_digest=dataset.digest,
                pairs_per_epoch=pairs_per_epoch,
                epochs=epochs,
                seed=seed,
                official_relative=True,
            )
            fold_b_results.append(fold_b)
        duration_b_scores = [result.pop("_duration_scores") for result in fold_b_results]
        official_b_scores = [result.pop("_official_scores") for result in fold_b_results]
        scorer_b = build_fold_scoring_context(
            ROOT / "kuairand-starter-kit",
            "B",
            data.fold_b.fold.digest,
            data.fold_b.query_inputs,
            data.fold_b.query_labels,
        )
        selected_weight, selected_grid_point, selection_grid = _select_shared_fusion(
            scorer=scorer_b,
            users=data.fold_b.query_inputs.user_id,
            videos=data.fold_b.query_inputs.video_id,
            duration_scores=duration_b_scores,
            official_scores=official_b_scores,
        )
        fold_a_results: list[dict[str, object]] = []
        for seed in seeds:
            fold_a = _fold_result(
                fold_name="A",
                data=data,
                dataset_digest=dataset.digest,
                pairs_per_epoch=pairs_per_epoch,
                epochs=epochs,
                seed=seed,
                official_relative=True,
                frozen_fusion_weight=selected_weight,
            )
            fold_a_results.append(fold_a)
        duration_a_scores = [result.pop("_duration_scores") for result in fold_a_results]
        official_a_scores = [result.pop("_official_scores") for result in fold_a_results]
        scorer_a = build_fold_scoring_context(
            ROOT / "kuairand-starter-kit",
            "A",
            data.fold_a.fold.digest,
            data.fold_a.query_inputs,
            data.fold_a.query_labels,
        )

        def portfolio_summary(
            *,
            scorer: object,
            users: tuple[str, ...],
            videos: tuple[str, ...],
            duration_scores: list[np.ndarray],
            official_scores: list[np.ndarray],
        ) -> dict[str, object]:
            equal_weights = tuple(1.0 / len(duration_scores) for _ in duration_scores)
            duration_only = fuse_ranked_members(
                users,
                videos,
                duration_scores,
                weights=equal_weights,
                phase=DataPhase.INNER_VALID,
            )
            official_only = fuse_ranked_members(
                users,
                videos,
                official_scores,
                weights=equal_weights,
                phase=DataPhase.INNER_VALID,
            )
            deployed = _deployable_seed_fusion(
                users=users,
                videos=videos,
                duration_scores=duration_scores,
                official_scores=official_scores,
                duration_weight=selected_weight,
            )
            duration_metrics = _metrics(scorer(duration_only.scores))
            official_metrics = _metrics(scorer(official_only.scores))
            deployed_metrics = _metrics(scorer(deployed.scores))
            return {
                "seed_count": len(duration_scores),
                "seed_weights": list(equal_weights),
                "duration_only": {
                    "metrics": duration_metrics,
                    "prediction_digest": duration_only.prediction_digest,
                    "fusion_digest": duration_only.fusion_digest,
                },
                "official_fm_only": {
                    "metrics": official_metrics,
                    "prediction_digest": official_only.prediction_digest,
                    "fusion_digest": official_only.fusion_digest,
                },
                "duration_official_deployed": {
                    "duration_weight": selected_weight,
                    "official_fm_weight": 1.0 - selected_weight,
                    "metrics": deployed_metrics,
                    "primary_delta_to_official_fm_only": deployed_metrics["primary"]
                    - official_metrics["primary"],
                    "prediction_digest": deployed.prediction_digest,
                    "fusion_digest": deployed.fusion_digest,
                },
                "duration_minus_official_fm_only": {
                    name: duration_metrics[name] - official_metrics[name]
                    for name in ("GAUC", "nDCG@5", "primary")
                },
            }

        duration_b_digests = [
            result["arms"]["duration_conditioned"]["prediction_digest"]
            for result in fold_b_results
        ]
        official_b_digests = [
            result["official_fm"]["prediction_digest"] for result in fold_b_results
        ]
        duration_a_digests = [
            result["arms"]["duration_conditioned"]["prediction_digest"]
            for result in fold_a_results
        ]
        official_a_digests = [
            result["official_fm"]["prediction_digest"] for result in fold_a_results
        ]
        if len(set(duration_b_digests)) != len(duration_b_digests) or len(
            set(official_b_digests)
        ) != len(official_b_digests):
            raise RuntimeError("distinct seed prediction digests unexpectedly collided on Fold B")
        if len(set(duration_a_digests)) != len(duration_a_digests) or len(
            set(official_a_digests)
        ) != len(official_a_digests):
            raise RuntimeError("distinct seed prediction digests unexpectedly collided on Fold A")
        official_relative_result = {
            "selection_fold": "B",
            "confirmation_fold": "A",
            "fusion_weight_grid": [list(point) for point in FUSION_WEIGHT_GRID],
            "selected_shared_duration_weight": selected_weight,
            "selected_shared_official_fm_weight": 1.0 - selected_weight,
            "selected_fold_b_grid_point": selected_grid_point,
            "fold_b_selection_grid": selection_grid,
            "fold_b": portfolio_summary(
                scorer=scorer_b,
                users=data.fold_b.query_inputs.user_id,
                videos=data.fold_b.query_inputs.video_id,
                duration_scores=duration_b_scores,
                official_scores=official_b_scores,
            ),
            "fold_a_frozen": portfolio_summary(
                scorer=scorer_a,
                users=data.fold_a.query_inputs.user_id,
                videos=data.fold_a.query_inputs.video_id,
                duration_scores=duration_a_scores,
                official_scores=official_a_scores,
            ),
            "fold_b_duration_member_prediction_digests": duration_b_digests,
            "fold_b_official_member_prediction_digests": official_b_digests,
            "fold_a_duration_member_prediction_digests": duration_a_digests,
            "fold_a_official_member_prediction_digests": official_a_digests,
            "seed_prediction_digests_distinct": True,
        }
        folds.extend((*fold_b_results, *fold_a_results))
    else:
        folds = [
            _fold_result(
                fold_name=fold_name,
                data=data,
                dataset_digest=dataset.digest,
                pairs_per_epoch=pairs_per_epoch,
                epochs=epochs,
                seed=seed,
            )
            for fold_name in ("A", "B")
            for seed in seeds
        ]
    delta_rows = [result["duration_minus_uniform"] for result in folds]
    mean_deltas = {
        name: float(np.mean([float(row[name]) for row in delta_rows]))
        for name in ("GAUC", "nDCG@5", "primary")
    }
    worst_deltas = {
        name: float(min(float(row[name]) for row in delta_rows))
        for name in ("GAUC", "nDCG@5", "primary")
    }
    if official_relative_result is not None:
        for result in folds:
            result.pop("_duration_scores", None)
            result.pop("_official_scores", None)
    output = {
        "schema_version": 1,
        "status": (
            "full_budget_train_derived_replicate"
            if args.full_replicate
            else "pilot_only_train_derived"
        ),
        "scope": "official training period only; frozen Fold A/B query windows",
        "control": "uniform_control",
        "treatment": "duration_conditioned",
        "equal_budget": True,
        "pairs_per_epoch": pairs_per_epoch,
        "epochs": epochs,
        "seeds": list(seeds),
        "folds": folds,
        "mean_duration_minus_uniform": mean_deltas,
        "worst_duration_minus_uniform": worst_deltas,
        "all_primary_deltas_positive": bool(worst_deltas["primary"] > 0.0),
        "official_relative": args.official_relative,
        "official_relative_comparison": official_relative_result,
        "wall_seconds": time.monotonic() - started,
    }
    if args.summary_only:
        compact_folds = [
            {
                "fold": result["fold"],
                "seed": result["seed"],
                "uniform_control": result["arms"]["uniform_control"]["metrics"],
                "duration_conditioned": result["arms"]["duration_conditioned"]["metrics"],
                "official_fm": result.get("official_fm", {}).get("metrics"),
                "duration_minus_official_fm": result.get("duration_minus_official_fm"),
                "duration_minus_uniform": result["duration_minus_uniform"],
                "frozen_duration_official_fusion": result.get(
                    "frozen_duration_official_fusion"
                ),
            }
            for result in folds
        ]
        output = {
            "schema_version": output["schema_version"],
            "status": output["status"],
            "pairs_per_epoch": output["pairs_per_epoch"],
            "epochs": output["epochs"],
            "seeds": output["seeds"],
            "official_relative": output["official_relative"],
            "folds": compact_folds,
            "mean_duration_minus_uniform": output["mean_duration_minus_uniform"],
            "worst_duration_minus_uniform": output["worst_duration_minus_uniform"],
            "official_relative_comparison": output["official_relative_comparison"],
            "wall_seconds": output["wall_seconds"],
        }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()

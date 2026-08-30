"""Train-only pilot for auxiliary-supervision ranking members.

The pilot never scores public-validation or final-period rows.  It fits the protected
native-categorical ranker to training-only auxiliary outcomes on the two rolling-origin folds,
then asks whether each auxiliary ranking adds incremental signal to the strongest previously
recorded train-fold portfolio.  The selection grid is deliberately tiny and predeclared.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "candidate_seed"))

from reference_categorical_ranker import (  # noqa: E402
    reference_categorical_ranker_scores,
    train_reference_categorical_ranker,
)

from kuairand_agent.baselines.fold_controls import build_fold_scoring_context  # noqa: E402
from kuairand_agent.campaign.full_campaign import (  # noqa: E402
    encode_numeric_user_groups,
    prepare_campaign_data_plane,
)
from kuairand_agent.campaign.pure_features import build_pure_feature_pair  # noqa: E402
from kuairand_agent.candidates.fusion import normalize_within_user_percentiles  # noqa: E402
from kuairand_agent.data.canonical import TrainingTargets, load_canonical_dataset  # noqa: E402
from kuairand_agent.data.capabilities import DataPhase  # noqa: E402

ATTEMPT_14_OBJECTS = (
    ROOT / "runs" / "improvement-20260830-attempt-14" / "artifacts" / "objects" / "sha256"
)
ARTIFACTS: Final = {
    "A": {
        "pointwise": "0f4d7e0357e9a2d395beba4774fb3868e360fbe9a68fe8e4c327552242c1cec4",
        "control": "1a3c8a19b200eac9d10c6d59e1f42aaf5ba9773498106516fa9c15798ce279ba",
    },
    "B": {
        "pointwise": "8eaf340138c49f97936a8704579e2781fd20657b49c337ae5467372cf92b7e99",
        "control": "11be6c8074de7dc3bc0d0b49ceb8c1a4d3565699416f8f68bb9a13921d2fd19c",
    },
}
BASE_WEIGHTS: Final = (0.15, 0.30, 0.55)
AUXILIARY_WEIGHT_GRID: Final = (0.0, 0.05, 0.10, 0.15, 0.20)
SUPPORTED_TARGETS: Final = ("is_click", "is_profile_enter")


def _object_path(digest: str) -> Path:
    return ATTEMPT_14_OBJECTS / digest[:2] / digest


def _metrics(score: object) -> dict[str, float]:
    return {
        "GAUC": float(score.gauc),
        "nDCG@5": float(score.ndcg_at_5),
        "primary": float(score.primary),
    }


def _normalized(
    users: tuple[str, ...],
    videos: tuple[str, ...],
    scores: np.ndarray,
) -> np.ndarray:
    return normalize_within_user_percentiles(
        users,
        videos,
        scores,
        phase=DataPhase.INNER_VALID,
    ).scores


def _base_portfolio(
    fold_name: str,
    users: tuple[str, ...],
    videos: tuple[str, ...],
) -> np.ndarray:
    pointwise = _normalized(
        users,
        videos,
        np.load(_object_path(ARTIFACTS[fold_name]["pointwise"]), allow_pickle=False),
    )
    video_type = _normalized(
        users,
        videos,
        np.load(
            ROOT
            / "runs"
            / "diagnostics"
            / f"video_type_fold_{fold_name.lower()}_generated.npy",
            allow_pickle=False,
        ),
    )
    control = _normalized(
        users,
        videos,
        np.load(_object_path(ARTIFACTS[fold_name]["control"]), allow_pickle=False),
    )
    return np.ascontiguousarray(
        BASE_WEIGHTS[0] * pointwise
        + BASE_WEIGHTS[1] * video_type
        + BASE_WEIGHTS[2] * control,
        dtype=np.float64,
    )


def _fold_result(
    *,
    fold_name: str,
    data: object,
    dataset_digest: str,
    target_name: str,
    full_target: tuple[object, ...],
    selected_auxiliary_weight: float | None,
) -> tuple[dict[str, object], float]:
    fold = data.fold_a if fold_name == "A" else data.fold_b
    pair = build_pure_feature_pair(
        prefix_inputs=fold.prefix_inputs,
        prefix_labels=fold.prefix_labels,
        prefix_click_labels=fold.prefix_click_labels,
        prefix_watch_progress=fold.prefix_watch_progress,
        query_inputs=fold.query_inputs,
        dataset_digest=dataset_digest,
        split_role=f"research_auxiliary_{target_name}_fold_{fold_name}",
        builder_source_digest=hashlib.sha256(
            (ROOT / "src" / "kuairand_agent" / "campaign" / "pure_features.py").read_bytes()
        ).hexdigest(),
        cache_dir=ROOT / "runs" / "diagnostics" / "feature-cache-v7",
    )
    auxiliary_targets = np.fromiter(
        (int(full_target[index]) for index in fold.fold.train_positions),
        dtype=np.float64,
        count=len(fold.fold.train_positions),
    )
    groups = encode_numeric_user_groups(fold.prefix_inputs.user_id)
    state = train_reference_categorical_ranker(
        pair.prefix.values,
        auxiliary_targets,
        groups,
    )
    raw_auxiliary = reference_categorical_ranker_scores(pair.query.values, state)
    users = fold.query_inputs.user_id
    videos = fold.query_inputs.video_id
    auxiliary = _normalized(users, videos, raw_auxiliary)
    base = _base_portfolio(fold_name, users, videos)
    scorer = build_fold_scoring_context(
        ROOT / "kuairand-starter-kit",
        fold_name,
        fold.fold.digest,
        fold.query_inputs,
        fold.query_labels,
    )
    base_metrics = _metrics(scorer(base))
    auxiliary_metrics = _metrics(scorer(auxiliary))
    weights = (
        AUXILIARY_WEIGHT_GRID
        if selected_auxiliary_weight is None
        else (selected_auxiliary_weight,)
    )
    variants: list[dict[str, object]] = []
    for weight in weights:
        predictions = np.ascontiguousarray(
            weight * auxiliary + (1.0 - weight) * base,
            dtype=np.float64,
        )
        metrics = _metrics(scorer(predictions))
        variants.append(
            {
                "auxiliary_weight": weight,
                "base_weight": 1.0 - weight,
                "metrics": metrics,
                "primary_delta_to_base": metrics["primary"] - base_metrics["primary"],
            }
        )
    selected = max(
        variants,
        key=lambda item: (
            float(item["metrics"]["primary"]),
            float(item["metrics"]["nDCG@5"]),
            -float(item["auxiliary_weight"]),
        ),
    )
    result = {
        "fold": fold_name,
        "target": target_name,
        "prefix_rows": int(pair.prefix.row_count),
        "query_rows": int(pair.query.row_count),
        "auxiliary_positive_rate": float(np.mean(auxiliary_targets)),
        "base_metrics": base_metrics,
        "auxiliary_only_metrics": auxiliary_metrics,
        "selected": selected,
        "variants": variants,
    }
    del pair, state, raw_auxiliary, auxiliary, base, auxiliary_targets, groups
    gc.collect()
    return result, float(selected["auxiliary_weight"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        action="append",
        choices=SUPPORTED_TARGETS,
        dest="targets",
        help="Training-only auxiliary target to test; repeat to test both.",
    )
    arguments = parser.parse_args()
    targets = tuple(arguments.targets or SUPPORTED_TARGETS)
    started = time.monotonic()
    dataset = load_canonical_dataset(ROOT / ".data" / "KuaiRand-Pure" / "data")
    if not isinstance(dataset.train.targets, TrainingTargets):
        raise RuntimeError("official training targets are unavailable")
    full_targets = {name: dataset.train.targets.column(name) for name in targets}
    data = prepare_campaign_data_plane(dataset, expected_dataset_digest=dataset.digest)
    dataset_digest = dataset.digest
    results: list[dict[str, object]] = []
    for target_name in targets:
        fold_b, selected_weight = _fold_result(
            fold_name="B",
            data=data,
            dataset_digest=dataset_digest,
            target_name=target_name,
            full_target=full_targets[target_name],
            selected_auxiliary_weight=None,
        )
        print(json.dumps({"stage": "selection", "result": fold_b}, sort_keys=True), flush=True)
        if selected_weight == 0.0:
            results.append(
                {
                    "target": target_name,
                    "status": "falsified_on_selection_fold",
                    "selected_auxiliary_weight_on_fold_b": selected_weight,
                    "fold_b": fold_b,
                    "frozen_fold_a": None,
                    "minimum_incremental_primary_delta": 0.0,
                }
            )
            continue
        fold_a, _ = _fold_result(
            fold_name="A",
            data=data,
            dataset_digest=dataset_digest,
            target_name=target_name,
            full_target=full_targets[target_name],
            selected_auxiliary_weight=selected_weight,
        )
        print(json.dumps({"stage": "confirmation", "result": fold_a}, sort_keys=True), flush=True)
        results.append(
            {
                "target": target_name,
                "status": (
                    "confirmed"
                    if float(fold_a["selected"]["primary_delta_to_base"]) > 0
                    else "falsified_on_confirmation_fold"
                ),
                "selected_auxiliary_weight_on_fold_b": selected_weight,
                "fold_b": fold_b,
                "frozen_fold_a": fold_a,
                "minimum_incremental_primary_delta": min(
                    float(fold_b["selected"]["primary_delta_to_base"]),
                    float(fold_a["selected"]["primary_delta_to_base"]),
                ),
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "official training period only",
                "selection_fold": "B",
                "confirmation_fold": "A",
                "base_portfolio_weights": {
                    "pointwise": BASE_WEIGHTS[0],
                    "video_type": BASE_WEIGHTS[1],
                    "official_control": BASE_WEIGHTS[2],
                },
                "auxiliary_weight_grid": list(AUXILIARY_WEIGHT_GRID),
                "results": results,
                "public_validation_used": False,
                "outer_query_used": False,
                "final_outcomes_read": False,
                "wall_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

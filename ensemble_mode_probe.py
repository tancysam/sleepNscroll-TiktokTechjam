"""Does the seed-ensemble gain need within-user rank averaging, or is raw averaging enough?

This decides whether a candidate can ever reproduce the measured five-seed gain. The controller
holds user ids and can rank normalise within user before averaging. A candidate cannot: its
prediction request carries the feature matrix and the checkpoint and nothing else, so it must
average raw scores. Run 17 tested exactly that and came out flat.

Scores the same five qualified official FM seeds three ways on public validation: each alone, the
raw score mean, and the within-user rank mean.

Offline read only. Repository root, outside the hash_source_tree slice.

Run:  python3 ensemble_mode_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from kuairand_agent.candidates.fusion import normalize_within_user_percentiles
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import load_canonical_dataset
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

_SEEDS = (0, 1, 2, 3, 4)
_QUALIFICATION = Path("runs/maki-qualification")


def main(data_dir: str, starter_dir: str) -> int:
    dataset = load_canonical_dataset(Path(data_dir))
    valid = dataset.split(SplitName.VALID)
    assert valid.targets is not None
    labels = np.asarray(valid.targets.reveal_for_scorer(), dtype=np.int8)

    split = SplitIdentity(
        name="outer_valid",
        token="ensemble-mode-probe-v1",
        expected_count=len(valid.inputs),
    )
    alignment = Alignment.from_ids(
        split=split,
        user_ids=valid.inputs.user_id,
        video_ids=valid.inputs.video_id,
    )
    scorer = ProtectedScorer(starter_dir=Path(starter_dir), trusted_alignment=alignment)

    def primary(scores: np.ndarray) -> float:
        return float(
            scorer.score_with_encoded_labels(
                alignment=alignment,
                split=split,
                labels=labels,
                scores=np.ascontiguousarray(scores, dtype=np.float64),
            ).primary
        )

    raw = {
        seed: np.ascontiguousarray(
            np.load(_QUALIFICATION / "fm" / f"seed-{seed}" / "validation-predictions.npy"),
            dtype=np.float64,
        )
        for seed in _SEEDS
    }
    ranked = {
        seed: normalize_within_user_percentiles(
            valid.inputs.user_id,
            valid.inputs.video_id,
            values,
            phase=DataPhase.OUTER_VALID,
        ).scores
        for seed, values in raw.items()
    }

    singles = {seed: primary(values) for seed, values in raw.items()}
    best_single = max(singles.values())
    raw_mean = primary(np.mean([raw[s] for s in _SEEDS], axis=0))
    rank_mean = primary(np.mean([ranked[s] for s in _SEEDS], axis=0))

    print("single seeds")
    for seed in _SEEDS:
        print(f"    seed {seed}                    {singles[seed]:.10f}")
    print(f"    best single                {best_single:.10f}")
    print()
    print("five seed ensembles")
    print(f"    raw score mean             {raw_mean:.10f}   {raw_mean - best_single:+.10f}")
    print(f"    within-user rank mean      {rank_mean:.10f}   {rank_mean - best_single:+.10f}")
    print()
    print(f"    rank mean minus raw mean   {rank_mean - raw_mean:+.10f}")
    print(f"    in sigma at sigma 0.0008   {(rank_mean - raw_mean) / 8e-4:+.2f}")
    print()
    print("A candidate can only do the raw-score row: predict_scores receives no user_groups,")
    print("so within-user rank normalisation is unavailable to it at prediction time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data",
            sys.argv[2] if len(sys.argv) > 2 else "kuairand-starter-kit",
        )
    )

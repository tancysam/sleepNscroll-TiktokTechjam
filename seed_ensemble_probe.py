"""Measure how much public validation headroom is pure FM seed variance.

The campaign's incumbent is a SINGLE official FM seed. Seed to seed sigma is 0.0008, which is
large relative to every margin this project has chased. If rank averaging the five already
qualified seeds beats the best single seed by a real margin, that headroom is sitting on disk and
has never been collected.

This reads only artifacts that already exist. It trains nothing, launches no campaign, and costs
nothing. It reports numbers; it does not change what any bundle ships.

Offline read only diagnostic. Lives at the repository root deliberately: ``hash_source_tree``
covers ``src``, ``configs``, ``scripts``, ``candidate_seed`` and ``candidate_templates`` plus a
fixed list of root files, so a probe here cannot strand a running campaign.

Run:  python3 seed_ensemble_probe.py
"""

from __future__ import annotations

import itertools
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
_ORGANIZER_BASELINE = 0.6016
_FALLBACK_OUTER_MEAN = 0.6014402508735657


def main(data_dir: str, starter_dir: str) -> int:
    dataset = load_canonical_dataset(Path(data_dir))
    valid = dataset.split(SplitName.VALID)
    assert valid.targets is not None
    labels = np.asarray(valid.targets.reveal_for_scorer(), dtype=np.int8)

    split = SplitIdentity(
        name="outer_valid",
        token="seed-ensemble-probe-v1",
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

    raw: dict[int, np.ndarray] = {}
    for seed in _SEEDS:
        path = _QUALIFICATION / "fm" / f"seed-{seed}" / "validation-predictions.npy"
        raw[seed] = np.ascontiguousarray(np.load(path), dtype=np.float64)
        if raw[seed].shape != (len(valid.inputs),):
            raise SystemExit(f"seed {seed} predictions do not match the validation split")

    # Rank normalise once per seed. Averaging raw FM logits across seeds is not meaningful because
    # each seed has its own scale; averaging within user percentiles is scale free and is exactly
    # what the controller's own fusion does.
    ranked = {
        seed: normalize_within_user_percentiles(
            valid.inputs.user_id,
            valid.inputs.video_id,
            values,
            phase=DataPhase.OUTER_VALID,
        ).scores
        for seed, values in raw.items()
    }

    print("single seeds, public validation primary")
    singles = {}
    for seed in _SEEDS:
        singles[seed] = primary(raw[seed])
        print(f"    seed {seed}   {singles[seed]:.10f}")
    best_single = max(singles.values())
    print(f"    best single      {best_single:.10f}")
    print(f"    mean of singles  {sum(singles.values()) / len(singles):.10f}")

    print("\nrank mean ensembles, public validation primary")
    for size in (2, 3, 5):
        best = None
        for combo in itertools.combinations(_SEEDS, size):
            score = primary(np.mean([ranked[seed] for seed in combo], axis=0))
            if best is None or score > best[0]:
                best = (score, combo)
            if size == 5:
                print(f"    all five {combo}   {score:.10f}")
        assert best is not None
        if size != 5:
            print(f"    best {size} of 5 {best[1]}   {best[0]:.10f}")

    full = primary(np.mean([ranked[seed] for seed in _SEEDS], axis=0))
    print("\ncomparisons for the five seed rank ensemble")
    print(f"    vs best single seed        {full - best_single:+.10f}")
    print(f"    vs organizer baseline      {full - _ORGANIZER_BASELINE:+.10f}")
    print(f"    vs campaign incumbent      {full - _FALLBACK_OUTER_MEAN:+.10f}")
    print(f"    in sigma at sigma 0.0008   {(full - _FALLBACK_OUTER_MEAN) / 8e-4:+.2f}")
    print("\nNote: best-of-N over seed subsets is selected on the split it is reported on.")
    print("Only the all-five ensemble is free of that selection effect. Report that one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data",
            sys.argv[2] if len(sys.argv) > 2 else "kuairand-starter-kit",
        )
    )

"""Does rank-ensembling the AGENT's own candidates reproduce the official FM ensemble's gain?

`ensemble_mode_probe.py` measured the mechanism on five qualified organizer FM seeds: averaging
their within-user rank percentiles scores 0.6026034 against 0.6020371 for the best single seed,
+0.0005664, while averaging raw scores is worth +0.0000772. That result ships as a submission, but
its provenance is honest and unflattering -- it ensembles the organizer's own baseline with itself
and records `agent_generated: false`.

The agent has since finalized five campaigns, each a separate honestly-converged search producing
its own model and its own validation predictions. Ensembling THOSE applies the same measured
mechanism to the agent's own output. It costs no new training, because the predictions already
exist inside the retained bundles.

Two things this can distinguish, and only one is good news:

- If the gain is comparable to the FM ensemble's, the campaigns are producing genuinely different
  orderings and their errors partly cancel. That is a real result and it is the agent's.
- If the gain is near zero, the five campaigns converged to nearly the same model. Their +0.0003
  deltas would then be five measurements of one thing rather than five independent ones -- which
  would matter for how the campaign record is reported, not only for the score.

Read only. Repository root, outside the hash_source_tree slice.

Run:  python3 agent_ensemble_probe.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from kuairand_agent.candidates.fusion import normalize_within_user_percentiles
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import load_canonical_dataset
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

#: One finalized bundle per completed campaign, in the order the campaigns ran.
_CAMPAIGNS: tuple[str, ...] = (
    "sol6-20260831T090026Z",
    "sol7-20260831T124700Z",
    "sol7-20260831T131245Z",
    "sol8-20260831T161102Z",
    "sol8-20260831T164106Z",
)
#: The official FM's own five-seed rank ensemble, for reference (ensemble_mode_probe.py).
_FM_ENSEMBLE_PRIMARY = 0.6026034355
_FM_MEAN_PRIMARY = 0.6014403


def main(data_dir: str, starter_dir: str, runs_root: str) -> int:
    dataset = load_canonical_dataset(Path(data_dir))
    valid = dataset.split(SplitName.VALID)
    assert valid.targets is not None
    labels = np.asarray(valid.targets.reveal_for_scorer(), dtype=np.int8)

    split = SplitIdentity(
        name="outer_valid",
        token="agent-ensemble-probe-v1",
        expected_count=len(valid.inputs),
    )
    alignment = Alignment.from_ids(
        split=split, user_ids=valid.inputs.user_id, video_ids=valid.inputs.video_id
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

    raw: dict[str, np.ndarray] = {}
    for name in _CAMPAIGNS:
        path = Path(runs_root) / name / "final" / "replay" / "validation-predictions.npy"
        raw[name] = np.ascontiguousarray(np.load(path), dtype=np.float64)
    # Percentiles, not raw scores. 86% of the ensembling effect is the normalisation, because the
    # metric reads a within-user ordering and independently trained members share no score scale.
    ranked = {
        name: normalize_within_user_percentiles(
            valid.inputs.user_id, valid.inputs.video_id, values, phase=DataPhase.OUTER_VALID
        ).scores
        for name, values in raw.items()
    }

    singles = {name: primary(values) for name, values in raw.items()}
    best_single = max(singles.values())
    print("individual campaigns")
    for name in _CAMPAIGNS:
        print(f"    {name:26s} {singles[name]:.7f}   {singles[name] - _FM_MEAN_PRIMARY:+.7f}")
    print(f"    {'best single':26s} {best_single:.7f}   {best_single - _FM_MEAN_PRIMARY:+.7f}")
    print()

    members = [ranked[name] for name in _CAMPAIGNS]
    raw_mean = primary(np.mean([raw[name] for name in _CAMPAIGNS], axis=0))
    rank_mean = primary(np.mean(members, axis=0))
    print("five-campaign ensembles")
    print(f"    {'raw score mean':26s} {raw_mean:.7f}   {raw_mean - _FM_MEAN_PRIMARY:+.7f}")
    print(
        f"    {'within-user rank mean':26s} {rank_mean:.7f}   {rank_mean - _FM_MEAN_PRIMARY:+.7f}"
    )
    print()
    print(f"    gain over the best single campaign     {rank_mean - best_single:+.7f}")
    print(
        f"    official FM five-seed ensemble         {_FM_ENSEMBLE_PRIMARY:.7f}   "
        f"{_FM_ENSEMBLE_PRIMARY - _FM_MEAN_PRIMARY:+.7f}"
    )
    print()

    # How much of the gain is diversity rather than luck: every subset size, best and mean.
    print("subset sizes, within-user rank mean")
    for size in range(2, len(_CAMPAIGNS) + 1):
        scores = [
            primary(np.mean([ranked[name] for name in subset], axis=0))
            for subset in combinations(_CAMPAIGNS, size)
        ]
        print(
            f"    n={size}   mean {np.mean(scores):.7f}   best {max(scores):.7f}   "
            f"worst {min(scores):.7f}"
        )
    print()
    correlations = [
        float(np.corrcoef(ranked[a], ranked[b])[0, 1]) for a, b in combinations(_CAMPAIGNS, 2)
    ]
    print(
        f"pairwise rank correlation between campaigns: "
        f"min {min(correlations):.4f}  mean {np.mean(correlations):.4f}  "
        f"max {max(correlations):.4f}"
    )
    if np.mean(correlations) > 0.99:
        print("    Near-identical orderings: the campaigns converged to one model, so their")
        print("    individual deltas are five measurements of the same thing, not five results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data",
            sys.argv[2] if len(sys.argv) > 2 else "kuairand-starter-kit",
            sys.argv[3] if len(sys.argv) > 3 else "runs",
        )
    )

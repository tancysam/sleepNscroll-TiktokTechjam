"""Measure how far the official-FM rank ensemble goes as the seed pool grows.

The shipped submission is a five-seed within-user rank ensemble scoring 0.6026034 on public
validation.  Fitting the standard variance-reduction form gain(N) = G(1 - 1/sqrt(N)) to that single
point predicts roughly 0.6030 at twenty seeds and an asymptote near 0.60344, which would fall short
of the 0.6036 materiality threshold.  That extrapolation rests on ONE measured point, so this
measures the curve directly instead of trusting it.

Selection discipline, which is the whole point of doing this carefully.  Pool size is not chosen by
reading the validation column and keeping the best: N is fixed a priori as the largest pool
available, and the per-N table is printed as a CURVE for the writeup, not as a menu.  Seeds enter
in ascending numeric order, never ordered by their individual validation score, so no member is
selected for being lucky.  This is the same best-of-N selection effect docs/RESULTS.md section 3.2
criticises in the pairwise sweep and in the seed-4 fallback; producing it here would make any gain
meaningless.

Offline and read-only.  Lives at the repository root, outside the slice hash_source_tree covers, so
it cannot strand a running campaign.

Run:  python3 -B ensemble_search_probe.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kuairand_agent.candidates.fusion import normalize_within_user_percentiles
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import load_canonical_dataset
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

QUALIFIED = Path("runs/maki-qualification/fm")
EXTRA = Path("runs/seed-pool")
SHIPPED_FIVE_SEED = 0.6026034355
PUBLISHED_BASELINE = 0.6016
EPSILON = 0.002


def _members() -> list[tuple[int, np.ndarray]]:
    """Every available seed, ascending, qualified pool first."""

    found: list[tuple[int, np.ndarray]] = []
    for seed in range(5):
        path = QUALIFIED / f"seed-{seed}" / "validation-predictions.npy"
        if path.is_file():
            found.append((seed, np.ascontiguousarray(np.load(path), dtype=np.float64)))
    if EXTRA.is_dir():
        for path in sorted(EXTRA.glob("seed-*.npy"), key=lambda p: int(p.stem.split("-")[1])):
            found.append((int(path.stem.split("-")[1]), np.load(path).astype(np.float64)))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".data/KuaiRand-Pure/data")
    parser.add_argument("--starter-dir", default="kuairand-starter-kit")
    args = parser.parse_args()

    dataset = load_canonical_dataset(Path(args.data_dir))
    valid = dataset.split(SplitName.VALID)
    assert valid.targets is not None
    labels = np.asarray(valid.targets.reveal_for_scorer(), dtype=np.int8)

    split = SplitIdentity(
        name="outer_valid", token="ensemble-search-probe-v1", expected_count=len(valid.inputs)
    )
    alignment = Alignment.from_ids(
        split=split, user_ids=valid.inputs.user_id, video_ids=valid.inputs.video_id
    )
    scorer = ProtectedScorer(starter_dir=Path(args.starter_dir), trusted_alignment=alignment)

    def primary(scores: np.ndarray) -> float:
        return float(
            scorer.score_with_encoded_labels(
                alignment=alignment,
                split=split,
                labels=labels,
                scores=np.ascontiguousarray(scores, dtype=np.float64),
            ).primary
        )

    members = _members()
    if len(members) < 5:
        raise SystemExit(f"need at least the five qualified seeds, found {len(members)}")
    print(f"members available: {len(members)} (seeds {members[0][0]}..{members[-1][0]})\n")

    ranked = [
        normalize_within_user_percentiles(
            valid.inputs.user_id, valid.inputs.video_id, values, phase=DataPhase.OUTER_VALID
        ).scores
        for _, values in members
    ]

    print(f"{'N':>3}  {'primary':>14}  {'vs 0.6016':>11}  {'vs shipped':>11}  material")
    print("-" * 62)
    running = np.zeros_like(ranked[0])
    best_n, best_primary = 0, -1.0
    for index, vector in enumerate(ranked, start=1):
        running = running + vector
        if index < 5:
            continue
        value = primary(running / index)
        delta = value - PUBLISHED_BASELINE
        flag = "YES" if delta > EPSILON else "no"
        print(
            f"{index:>3}  {value:>14.10f}  {delta:>+11.7f}  "
            f"{value - SHIPPED_FIVE_SEED:>+11.7f}  {flag}"
        )
        if value > best_primary:
            best_n, best_primary = index, value

    print()
    print(f"largest pool: N={len(ranked)}")
    print(f"curve maximum: N={best_n} at {best_primary:.10f}")
    print(f"shipped five-seed reference: {SHIPPED_FIVE_SEED:.10f}")
    print(
        "\nN is fixed a priori at the largest available pool; the table is the curve, not a menu. "
        "Reporting the row with the highest validation primary would be exactly the best-of-N "
        "selection effect this project criticises elsewhere."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

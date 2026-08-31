"""Does the seed-ensemble gain need within-user rank averaging, and can a candidate do it?

Two questions, and the second one was answered wrongly for four campaigns.

The controller holds true user ids and can rank normalise within user before averaging. The
original reading was that a candidate cannot, because its prediction request carries the feature
matrix and the checkpoint and nothing else -- so it must average raw scores, which run 17 tested
and found flat.

That reading is wrong. ``user_id_code`` is a COLUMN OF THE FEATURE MATRIX (pure_features.py
ID_CODE_FEATURE_NAMES), so it arrives at prediction time like any other feature. A candidate can
group its own rows by that code and rank normalise inside each group. The only rows it cannot
group are the ones whose user was absent from the training fold: they all collapse onto the single
trailing unknown slot and become one pseudo-group of unrelated users.

This probe measures whether the gain survives that collapse. Fallback for the unknown pool: rank
those rows among THEMSELVES rather than pooling them with a real slate. Any monotone map of the
raw score preserves each hidden user's own ordering, which is all the within-user metrics read.

Scores the five qualified official FM seeds: each alone, the raw score mean, the within-user rank
mean on true user ids (the controller's ceiling), and the same on user_id_code (what a candidate
can actually reach).

Offline read only. Repository root, outside the hash_source_tree slice.

Run:  python3 ensemble_mode_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from kuairand_agent.campaign.pure_features import fit_code_vocabulary
from kuairand_agent.candidates.fusion import normalize_within_user_percentiles
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import load_canonical_dataset
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

_SEEDS = (0, 1, 2, 3, 4)
_QUALIFICATION = Path("runs/wp3-official-qualification")
#: Position of user_id_code within ID_CODE_FEATURE_NAMES.
_USER_CODE_FIELD = 0


def _user_codes(train_inputs: object, valid_inputs: object) -> tuple[np.ndarray, int]:
    """Encode validation user ids through a vocabulary fitted on training rows only.

    Mirrors ``pure_features._code_matrix``: a value absent from the fitted table encodes to
    ``len(table)``, the trailing unknown slot. Fitting on prefix rows only is the same
    frozen-query discipline the real bundle uses, so the unknown rate here is the real one.
    """

    table = fit_code_vocabulary(train_inputs)[_USER_CODE_FIELD]
    unknown = len(table)
    codes = np.asarray(
        [table.get(str(value), unknown) for value in valid_inputs.user_id],
        dtype=np.int64,
    )
    return codes, unknown


def _percentiles_within(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rank ``values`` to ``[0, 1]`` inside each group, ties sharing their average rank.

    Matches ``normalize_within_user_percentiles`` semantics: a singleton group takes the
    neutral 0.5, so a lone row never contributes a spurious extreme.
    """

    out = np.empty(len(values), dtype=np.float64)
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    for block in np.split(order, boundaries):
        if len(block) == 1:
            out[block[0]] = 0.5
            continue
        ranks = _average_ranks(values[block])
        out[block] = (ranks - 1.0) / (len(block) - 1.0)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ordered = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + index + 1) / 2.0
            start = index
    return ranks


def _candidate_percentiles(codes: np.ndarray, unknown: int, values: np.ndarray) -> np.ndarray:
    """What a candidate can compute in ``predict_scores`` from the feature matrix alone."""

    scores = _percentiles_within(codes, values)
    pool = codes == unknown
    if pool.any():
        # One pseudo-group of unrelated users. Ranking it against itself is still monotone in the
        # raw score, so every hidden user's own ordering survives; ranking it as one slate would
        # instead compare users to each other, which the metric never does.
        scores[pool] = _percentiles_within(np.zeros(int(pool.sum()), dtype=np.int64), values[pool])
    return scores


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

    train = dataset.split(SplitName.TRAIN)
    codes, unknown = _user_codes(train.inputs, valid.inputs)
    unknown_rows = int((codes == unknown).sum())
    coded = {seed: _candidate_percentiles(codes, unknown, values) for seed, values in raw.items()}

    singles = {seed: primary(values) for seed, values in raw.items()}
    best_single = max(singles.values())
    raw_mean = primary(np.mean([raw[s] for s in _SEEDS], axis=0))
    rank_mean = primary(np.mean([ranked[s] for s in _SEEDS], axis=0))
    code_mean = primary(np.mean([coded[s] for s in _SEEDS], axis=0))

    print("single seeds")
    for seed in _SEEDS:
        print(f"    seed {seed}                    {singles[seed]:.10f}")
    print(f"    best single                {best_single:.10f}")
    print()
    print("five seed ensembles")
    print(f"    raw score mean             {raw_mean:.10f}   {raw_mean - best_single:+.10f}")
    print(f"    rank mean, true user_id    {rank_mean:.10f}   {rank_mean - best_single:+.10f}")
    print(f"    rank mean, user_id_code    {code_mean:.10f}   {code_mean - best_single:+.10f}")
    print()
    print(
        f"    unknown-slot rows          {unknown_rows:,} of {len(codes):,}"
        f"   ({unknown_rows / len(codes):.2%})"
    )
    print(f"    code mean minus raw mean   {code_mean - raw_mean:+.10f}")
    print(f"    cost of the unknown pool   {code_mean - rank_mean:+.10f}")
    print(f"    fraction of ceiling kept   {(code_mean - raw_mean) / (rank_mean - raw_mean):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data",
            sys.argv[2] if len(sys.argv) > 2 else "kuairand-starter-kit",
        )
    )

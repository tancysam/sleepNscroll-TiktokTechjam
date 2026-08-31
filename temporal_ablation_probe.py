"""Does adding temporal columns to the parent move the Fold B score? The actual ablation.

``temporal_signal_probe.py`` answered a correlation question and its verdict overreached in both
directions before settling. Correlation against the incumbent column is not the incremental test
either: the bundle already owns ``video_past_rate``'s signal, and what matters about a new feature
is what it adds ORTHOGONALLY to what the model already fits. A within-user r of 0.035 that is
independent of the current signal is worth roughly +0.001 GAUC, about +0.0005 primary -- the same
magnitude as the rank-ensembling gain, which was judged real.

So this fits the actual parent, twice, on the actual Fold B split, and scores both with the
protected organizer scorer. No proxies.

    baseline   the 76-column bundle exactly as candidates receive it
    treatment  the same 76 plus three impression-side temporal columns

The three are properties of the impression rather than of its outcome, so they are inside the
causal cutoff: position within the user's session, log time since their previous impression, and
the sine of the time of day. Fold B is the screen every candidate faces: prefix 2022-04-08..18,
query 04-19..21.

Read only, training rows only. Repository root, outside the hash_source_tree slice.

Run:  python3 temporal_ablation_probe.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from kuairand_agent.campaign.pure_features import build_pure_feature_pair, subset_canonical_inputs
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import load_canonical_dataset
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

sys.path.insert(0, str(Path(__file__).parent / "candidate_seed"))

#: Fold B as the campaign defines it in ``campaign.scientific.FOLD_DATES``.
_PREFIX_DATES = (20220408, 20220418)
_QUERY_DATES = (20220419, 20220421)
_SESSION_GAP_MS = 30 * 60 * 1000
_BUILDER_DIGEST = "0" * 64


def _temporal_columns(inputs: object, order: np.ndarray) -> np.ndarray:
    """Three impression-side temporal columns, in the canonical row order of ``inputs``.

    Session position and gap are per user in time order, so the arrays are filled by scattering
    back to canonical positions rather than by sorting the matrix.
    """

    users = np.asarray(inputs.user_id)  # type: ignore[attr-defined]
    times = np.asarray(inputs.time_ms, dtype=np.float64)  # type: ignore[attr-defined]
    position = np.zeros(len(times), dtype=np.float64)
    gap = np.zeros(len(times), dtype=np.float64)
    previous_user: object = None
    previous_time = 0.0
    run = 0
    for index in order:
        if users[index] != previous_user:
            run, elapsed = 0, 0.0
        else:
            elapsed = times[index] - previous_time
            run = 0 if elapsed > _SESSION_GAP_MS else run + 1
        position[index] = float(run)
        gap[index] = math.log1p(max(elapsed, 0.0) / 1000.0)
        previous_user, previous_time = users[index], times[index]
    # Time of day from the millisecond timestamp, as a smooth cyclic term.
    seconds = np.mod(times / 1000.0, 86400.0)
    hour_sin = np.sin(2.0 * np.pi * seconds / 86400.0)
    return np.column_stack((position, gap, hour_sin)).astype(np.float64)


def _order_by_user_then_time(inputs: object) -> np.ndarray:
    users = np.asarray(inputs.user_id)  # type: ignore[attr-defined]
    times = np.asarray(inputs.time_ms, dtype=np.float64)  # type: ignore[attr-defined]
    return np.lexsort((times, users))


def _fit_and_score(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_groups: np.ndarray,
    query_features: np.ndarray,
    scorer: ProtectedScorer,
    alignment: Alignment,
    split: SplitIdentity,
    labels: np.ndarray,
    config: dict[str, object],
) -> float:
    """Fit the trusted parent on the prefix and score its query predictions."""

    import model_impl  # the candidate seed's own scorer, unmodified

    checkpoint = model_impl.train_model(train_features, train_targets, train_groups, config, 0)
    scores = model_impl.predict_scores(query_features, checkpoint)
    return float(
        scorer.score_with_encoded_labels(
            alignment=alignment,
            split=split,
            labels=labels,
            scores=np.ascontiguousarray(scores, dtype=np.float64),
        ).primary
    )


def main(data_dir: str, starter_dir: str) -> int:
    dataset = load_canonical_dataset(Path(data_dir))
    train = dataset.split(SplitName.TRAIN)
    assert train.targets is not None
    dates = np.asarray(train.inputs.date)
    all_labels = np.asarray(train.targets.long_view, dtype=np.int8)

    prefix_rows = np.flatnonzero((dates >= _PREFIX_DATES[0]) & (dates <= _PREFIX_DATES[1]))
    query_rows = np.flatnonzero((dates >= _QUERY_DATES[0]) & (dates <= _QUERY_DATES[1]))
    prefix_inputs = subset_canonical_inputs(train.inputs, prefix_rows.tolist())
    query_inputs = subset_canonical_inputs(train.inputs, query_rows.tolist())
    prefix_labels = [int(value) for value in all_labels[prefix_rows]]
    query_labels = all_labels[query_rows]
    print(f"fold B    prefix {len(prefix_rows):,} rows    query {len(query_rows):,} rows")

    pair = build_pure_feature_pair(
        prefix_inputs=prefix_inputs,
        prefix_labels=prefix_labels,
        query_inputs=query_inputs,
        dataset_digest=dataset.digest,
        split_role="inner_train",
        builder_source_digest=_BUILDER_DIGEST,
        # Without these the bundle collapses from 76 columns to 38: the is_click and is_like
        # aggregate families are built from approved auxiliary outcomes, not from long_view.
        prefix_auxiliary={
            "is_click": [
                int(value) for value in np.asarray(train.targets.column("is_click"))[prefix_rows]
            ],
            "is_like": [
                int(value) for value in np.asarray(train.targets.column("is_like"))[prefix_rows]
            ],
        },
    )
    baseline_train = np.asarray(pair.prefix.values, dtype=np.float64)
    baseline_query = np.asarray(pair.query.values, dtype=np.float64)
    print(f"bundle    {baseline_train.shape[1]} columns\n")

    split = SplitIdentity(
        name="inner_valid",
        token="temporal-ablation-probe-v1",
        expected_count=len(query_rows),
    )
    alignment = Alignment.from_ids(
        split=split,
        user_ids=query_inputs.user_id,
        video_ids=query_inputs.video_id,
    )
    scorer = ProtectedScorer(starter_dir=Path(starter_dir), trusted_alignment=alignment)

    config = {"epochs": 64, "learning_rate": 0.25, "l2": 0.001, "logit_clip": 40.0}
    train_targets = np.asarray(all_labels[prefix_rows], dtype=np.float64)
    train_groups = np.asarray(
        [hash(value) % (2**31) for value in prefix_inputs.user_id], dtype=np.float64
    )

    baseline = _fit_and_score(
        baseline_train,
        train_targets,
        train_groups,
        baseline_query,
        scorer,
        alignment,
        split,
        query_labels,
        config,
    )
    treatment_train = np.hstack(
        (baseline_train, _temporal_columns(prefix_inputs, _order_by_user_then_time(prefix_inputs)))
    )
    treatment_query = np.hstack(
        (baseline_query, _temporal_columns(query_inputs, _order_by_user_then_time(query_inputs)))
    )
    treatment = _fit_and_score(
        treatment_train,
        train_targets,
        train_groups,
        treatment_query,
        scorer,
        alignment,
        split,
        query_labels,
        config,
    )

    delta = treatment - baseline
    print(f"    parent, 76 columns                  {baseline:.7f}")
    print(f"    parent + 3 temporal columns         {treatment:.7f}")
    print(f"    delta                               {delta:+.7f}")
    print()
    # 0.000316 is the measured seed sigma (docs/RESULTS.md 3.4); epsilon is 0.002.
    print(f"    in measured sigma                   {delta / 0.000316:+.2f}")
    if delta <= -0.000316:
        print("\n    VERDICT: NEGATIVE. The columns cost more than they add on this base. A")
        print("    linear scorer cannot use a non-monotone term like session position, so this")
        print("    retires the axis for the current parent and would need re-running against a")
        print("    non-linear one before the axis could be called dead in general.")
    elif delta < 0.000316:
        print("\n    VERDICT: below one sigma. Temporal position is not a lever worth an")
        print("    iteration, and the plateau finding keeps the pillar this was meant to test.")
    elif delta < 0.002:
        print("\n    VERDICT: real but below epsilon, on the order of the rank-ensembling gain.")
        print("    Worth carrying on a competitive base, where increments of this size decide")
        print("    whether a standalone win happens; not worth a scientific iteration alone.")
    else:
        print("\n    VERDICT: clears epsilon on its own. This is the mechanism to build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data",
            sys.argv[2] if len(sys.argv) > 2 else "kuairand-starter-kit",
        )
    )

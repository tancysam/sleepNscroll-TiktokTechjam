"""Does the promoted parent reproduce run 16's measurement? Blocking acceptance test.

``candidate_seed/model_impl.py`` was a bare logistic scorer and is now an identity-code
factorization machine, because every campaign was spending its one shot re-deriving the scoring
structure before its own hypothesis could be measured. Promoting a parent changes what every
future ``parent_fold_b_primary`` MEANS, so it must be verified against the measurement it claims
to reproduce rather than assumed to work.

docs/RESULTS.md 3.3a records run 16 at Fold B standalone 0.5745 against a 0.5754 control -- about
one sigma down, and the best standalone this project has produced. If the rewired parent does not
land within roughly one measured sigma of that, every later verdict would read "versus an
unverified reimplementation", which is the retracted-headline problem rebuilt at the foundation.

Read only, training rows only. Repository root, outside the hash_source_tree slice.

Run:  python3 parent_acceptance_probe.py
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
    model_impl: object,
) -> float:
    """Fit the supplied candidate implementation on the prefix and score its query predictions."""

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


def _load_candidate(directory: Path) -> tuple[object, dict[str, object]]:
    """Import a candidate tree's ``model_impl`` and its own ``config.json``.

    Any candidate directory works, not only ``candidate_seed``: the point of this probe is to run
    the recipe that a recorded measurement actually used, and the best FM-lineage implementations
    survive as complete source under a run's ``generated-source/``.
    """

    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        f"probe_model_impl_{abs(hash(str(directory)))}", directory / "model_impl.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{directory}/model_impl.py is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((directory / "config.json").read_text(encoding="utf-8"))


def evaluate_candidate(
    candidate_dir: Path | str,
    data_dir: str = ".data/KuaiRand-Pure/data",
    starter_dir: str = "kuairand-starter-kit",
) -> float:
    """Fold B standalone primary for one candidate tree, scored by the protected scorer.

    Extracted so `advance_seed.py` gates a cross-campaign promotion with the same harness that
    gates the parent, rather than a second implementation that could drift from it.
    """

    return _fold_b_primary(Path(data_dir), Path(starter_dir), Path(candidate_dir))


def _fold_b_primary(data_dir: Path, starter_dir: Path, candidate_dir: Path) -> float:
    dataset = load_canonical_dataset(data_dir)
    train = dataset.split(SplitName.TRAIN)
    assert train.targets is not None
    dates = np.asarray(train.inputs.date)
    all_labels = np.asarray(train.targets.long_view, dtype=np.int8)
    prefix_rows = np.flatnonzero((dates >= _PREFIX_DATES[0]) & (dates <= _PREFIX_DATES[1]))
    query_rows = np.flatnonzero((dates >= _QUERY_DATES[0]) & (dates <= _QUERY_DATES[1]))
    prefix_inputs = subset_canonical_inputs(train.inputs, prefix_rows.tolist())
    query_inputs = subset_canonical_inputs(train.inputs, query_rows.tolist())
    pair = build_pure_feature_pair(
        prefix_inputs=prefix_inputs,
        prefix_labels=[int(value) for value in all_labels[prefix_rows]],
        query_inputs=query_inputs,
        dataset_digest=dataset.digest,
        split_role="inner_train",
        builder_source_digest=_BUILDER_DIGEST,
        prefix_auxiliary={
            name: [int(v) for v in np.asarray(train.targets.column(name))[prefix_rows]]
            for name in ("is_click", "is_like")
        },
    )
    split = SplitIdentity(name="inner_valid", token="seed-gate-v1", expected_count=len(query_rows))
    alignment = Alignment.from_ids(
        split=split, user_ids=query_inputs.user_id, video_ids=query_inputs.video_id
    )
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)
    model_impl, config = _load_candidate(candidate_dir)
    return _fit_and_score(
        np.asarray(pair.prefix.values, dtype=np.float64),
        np.asarray(all_labels[prefix_rows], dtype=np.float64),
        np.asarray([hash(value) % (2**31) for value in prefix_inputs.user_id], dtype=np.float64),
        np.asarray(pair.query.values, dtype=np.float64),
        scorer,
        alignment,
        split,
        all_labels[query_rows],
        config,
        model_impl,
    )


def main(data_dir: str, starter_dir: str, candidate_dir: str = "candidate_seed") -> int:
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

    model_impl, config = _load_candidate(Path(candidate_dir))
    print(f"candidate {candidate_dir}")
    print(f"family    {config.get('candidate_family')}\n")
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
        model_impl,
    )
    # docs/RESULTS.md 3.3a. The control is the official FM's own Fold B primary.
    run_16 = 0.5745312
    control = 0.5754240
    sigma = 0.000316
    print(f"    promoted parent, Fold B standalone  {baseline:.7f}")
    print(f"    run 16, the target                  {run_16:.7f}")
    print(f"    official FM control                 {control:.7f}")
    print(
        f"    parent minus run 16                 {baseline - run_16:+.7f}"
        f"   ({(baseline - run_16) / sigma:+.2f} sigma)"
    )
    print(
        f"    parent minus control                {baseline - control:+.7f}"
        f"   ({(baseline - control) / sigma:+.2f} sigma)"
    )
    print()
    if abs(baseline - run_16) <= sigma:
        print("    ACCEPTED. The promoted parent reproduces the measurement it claims.")
        return 0
    if baseline > run_16:
        print("    ACCEPTED, and stronger than the target. Verify the gain is real before")
        print("    relying on it, but nothing here blocks a campaign.")
        return 0
    print("    REJECTED. The promoted parent does not reproduce run 16 within one sigma.")
    print("    Do not run a campaign on it: every parent_fold_b_primary would then mean")
    print("    'versus an unverified reimplementation'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data",
            sys.argv[2] if len(sys.argv) > 2 else "kuairand-starter-kit",
            sys.argv[3] if len(sys.argv) > 3 else "candidate_seed",
        )
    )

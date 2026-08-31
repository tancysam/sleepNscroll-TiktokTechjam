"""Regenerate the five-seed official-FM rank-ensemble submission, byte for byte.

WHAT THIS IS, STATED PLAINLY. This is a controller-side ensemble of the ORGANIZER'S OWN baseline
with itself. It is not agent-generated, and it is not produced by the autonomous campaign. It is
included because it is the only result this project has that is measurably above the published
baseline, and because it is fully reproducible from artifacts already in the repository.

    five-seed within-user rank ensemble   0.6026034355   public validation
    published organizer baseline          0.6016         +0.0010
    best single seed (seed 4, our shipped fallback)
                                          0.6020370722   +0.0006

The gain is a variance-reduction effect, not a favourable draw: it uses all five qualified seeds,
so unlike a best-of-N result it carries no selection effect, and it beats every individual seed
including the luckiest. It is nonetheless BELOW this project's own materiality threshold of
epsilon = 0.002, and it must not be described as a material improvement.

Members are combined on WITHIN-USER RANK PERCENTILES, never on raw scores. Measured with
ensemble_mode_probe.py: rank averaging scores 0.6026034 while raw score averaging scores
0.6021143, so raw averaging recovers about one seventh of the gain. The metric is a within-user
ordering and the members do not share a score scale.

No new training happens here. Every member is an already-qualified organizer FM run from
runs/maki-qualification, each checkpoint is verified against the digest recorded at qualification,
and inference runs through the hash-pinned organizer source rather than a reimplementation.

Offline and read-only with respect to the repository: it writes only the output directory it is
given. It lives at the repository root, outside the slice hash_source_tree covers, so it cannot
strand a running campaign.

Run:
    python3 build_ensemble_submission.py                 # writes ./ensemble-submission/
    python3 build_ensemble_submission.py --out DIR --data-dir DIR --starter-dir DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kuairand_agent.baselines.artifacts import load_checkpoint
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import StarterFMConfig
from kuairand_agent.campaign.full_campaign import build_finalization_candidate_inputs
from kuairand_agent.candidates.fusion import normalize_within_user_percentiles
from kuairand_agent.contract import SplitName, verify_starter_kit
from kuairand_agent.data.canonical import load_canonical_dataset
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.finalization.backends import _fm_predictions
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity
from kuairand_agent.scoring.submission import AlignmentRow, write_submission

SEEDS = (0, 1, 2, 3, 4)
QUALIFICATION = Path("runs/maki-qualification")
PUBLISHED_BASELINE_VALIDATION = 0.6016


def _members(starter_manifest_sha256: str) -> list[tuple[int, Path, StarterFMConfig]]:
    members = []
    for seed in SEEDS:
        checkpoint = QUALIFICATION / "fm" / f"seed-{seed}" / "checkpoint.npz"
        if not checkpoint.is_file():
            raise SystemExit(f"missing qualified checkpoint for seed {seed}: {checkpoint}")
        members.append((seed, checkpoint, StarterFMConfig(seed=seed)))
    del starter_manifest_sha256
    return members


def _rank_ensemble(
    *,
    inputs: object,
    user_ids: tuple[str, ...],
    video_ids: tuple[str, ...],
    phase: DataPhase,
    encoding: StarterEncoding,
    encoding_path: Path,
    starter_dir: Path,
    starter_manifest_sha256: str,
) -> np.ndarray:
    """Average within-user rank percentiles across every qualified seed."""

    normalized = []
    for seed, checkpoint_path, config in _members(starter_manifest_sha256):
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_encoding_digest=encoding.digest,
            expected_starter_manifest_digest=starter_manifest_sha256,
            expected_config_digest=config.digest,
            expected_seed=seed,
        )
        scores = _fm_predictions(
            source_dir=starter_dir,
            inputs=inputs,  # type: ignore[arg-type]
            encoding=encoding,
            checkpoint=checkpoint,
            config=config,
            expected_starter_manifest_sha256=starter_manifest_sha256,
        )
        ranked = normalize_within_user_percentiles(user_ids, video_ids, scores, phase=phase)
        normalized.append(ranked.scores)
        print(f"    seed {seed}: {len(scores)} rows scored through the pinned organizer source")
    del encoding_path
    return np.ascontiguousarray(np.mean(np.stack(normalized, axis=0), axis=0), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ensemble-submission")
    parser.add_argument("--data-dir", default=".data/KuaiRand-Pure/data")
    parser.add_argument("--starter-dir", default="kuairand-starter-kit")
    args = parser.parse_args()

    starter_dir = Path(args.starter_dir).resolve(strict=True)
    starter_manifest_sha256 = verify_starter_kit(starter_dir).manifest_sha256
    encoding_path = QUALIFICATION / "fm" / "seed-0" / "encoding.npz"
    encoding = StarterEncoding.load(encoding_path)

    # Every seed must share one vocabulary; the organizer encoding is deterministic and only the
    # weight initialisation differs by seed. Verify rather than assume.
    for seed in SEEDS:
        other = StarterEncoding.load(QUALIFICATION / "fm" / f"seed-{seed}" / "encoding.npz")
        if other.digest != encoding.digest:
            raise SystemExit(f"seed {seed} encoding differs from seed 0; refusing to ensemble")
    print(f"encoding shared across all {len(SEEDS)} seeds: {encoding.digest[:16]}")

    dataset = load_canonical_dataset(Path(args.data_dir).resolve(strict=True))

    print("\nscoring public validation (this is the number we report)")
    valid = dataset.split(SplitName.VALID)
    assert valid.targets is not None
    valid_inputs = build_finalization_candidate_inputs(DataPhase.OUTER_VALID, valid.inputs)
    valid_scores = _rank_ensemble(
        inputs=valid_inputs,
        user_ids=valid.inputs.user_id,
        video_ids=valid.inputs.video_id,
        phase=DataPhase.OUTER_VALID,
        encoding=encoding,
        encoding_path=encoding_path,
        starter_dir=starter_dir,
        starter_manifest_sha256=starter_manifest_sha256,
    )
    split = SplitIdentity(
        name="outer_valid",
        token="ensemble-submission-v1",
        expected_count=len(valid.inputs),
    )
    alignment = Alignment.from_ids(
        split=split, user_ids=valid.inputs.user_id, video_ids=valid.inputs.video_id
    )
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)
    result = scorer.score_with_encoded_labels(
        alignment=alignment,
        split=split,
        labels=np.asarray(valid.targets.reveal_for_scorer(), dtype=np.int8),
        scores=valid_scores,
    )
    delta = result.primary - PUBLISHED_BASELINE_VALIDATION
    print(f"    GAUC     {result.gauc:.10f}")
    print(f"    nDCG@5   {result.ndcg_at_5:.10f}")
    print(f"    primary  {result.primary:.10f}   vs published {PUBLISHED_BASELINE_VALIDATION} "
          f"= {delta:+.10f}")

    print("\nscoring the hidden test split (labels never read)")
    final = dataset.split(SplitName.TEST)
    if final.targets is not None:
        raise SystemExit("final split must not expose targets")
    final_inputs = build_finalization_candidate_inputs(DataPhase.FINAL, final.inputs)
    final_scores = _rank_ensemble(
        inputs=final_inputs,
        user_ids=final.inputs.user_id,
        video_ids=final.inputs.video_id,
        phase=DataPhase.FINAL,
        encoding=encoding,
        encoding_path=encoding_path,
        starter_dir=starter_dir,
        starter_manifest_sha256=starter_manifest_sha256,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = tuple(
        AlignmentRow(row_id=index, user_id=user, video_id=video)
        for index, (user, video) in enumerate(zip(final.inputs.user_id, final.inputs.video_id))
    )
    written = write_submission(out / "submission.csv", rows, final_scores)
    print(f"\nwrote {out / 'submission.csv'}")
    print(f"    rows            {len(rows)}")
    print(f"    sha256          {written.submission_digest}")

    (out / "provenance.json").write_text(
        json.dumps(
            {
                "what": "five-seed within-user rank ensemble of the official organizer FM",
                "agent_generated": False,
                "controller_side": True,
                "new_training_performed": False,
                "seeds": list(SEEDS),
                "combination": "within_user_descending_midrank_percentile_mean_v1",
                "encoding_digest": encoding.digest,
                "starter_manifest_sha256": starter_manifest_sha256,
                "dataset_digest": dataset.digest,
                "public_validation": {
                    "GAUC": result.gauc,
                    "nDCG@5": result.ndcg_at_5,
                    "primary": result.primary,
                    "delta_vs_published_baseline": delta,
                    "published_baseline": PUBLISHED_BASELINE_VALIDATION,
                },
                "materiality": {
                    "epsilon": 0.002,
                    "clears_epsilon": bool(delta > 0.002),
                    "seed_sigma": 0.0008,
                },
                "submission_digest": written.submission_digest,
                "prediction_digest": written.prediction_digest,
                "submission_rows": len(rows),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {out / 'provenance.json'}")
    print("\nValidate with the organizer checker:")
    print(f"    cd {starter_dir} && python submit.py "
          f"{(out / 'submission.csv').resolve()} --data_dir <DATA> --split test --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Promote a campaign's best candidate into the trusted seed, so searches compound.

Within a campaign a promoted candidate becomes the next iteration's parent. Across campaigns the
seed resets, so every campaign re-explores from the same root. That is measurable rather than
theoretical: `agent_ensemble_probe.py` found the five completed campaigns agree with each other at
rank correlation 0.95 to 0.99, and their ensemble gains +0.00079 where the organizers' own five
seeds gain +0.00116. Near-copies do not cancel each other's errors, and they are near-copies
because they all start in the same place.

This turns a series of independent restarts into the greedy tree search the AIDE-class agents
win with: the best node becomes the root of the next search. It is deliberately a between-batch
tool rather than an automatic step inside one. `scripts/auto_retry_campaigns.sh` notes that
cross-run memory only compounds "while the trusted controller source is unchanged between"
attempts, and `candidate_seed/` is inside the hashed source tree, so advancing the seed mid-batch
would change the digest under a running campaign.

Promotion is earned, not assumed. The candidate must beat the incumbent seed on the Fold B screen
by at least one measured seed sigma, using the same offline harness that gates the parent. A
candidate that merely ties inherits nothing: this is the mechanism that would otherwise let a
run of luck ratchet the seed in the wrong direction, permanently.

Run:  python3 advance_seed.py runs/<completed-run-dir>
      python3 advance_seed.py runs/<completed-run-dir> --apply
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from kuairand_agent.contract import MEASURED_SEED_SIGMA
from parent_acceptance_probe import evaluate_candidate

_SEED_DIR = Path("candidate_seed")
#: A candidate must clear the incumbent by this much on Fold B to become the next seed. One
#: measured sigma: below it the two are indistinguishable and inheriting adds noise, not progress.
_ADVANCE_MARGIN = MEASURED_SEED_SIGMA


def _selected_candidate_source(run_dir: Path) -> Path:
    """Locate the code of the candidate a completed campaign selected.

    Falls back to the durable archive, because a pruned run keeps its manifest and its archived
    source but not its working tree.
    """

    manifest = json.loads((run_dir / "final" / "manifest.json").read_text(encoding="utf-8"))
    selected = manifest.get("selection", {}).get("selected_experiment")
    if not isinstance(selected, str) or not selected:
        raise SystemExit(f"{run_dir} declares no selected experiment")
    for root in (
        run_dir / "production" / "generated-source" / selected,
        run_dir.parent / "archive" / "generated-source" / run_dir.name / selected,
    ):
        if (root / "model_impl.py").is_file() and (root / "config.json").is_file():
            return root
    raise SystemExit(
        f"selected candidate {selected} has no retained source; "
        "archive_generated_source runs before prune, so this run predates retention"
    )


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    run_dir = Path(argv[0])
    apply = "--apply" in argv[1:]
    candidate_dir = _selected_candidate_source(run_dir)
    print(f"incumbent seed   {_SEED_DIR}")
    print(f"challenger       {candidate_dir}\n")

    incumbent = evaluate_candidate(_SEED_DIR)
    challenger = evaluate_candidate(candidate_dir)
    delta = challenger - incumbent
    print(f"    incumbent Fold B standalone   {incumbent:.7f}")
    print(f"    challenger Fold B standalone  {challenger:.7f}")
    print(f"    delta                         {delta:+.7f}   ({delta / MEASURED_SEED_SIGMA:+.2f}s)")
    print(f"    margin required               {_ADVANCE_MARGIN:+.7f}   (+1.00s)\n")

    if delta <= _ADVANCE_MARGIN:
        print("    REFUSED. The challenger does not clear the incumbent by a measured sigma, so")
        print("    the seed stays as it is. Inheriting a tie ratchets noise into the root of")
        print("    every future search, and that is not reversible by a later campaign.")
        return 1

    print("    ACCEPTED. The challenger becomes the trusted seed for the next campaign.")
    if not apply:
        print("    Dry run: pass --apply to copy it into candidate_seed/.")
        return 0
    for name in ("model_impl.py", "config.json"):
        shutil.copy2(candidate_dir / name, _SEED_DIR / name)
        print(f"    wrote candidate_seed/{name}")
    print("\n    candidate_seed is inside the hashed source tree, so this changes the source")
    print("    digest. Re-run parent_acceptance_probe.py before launching, and never do this")
    print("    while a campaign is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

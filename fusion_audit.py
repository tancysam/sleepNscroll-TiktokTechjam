"""Print what rank fusion actually did to each generated candidate in a campaign run.

The primary recorded for a candidate is the *selected blend's*, never the model's own.  This
prints the whole five-point Fold B grid so the two can be told apart, which is the difference
between "matched the baseline" and "was rejected outright".

Offline read-only diagnostic.  Lives at the repository root deliberately: ``hash_source_tree``
covers ``src``, ``configs``, ``scripts``, ``candidate_seed`` and ``candidate_templates`` plus a
fixed list of root files, so a probe here cannot strand a running campaign the way one under
``scripts/`` would.

Run:  python3 fusion_audit.py maki-overnight-15
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MODEL_ONLY = [1.0, 0.0]
_CONTROL_ONLY = [0.0, 1.0]


def audit(run: str) -> None:
    records = sorted(Path("runs", run, "production", "scientific-records").glob("*.json"))
    print("=" * 78)
    print(f"{run}   scientific records: {len(records)}")
    if not records:
        print("  no scientific records: no candidate ever executed in this run")
        return

    discarded = 0
    for path in records:
        record = json.loads(path.read_text())["record"]
        primary = record["evidence"]["metrics"]["primary"]
        selection = record.get("fusion_selection")
        if selection is None:
            digest = record.get("fixed_fusion_result_digest") or "none"
            print(f"\n  reported primary {primary:.10f}   (frozen weight reused, {digest[:12]})")
            continue

        points = {tuple(point["weights"]): point for point in selection["points"]}
        model = points[tuple(_MODEL_ONLY)]["metrics"]["primary"]
        control = points[tuple(_CONTROL_ONLY)]["metrics"]["primary"]
        selected = selection["selected_weights"]
        rejected = selected == _CONTROL_ONLY
        discarded += rejected

        print(f"\n  reported primary {primary:.10f}   Fold B screen")
        print(f"    model alone      {model:.10f}")
        print(f"    control alone    {control:.10f}")
        gap = model - control
        print(f"    model minus control {gap:+.10f}   ({gap / 8e-4:+.2f} sigma at sigma 0.0008)")
        print(f"    selected weights {selected}" + ("   MODEL DISCARDED" if rejected else ""))
        for weights, point in points.items():
            print(f"      {str(list(weights)):14s} primary={point['metrics']['primary']:.10f}")

    if discarded:
        print(f"\n  {discarded} candidate(s) were weighted out entirely; their reported primary")
        print("  is the official FM control's own score. That is a rejection, not a tie.")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["maki-overnight-15"]:
        audit(name)

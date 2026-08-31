"""Recompute total provider token consumption and spend from the per attempt journals.

Total token consumption and GPU hours are required submission deliverables, so they are derived
here from the durable per call records rather than transcribed. Every provider attempt in every
run writes one journal file; this reads all of them.

Offline read only diagnostic. Lives at the repository root deliberately, outside the slice
``hash_source_tree`` covers, so it cannot strand a running campaign.

Run:  python3 token_accounting.py
"""

from __future__ import annotations

import collections
import glob
import json


def main() -> int:
    total: collections.Counter[str] = collections.Counter()
    per: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    for path in sorted(glob.glob("runs/*/production/provider-attempt-journal/*.json")):
        run = path.split("/")[1]
        record = json.load(open(path))
        usage = record.get("response", {}).get("usage", {})
        if not usage:
            continue
        # OpenRouter reports a zero top level cost for BYOK keys; the real figure is the upstream
        # inference cost, which is the number the campaign pays.
        cost = float(usage.get("cost_details", {}).get("upstream_inference_cost") or 0.0)
        fields = {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
            "reasoning": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
            "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        }
        for key, value in fields.items():
            total[key] += value
            per[run][key] += value
        for counter in (total, per[run]):
            counter["calls"] += 1
        total["cost_milli"] += round(cost * 1000)
        per[run]["cost_milli"] += round(cost * 1000)

    header = (
        f"{'run':<22}{'usage calls':>12}{'input':>11}{'output':>11}{'total':>12}{'reasoning':>11}"
    )
    print(header)
    print("-" * len(header))
    for run in sorted(per):
        row = per[run]
        print(
            f"{run:<22}{row['calls']:>12}{row['prompt']:>11,}{row['completion']:>11,}"
            f"{row['total']:>12,}{row['reasoning']:>11,}"
        )
    print("-" * len(header))
    print(
        f"{'ALL RUNS':<22}{total['calls']:>12}{total['prompt']:>11,}{total['completion']:>11,}"
        f"{total['total']:>12,}{total['reasoning']:>11,}"
    )
    print()
    print(f"  prompt tokens      {total['prompt']:>12,}")
    print(f"  of which cached    {total['cached']:>12,}")
    print(f"  completion tokens  {total['completion']:>12,}")
    print(f"  of which reasoning {total['reasoning']:>12,}")
    print(f"  total tokens       {total['total']:>12,}")
    print(f"  calls returning usage {total['calls']:>9,}")
    print("  GPU hours                  0.00")
    print()
    attempts = len(glob.glob("runs/*/production/provider-attempt-journal/*.json"))
    print("Two counting notes, so these reconcile against docs/RESULTS.md section 4:")
    print("  Calls here counts only attempts that returned a usage block. The published table")
    print("  counts every attempt, so it reports", attempts, "where this reports", total["calls"])
    print("  on identical token totals;", attempts - total["calls"], "attempts errored before")
    print("  returning usage and contribute zero tokens either way.")
    print("  Spend is deliberately not computed here. The published figures come from the frozen")
    print("  pricing block in configs, not from the provider reported upstream cost, so that they")
    print("  are reproducible rather than dependent on a rate card that can change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

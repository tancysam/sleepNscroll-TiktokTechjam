"""Report the real production feature-bundle shape and identity-code cardinalities.

Offline read-only diagnostic.  Lives at the repository root deliberately: ``hash_source_tree``
covers ``src``, ``configs``, ``scripts``, ``candidate_seed`` and ``candidate_templates`` plus a
fixed list of root files, so a probe here cannot strand a running campaign the way one under
``scripts/`` would.

Run:  python3 feature_bundle_probe.py .data/KuaiRand-Pure/data
"""

from __future__ import annotations

import sys
from pathlib import Path

from kuairand_agent.campaign.pure_features import (
    ID_CODE_FEATURE_NAMES,
    build_pure_feature_pair,
)
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import load_canonical_dataset


def main(data_dir: str) -> int:
    dataset = load_canonical_dataset(Path(data_dir))
    train = dataset.split(SplitName.TRAIN)
    valid = dataset.split(SplitName.VALID)
    assert train.targets is not None

    pair = build_pure_feature_pair(
        prefix_inputs=train.inputs,
        prefix_labels=tuple(int(value) for value in train.targets.long_view),
        query_inputs=valid.inputs,
        dataset_digest=dataset.digest,
        split_role="probe-outer",
        builder_source_digest="0" * 64,
    )

    print("prefix rows      :", pair.prefix.row_count)
    print("query rows       :", pair.query.row_count)
    print("feature count    :", pair.prefix.feature_count)
    print("code columns     :", ", ".join(ID_CODE_FEATURE_NAMES))
    print("code cardinality :")
    for name, cardinality in zip(ID_CODE_FEATURE_NAMES, pair.code_cardinalities, strict=True):
        print(f"    {name:24s} {cardinality}")

    # A query identity absent from the train-fitted vocabulary must land on the trailing unknown
    # slot, never on a code of its own.  This is the leak the frozen-query contract prevents.
    names = pair.prefix.feature_names
    for name, cardinality in zip(ID_CODE_FEATURE_NAMES, pair.code_cardinalities, strict=True):
        column = pair.query.values[:, names.index(name)]
        unknown = int((column == cardinality - 1).sum())
        print(
            f"    {name:24s} query rows on unknown slot: {unknown}"
            f" ({100.0 * unknown / column.size:.2f}%)"
        )
        assert column.max() < cardinality, name
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ".data/KuaiRand-Pure/data"))

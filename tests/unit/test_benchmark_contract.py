from __future__ import annotations

import dataclasses
import json

import pytest

from kuairand_agent.contract import (
    BENCHMARK_CONTRACT,
    BenchmarkContractError,
    RankingTaskContract,
    benchmark_digest,
    benchmark_manifest,
)


def test_frozen_contract_pins_task_splits_metrics_and_public_rungs() -> None:
    contract = BENCHMARK_CONTRACT
    assert contract.task.target == "long_view"
    assert [(split.name.value, split.start_date, split.end_date) for split in contract.splits] == [
        ("train", 20220408, 20220421),
        ("valid", 20220422, 20220428),
        ("test", 20220429, 20220508),
    ]
    assert contract.metrics.gauc_weight == "positive count"
    assert contract.metrics.ndcg_cutoff == 5
    assert contract.metrics.ndcg_zero_positive_value == 0.0
    assert contract.metrics.primary_formula == "(GAUC + nDCG@5) / 2"
    assert [rung.name for rung in contract.reference_rungs] == [
        "random",
        "item_popularity",
        "fm_official",
    ]
    assert contract.reference_rungs[-1].validation.primary == 0.6016


def test_manifest_is_canonical_complete_and_does_not_publish_test_outcomes() -> None:
    first = benchmark_manifest()
    second = benchmark_manifest()
    assert first == second
    assert first is not second
    assert len(benchmark_digest()) == 64
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert "starter_file_sha256" in encoded
    assert "dataset_archive_sha256" in encoded
    assert '"split":"valid"' in encoded
    assert '"scores"' in encoded
    assert '"split":"test","scores"' not in encoded


def test_contract_validation_rejects_stale_task_and_metric_drift() -> None:
    stale_task = RankingTaskContract(
        dataset="KuaiRand-Pure",
        target="is_click",
        relevance="native binary",
        ranking_unit="within-user ranking over logged impressions",
    )
    with pytest.raises(BenchmarkContractError, match="long_view"):
        dataclasses.replace(BENCHMARK_CONTRACT, task=stale_task).validate()

    stale_metric = dataclasses.replace(BENCHMARK_CONTRACT.metrics, ndcg_cutoff=10)
    with pytest.raises(BenchmarkContractError, match="metric semantics"):
        dataclasses.replace(BENCHMARK_CONTRACT, metrics=stale_metric).validate()


def test_manifest_copy_cannot_mutate_frozen_contract() -> None:
    manifest = benchmark_manifest()
    task = manifest["task"]
    assert isinstance(task, dict)
    task["target"] = "is_click"
    assert benchmark_manifest()["task"] == BENCHMARK_CONTRACT.task.manifest()

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kuairand_agent.data.audit import DataAuditReport, audit_dataset
from kuairand_agent.data.canonical import CanonicalDataset, load_canonical_dataset

DATA_ENV = "KUAIRAND_PURE_DATA_DIR"


@pytest.fixture(scope="session")
def official_data_dir() -> Path:
    raw = os.environ.get(DATA_ENV)
    if raw is None:
        pytest.skip(f"set {DATA_ENV} to the verified KuaiRand-Pure data directory")
    try:
        data_dir = Path(raw).resolve(strict=True)
    except OSError as exc:
        pytest.skip(f"{DATA_ENV} does not resolve to an existing directory: {exc}")
    if not data_dir.is_dir():
        pytest.skip(f"{DATA_ENV} is not a directory: {data_dir}")
    return data_dir


@pytest.fixture(scope="session")
def official_audit(official_data_dir: Path) -> DataAuditReport:
    report = audit_dataset(official_data_dir)
    trace = report.final_outcome_trace.manifest()
    counters = (
        "outcome_cells_materialized",
        "outcome_cells_decoded",
        "outcome_cells_converted",
        "outcome_cells_validated",
        "outcome_cells_aggregated",
        "outcome_cells_logged",
        "outcome_cells_scored",
    )
    assert all(trace[name] == 0 for name in counters)
    assert trace["skipped_values_recorded"] is False
    return report


@pytest.fixture(scope="session")
def official_dataset(
    official_data_dir: Path,
    official_audit: DataAuditReport,
) -> CanonicalDataset:
    dataset = load_canonical_dataset(official_data_dir)
    assert dataset.final.targets is None
    assert dataset.final.outcome_trace.parsed_cell_count == 0
    assert dataset.train.row_count == 1_141_112
    assert dataset.valid.row_count == 124_909
    assert dataset.final.row_count == 170_588
    assert official_audit.final_outcome_trace.row_count == dataset.final.row_count
    return dataset

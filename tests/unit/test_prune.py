from __future__ import annotations

from pathlib import Path

import pytest

from kuairand_agent.campaign.prune import PruneError, iter_run_dirs, plan_prune, prune_run


def _run_dir(root: Path, name: str, *, finalized: bool) -> Path:
    run = root / name
    for relative in (
        "production/provider-attempt-journal",
        "production/generated-source",
        "production/scientific-records",
        "production/feature-cache",
        "artifacts",
    ):
        (run / relative).mkdir(parents=True)
    (run / "campaign.sqlite3").write_bytes(b"store")
    (run / "production/feature-cache/cache.npz").write_bytes(b"x" * 2048)
    (run / "artifacts/blob").write_bytes(b"y" * 4096)
    (run / "production/provider-attempt-journal/a.json").write_text("{}", encoding="utf-8")
    (run / "production/scientific-records/a.json").write_text("{}", encoding="utf-8")
    if finalized:
        (run / "final").mkdir()
        (run / "final/report.md").write_text("# report", encoding="utf-8")
        (run / "final/submission.csv").write_text("row_id,score\n", encoding="utf-8")
    return run


def test_finalized_run_drops_cache_and_artifacts_but_keeps_every_evidence_path(
    tmp_path: Path,
) -> None:
    run = _run_dir(tmp_path, "finalized", finalized=True)

    plan = prune_run(run)

    assert plan.finalized is True
    assert plan.reclaimed_bytes == 2048 + 4096
    assert not (run / "production/feature-cache").exists()
    assert not (run / "artifacts").exists()
    # Everything the results documents rest on survives.
    for keep in (
        "final/report.md",
        "final/submission.csv",
        "campaign.sqlite3",
        "production/provider-attempt-journal/a.json",
        "production/scientific-records/a.json",
        "production/generated-source",
    ):
        assert (run / keep).exists(), keep


def test_unfinalized_run_keeps_artifacts_because_nothing_else_records_its_output(
    tmp_path: Path,
) -> None:
    """Without a bundle the artifact store is the only record of what the campaign produced."""

    run = _run_dir(tmp_path, "crashed", finalized=False)

    plan = prune_run(run)

    assert plan.finalized is False
    assert plan.reclaimed_bytes == 2048
    assert not (run / "production/feature-cache").exists()
    assert (run / "artifacts/blob").exists()


def test_plan_removes_nothing(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "planned", finalized=True)

    plan = plan_prune(run)

    assert plan.reclaimed_bytes == 2048 + 4096
    assert (run / "artifacts/blob").exists()
    assert (run / "production/feature-cache/cache.npz").exists()


def test_prune_succeeds_for_a_scripted_run_with_no_provider_journal(tmp_path: Path) -> None:
    """A scripted campaign makes no provider calls, so demanding a journal would refuse it."""

    run = _run_dir(tmp_path, "scripted", finalized=True)
    for child in (run / "production/provider-attempt-journal").iterdir():
        child.unlink()
    (run / "production/provider-attempt-journal").rmdir()

    plan = prune_run(run)

    assert plan.reclaimed_bytes == 2048 + 4096
    assert (run / "final/report.md").exists()


def test_prune_refuses_a_run_missing_protected_evidence(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "damaged", finalized=True)
    (run / "campaign.sqlite3").unlink()

    with pytest.raises(PruneError, match="missing protected evidence"):
        prune_run(run)

    # Nothing was removed before the refusal.
    assert (run / "artifacts/blob").exists()
    assert (run / "production/feature-cache/cache.npz").exists()


def test_prune_refuses_a_symlinked_target_that_escapes_the_run(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "escaping", finalized=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run / "artifacts").rmdir() if not any((run / "artifacts").iterdir()) else None
    escape = run / "production" / "feature-cache"
    for child in escape.iterdir():
        child.unlink()
    escape.rmdir()
    escape.symlink_to(outside, target_is_directory=True)

    plan = plan_prune(run)

    # A symlinked target is skipped, never followed.
    assert all("feature-cache" not in str(target) for target in plan.targets)
    assert outside.exists()


def test_prune_rejects_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    (tmp_path / "not-a-run").mkdir()

    with pytest.raises(PruneError, match="does not look like a campaign run"):
        plan_prune(tmp_path / "not-a-run")


def test_iter_run_dirs_finds_only_real_campaign_runs(tmp_path: Path) -> None:
    _run_dir(tmp_path, "alpha", finalized=True)
    _run_dir(tmp_path, "beta", finalized=False)
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "ledger.sqlite3").write_bytes(b"x")

    assert [path.name for path in iter_run_dirs(tmp_path)] == ["alpha", "beta"]

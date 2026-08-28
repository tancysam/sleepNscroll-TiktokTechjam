from __future__ import annotations

from dataclasses import replace

import pytest

from kuairand_agent.campaign.recovery import (
    MAX_DIAGNOSTIC_CHARACTERS,
    FailureCategory,
    FailureEvidence,
    FailureSignal,
    RecoveryAction,
    RecoveryHistory,
    RecoveryPolicy,
    RecoveryResources,
)


def _evidence(
    signal: FailureSignal,
    *,
    message: str = "candidate failed",
    traceback: str = "traceback",
    source_digest: str = "source-a",
    config_digest: str = "config-a",
    launched_training: bool = False,
) -> FailureEvidence:
    return FailureEvidence(
        signal=signal,
        stage="inner_screen",
        exception_type="TestFailure",
        message=message,
        traceback=traceback,
        source_digest=source_digest,
        config_digest=config_digest,
        data_digest="data-a",
        launched_training=launched_training,
    )


def _resources(**changes: object) -> RecoveryResources:
    defaults: dict[str, object] = {
        "remaining_repairs": 2,
        "recovery_launches_remaining": 2,
        "provider_retries_remaining": 1,
        "matching_checkpoint_identity": False,
        "changed_configuration_available": True,
        "smaller_repair_predicted_to_fit": True,
        "eligible_incumbent_available": True,
    }
    defaults.update(changes)
    return RecoveryResources(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("signal", "category"),
    [
        (FailureSignal.MALFORMED_PROVIDER_OUTPUT, FailureCategory.MALFORMED_MODEL_RESPONSE),
        (FailureSignal.SYNTAX_ERROR, FailureCategory.STATIC_IMPORT_POLICY),
        (FailureSignal.IMPORT_ERROR, FailureCategory.STATIC_IMPORT_POLICY),
        (FailureSignal.STATIC_POLICY_VIOLATION, FailureCategory.STATIC_IMPORT_POLICY),
        (FailureSignal.DETERMINISTIC_DATA_BUG, FailureCategory.DETERMINISTIC_DATA_SHAPE),
        (FailureSignal.SHAPE_BUG, FailureCategory.DETERMINISTIC_DATA_SHAPE),
        (FailureSignal.NON_FINITE_OUTPUT, FailureCategory.NON_FINITE_SCORES),
        (FailureSignal.TIMEOUT, FailureCategory.TIMEOUT),
        (FailureSignal.MEMORY_LIMIT, FailureCategory.OUT_OF_MEMORY),
        (FailureSignal.INTERRUPTED, FailureCategory.INTERRUPTED_PROCESS),
        (FailureSignal.CANDIDATE_INVALID_METRIC, FailureCategory.CANDIDATE_INVALID_METRIC),
        (FailureSignal.PROTECTED_SCORER_ERROR, FailureCategory.PROTECTED_SCORER_CONTRACT),
        (FailureSignal.TRUSTED_CONTRACT_ERROR, FailureCategory.PROTECTED_SCORER_CONTRACT),
        (FailureSignal.PROVIDER_UNAVAILABLE, FailureCategory.PROVIDER_UNAVAILABLE),
        (FailureSignal.FINAL_REPLAY_MISMATCH, FailureCategory.FINAL_REPLAY_MISMATCH),
        (FailureSignal.UNKNOWN_CANDIDATE_FAILURE, FailureCategory.UNKNOWN_CANDIDATE_FAILURE),
    ],
)
def test_every_failure_signal_has_a_typed_category(
    signal: FailureSignal, category: FailureCategory
) -> None:
    failure = RecoveryPolicy().classify(_evidence(signal))
    assert failure.category is category
    assert len(failure.fingerprint) == 64


def test_fingerprint_is_stable_after_path_pid_address_and_secret_redaction() -> None:
    policy = RecoveryPolicy()
    first = policy.classify(
        _evidence(
            FailureSignal.SHAPE_BUG,
            message="pid=123 at 0xabc with token=sk-abcdefghijk",
            traceback='File "/private/tmp/run-a/candidate.py", line 10',
        )
    )
    second = policy.classify(
        _evidence(
            FailureSignal.SHAPE_BUG,
            message="pid=987 at 0xdef with token=sk-zzzzzzzzzzz",
            traceback='File "/private/tmp/run-b/candidate.py", line 10',
        )
    )
    assert first.fingerprint == second.fingerprint
    assert "sk-" not in first.bounded_diagnostic
    assert "/private/tmp" not in first.bounded_diagnostic
    assert "pid=123" not in first.bounded_diagnostic


def test_repair_diagnostic_is_bounded() -> None:
    failure = RecoveryPolicy().classify(
        _evidence(FailureSignal.SYNTAX_ERROR, traceback="x" * 10_000)
    )
    assert len(failure.bounded_diagnostic) == MAX_DIAGNOSTIC_CHARACTERS


def test_malformed_response_reparse_once_then_one_schema_retry_without_launch() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.MALFORMED_PROVIDER_OUTPUT))
    history = RecoveryHistory()

    reparse = policy.decide(failure, history, _resources())
    assert reparse.action is RecoveryAction.REPARSE_RESPONSE
    assert not reparse.may_launch_training
    history = history.append(failure, reparse)

    retry = policy.decide(failure, history, _resources())
    assert retry.action is RecoveryAction.SCHEMA_PROVIDER_RETRY
    assert not retry.may_launch_training
    history = history.append(failure, retry)

    exhausted = policy.decide(failure, history, _resources())
    assert exhausted.action is RecoveryAction.CLOSE_BRANCH


def test_static_repairs_are_bounded_to_two_children() -> None:
    policy = RecoveryPolicy()
    history = RecoveryHistory()
    first_failure = policy.classify(_evidence(FailureSignal.SYNTAX_ERROR))
    first = policy.decide(first_failure, history, _resources())
    assert first.action is RecoveryAction.REPAIR_CHILD
    history = history.append(first_failure, first)

    second_failure = policy.classify(
        _evidence(
            FailureSignal.IMPORT_ERROR,
            message="new import error",
            source_digest="source-b",
        )
    )
    second = policy.decide(second_failure, history, _resources(remaining_repairs=1))
    assert second.action is RecoveryAction.REPAIR_CHILD
    history = history.append(second_failure, second)
    assert history.repairs_used == 2

    third_failure = policy.classify(
        _evidence(
            FailureSignal.STATIC_POLICY_VIOLATION,
            message="third static failure",
            source_digest="source-c",
        )
    )
    assert (
        policy.decide(third_failure, history, _resources(remaining_repairs=0)).action
        is RecoveryAction.CLOSE_BRANCH
    )


def test_identical_deterministic_shape_failure_is_not_retried_indefinitely() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.SHAPE_BUG, launched_training=True))
    repair = policy.decide(failure, RecoveryHistory(), _resources())
    assert repair.action is RecoveryAction.REPAIR_CHILD
    assert repair.may_launch_training
    history = RecoveryHistory().append(failure, repair)
    repeated = policy.decide(failure, history, _resources(remaining_repairs=1))
    assert repeated.action is RecoveryAction.CLOSE_BRANCH


def test_non_finite_scores_get_one_stabilization_repair() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.NON_FINITE_OUTPUT))
    repair = policy.decide(failure, RecoveryHistory(), _resources())
    assert repair.action is RecoveryAction.NUMERICAL_STABILIZATION_REPAIR
    assert repair.requires_changed_source_or_config
    repeated = policy.decide(
        failure,
        RecoveryHistory().append(failure, repair),
        _resources(remaining_repairs=1),
    )
    assert repeated.action is RecoveryAction.CLOSE_BRANCH


def test_timeout_never_relaunches_identical_config() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.TIMEOUT, launched_training=True))
    reduced = policy.decide(failure, RecoveryHistory(), _resources())
    assert reduced.action is RecoveryAction.REDUCE_COST_REPAIR
    assert reduced.requires_changed_source_or_config
    repeated = policy.decide(
        failure,
        RecoveryHistory().append(failure, reduced),
        _resources(remaining_repairs=1),
    )
    assert repeated.action is RecoveryAction.CLOSE_BRANCH
    unavailable = policy.decide(
        failure,
        RecoveryHistory(),
        _resources(changed_configuration_available=False),
    )
    assert unavailable.action is RecoveryAction.CLOSE_BRANCH


def test_oom_allows_one_explicitly_smaller_predicted_to_fit_repair() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.MEMORY_LIMIT, launched_training=True))
    smaller = policy.decide(failure, RecoveryHistory(), _resources())
    assert smaller.action is RecoveryAction.SMALLER_MODEL_REPAIR
    assert smaller.may_launch_training
    cannot_fit = policy.decide(
        failure,
        RecoveryHistory(),
        _resources(smaller_repair_predicted_to_fit=False),
    )
    assert cannot_fit.action is RecoveryAction.CLOSE_BRANCH


def test_interrupted_process_resumes_only_matching_checkpoint() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.INTERRUPTED, launched_training=True))
    resume = policy.decide(
        failure, RecoveryHistory(), _resources(matching_checkpoint_identity=True)
    )
    assert resume.action is RecoveryAction.RESUME_MATCHING_CHECKPOINT
    assert resume.may_launch_training
    mismatch = policy.decide(failure, RecoveryHistory(), _resources())
    assert mismatch.action is RecoveryAction.CLOSE_BRANCH


def test_candidate_metric_is_ignored_but_protected_scorer_failure_stops_research() -> None:
    policy = RecoveryPolicy()
    invalid_metric = policy.classify(_evidence(FailureSignal.CANDIDATE_INVALID_METRIC))
    ignored = policy.decide(invalid_metric, RecoveryHistory(), _resources())
    assert ignored.action is RecoveryAction.IGNORE_CANDIDATE_METRIC
    assert not ignored.stop_research

    trusted_failure = policy.classify(_evidence(FailureSignal.PROTECTED_SCORER_ERROR))
    stopped = policy.decide(trusted_failure, RecoveryHistory(), _resources())
    assert stopped.action is RecoveryAction.STOP_RESEARCH_PRESERVE_STATE
    assert stopped.stop_research
    assert stopped.preserve_incumbent


def test_provider_unavailable_retries_once_then_finalizes_incumbent() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.PROVIDER_UNAVAILABLE))
    retry = policy.decide(failure, RecoveryHistory(), _resources())
    assert retry.action is RecoveryAction.RETRY_PROVIDER
    assert not retry.may_launch_training
    history = RecoveryHistory().append(failure, retry)
    final = policy.decide(failure, history, _resources(provider_retries_remaining=0))
    assert final.action is RecoveryAction.FINALIZE_INCUMBENT
    assert final.stop_research

    no_incumbent = policy.decide(
        failure,
        history,
        _resources(provider_retries_remaining=0, eligible_incumbent_available=False),
    )
    assert no_incumbent.action is RecoveryAction.STOP_RESEARCH_PRESERVE_STATE


def test_replay_mismatch_falls_back_and_unknown_failure_closes_branch() -> None:
    policy = RecoveryPolicy()
    replay = policy.classify(_evidence(FailureSignal.FINAL_REPLAY_MISMATCH))
    fallback = policy.decide(replay, RecoveryHistory(), _resources())
    assert fallback.action is RecoveryAction.FALLBACK_REPLAYABLE_INCUMBENT
    assert fallback.stop_research

    unknown = policy.classify(_evidence(FailureSignal.UNKNOWN_CANDIDATE_FAILURE))
    assert (
        policy.decide(unknown, RecoveryHistory(), _resources()).action
        is RecoveryAction.CLOSE_BRANCH
    )


def test_recovery_never_demotes_incumbent_or_counts_an_extra_scientific_iteration() -> None:
    policy = RecoveryPolicy()
    for signal in FailureSignal:
        failure = policy.classify(_evidence(signal))
        resources = _resources(
            matching_checkpoint_identity=True,
            smaller_repair_predicted_to_fit=True,
        )
        decision = policy.decide(failure, RecoveryHistory(), resources)
        assert decision.preserve_incumbent
        assert decision.scientific_iteration_delta == 0


def test_a_new_source_changes_the_failure_fingerprint() -> None:
    policy = RecoveryPolicy()
    failure = policy.classify(_evidence(FailureSignal.SHAPE_BUG))
    changed = policy.classify(replace(_evidence(FailureSignal.SHAPE_BUG), source_digest="source-b"))
    assert failure.fingerprint != changed.fingerprint

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace

import pytest

from kuairand_agent.campaign.convergence import MAX_UNMEASURED_STREAK
from kuairand_agent.campaign.scientific import (
    CampaignStopReason,
    CandidateOutcome,
    ExecutableChangeEvidence,
    OuterPromotionCompletion,
    OuterPromotionLedgerSnapshot,
    OuterPromotionRequest,
    OuterPromotionReservation,
    PublicAggregateFeedback,
    ResourceEvidence,
    RunIdentityEvidence,
    ScientificCampaignConfig,
    ScientificCampaignError,
    ScientificCandidate,
    ScientificRunEvidence,
    ScientificRunRequest,
    ScientificTier,
    run_scientific_campaign,
)
from kuairand_agent.campaign.selector import (
    GateEvidence,
    IncumbentEvidence,
    OrganizerMetrics,
    SeedMetrics,
)

_DIGESTS = tuple(f"{index:x}" * 64 for index in range(1, 16))


def _metrics(primary: float) -> OrganizerMetrics:
    return OrganizerMetrics(gauc=primary, ndcg_at_5=primary)


def _fallback(primary: float = 0.6) -> IncumbentEvidence:
    return IncumbentEvidence(
        candidate_id="official-fm",
        inner_by_fold=(("A", _metrics(primary)), ("B", _metrics(primary))),
        outer_by_seed=tuple(SeedMetrics(seed, _metrics(primary)) for seed in (0, 1, 2)),
        evidence_receipt_digest=_DIGESTS[14],
        replayable=True,
        eligible=True,
        official_fm=True,
    )


def _config() -> ScientificCampaignConfig:
    return ScientificCampaignConfig(
        benchmark_digest=_DIGESTS[0],
        dataset_digest=_DIGESTS[1],
        scorer_digest=_DIGESTS[2],
        fold_a_data_digest=_DIGESTS[3],
        fold_b_data_digest=_DIGESTS[4],
        outer_data_digest=_DIGESTS[5],
        environment_digest=_DIGESTS[6],
        campaign_digest=_DIGESTS[7],
        qualified_fallback_receipt_digest=_DIGESTS[14],
        max_scientific_iterations=12,
        launches_already_used=6,
    )


def test_official_fallback_receipt_must_match_frozen_qualification_identity() -> None:
    fallback = replace(_fallback(), evidence_receipt_digest=_DIGESTS[13])

    with pytest.raises(ScientificCampaignError, match="fallback receipt identity mismatch"):
        run_scientific_campaign(
            config=_config(),
            fallback=fallback,
            candidates=(),
            runner=lambda request: _evidence(request, 0.6),
            outer_ledger=_Ledger(),
        )


def _material_change(
    *,
    parent_source_digest: str = _DIGESTS[11],
    candidate_source_digest: str = _DIGESTS[8],
    executable_diff_digest: str = _DIGESTS[12],
    changed_symbols: tuple[str, ...] = ("candidate.py:score",),
    reachable_python_files: tuple[str, ...] = ("candidate.py",),
) -> ExecutableChangeEvidence:
    return ExecutableChangeEvidence(
        parent_source_digest=parent_source_digest,
        candidate_source_digest=candidate_source_digest,
        executable_diff_digest=executable_diff_digest,
        controller_attestation_digest=_DIGESTS[13],
        changed_symbols=changed_symbols,
        reachable_python_files=reachable_python_files,
    )


def _candidate(candidate_id: str = "pairwise") -> ScientificCandidate:
    return ScientificCandidate(
        candidate_id=candidate_id,
        parent_id="official-fm",
        family="pairwise_fm",
        source_digest=_DIGESTS[8],
        parent_source_digest=_DIGESTS[11],
        executable_change=_material_change(),
        config_digest=_DIGESTS[9],
        training_policy_digest=_DIGESTS[10],
        gates=GateEvidence(),
    )


def _evidence(
    request: ScientificRunRequest,
    primary: float,
    *,
    gates: GateEvidence | None = None,
    replay_verified: bool = True,
    request_digest: str | None = None,
) -> ScientificRunEvidence:
    return ScientificRunEvidence(
        request_digest=request.digest if request_digest is None else request_digest,
        metrics=_metrics(primary),
        gates=GateEvidence() if gates is None else gates,
        identities=RunIdentityEvidence(
            source_digest=request.source_digest,
            parent_source_digest=request.parent_source_digest,
            executable_diff_digest=request.executable_diff_digest,
            material_change_digest=request.material_change_digest,
            controller_attestation_digest=request.controller_attestation_digest,
            config_digest=request.config_digest,
            training_policy_digest=request.training_policy_digest,
            data_digest=request.data_digest,
            environment_digest=request.environment_digest,
            execution_digest=_DIGESTS[11],
            checkpoint_digest=_DIGESTS[12],
            prediction_digest=_DIGESTS[13],
            scorer_digest=request.scorer_digest,
            artifact_closure_digest=_DIGESTS[14],
        ),
        resources=ResourceEvidence(
            wall_seconds=1.25,
            peak_rss_bytes=1024,
            disk_bytes=2048,
        ),
        replay_verified=replay_verified,
    )


@dataclass
class _Ledger:
    reservations: list[OuterPromotionRequest] = field(default_factory=list)
    completions: list[OuterPromotionCompletion] = field(default_factory=list)

    def snapshot(self) -> OuterPromotionLedgerSnapshot:
        return OuterPromotionLedgerSnapshot(
            revision=len(self.reservations) + len(self.completions),
            campaign_digest=_DIGESTS[7],
            benchmark_digest=_DIGESTS[0],
            dataset_digest=_DIGESTS[1],
            scorer_digest=_DIGESTS[2],
            max_distinct_candidates=6,
            candidate_fingerprints=tuple(
                dict.fromkeys(item.candidate_fingerprint for item in self.reservations)
            ),
        )

    def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation:
        consumes_slot = request.candidate_fingerprint not in {
            item.candidate_fingerprint for item in self.reservations
        }
        self.reservations.append(request)
        return OuterPromotionReservation(
            reservation_id=f"outer-{len(self.reservations)}",
            request_digest=request.digest,
            ledger_revision=len(self.reservations) + len(self.completions),
            candidate_id=request.candidate_id,
            candidate_fingerprint=request.candidate_fingerprint,
            consumes_slot=consumes_slot,
        )

    def complete(
        self,
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
    ) -> None:
        assert completion.request_digest == reservation.request_digest
        assert completion.reservation_revision == reservation.ledger_revision
        self.completions.append(completion)


def test_fold_b_screen_failure_stops_before_fold_a_or_outer_and_preserves_fm() -> None:
    requests: list[ScientificRunRequest] = []

    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        requests.append(request)
        assert request.tier is ScientificTier.FOLD_B_SCREEN
        return _evidence(request, 0.599)

    ledger = _Ledger()
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=runner,
        outer_ledger=ledger,
    )

    assert [(request.tier, request.fold_id, request.seed) for request in requests] == [
        (ScientificTier.FOLD_B_SCREEN, "B", 0)
    ]
    assert result.candidates[0].outcome is CandidateOutcome.SCREEN_REJECTED
    assert result.incumbent.candidate_id == "official-fm"
    assert result.fallback.candidate_id == "official-fm"
    assert result.launches_used == 7
    assert result.convergence.completed_iterations == 1
    # Rejected: no eligible outer primary, so it is unmeasured rather than non-material.
    assert result.convergence.non_material_streak == 0
    assert result.convergence.unmeasured_streak == 1
    assert not ledger.reservations
    assert not ledger.completions

    request_parameters = str(inspect.signature(ScientificRunRequest)).lower()
    assert "final" not in request_parameters
    assert "outcome" not in request_parameters
    assert "label" not in request_parameters


def test_fold_a_b_confirmation_then_matched_seeds_materially_promotes() -> None:
    requests: list[ScientificRunRequest] = []
    outer_values = {0: 0.604, 1: 0.603, 2: 0.605}

    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        requests.append(request)
        if request.tier is ScientificTier.FOLD_B_SCREEN:
            return _evidence(request, 0.604)
        if request.tier is ScientificTier.FOLD_A_CONFIRMATION:
            return _evidence(request, 0.603)
        return _evidence(request, outer_values[request.seed])

    ledger = _Ledger()
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=runner,
        outer_ledger=ledger,
    )

    assert [(request.tier, request.fold_id, request.seed) for request in requests] == [
        (ScientificTier.FOLD_B_SCREEN, "B", 0),
        (ScientificTier.FOLD_A_CONFIRMATION, "A", 0),
        (ScientificTier.OUTER_MATCHED_SEED, None, 0),
        (ScientificTier.OUTER_MATCHED_SEED, None, 1),
        (ScientificTier.OUTER_MATCHED_SEED, None, 2),
    ]
    candidate_result = result.candidates[0]
    assert candidate_result.outcome is CandidateOutcome.PROMOTED_CONFIRMED
    assert result.incumbent.candidate_id == "pairwise"
    assert result.fallback.candidate_id == "official-fm"
    assert result.fallback.official_fm
    assert result.launches_used == 11
    assert result.convergence.completed_iterations == 1
    assert result.convergence.non_material_streak == 0
    assert result.convergence.best_primary == 0.604
    assert len(ledger.reservations) == 1
    assert len(ledger.completions) == 1
    assert ledger.completions[0].successful
    assert tuple(item.seed for item in ledger.completions[0].seed_metrics) == (0, 1, 2)

    assert all(isinstance(item, PublicAggregateFeedback) for item in result.public_feedback)
    assert [item.primary for item in result.public_feedback] == [0.604, 0.603, 0.605]
    assert {item.primary_direction for item in result.public_feedback} == {"higher"}
    assert set(result.public_feedback[0].manifest()) == {
        "candidate_id",
        "seed",
        "GAUC",
        "nDCG@5",
        "primary",
        "GAUC_direction",
        "nDCG@5_direction",
        "primary_direction",
    }
    outer_requests = [
        request for request in requests if request.tier is ScientificTier.OUTER_MATCHED_SEED
    ]
    assert len({request.config_digest for request in outer_requests}) == 1
    assert len({request.training_policy_digest for request in outer_requests}) == 1
    assert not any("epoch" in request.manifest() for request in outer_requests)


def test_callback_failure_is_charged_but_cannot_displace_replayable_fallback() -> None:
    def runner(_: ScientificRunRequest) -> ScientificRunEvidence:
        raise RuntimeError("ordinary isolated candidate failure")

    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate("broken"),),
        runner=runner,
        outer_ledger=_Ledger(),
    )

    assert result.candidates[0].outcome is CandidateOutcome.CALLBACK_FAILED
    assert result.candidates[0].runs == ()
    # The reason must carry the exception MESSAGE, not only its class name. It becomes
    # campaign_records[].candidate_reason, the only account of the crash the research model ever
    # sees, and a campaign once spent all six of its iterations guessing at a defect it had been
    # told nothing about beyond the bare string "CandidateExecutionError".
    assert result.candidates[0].reason == (
        "callback_failed:RuntimeError: ordinary isolated candidate failure"
    )
    assert result.incumbent == result.fallback == _fallback()
    assert result.launches_used == 7
    assert result.convergence.completed_iterations == 1
    # Rejected: no eligible outer primary, so it is unmeasured rather than non-material.
    assert result.convergence.non_material_streak == 0
    assert result.convergence.unmeasured_streak == 1


def test_callback_failure_reason_is_bounded_and_survives_an_empty_message() -> None:
    def noisy(_: ScientificRunRequest) -> ScientificRunEvidence:
        raise RuntimeError("x" * 10_000)

    loud = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate("broken"),),
        runner=noisy,
        outer_ledger=_Ledger(),
    )
    reason = loud.candidates[0].reason
    assert reason.startswith("callback_failed:RuntimeError: xxx")
    assert len(reason) <= len("callback_failed:") + 4_000

    def silent(_: ScientificRunRequest) -> ScientificRunEvidence:
        raise RuntimeError

    quiet = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate("broken"),),
        runner=silent,
        outer_ledger=_Ledger(),
    )
    assert quiet.candidates[0].reason == "callback_failed:RuntimeError"


def test_three_rejected_screens_do_not_stop_a_fourth_candidate() -> None:
    # Screen-rejected candidates produce no eligible outer primary.  They used to end the campaign
    # at three under the label "converged"; the fourth candidate must now still get its turn.
    seen: list[str] = []

    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        seen.append(request.candidate_id)
        return _evidence(request, 0.599)

    candidates = tuple(_candidate(f"candidate-{index}") for index in range(4))
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=candidates,
        runner=runner,
        outer_ledger=_Ledger(),
    )

    assert seen == ["candidate-0", "candidate-1", "candidate-2", "candidate-3"]
    assert len(result.candidates) == 4
    assert result.stop_reason is CampaignStopReason.CANDIDATES_EXHAUSTED
    assert not result.convergence.converged
    assert result.convergence.unmeasured_streak == 4
    assert result.incumbent.candidate_id == "official-fm"


def test_exactly_three_rejected_candidates_report_exhaustion_not_convergence() -> None:
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=tuple(_candidate(f"candidate-{index}") for index in range(3)),
        runner=lambda request: _evidence(request, 0.599),
        outer_ledger=_Ledger(),
    )

    # Nothing was measured, so nothing plateaued: this must not be reported as convergence.
    assert not result.convergence.converged
    assert not result.convergence.should_stop
    assert result.stop_reason is CampaignStopReason.CANDIDATES_EXHAUSTED


def test_sustained_rejection_stops_without_claiming_convergence() -> None:
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=tuple(_candidate(f"candidate-{index}") for index in range(8)),
        runner=lambda request: _evidence(request, 0.599),
        outer_ledger=_Ledger(),
    )

    assert result.convergence.unmeasured_streak == MAX_UNMEASURED_STREAK
    assert result.convergence.should_stop
    assert not result.convergence.converged
    assert result.stop_reason is CampaignStopReason.CANDIDATES_NOT_PROMOTABLE
    assert result.incumbent.candidate_id == "official-fm"


def test_finalization_reserve_rejects_new_scientific_launch_without_charging() -> None:
    config = ScientificCampaignConfig(
        benchmark_digest=_DIGESTS[0],
        dataset_digest=_DIGESTS[1],
        scorer_digest=_DIGESTS[2],
        fold_a_data_digest=_DIGESTS[3],
        fold_b_data_digest=_DIGESTS[4],
        outer_data_digest=_DIGESTS[5],
        environment_digest=_DIGESTS[6],
        campaign_digest=_DIGESTS[7],
        qualified_fallback_receipt_digest=_DIGESTS[14],
        max_scientific_iterations=12,
        launches_already_used=6,
        elapsed_seconds_at_start=18_000.0,
    )

    def must_not_run(_: ScientificRunRequest) -> ScientificRunEvidence:
        raise AssertionError("reserve should block this launch")

    result = run_scientific_campaign(
        config=config,
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=must_not_run,
        outer_ledger=_Ledger(),
    )

    assert result.candidates[0].outcome is CandidateOutcome.BUDGET_REJECTED
    assert result.stop_reason is CampaignStopReason.FINALIZATION_RESERVE
    assert result.launches_used == 6
    assert result.convergence.completed_iterations == 0


def test_unverified_replay_is_a_structural_failure_and_never_reaches_outer() -> None:
    requests: list[ScientificRunRequest] = []

    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        requests.append(request)
        return _evidence(request, 0.61, replay_verified=False)

    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=runner,
        outer_ledger=_Ledger(),
    )

    assert [request.tier for request in requests] == [ScientificTier.FOLD_B_SCREEN]
    assert result.candidates[0].outcome is CandidateOutcome.SCREEN_REJECTED
    assert result.incumbent == result.fallback


def test_candidate_fingerprint_is_content_addressed_not_rename_addressed() -> None:
    first = _candidate("alias-one")
    second = _candidate("alias-two")

    assert first.fingerprint == second.fingerprint


def test_malformed_callback_evidence_is_charged_and_fails_closed() -> None:
    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        return _evidence(request, 0.61, request_digest=_DIGESTS[0])

    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=runner,
        outer_ledger=_Ledger(),
    )

    assert result.candidates[0].outcome is CandidateOutcome.CALLBACK_FAILED
    assert result.incumbent == result.fallback
    assert result.launches_used == 7


def test_outer_callback_failure_completes_reserved_slot_unsuccessfully() -> None:
    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        if request.tier is ScientificTier.OUTER_MATCHED_SEED:
            raise RuntimeError("isolated outer worker failed")
        return _evidence(request, 0.604)

    ledger = _Ledger()
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=runner,
        outer_ledger=ledger,
    )

    assert result.candidates[0].outcome is CandidateOutcome.OUTER_FAILED
    assert result.incumbent == result.fallback
    assert result.launches_used == 9
    assert len(ledger.reservations) == 1
    assert len(ledger.completions) == 1
    assert not ledger.completions[0].successful
    assert ledger.completions[0].seed_metrics == ()


def test_launch_cap_after_inner_confirmation_does_not_consume_outer_slot() -> None:
    config = ScientificCampaignConfig(
        benchmark_digest=_DIGESTS[0],
        dataset_digest=_DIGESTS[1],
        scorer_digest=_DIGESTS[2],
        fold_a_data_digest=_DIGESTS[3],
        fold_b_data_digest=_DIGESTS[4],
        outer_data_digest=_DIGESTS[5],
        environment_digest=_DIGESTS[6],
        campaign_digest=_DIGESTS[7],
        qualified_fallback_receipt_digest=_DIGESTS[14],
        max_scientific_iterations=12,
        launches_already_used=48,
    )
    ledger = _Ledger()

    result = run_scientific_campaign(
        config=config,
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=lambda request: _evidence(request, 0.604),
        outer_ledger=ledger,
    )

    assert result.candidates[0].outcome is CandidateOutcome.BUDGET_REJECTED
    assert result.stop_reason is CampaignStopReason.LAUNCH_CAP
    assert result.launches_used == 50
    assert not ledger.reservations
    assert not ledger.completions


def test_outer_reservation_requires_capacity_for_entire_matched_seed_bundle() -> None:
    config = ScientificCampaignConfig(
        benchmark_digest=_DIGESTS[0],
        dataset_digest=_DIGESTS[1],
        scorer_digest=_DIGESTS[2],
        fold_a_data_digest=_DIGESTS[3],
        fold_b_data_digest=_DIGESTS[4],
        outer_data_digest=_DIGESTS[5],
        environment_digest=_DIGESTS[6],
        campaign_digest=_DIGESTS[7],
        qualified_fallback_receipt_digest=_DIGESTS[14],
        max_scientific_iterations=12,
        launches_already_used=47,
    )
    ledger = _Ledger()

    result = run_scientific_campaign(
        config=config,
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=lambda request: _evidence(request, 0.604),
        outer_ledger=ledger,
    )

    assert result.candidates[0].outcome is CandidateOutcome.BUDGET_REJECTED
    assert result.stop_reason is CampaignStopReason.LAUNCH_CAP
    assert result.launches_used == 49
    assert not ledger.reservations


def test_outer_reservation_preserves_finalization_time_for_full_seed_bundle() -> None:
    config = ScientificCampaignConfig(
        benchmark_digest=_DIGESTS[0],
        dataset_digest=_DIGESTS[1],
        scorer_digest=_DIGESTS[2],
        fold_a_data_digest=_DIGESTS[3],
        fold_b_data_digest=_DIGESTS[4],
        outer_data_digest=_DIGESTS[5],
        environment_digest=_DIGESTS[6],
        campaign_digest=_DIGESTS[7],
        qualified_fallback_receipt_digest=_DIGESTS[14],
        max_scientific_iterations=12,
        launches_already_used=6,
        elapsed_seconds_at_start=17_800.0,
    )
    candidate = ScientificCandidate(
        candidate_id="pairwise",
        parent_id="official-fm",
        family="pairwise_fm",
        source_digest=_DIGESTS[8],
        parent_source_digest=_DIGESTS[11],
        executable_change=_material_change(),
        config_digest=_DIGESTS[9],
        training_policy_digest=_DIGESTS[10],
        gates=GateEvidence(),
        p95_runtime_seconds=100.0,
        cleanup_seconds=5.0,
    )
    ledger = _Ledger()

    result = run_scientific_campaign(
        config=config,
        fallback=_fallback(),
        candidates=(candidate,),
        runner=lambda request: _evidence(request, 0.604),
        outer_ledger=ledger,
    )

    assert result.candidates[0].outcome is CandidateOutcome.BUDGET_REJECTED
    assert result.stop_reason is CampaignStopReason.FINALIZATION_RESERVE
    assert result.launches_used == 8
    assert not ledger.reservations


def test_public_feedback_precision_is_locked_to_safe_context_four_decimals() -> None:
    assert _config().reporting_precision == 4
    assert "reporting_precision" not in str(inspect.signature(ScientificCampaignConfig))


def test_outer_reservation_must_echo_exact_request_and_advance_revision() -> None:
    class CorruptLedger(_Ledger):
        def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation:
            valid = super().reserve(request)
            return OuterPromotionReservation(
                reservation_id=valid.reservation_id,
                request_digest=_DIGESTS[0],
                ledger_revision=valid.ledger_revision,
                candidate_id=valid.candidate_id,
                candidate_fingerprint=valid.candidate_fingerprint,
                consumes_slot=valid.consumes_slot,
            )

    with pytest.raises(ScientificCampaignError, match="reservation identity mismatch"):
        run_scientific_campaign(
            config=_config(),
            fallback=_fallback(),
            candidates=(_candidate(),),
            runner=lambda request: _evidence(request, 0.604),
            outer_ledger=CorruptLedger(),
        )


def test_restart_reuses_exact_durable_reservation_without_consuming_new_slot() -> None:
    candidate = _candidate()

    @dataclass
    class RestartLedger:
        completions: list[OuterPromotionCompletion] = field(default_factory=list)

        def snapshot(self) -> OuterPromotionLedgerSnapshot:
            return OuterPromotionLedgerSnapshot(
                revision=4,
                campaign_digest=_DIGESTS[7],
                benchmark_digest=_DIGESTS[0],
                dataset_digest=_DIGESTS[1],
                scorer_digest=_DIGESTS[2],
                max_distinct_candidates=6,
                candidate_fingerprints=(candidate.fingerprint,),
            )

        def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation:
            return OuterPromotionReservation(
                reservation_id="outer-existing",
                request_digest=request.digest,
                ledger_revision=4,
                candidate_id=request.candidate_id,
                candidate_fingerprint=request.candidate_fingerprint,
                consumes_slot=True,
            )

        def complete(
            self,
            reservation: OuterPromotionReservation,
            completion: OuterPromotionCompletion,
        ) -> None:
            assert reservation.reservation_id == "outer-existing"
            self.completions.append(completion)

    ledger = RestartLedger()
    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(candidate,),
        runner=lambda request: _evidence(request, 0.604),
        outer_ledger=ledger,
    )

    assert result.candidates[0].outcome is CandidateOutcome.PROMOTED_CONFIRMED
    assert result.incumbent.candidate_id == candidate.candidate_id
    assert len(ledger.completions) == 1
    assert ledger.completions[0].reservation_revision == 4


def test_outer_snapshot_is_bound_to_campaign_and_benchmark_identity() -> None:
    class WrongCampaignLedger(_Ledger):
        def snapshot(self) -> OuterPromotionLedgerSnapshot:
            valid = super().snapshot()
            return OuterPromotionLedgerSnapshot(
                revision=valid.revision,
                campaign_digest=_DIGESTS[0],
                benchmark_digest=valid.benchmark_digest,
                dataset_digest=valid.dataset_digest,
                scorer_digest=valid.scorer_digest,
                max_distinct_candidates=valid.max_distinct_candidates,
                candidate_fingerprints=valid.candidate_fingerprints,
            )

    with pytest.raises(ScientificCampaignError, match="campaign_digest mismatch"):
        run_scientific_campaign(
            config=_config(),
            fallback=_fallback(),
            candidates=(),
            runner=lambda request: _evidence(request, 0.604),
            outer_ledger=WrongCampaignLedger(),
        )


def test_alias_or_config_only_candidate_is_rejected_before_any_launch() -> None:
    with pytest.raises(ScientificCampaignError, match="must differ"):
        _material_change(candidate_source_digest=_DIGESTS[11])

    with pytest.raises(ScientificCampaignError, match="changed_symbols cannot be empty"):
        _material_change(changed_symbols=())

    with pytest.raises(ScientificCampaignError, match="reachable executable Python"):
        _material_change(
            changed_symbols=("candidate.py:score",),
            reachable_python_files=("config.json",),
        )


def test_nonexecutable_or_unreachable_declared_change_is_rejected() -> None:
    with pytest.raises(ScientificCampaignError, match="reachable executable Python"):
        _material_change(
            changed_symbols=("README.md:score",),
            reachable_python_files=("README.md",),
        )

    with pytest.raises(ScientificCampaignError, match="must name a reachable Python file"):
        _material_change(
            changed_symbols=("helper.py:score",),
            reachable_python_files=("candidate.py",),
        )


def test_material_change_identity_is_bound_into_candidate_and_run_request() -> None:
    candidate = _candidate()
    renamed = _candidate("renamed")

    assert candidate.fingerprint == renamed.fingerprint
    assert candidate.executable_change.candidate_source_digest == candidate.source_digest
    assert candidate.executable_change.parent_source_digest == candidate.parent_source_digest

    requests: list[ScientificRunRequest] = []

    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        requests.append(request)
        return _evidence(request, 0.599)

    run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(candidate,),
        runner=runner,
        outer_ledger=_Ledger(),
    )
    request = requests[0]
    assert request.parent_source_digest == candidate.parent_source_digest
    assert request.executable_diff_digest == candidate.executable_change.executable_diff_digest
    assert request.material_change_digest == candidate.executable_change.digest
    assert (
        request.controller_attestation_digest
        == candidate.executable_change.controller_attestation_digest
    )


def test_mismatched_material_change_receipt_is_a_charged_callback_failure() -> None:
    def runner(request: ScientificRunRequest) -> ScientificRunEvidence:
        valid = _evidence(request, 0.61)
        wrong_identities = replace(
            valid.identities,
            material_change_digest=_DIGESTS[0],
        )
        return ScientificRunEvidence(
            request_digest=valid.request_digest,
            metrics=valid.metrics,
            gates=valid.gates,
            identities=wrong_identities,
            resources=valid.resources,
            replay_verified=valid.replay_verified,
        )

    result = run_scientific_campaign(
        config=_config(),
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=runner,
        outer_ledger=_Ledger(),
    )

    assert result.candidates[0].outcome is CandidateOutcome.CALLBACK_FAILED
    assert result.launches_used == 7
    assert result.incumbent == result.fallback

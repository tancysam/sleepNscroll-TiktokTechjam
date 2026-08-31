"""Trainer preflight and exact same-backend deterministic replay qualification."""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import numpy as np

from kuairand_agent.domain.identity import AttemptId
from kuairand_agent.training.protocol import (
    EnvironmentReceipt,
    QualificationReceipt,
    QualificationStatus,
    QualifiedTrainer,
    ResourceReceipt,
    TrainerError,
    TrainerFailureCode,
    TrialRequest,
    TrialResult,
)


def _terminal_failure(
    trainer: QualifiedTrainer,
    request: TrialRequest,
    code: TrainerFailureCode,
    detail: str,
    *,
    checks: tuple[str, ...],
) -> QualificationReceipt:
    return QualificationReceipt(
        trainer_identity=trainer.identity,
        trial_id=request.trial_id,
        status=QualificationStatus.FAILED,
        checks=checks,
        environment=EnvironmentReceipt.capture(trainer.identity),
        failure_code=code,
        detail=detail,
    )


def _replay_request(request: TrialRequest) -> TrialRequest:
    replay_number = request.infrastructure_attempt + 1
    return replace(
        request,
        infrastructure_attempt=replay_number,
        attempt_id=AttemptId.derive(
            trial_id=request.trial_id,
            infrastructure_attempt=replay_number,
        ),
    )


def _percentiles(resources: tuple[ResourceReceipt, ...]) -> tuple[float, float]:
    ordered = sorted(item.wall_seconds for item in resources)
    p50 = float(statistics.median(ordered))
    p95 = ordered[max(1, math.ceil(0.95 * len(ordered))) - 1]
    return p50, p95


def _same_backend_replay(first: TrialResult, replay: TrialResult) -> bool:
    return bool(
        first.trial_id == replay.trial_id
        and first.attempt_id != replay.attempt_id
        and first.trainer_identity == replay.trainer_identity
        and first.environment.backend == replay.environment.backend
        and first.environment.device == replay.environment.device
        and first.environment.precision == replay.environment.precision
        and first.model.backend == replay.model.backend
        and first.model.config_sha256 == replay.model.config_sha256
        and first.model.model_sha256 == replay.model.model_sha256
        and first.ordered_row_ids == replay.ordered_row_ids
        and first.prediction_sha256 == replay.prediction_sha256
        and np.array_equal(first.predictions, replay.predictions)
    )


def qualify_trainer(
    trainer: QualifiedTrainer,
    request: TrialRequest,
    *,
    replay_request: TrialRequest | None = None,
) -> QualificationReceipt:
    """Run preflight and two distinct attempts on one exact backend/trial identity.

    This helper does not perform cross-backend comparison, spend a protected query, or make a
    scientific decision.  It returns a campaign-admissible receipt only when model bytes and exact
    predictions replay on the same backend.
    """

    preflight = trainer.preflight(request)
    if preflight.status is not QualificationStatus.PREFLIGHT_PASSED:
        return preflight
    replay = _replay_request(request) if replay_request is None else replay_request
    if replay.trial_id != request.trial_id:
        return QualificationReceipt(
            trainer_identity=trainer.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.ADMISSION_REJECTED,
            checks=(*preflight.checks, "same-trial-replay-request"),
            environment=preflight.environment,
            failure_code=TrainerFailureCode.ADMISSION_REJECTED,
            detail="replay request must retain the exact TrialId",
        )
    if replay.attempt_id == request.attempt_id:
        return QualificationReceipt(
            trainer_identity=trainer.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.ADMISSION_REJECTED,
            checks=(*preflight.checks, "distinct-replay-attempt"),
            environment=preflight.environment,
            failure_code=TrainerFailureCode.ADMISSION_REJECTED,
            detail="same-backend replay requires a distinct AttemptId",
        )
    replay_preflight = trainer.preflight(replay)
    if replay_preflight.status is not QualificationStatus.PREFLIGHT_PASSED:
        return _terminal_failure(
            trainer,
            request,
            replay_preflight.failure_code or TrainerFailureCode.INTERNAL_ERROR,
            replay_preflight.detail or "replay attempt failed preflight",
            checks=(*preflight.checks, "replay-preflight"),
        )

    try:
        first = trainer.fit_predict(request)
        second = trainer.fit_predict(replay)
    except TrainerError as exc:
        return _terminal_failure(
            trainer,
            request,
            exc.code,
            exc.detail,
            checks=(*preflight.checks, "same-backend-fit-replay"),
        )
    except Exception as exc:  # pragma: no cover - defensive protocol boundary.
        return _terminal_failure(
            trainer,
            request,
            TrainerFailureCode.INTERNAL_ERROR,
            f"unexpected qualification failure: {type(exc).__name__}",
            checks=(*preflight.checks, "same-backend-fit-replay"),
        )

    if not _same_backend_replay(first, second):
        return QualificationReceipt(
            trainer_identity=trainer.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.ADMISSION_REJECTED,
            checks=(*preflight.checks, "same-backend-exact-replay"),
            environment=preflight.environment,
            failure_code=TrainerFailureCode.ADMISSION_REJECTED,
            detail="same-backend replay changed model bytes, row order, or exact predictions",
        )
    resources = (first.resources, second.resources)
    p50, p95 = _percentiles(resources)
    return QualificationReceipt(
        trainer_identity=trainer.identity,
        trial_id=request.trial_id,
        status=QualificationStatus.QUALIFIED,
        checks=(
            *preflight.checks,
            "distinct-replay-attempt",
            "same-backend-identity",
            "same-backend-model-bytes",
            "same-backend-exact-predictions",
            "resource-receipts",
        ),
        environment=preflight.environment,
        same_backend_replay_verified=True,
        first_prediction_sha256=first.prediction_sha256,
        replay_prediction_sha256=second.prediction_sha256,
        result_receipt_sha256=first.digest,
        replay_result_receipt_sha256=second.digest,
        resource_receipts=resources,
        p50_wall_seconds=p50,
        p95_wall_seconds=p95,
    )


__all__ = ["qualify_trainer"]

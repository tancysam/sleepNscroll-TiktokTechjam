from __future__ import annotations

from pathlib import Path

from kuairand_agent.campaign import full_campaign_runtime as runtime
from kuairand_agent.campaign.controller import CAMPAIGN_DATABASE_NAME, CampaignEngine
from kuairand_agent.campaign.scientific import OuterPromotionRequest
from kuairand_agent.campaign.scientific_store import DurableScientificLedgerAdapter
from kuairand_agent.campaign.store import CampaignStore, OuterQueryLedger
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.research.production import SCRIPTED_CANDIDATE_ID
from tests.integration.test_campaign_controller import build_request


def test_only_exact_durable_outer_reservation_authorizes_missing_seed_resume(
    tmp_path: Path,
) -> None:
    request = build_request(tmp_path)
    CampaignEngine().create(request)
    ledger_path = tmp_path / "outer-query-ledger.sqlite3"
    scorer_digest = STARTER_FILE_SHA256["evaluate.py"]

    with (
        CampaignStore.open(
            request.run_dir / CAMPAIGN_DATABASE_NAME,
            campaign_id=request.campaign_id,
        ) as store,
        OuterQueryLedger.create(
            ledger_path,
            max_queries=request.config.validation.outer_promotion_limit,
        ) as ledger,
    ):
        promotion = OuterPromotionRequest(
            campaign_digest=store.identity().config_digest,
            candidate_id=SCRIPTED_CANDIDATE_ID,
            candidate_fingerprint="1" * 64,
            source_digest="2" * 64,
            parent_source_digest="3" * 64,
            executable_diff_digest="4" * 64,
            material_change_digest="5" * 64,
            controller_attestation_digest="6" * 64,
            benchmark_digest=request.benchmark_digest,
            dataset_digest=request.dataset_manifest_digest,
            scorer_digest=scorer_digest,
            training_policy_digest="7" * 64,
        )
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=scorer_digest,
            evidence_registry={},
        )
        adapter.reserve(promotion)

        authorization = runtime._outer_resume_authorization(
            ledger_path=ledger_path,
            request=request,
            scorer_digest=scorer_digest,
            campaign_store=store,
        )

        assert authorization is not None
        assert authorization.request_digest == promotion.digest
        assert authorization.missing_seeds == (0, 1, 2)
        assert store.snapshot().outer_queries_used == 1

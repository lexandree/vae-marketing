from pathlib import Path

from src.services.validation_reporting import (
    build_claim_recommendations,
    build_validation_report_markdown,
)


def test_build_claim_recommendations_preserves_negative_findings() -> None:
    campaign_results = [
        {"CAMPAIGN": 18, "outcome_name": "total_spend", "evidence_classification": "supported"},
        {"CAMPAIGN": 13, "outcome_name": "promo_share", "evidence_classification": "unsupported"},
    ]
    factor_mappings = [
        {"run_id": "beta-best", "latent_dimension": 0, "mapping_decision": "validated"},
        {"run_id": "beta-best", "latent_dimension": 1, "mapping_decision": "rejected"},
    ]

    recommendations = build_claim_recommendations(campaign_results, factor_mappings)

    assert recommendations["campaign_claims"]["keep"]
    assert recommendations["campaign_claims"]["withdraw_or_soften"]
    assert recommendations["latent_claims"]["keep"]
    assert recommendations["latent_claims"]["withdraw_or_soften"]


def test_build_validation_report_markdown_contains_mixed_findings() -> None:
    report = build_validation_report_markdown(
        campaign_results=[
            {"CAMPAIGN": 18, "outcome_name": "total_spend", "effect_size": 1.2, "evidence_classification": "supported"},
            {"CAMPAIGN": 13, "outcome_name": "promo_share", "effect_size": -0.3, "evidence_classification": "weak"},
        ],
        campaign_diagnostics=[
            {"CAMPAIGN": 18, "outcome_name": "total_spend", "balance_pass": True, "placebo_pass": True},
        ],
        latent_results=[
            {"run_id": "beta-best", "latent_dimension": 0, "candidate_attribute": "promo_share", "mapping_decision": "validated"},
            {"run_id": "beta-best", "latent_dimension": 1, "candidate_attribute": "trip_count", "mapping_decision": "rejected"},
        ],
        latent_metrics=[
            {"run_id": "beta-best", "model_type": "beta_vae", "mig_score": 0.2, "sap_score": 0.3},
        ],
        claim_recommendations={
            "campaign_claims": {"keep": ["Campaign 18"], "withdraw_or_soften": ["Campaign 13"]},
            "latent_claims": {"keep": ["Latent 0"], "withdraw_or_soften": ["Latent 1"]},
        },
    )

    assert "supported" in report
    assert "weak" in report
    assert "validated" in report
    assert "rejected" in report

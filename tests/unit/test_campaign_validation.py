import pandas as pd

from src.services.campaign_validation import (
    DEFAULT_PRIMARY_OUTCOMES,
    compute_campaign_effects,
)
from src.utils.statistics import classify_evidence, enforce_label_completeness


def test_classify_evidence_supported() -> None:
    label = classify_evidence(
        treated_sample_size=40,
        comparison_sample_size=45,
        diagnostics={"balance_pass": True, "placebo_pass": True},
    )
    assert label == "supported"


def test_enforce_label_completeness() -> None:
    findings = pd.DataFrame(
        {
            "evidence_classification": ["supported", "weak", None, "unsupported"],
        }
    )
    completeness = enforce_label_completeness(findings)
    assert completeness == 0.75


def test_compute_campaign_effects_uses_primary_outcomes() -> None:
    analysis_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2],
            "CAMPAIGN": [18, 18],
            "treatment_status": ["treated", "comparison"],
            "pre_total_spend": [10.0, 8.0],
            "campaign_total_spend": [18.0, 9.0],
            "post_total_spend": [11.0, 7.0],
            "pre_trip_count": [1.0, 1.0],
            "campaign_trip_count": [2.0, 1.0],
            "post_trip_count": [1.0, 1.0],
            "pre_category_diversity": [1.0, 1.0],
            "campaign_category_diversity": [2.0, 1.0],
            "post_category_diversity": [1.0, 1.0],
            "pre_promo_share": [0.0, 0.0],
            "campaign_promo_share": [0.5, 0.0],
            "post_promo_share": [0.0, 0.0],
        }
    )

    effects_df, diagnostics_df, event_study_df = compute_campaign_effects(
        analysis_df=analysis_df,
        campaign_ids=[18],
        outcomes=None,
        min_treated=1,
        min_comparison=1,
    )

    assert set(effects_df["outcome_name"]) == set(DEFAULT_PRIMARY_OUTCOMES)
    assert effects_df["evidence_classification"].notna().all()
    assert not diagnostics_df.empty
    assert not event_study_df.empty


def test_compute_campaign_effects_supports_propensity_matching() -> None:
    analysis_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "CAMPAIGN": [26, 26, 26, 26],
            "treatment_status": ["treated", "treated", "comparison", "comparison"],
            "eligibility_status": ["eligible", "eligible", "eligible", "eligible"],
            "pre_total_spend": [10.0, 12.0, 10.5, 11.5],
            "campaign_total_spend": [11.0, 13.0, 10.8, 11.8],
            "post_total_spend": [10.5, 12.5, 10.2, 11.2],
            "pre_trip_count": [2.0, 2.5, 2.1, 2.4],
            "campaign_trip_count": [2.0, 3.0, 2.1, 2.5],
            "post_trip_count": [2.0, 2.0, 2.0, 2.0],
            "pre_category_diversity": [4.0, 5.0, 4.1, 5.1],
            "campaign_category_diversity": [4.5, 5.5, 4.2, 5.2],
            "post_category_diversity": [4.1, 5.0, 4.0, 5.0],
            "pre_promo_share": [0.0, 0.0, 0.0, 0.0],
            "campaign_promo_share": [0.1, 0.1, 0.0, 0.0],
            "post_promo_share": [0.0, 0.0, 0.0, 0.0],
            "pre_avg_price_per_unit": [1.0, 1.1, 1.0, 1.1],
            "pre_coupon_redemption_count": [0.0, 0.0, 0.0, 0.0],
            "pre_spend_concentration": [0.4, 0.5, 0.4, 0.5],
            "pre_target_product_share": [0.1, 0.1, 0.1, 0.1],
        }
    )

    effects_df, diagnostics_df, _ = compute_campaign_effects(
        analysis_df=analysis_df,
        campaign_ids=[26],
        outcomes=["total_spend"],
        min_treated=1,
        min_comparison=1,
        matching_method="propensity",
        propensity_caliper=0.2,
    )

    assert not effects_df.empty
    assert not diagnostics_df.empty
    assert diagnostics_df["matching_method"].eq("propensity").all()

"""Quasi-causal campaign validation workflows.

The current validator uses simple campaign-aligned windows:

- `pre_*`: behavior before the campaign window
- `campaign_*`: behavior during the campaign window
- `post_*`: behavior immediately after the campaign window

The working quasi-causal idea is a restrained difference-in-differences style
comparison:

1. make treated and untreated cohorts comparable in the pre-period
2. compare the change from pre -> campaign
3. reject claims when post-period gaps still suggest the cohorts are drifting

This is still a quasi-causal design, not randomized causal proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.services.validation_reporting import serialize_campaign_validation_outputs
from src.utils.matching import (
    attach_propensity_scores,
    build_caliper_matched_cohort,
    summarize_balance,
)
from src.utils.statistics import (
    DEFAULT_LABEL_COMPLETENESS_THRESHOLD,
    classify_evidence,
    enforce_label_completeness,
)

DEFAULT_PRIMARY_OUTCOMES = [
    "total_spend",
    "trip_count",
    "category_diversity",
    "promo_share",
]
DEFAULT_PLACEBO_TOLERANCE = 0.1
DEFAULT_MATCHING_COVARIATES = [
    "pre_total_spend",
    "pre_trip_count",
    "pre_category_diversity",
    "pre_avg_price_per_unit",
    "pre_promo_share",
    "pre_coupon_redemption_count",
    "pre_spend_concentration",
    "pre_target_product_share",
]


def _estimate_did(
    treated_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    outcome_name: str,
) -> float:
    """Estimate a simple difference-in-differences effect.

    This is the core quantity we report:

    (treated campaign - treated pre) - (comparison campaign - comparison pre)

    Positive means the treated cohort moved up more than the comparison cohort
    inside the campaign window; negative means it moved down more.
    """
    treated_delta = (
        treated_df[f"campaign_{outcome_name}"] - treated_df[f"pre_{outcome_name}"]
    ).mean()
    comparison_delta = (
        comparison_df[f"campaign_{outcome_name}"] - comparison_df[f"pre_{outcome_name}"]
    ).mean()
    return float(treated_delta - comparison_delta)


def _event_study_rows(
    campaign_id: int,
    outcome_name: str,
    treated_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Create compact event-study style summaries."""
    rows = []
    for window_type in ("pre", "campaign", "post"):
        treated_mean = float(treated_df[f"{window_type}_{outcome_name}"].mean())
        comparison_mean = float(comparison_df[f"{window_type}_{outcome_name}"].mean())
        rows.append(
            {
                "CAMPAIGN": campaign_id,
                "outcome_name": outcome_name,
                "window_type": window_type,
                "treated_mean": treated_mean,
                "comparison_mean": comparison_mean,
                "difference": treated_mean - comparison_mean,
            }
        )
    return rows


def _prepare_restricted_cohorts(
    campaign_df: pd.DataFrame,
    matching_method: str,
    propensity_caliper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare treated/comparison cohorts for the requested design.

    `none`:
    use the raw treated vs untreated split.

    `propensity`:
    keep only eligible rows, estimate propensity scores from pre-period
    covariates, and then apply caliper matching to enforce tighter common
    support before effect estimation.
    """
    if matching_method == "none":
        treated_df = campaign_df[campaign_df["treatment_status"] == "treated"].copy()
        comparison_df = campaign_df[campaign_df["treatment_status"] == "comparison"].copy()
        return treated_df, comparison_df

    restricted_df = campaign_df[campaign_df["eligibility_status"] == "eligible"].copy()
    if restricted_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    scored_df = attach_propensity_scores(
        restricted_df,
        covariates=DEFAULT_MATCHING_COVARIATES,
    )
    treated_df = scored_df[scored_df["treatment_status"] == "treated"].copy()
    comparison_df = scored_df[scored_df["treatment_status"] == "comparison"].copy()
    matched_df = build_caliper_matched_cohort(
        treated_df,
        comparison_df,
        score_column="propensity_score",
        caliper=propensity_caliper,
    )
    return (
        matched_df[matched_df["cohort_role"] == "treated"].copy(),
        matched_df[matched_df["cohort_role"] == "comparison"].copy(),
    )


def compute_campaign_effects(
    analysis_df: pd.DataFrame,
    campaign_ids: Iterable[int],
    outcomes: list[str] | None,
    min_treated: int,
    min_comparison: int,
    matching_method: str = "none",
    propensity_caliper: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute campaign effects, diagnostics, and event-study summaries.

    In restricted mode this function becomes more selective:

    - rows with incomplete pre/campaign/post activity are dropped
    - untreated households are matched to treated ones on pre-period behavior
    - a caliper removes pairs that are too far apart in propensity space
    """
    selected_outcomes = outcomes or list(DEFAULT_PRIMARY_OUTCOMES)

    effect_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []

    for campaign_id in campaign_ids:
        campaign_df = analysis_df[analysis_df["CAMPAIGN"] == campaign_id].copy()
        raw_treated_df = campaign_df[campaign_df["treatment_status"] == "treated"].copy()
        raw_comparison_df = campaign_df[campaign_df["treatment_status"] == "comparison"].copy()
        treated_df, comparison_df = _prepare_restricted_cohorts(
            campaign_df,
            matching_method=matching_method,
            propensity_caliper=propensity_caliper,
        )
        cohort_rows.append(
            {
                "CAMPAIGN": campaign_id,
                "matching_method": matching_method,
                "raw_treated_size": len(raw_treated_df),
                "raw_comparison_size": len(raw_comparison_df),
                "matched_treated_size": len(treated_df),
                "matched_comparison_size": len(comparison_df),
                "matched_retention_rate": (
                    float(len(treated_df)) / float(len(raw_treated_df))
                    if len(raw_treated_df) > 0 else 0.0
                ),
            }
        )

        for outcome_name in selected_outcomes:
            required_cols = [
                f"pre_{outcome_name}",
                f"campaign_{outcome_name}",
                f"post_{outcome_name}",
            ]
            if not all(col in campaign_df.columns for col in required_cols):
                continue

            balance_df = summarize_balance(
                treated_df,
                comparison_df,
                [column for column in DEFAULT_MATCHING_COVARIATES if column in campaign_df.columns],
            )
            balance_pass = (
                not balance_df.empty
                and balance_df["balance_status"].eq("balanced").all()
            )
            if not balance_df.empty:
                for row in balance_df.to_dict(orient="records"):
                    balance_rows.append(
                        {
                            "CAMPAIGN": campaign_id,
                            "outcome_name": outcome_name,
                            **row,
                        }
                    )

            placebo_effect = float(
                treated_df[f"post_{outcome_name}"].mean() - comparison_df[f"post_{outcome_name}"].mean()
            )
            placebo_pass = abs(placebo_effect) <= max(
                DEFAULT_PLACEBO_TOLERANCE,
                abs(_estimate_did(treated_df, comparison_df, outcome_name)),
            )

            diagnostics = {
                "balance_pass": bool(balance_pass),
                "placebo_pass": bool(placebo_pass),
            }
            evidence = classify_evidence(
                treated_sample_size=len(treated_df),
                comparison_sample_size=len(comparison_df),
                diagnostics=diagnostics,
                min_sample_size=min(min_treated, min_comparison),
            )
            effect_size = _estimate_did(treated_df, comparison_df, outcome_name)
            effect_direction = "increase" if effect_size > 0 else "decrease" if effect_size < 0 else "neutral"

            effect_rows.append(
                {
                    "CAMPAIGN": campaign_id,
                    "outcome_name": outcome_name,
                    "treated_sample_size": len(treated_df),
                    "comparison_sample_size": len(comparison_df),
                    "effect_size": effect_size,
                    "effect_direction": effect_direction,
                    "evidence_classification": evidence,
                }
            )
            diagnostic_rows.append(
                {
                    "CAMPAIGN": campaign_id,
                    "outcome_name": outcome_name,
                    "matching_method": matching_method,
                    "balance_pass": bool(balance_pass),
                    "placebo_pass": bool(placebo_pass),
                    "placebo_effect": placebo_effect,
                }
            )
            event_rows.extend(_event_study_rows(campaign_id, outcome_name, treated_df, comparison_df))

    effects_df = pd.DataFrame(effect_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    event_study_df = pd.DataFrame(event_rows)

    if not effects_df.empty:
        completeness = enforce_label_completeness(effects_df)
        if completeness < DEFAULT_LABEL_COMPLETENESS_THRESHOLD:
            raise ValueError(
                f"Evidence label coverage dropped below the required threshold: {completeness:.2f}"
            )

    balance_details_df = pd.DataFrame(balance_rows)
    cohort_summary_df = pd.DataFrame(cohort_rows)
    return effects_df, diagnostics_df, event_study_df, balance_details_df, cohort_summary_df


def validate_campaign_effects(*, args: Any) -> None:
    """Run campaign validation and diagnostics from CLI arguments."""
    analysis_df = pd.read_parquet(args.analysis_data)
    effects_df, diagnostics_df, event_study_df, balance_details_df, cohort_summary_df = compute_campaign_effects(
        analysis_df=analysis_df,
        campaign_ids=args.campaign_ids,
        outcomes=args.outcomes,
        min_treated=args.min_treated,
        min_comparison=args.min_comparison,
        matching_method=getattr(args, "matching_method", "none"),
        propensity_caliper=getattr(args, "propensity_caliper", 0.02),
    )
    output_dir = Path(args.output_dir)
    serialize_campaign_validation_outputs(
        output_dir,
        effects_df,
        diagnostics_df,
        event_study_df,
        balance_details_df=balance_details_df,
        cohort_summary_df=cohort_summary_df,
    )

    print("\n" + "=" * 50 + "\nCAMPAIGN VALIDATION SUMMARY\n" + "=" * 50)
    print(f"Campaigns: {list(args.campaign_ids)}")
    print(f"Findings: {len(effects_df)}")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 50 + "\n")

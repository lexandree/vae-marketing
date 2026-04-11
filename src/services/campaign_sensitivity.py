"""Sensitivity analysis for campaign validation settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.campaign_dataset import build_campaign_analysis_dataset
from src.data.dataset import load_validation_source
from src.services.campaign_validation import compute_campaign_effects
from src.services.validation_reporting import write_json_artifact


def _to_pandas_frame(frame: Any) -> pd.DataFrame:
    """Return a pandas DataFrame from either pandas- or Polars-like inputs."""
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise TypeError(f"Unsupported analysis frame type: {type(frame)!r}")


def run_campaign_sensitivity(*, args: Any) -> None:
    """Run a reproducible sensitivity grid over window sizes and matching settings."""
    transactions = load_validation_source(args.transactions, "transactions")
    products = load_validation_source(args.products, "products")
    campaign_table = load_validation_source(args.campaign_table, "campaign_table")
    campaign_desc = load_validation_source(args.campaign_desc, "campaign_desc")
    coupon = load_validation_source(args.coupon, "coupon")
    coupon_redempt = load_validation_source(args.coupon_redempt, "coupon_redempt")
    demographics = (
        load_validation_source(args.demographics, "demographics")
        if args.demographics is not None
        else None
    )

    rows: list[dict[str, object]] = []
    for campaign_id in args.campaign_ids:
        for weeks in args.weeks_grid:
            analysis_df, _, _, build_summary = build_campaign_analysis_dataset(
                transactions=transactions,
                products=products,
                campaign_table=campaign_table,
                campaign_desc=campaign_desc,
                coupon=coupon,
                coupon_redempt=coupon_redempt,
                demographics=demographics,
                campaign_ids=[campaign_id],
                pre_weeks=weeks,
                post_weeks=weeks,
            )
            for matching_method in args.matching_methods:
                for caliper in args.propensity_calipers:
                    effects_df, diagnostics_df, _, balance_df, cohort_df = compute_campaign_effects(
                        analysis_df=_to_pandas_frame(analysis_df),
                        campaign_ids=[campaign_id],
                        outcomes=args.outcomes,
                        min_treated=args.min_treated,
                        min_comparison=args.min_comparison,
                        matching_method=matching_method,
                        propensity_caliper=caliper,
                    )
                    diagnostics = {
                        row["outcome_name"]: row for row in diagnostics_df.to_dict(orient="records")
                    }
                    balance_map: dict[str, list[dict[str, object]]] = {}
                    for row in balance_df.to_dict(orient="records"):
                        balance_map.setdefault(str(row["outcome_name"]), []).append(row)
                    cohort_row = cohort_df.to_dict(orient="records")[0] if not cohort_df.empty else {}
                    for effect in effects_df.to_dict(orient="records"):
                        outcome = str(effect["outcome_name"])
                        balance_rows = balance_map.get(outcome, [])
                        worst_smd = (
                            max(abs(float(row["standardized_mean_difference"])) for row in balance_rows)
                            if balance_rows else None
                        )
                        rows.append(
                            {
                                "campaign_id": campaign_id,
                                "weeks": weeks,
                                "matching_method": matching_method,
                                "propensity_caliper": caliper,
                                "outcome_name": outcome,
                                "effect_size": effect["effect_size"],
                                "effect_direction": effect["effect_direction"],
                                "evidence_classification": effect["evidence_classification"],
                                "balance_pass": diagnostics[outcome]["balance_pass"],
                                "placebo_pass": diagnostics[outcome]["placebo_pass"],
                                "placebo_effect": diagnostics[outcome]["placebo_effect"],
                                "worst_pre_smd": worst_smd,
                                "matched_treated_size": cohort_row.get("matched_treated_size"),
                                "matched_comparison_size": cohort_row.get("matched_comparison_size"),
                                "matched_retention_rate": cohort_row.get("matched_retention_rate"),
                                "analysis_records": build_summary["analysis_records"],
                                "excluded_records": build_summary["excluded_records"],
                            }
                        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "campaign_sensitivity.json", rows)
    pd.DataFrame(rows).to_parquet(output_dir / "campaign_sensitivity.parquet", index=False)

    print("\n" + "=" * 50 + "\nCAMPAIGN SENSITIVITY SUMMARY\n" + "=" * 50)
    print(f"Campaigns: {list(args.campaign_ids)}")
    print(f"Rows: {len(rows)}")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 50 + "\n")

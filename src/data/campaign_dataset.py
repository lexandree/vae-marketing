"""Campaign-linked dataset assembly helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.data.campaign_windows import build_campaign_windows
from src.data.validation_attributes import (
    compute_window_attributes,
    to_validation_attribute_rows,
)


def _prepare_transactions(
    transactions: pl.DataFrame,
    products: pl.DataFrame,
    causal_data: pl.DataFrame | None,
) -> pl.DataFrame:
    """Prepare transactions with product hierarchy and promo proxy flags."""
    tx = transactions.join(
        products.select(["PRODUCT_ID", "COMMODITY_DESC"]),
        on="PRODUCT_ID",
        how="left",
    )
    tx = tx.with_columns(pl.col("COMMODITY_DESC").fill_null("UNKNOWN"))

    if causal_data is not None and not causal_data.is_empty():
        promoted = causal_data.with_columns(
            ((pl.col("display") > 0) | (pl.col("mailer").is_not_null())).cast(pl.Int64).alias("PROMOTED")
        ).select(["PRODUCT_ID", "STORE_ID", "WEEK_NO", "PROMOTED"])
        tx = tx.join(promoted, on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"], how="left")
    else:
        tx = tx.with_columns(pl.lit(0).alias("PROMOTED"))

    return tx.with_columns(pl.col("PROMOTED").fill_null(0))


def _slice_window(df: pl.DataFrame, start_day: int, end_day: int) -> pl.DataFrame:
    """Slice a transaction frame by day window."""
    return df.filter((pl.col("DAY") >= start_day) & (pl.col("DAY") <= end_day))


def build_campaign_analysis_dataset(
    *,
    transactions: pl.DataFrame,
    products: pl.DataFrame,
    campaign_table: pl.DataFrame,
    campaign_desc: pl.DataFrame,
    coupon: pl.DataFrame,
    coupon_redempt: pl.DataFrame,
    demographics: pl.DataFrame | None = None,
    causal_data: pl.DataFrame | None = None,
    campaign_ids: list[int] | None = None,
    pre_weeks: int = 8,
    post_weeks: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Build campaign-linked treated and comparison records plus attributes."""
    selected_ids = campaign_ids or campaign_desc["CAMPAIGN"].unique().to_list()
    selected_campaigns = campaign_desc.filter(pl.col("CAMPAIGN").is_in(selected_ids))
    windows = build_campaign_windows(selected_campaigns, pre_weeks=pre_weeks, post_weeks=post_weeks)
    prepared_transactions = _prepare_transactions(transactions, products, causal_data)

    analysis_rows: list[dict[str, Any]] = []
    attribute_rows: list[dict[str, Any]] = []
    exclusions = 0

    all_households = set(prepared_transactions["HOUSEHOLD_KEY"].unique().to_list())

    demographics_lookup = demographics if demographics is not None else pl.DataFrame()

    for campaign in windows.iter_rows(named=True):
        campaign_id = int(campaign["CAMPAIGN"])
        treated_households = set(
            campaign_table.filter(pl.col("CAMPAIGN") == campaign_id)["household_key"].unique().to_list()
        )
        comparison_households = sorted(all_households.difference(treated_households))
        coupon_product_ids = set(
            coupon.filter(pl.col("CAMPAIGN") == campaign_id)["PRODUCT_ID"].unique().to_list()
        )

        candidate_households = (
            [(household, "treated") for household in sorted(treated_households)]
            + [(household, "comparison") for household in comparison_households]
        )

        for household_key, treatment_status in candidate_households:
            household_tx = prepared_transactions.filter(pl.col("HOUSEHOLD_KEY") == household_key)
            pre_tx = _slice_window(
                household_tx,
                int(campaign["pre_window_start_day"]),
                int(campaign["pre_window_end_day"]),
            )
            in_tx = _slice_window(
                household_tx,
                int(campaign["campaign_start_day"]),
                int(campaign["campaign_end_day"]),
            )
            post_tx = _slice_window(
                household_tx,
                int(campaign["post_window_start_day"]),
                int(campaign["post_window_end_day"]),
            )

            eligibility_status = "eligible"
            if pre_tx.is_empty() or in_tx.is_empty() or post_tx.is_empty():
                eligibility_status = "insufficient_activity"
                exclusions += 1

            coupon_redemption_window = coupon_redempt.filter(
                (pl.col("household_key") == household_key)
                & (pl.col("CAMPAIGN") == campaign_id)
            )

            pre_attrs = compute_window_attributes(
                pre_tx,
                _slice_window(
                    coupon_redemption_window,
                    int(campaign["pre_window_start_day"]),
                    int(campaign["pre_window_end_day"]),
                ),
                coupon_product_ids,
            )
            in_attrs = compute_window_attributes(
                in_tx,
                _slice_window(
                    coupon_redemption_window,
                    int(campaign["campaign_start_day"]),
                    int(campaign["campaign_end_day"]),
                ),
                coupon_product_ids,
            )
            post_attrs = compute_window_attributes(
                post_tx,
                _slice_window(
                    coupon_redemption_window,
                    int(campaign["post_window_start_day"]),
                    int(campaign["post_window_end_day"]),
                ),
                coupon_product_ids,
            )

            total_attrs = {key: pre_attrs[key] + in_attrs[key] + post_attrs[key] for key in pre_attrs}
            prefixed_attrs = {
                **{f"pre_{key}": value for key, value in pre_attrs.items()},
                **{f"campaign_{key}": value for key, value in in_attrs.items()},
                **{f"post_{key}": value for key, value in post_attrs.items()},
            }
            analysis_row = {
                "HOUSEHOLD_KEY": household_key,
                "CAMPAIGN": campaign_id,
                "DESCRIPTION": campaign.get("DESCRIPTION", None),
                "treatment_status": treatment_status,
                "pre_window_start_day": int(campaign["pre_window_start_day"]),
                "pre_window_end_day": int(campaign["pre_window_end_day"]),
                "campaign_start_day": int(campaign["campaign_start_day"]),
                "campaign_end_day": int(campaign["campaign_end_day"]),
                "post_window_start_day": int(campaign["post_window_start_day"]),
                "post_window_end_day": int(campaign["post_window_end_day"]),
                "eligibility_status": eligibility_status,
                **prefixed_attrs,
                **total_attrs,
            }

            if not demographics_lookup.is_empty() and "household_key" in demographics_lookup.columns:
                match = demographics_lookup.filter(pl.col("household_key") == household_key)
                if not match.is_empty() and "AGE_DESC" in match.columns:
                    analysis_row["AGE_DESC"] = match["AGE_DESC"][0]

            analysis_rows.append(analysis_row)
            attribute_rows.extend(to_validation_attribute_rows(household_key, campaign_id, "pre", pre_attrs))
            attribute_rows.extend(to_validation_attribute_rows(household_key, campaign_id, "campaign", in_attrs))
            attribute_rows.extend(to_validation_attribute_rows(household_key, campaign_id, "post", post_attrs))

    analysis_df = pl.DataFrame(analysis_rows).sort(["CAMPAIGN", "HOUSEHOLD_KEY"])
    comparison_df = analysis_df.filter(pl.col("treatment_status") == "comparison")
    attributes_df = pl.DataFrame(attribute_rows).sort(["CAMPAIGN", "HOUSEHOLD_KEY", "window_type", "attribute_name"])

    summary = {
        "selected_campaigns": selected_ids,
        "analysis_records": analysis_df.height,
        "treated_records": analysis_df.filter(pl.col("treatment_status") == "treated").height,
        "comparison_records": comparison_df.height,
        "validation_attribute_rows": attributes_df.height,
        "excluded_records": exclusions,
    }

    return analysis_df, comparison_df, attributes_df, summary


def write_validation_artifacts(
    output_dir: Path,
    analysis_df: pl.DataFrame,
    comparison_df: pl.DataFrame,
    attributes_df: pl.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write campaign validation artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df.write_parquet(output_dir / "campaign_analysis.parquet")
    comparison_df.write_parquet(output_dir / "comparison_pool.parquet")
    attributes_df.write_parquet(output_dir / "validation_attributes.parquet")
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

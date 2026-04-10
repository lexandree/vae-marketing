"""Observable validation attribute aggregation."""

from __future__ import annotations

import math

import polars as pl


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide numeric values."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_window_attributes(
    window_transactions: pl.DataFrame,
    coupon_redemptions: pl.DataFrame,
    coupon_product_ids: set[int],
) -> dict[str, float]:
    """Aggregate behavioral outcomes for a campaign-aligned window."""
    if window_transactions.is_empty():
        return {
            "total_spend": 0.0,
            "total_quantity": 0.0,
            "trip_count": 0.0,
            "avg_price_per_unit": 0.0,
            "promo_share": 0.0,
            "coupon_redemption_count": 0.0,
            "category_diversity": 0.0,
            "spend_concentration": 0.0,
            "target_product_share": 0.0,
        }

    total_spend = float(window_transactions["SALES_VALUE"].sum())
    total_quantity = float(window_transactions["QUANTITY"].sum())
    trip_count = float(window_transactions["BASKET_ID"].n_unique())
    avg_price = _safe_divide(total_spend, total_quantity)

    promo_spend = 0.0
    if "PROMOTED" in window_transactions.columns:
        promo_spend = float(
            window_transactions.filter(pl.col("PROMOTED") > 0)["SALES_VALUE"].sum()
        )

    target_spend = 0.0
    if coupon_product_ids:
        target_spend = float(
            window_transactions.filter(pl.col("PRODUCT_ID").is_in(coupon_product_ids))["SALES_VALUE"].sum()
        )

    commodity_spend = (
        window_transactions.group_by("COMMODITY_DESC")
        .agg(pl.col("SALES_VALUE").sum().alias("commodity_spend"))
    )
    spend_values = commodity_spend["commodity_spend"].to_list()
    spend_concentration = 0.0
    if spend_values and total_spend > 0:
        shares = [value / total_spend for value in spend_values]
        spend_concentration = float(sum(value * value for value in shares))

    return {
        "total_spend": total_spend,
        "total_quantity": total_quantity,
        "trip_count": trip_count,
        "avg_price_per_unit": avg_price,
        "promo_share": _safe_divide(promo_spend, total_spend),
        "coupon_redemption_count": float(coupon_redemptions.height),
        "category_diversity": float(commodity_spend.height),
        "spend_concentration": spend_concentration,
        "target_product_share": _safe_divide(target_spend, total_spend),
    }


def to_validation_attribute_rows(
    household_key: int,
    campaign_id: int,
    window_type: str,
    attributes: dict[str, float],
) -> list[dict[str, float | int | str]]:
    """Convert a window attribute dictionary into row-oriented validation attributes."""
    family_map = {
        "total_spend": "spending",
        "total_quantity": "spending",
        "trip_count": "frequency",
        "avg_price_per_unit": "pricing",
        "promo_share": "promotion",
        "coupon_redemption_count": "promotion",
        "category_diversity": "diversity",
        "spend_concentration": "concentration",
        "target_product_share": "promotion",
    }
    return [
        {
            "HOUSEHOLD_KEY": household_key,
            "CAMPAIGN": campaign_id,
            "window_type": window_type,
            "attribute_name": name,
            "attribute_value": value,
            "attribute_family": family_map.get(name, "other"),
        }
        for name, value in attributes.items()
        if not math.isnan(value)
    ]

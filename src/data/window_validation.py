"""Window-aligned observable attribute builders for latent validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def build_window_attribute_dataset(
    *,
    transactions: pl.DataFrame,
    products: pl.DataFrame,
    prepared_frame: pl.DataFrame,
) -> pl.DataFrame:
    """Build observable attributes aligned to prepared household-week windows.

    This builder is for latent validation on prepared `HOUSEHOLD_KEY /
    WINDOW_START_DAY` frames. Unlike campaign validation attributes, it does not
    assume campaign labels; it computes general weekly behavior summaries.
    """
    if prepared_frame.is_empty():
        return pl.DataFrame(
            schema={
                "HOUSEHOLD_KEY": pl.Utf8,
                "WINDOW_START_DAY": pl.Int64,
                "total_spend": pl.Float64,
                "total_quantity": pl.Float64,
                "trip_count": pl.Float64,
                "avg_price_per_unit": pl.Float64,
                "category_diversity": pl.Float64,
                "spend_concentration": pl.Float64,
            }
        )

    keys = prepared_frame.select(["HOUSEHOLD_KEY", "WINDOW_START_DAY"]).unique()
    tx = (
        transactions
        .filter((pl.col("SALES_VALUE") > 0) & (pl.col("QUANTITY") > 0))
        .join(products.select(["PRODUCT_ID", "COMMODITY_DESC"]), on="PRODUCT_ID", how="left")
        .with_columns(
            pl.col("COMMODITY_DESC").fill_null("UNKNOWN"),
            ((pl.col("DAY") // 7) * 7).cast(pl.Int64).alias("WINDOW_START_DAY"),
            pl.col("HOUSEHOLD_KEY").cast(pl.Utf8),
        )
    )

    commodity_spend = (
        tx.group_by(["HOUSEHOLD_KEY", "WINDOW_START_DAY", "COMMODITY_DESC"])
        .agg(pl.col("SALES_VALUE").sum().alias("commodity_spend"))
        .with_columns(
            (pl.col("commodity_spend") * pl.col("commodity_spend")).alias("commodity_spend_sq")
        )
    )
    spend_totals = commodity_spend.group_by(["HOUSEHOLD_KEY", "WINDOW_START_DAY"]).agg(
        pl.col("commodity_spend").sum().alias("total_spend"),
        pl.col("commodity_spend_sq").sum().alias("sum_spend_sq"),
        pl.len().alias("category_diversity"),
    )

    base_agg = tx.group_by(["HOUSEHOLD_KEY", "WINDOW_START_DAY"]).agg(
        pl.col("QUANTITY").sum().alias("total_quantity"),
        pl.col("BASKET_ID").n_unique().alias("trip_count"),
    )

    attributes = (
        keys.with_columns(pl.col("HOUSEHOLD_KEY").cast(pl.Utf8), pl.col("WINDOW_START_DAY").cast(pl.Int64))
        .join(spend_totals, on=["HOUSEHOLD_KEY", "WINDOW_START_DAY"], how="left")
        .join(base_agg, on=["HOUSEHOLD_KEY", "WINDOW_START_DAY"], how="left")
        .with_columns(
            pl.col("total_spend").fill_null(0.0),
            pl.col("sum_spend_sq").fill_null(0.0),
            pl.col("category_diversity").fill_null(0).cast(pl.Float64),
            pl.col("total_quantity").fill_null(0.0),
            pl.col("trip_count").fill_null(0).cast(pl.Float64),
        )
        .with_columns(
            pl.when(pl.col("total_quantity") > 0)
            .then(pl.col("total_spend") / pl.col("total_quantity"))
            .otherwise(0.0)
            .alias("avg_price_per_unit"),
            pl.when(pl.col("total_spend") > 0)
            .then(pl.col("sum_spend_sq") / (pl.col("total_spend") * pl.col("total_spend")))
            .otherwise(0.0)
            .alias("spend_concentration"),
        )
        .select(
            [
                "HOUSEHOLD_KEY",
                "WINDOW_START_DAY",
                "total_spend",
                "total_quantity",
                "trip_count",
                "avg_price_per_unit",
                "category_diversity",
                "spend_concentration",
            ]
        )
        .sort(["HOUSEHOLD_KEY", "WINDOW_START_DAY"])
    )
    return attributes


def write_window_attribute_artifact(attributes_df: pl.DataFrame, output_path: Path) -> None:
    """Write aligned window attribute parquet to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attributes_df.write_parquet(output_path)


def build_and_write_window_attributes(*, args: Any) -> None:
    """CLI entry point for aligned window attribute generation."""
    from src.data.dataset import load_validation_source

    transactions = load_validation_source(args.transactions, "transactions")
    products = load_validation_source(args.products, "products")
    prepared_frame = pl.read_parquet(args.prepared_data)
    attributes_df = build_window_attribute_dataset(
        transactions=transactions,
        products=products,
        prepared_frame=prepared_frame,
    )
    output_path = Path(args.output)
    write_window_attribute_artifact(attributes_df, output_path)

    print("\n" + "=" * 50 + "\nWINDOW ATTRIBUTE SUMMARY\n" + "=" * 50)
    print(f"Rows: {attributes_df.height}")
    print(f"Output: {output_path}")
    print("=" * 50 + "\n")

"""Extractors module for data preparation.

This module provides functions for temporal and hierarchical feature extraction
from raw transaction and product data using Polars.
"""

import numpy as np
import polars as pl

def filter_transactions(transactions: pl.DataFrame) -> pl.DataFrame:
    """Filter out non-purchase transactions (returns/refunds/coupons).
    
    Returns are identified by negative SALES_VALUE or QUANTITY.
    Coupons are identified by negative SALES_VALUE and zero QUANTITY, or similar.
    We just keep records where SALES_VALUE > 0 and QUANTITY > 0.
    """
    return transactions.filter(
        (pl.col("SALES_VALUE") > 0) & (pl.col("QUANTITY") > 0)
    )

def extract_vocabulary(products: pl.DataFrame) -> list[str]:
    """Extract a sorted list of valid unique commodity descriptions."""
    vocab = products.filter(
        (pl.col("COMMODITY_DESC").is_not_null()) & 
        (pl.col("COMMODITY_DESC") != "NO COMMODITY DESCRIPTION")
    )["COMMODITY_DESC"].unique().to_list()
    
    # Clean up and sort
    vocab = [str(v) for v in vocab if str(v).strip()]
    
    # Ensure UNKNOWN is always part of the vocabulary for fallback
    if "UNKNOWN" not in vocab:
        vocab.append("UNKNOWN")
        
    return sorted(list(set(vocab)))

def map_hierarchical_features(
    transactions: pl.DataFrame, 
    products: pl.DataFrame,
    vocabulary: list[str] | None = None
) -> pl.DataFrame:
    """Map transactions to product hierarchy.
    
    Aggregates spend and quantity by COMMODITY_DESC.
    If vocabulary is provided, non-vocabulary categories are mapped to 'UNKNOWN'.
    """
    mapped = transactions.join(
        products.select(["PRODUCT_ID", "COMMODITY_DESC"]),
        on="PRODUCT_ID",
        how="left"
    )
    
    if vocabulary is not None:
        mapped = mapped.with_columns(
            pl.when(pl.col("COMMODITY_DESC").is_in(vocabulary))
            .then(pl.col("COMMODITY_DESC"))
            .otherwise(pl.lit("UNKNOWN"))
            .alias("COMMODITY_DESC")
        )
        
    return mapped

def add_temporal_encodings(df: pl.DataFrame, day_col: str = "DAY") -> pl.DataFrame:
    """Add cyclical continuous encoding for temporal features.
    
    Assuming DAY is day of the dataset (starting from 1).
    We can derive:
    - week_of_year = (DAY // 7) % 52
    - day_of_week = DAY % 7
    - month_of_year = (DAY // 30) % 12 (approximate for modeling)
    """
    return df.with_columns([
        (np.sin(2 * np.pi * ((pl.col(day_col) // 7) % 52) / 52.0)).alias("TEMPORAL_WEEK_SIN"),
        (np.cos(2 * np.pi * ((pl.col(day_col) // 7) % 52) / 52.0)).alias("TEMPORAL_WEEK_COS"),
        (np.sin(2 * np.pi * (pl.col(day_col) % 7) / 7.0)).alias("TEMPORAL_DAY_SIN"),
        (np.cos(2 * np.pi * (pl.col(day_col) % 7) / 7.0)).alias("TEMPORAL_DAY_COS"),
        (np.sin(2 * np.pi * ((pl.col(day_col) // 30) % 12) / 12.0)).alias("TEMPORAL_MONTH_SIN"),
        (np.cos(2 * np.pi * ((pl.col(day_col) // 30) % 12) / 12.0)).alias("TEMPORAL_MONTH_COS"),
    ])

def extract_features(
    transactions: pl.DataFrame, 
    products: pl.DataFrame,
    vocabulary: list[str] | None = None
) -> pl.DataFrame:
    """Extract features with 7-day rolling window aggregation.
    
    Steps:
    1. Filter out non-purchases.
    2. Map products to vocabulary categories.
    3. Determine the start and end days for each household.
    4. Create a dense grid of households x 7-day windows.
    5. Aggregate spend and quantity into 7-day windows per household.
    6. Pivot commodities to columns.
    7. Join the temporal features.
    """
    # 1. Filter
    df = filter_transactions(transactions)
    
    # 2. Map
    df = map_hierarchical_features(df, products, vocabulary)
    
    # Add WINDOW_START_DAY based on DAY. 
    df = df.with_columns(
        ((pl.col("DAY") // 7) * 7).alias("WINDOW_START_DAY")
    )
    
    # Check if df is empty
    if len(df) == 0:
        return pl.DataFrame()
        
    # 4 & 5. Aggregate by Household, Window, and Commodity
    agg_df = df.group_by(["HOUSEHOLD_KEY", "WINDOW_START_DAY", "COMMODITY_DESC"]).agg([
        pl.col("SALES_VALUE").sum().alias("SPEND"),
        pl.col("QUANTITY").sum().alias("QTY"),
    ])
    
    # 6. Pivot commodities to wide format
    # Pivot for SPEND
    spend_wide = agg_df.pivot(
        values="SPEND",
        index=["HOUSEHOLD_KEY", "WINDOW_START_DAY"],
        on="COMMODITY_DESC",
        aggregate_function="sum"
    ).fill_null(0.0)
    
    spend_wide = spend_wide.rename({
        col: f"COMMODITY_{col}_SPEND" 
        for col in spend_wide.columns 
        if col not in ["HOUSEHOLD_KEY", "WINDOW_START_DAY"]
    })
    
    # Pivot for QTY
    qty_wide = agg_df.pivot(
        values="QTY",
        index=["HOUSEHOLD_KEY", "WINDOW_START_DAY"],
        on="COMMODITY_DESC",
        aggregate_function="sum"
    ).fill_null(0.0)
    
    qty_wide = qty_wide.rename({
        col: f"COMMODITY_{col}_QTY" 
        for col in qty_wide.columns 
        if col not in ["HOUSEHOLD_KEY", "WINDOW_START_DAY"]
    })
    
    # Join wide dfs
    wide_df = spend_wide.join(
        qty_wide, 
        on=["HOUSEHOLD_KEY", "WINDOW_START_DAY"],
        how="full"
    ).fill_null(0.0)
    
    # Force expected columns to exist if vocabulary is provided
    if vocabulary is not None:
        expected_commodities = vocabulary + ["UNKNOWN"]
        for comm in expected_commodities:
            spend_col = f"COMMODITY_{comm}_SPEND"
            qty_col = f"COMMODITY_{comm}_QTY"
            
            if spend_col not in wide_df.columns:
                wide_df = wide_df.with_columns(pl.lit(0.0).alias(spend_col))
            if qty_col not in wide_df.columns:
                wide_df = wide_df.with_columns(pl.lit(0.0).alias(qty_col))
    
    # Add dense grid (impute gaps as 0s)
    min_window = wide_df["WINDOW_START_DAY"].min()
    max_window = wide_df["WINDOW_START_DAY"].max()
    
    if min_window is None or max_window is None:
        return pl.DataFrame()
        
    windows = list(range(int(min_window), int(max_window) + 7, 7))
    households = wide_df["HOUSEHOLD_KEY"].unique()
    
    # Create grid
    grid = pl.DataFrame({
        "HOUSEHOLD_KEY": np.repeat(households.to_list(), len(windows)),
        "WINDOW_START_DAY": windows * len(households)
    }).with_columns(pl.col("WINDOW_START_DAY").cast(pl.Int64))
    
    wide_df = wide_df.with_columns(pl.col("WINDOW_START_DAY").cast(pl.Int64))
    
    # Join to get dense data
    dense_df = grid.join(
        wide_df,
        on=["HOUSEHOLD_KEY", "WINDOW_START_DAY"],
        how="left"
    ).fill_null(0.0)
    
    # 7. Add temporal encodings
    final_df = add_temporal_encodings(dense_df, day_col="WINDOW_START_DAY")
    
    # To ensure consistent column ordering, let's sort columns alphabetically 
    # for commodities if vocabulary is used.
    if vocabulary is not None:
        base_cols = ["HOUSEHOLD_KEY", "WINDOW_START_DAY"]
        temp_cols = [c for c in final_df.columns if c.startswith("TEMPORAL_")]
        comm_cols = sorted([c for c in final_df.columns if c.startswith("COMMODITY_")])
        final_cols = base_cols + comm_cols + temp_cols
        final_df = final_df.select(final_cols)
    
    return final_df.sort(["HOUSEHOLD_KEY", "WINDOW_START_DAY"])

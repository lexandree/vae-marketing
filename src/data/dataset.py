"""Dataset loading utilities for model and validation workflows."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from src.data.schema import (
    CAMPAIGN_DESC_SCHEMA,
    CAMPAIGN_TABLE_SCHEMA,
    CAUSAL_DATA_SCHEMA,
    COUPON_REDEMPTION_SCHEMA,
    COUPON_SCHEMA,
    DEMOGRAPHIC_SCHEMA,
    PRODUCT_SCHEMA,
    RAW_TRANSACTION_SCHEMA,
)


TRANSACTION_SCHEMA = {
    "transaction_id": pl.Utf8,
    "household_id": pl.Utf8,
    "timestamp": pl.Datetime,
    "product_category": pl.Categorical,
    "quantity": pl.Float32,
    "price": pl.Float32,
    "month_of_year": pl.UInt8,
    "week_of_year": pl.UInt8,
}

HOUSEHOLD_PROFILE_SCHEMA = {
    "household_id": pl.Utf8,
    "baseline_profile": pl.List(pl.Float32),
}

EXTERNAL_STIMULUS_SCHEMA = {
    "stimulus_id": pl.Utf8,
    "stimulus_type": pl.Categorical,
    "start_time": pl.Datetime,
    "end_time": pl.Datetime,
}

BEHAVIORAL_SHIFT_SCHEMA = {
    "household_id": pl.Utf8,
    "stimulus_id": pl.Utf8,
    "quantitative_magnitude": pl.Float32,
    "qualitative_nature": pl.Categorical,
    "persistence_duration_days": pl.Int32,
}

VALIDATION_SOURCE_SCHEMAS = {
    "transactions": RAW_TRANSACTION_SCHEMA,
    "products": PRODUCT_SCHEMA,
    "campaign_table": CAMPAIGN_TABLE_SCHEMA,
    "campaign_desc": CAMPAIGN_DESC_SCHEMA,
    "coupon": COUPON_SCHEMA,
    "coupon_redempt": COUPON_REDEMPTION_SCHEMA,
    "demographics": DEMOGRAPHIC_SCHEMA,
    "causal_data": CAUSAL_DATA_SCHEMA,
}


def load_data(file_path: Union[str, Path]) -> pl.DataFrame:
    """Load a CSV or Parquet dataset using Polars.

    Args:
        file_path: The path to the data file.

    Returns:
        A Polars DataFrame containing the loaded data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".csv":
        df = pl.read_csv(path, infer_schema_length=10000, ignore_errors=False)
    elif path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .parquet")

    cast_exprs = []
    for col_name, dtype in TRANSACTION_SCHEMA.items():
        if col_name in df.columns:
            cast_exprs.append(pl.col(col_name).cast(dtype))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def load_validation_source(file_path: Union[str, Path], source_type: str) -> pl.DataFrame:
    """Load and cast a validation data source using its declared schema.

    Args:
        file_path: Path to the source file.
        source_type: Named validation source type.

    Returns:
        A Polars DataFrame cast to the configured schema where possible.
    """
    if source_type not in VALIDATION_SOURCE_SCHEMAS:
        raise ValueError(f"Unsupported validation source type: {source_type}")

    schema = VALIDATION_SOURCE_SCHEMAS[source_type]
    path = Path(file_path)

    if path.suffix == ".csv":
        df = pl.read_csv(
            path,
            schema_overrides=schema,
            infer_schema_length=10000,
            ignore_errors=False,
        )
    else:
        df = load_data(path)
        cast_exprs = []
        for col_name, dtype in schema.items():
            if col_name in df.columns:
                cast_exprs.append(pl.col(col_name).cast(dtype))
        if cast_exprs:
            df = df.with_columns(cast_exprs)
    return df


def extract_validation_attributes(df: Union[pl.DataFrame, pd.DataFrame]) -> np.ndarray:
    """Extract observable attributes for MIG/SAP-style evaluation.

    Args:
        df: DataFrame (Polars or Pandas) with preprocessed data.

    Returns:
        Numpy array containing available spend and quantity attributes.
    """
    attribute_cols = [c for c in df.columns if c.endswith("_SPEND") or c.endswith("_QTY")]
    if not attribute_cols:
        return np.array([])

    if hasattr(df, "select"):
        return df.select(attribute_cols).to_numpy()
    return df[attribute_cols].to_numpy()

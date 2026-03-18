from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

# Define memory-efficient schema for transactions
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

# Define schema for household baseline profiles (if needed for output)
HOUSEHOLD_PROFILE_SCHEMA = {
    "household_id": pl.Utf8,
    "baseline_profile": pl.List(pl.Float32),
}

# Define memory-efficient schema for external stimuli
EXTERNAL_STIMULUS_SCHEMA = {
    "stimulus_id": pl.Utf8,
    "stimulus_type": pl.Categorical,
    "start_time": pl.Datetime,
    "end_time": pl.Datetime,
    # affected_categories can remain uncasted string lists or categorical lists
    # depending on Polars support
}

# Define schema for behavioral shift
BEHAVIORAL_SHIFT_SCHEMA = {
    "household_id": pl.Utf8,
    "stimulus_id": pl.Utf8,
    "quantitative_magnitude": pl.Float32,
    "qualitative_nature": pl.Categorical,  # 'Stockpiling', 'Trading Up', 'Brand Switching'
    "persistence_duration_days": pl.Int32,
}


def load_data(file_path: Union[str, Path]) -> pl.DataFrame:
    """Loads a dataset from a CSV or Parquet file using Polars.

    Args:
        file_path: The path to the data file.

    Returns:
        A Polars DataFrame containing the loaded data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".csv":
        # Don't strictly enforce schema on load to avoid errors if some cols missing,
        # but can cast later if needed. For now just load.
        df = pl.read_csv(path)
    elif path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .parquet")

    # Cast to memory-efficient types where columns exist
    cast_exprs = []
    for col_name, dtype in TRANSACTION_SCHEMA.items():
        if col_name in df.columns:
            cast_exprs.append(pl.col(col_name).cast(dtype))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def extract_validation_attributes(df: Union[pl.DataFrame, pd.DataFrame]) -> np.ndarray:
    """Extracts ground truth attributes from the dataframe for metric calculation (MIG/SAP).

    Assumes the dataframe has been preprocessed and contains '_SPEND' and '_QTY' columns.

    Args:
        df: DataFrame (Polars or Pandas) with the preprocessed data.

    Returns:
        Numpy array of shape (N, K) containing the attributes.
    """
    if hasattr(df, "select"):  # Polars
        attribute_cols = [c for c in df.columns if c.endswith("_SPEND") or c.endswith("_QTY")]
        if not attribute_cols:
            return np.array([])
        return df.select(attribute_cols).to_numpy()
    else:  # Pandas
        attribute_cols = [c for c in df.columns if c.endswith("_SPEND") or c.endswith("_QTY")]
        if not attribute_cols:
            return np.array([])
        return df[attribute_cols].to_numpy()

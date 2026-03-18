"""Normalizers module for data preparation.

This module provides functions for robust normalization (log-scaling, z-score)
and forward-chaining time-series splits, while avoiding data leakage.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl


def create_time_series_splits(
    df: pl.DataFrame,
    train_weeks: int,
    val_weeks: int,
    window_col: str = "WINDOW_START_DAY"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data using forward-chaining based on the starting window day.

    Assuming 7 days per week.
    Train split: days 0 to (train_weeks * 7) - 1
    Val split: days (train_weeks * 7) to ((train_weeks + val_weeks) * 7) - 1
    Test split: remaining days
    """
    train_end_day = train_weeks * 7
    val_end_day = train_end_day + (val_weeks * 7)

    train_df = df.filter(pl.col(window_col) < train_end_day)
    val_df = df.filter(
        (pl.col(window_col) >= train_end_day) &
        (pl.col(window_col) < val_end_day)
    )
    test_df = df.filter(pl.col(window_col) >= val_end_day)

    return train_df, val_df, test_df

def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Get continuous feature columns that need scaling."""
    return [col for col in df.columns if col.endswith("_SPEND") or col.endswith("_QTY")]

def fit_scalers(train_df: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Fit normalization parameters (mean, std) purely on the training window.

    Applies log1p transformation first: log(x + 1)
    Then calculates mean and std for z-score scaling.
    """
    feature_cols = _get_feature_cols(train_df)

    # We first apply log1p to the features to calculate the mean and std
    log_df = train_df.select([
        pl.col(col).log1p().alias(col) for col in feature_cols
    ])

    params = {}

    if len(log_df) == 0:
        return {col: {"mean": 0.0, "std": 1.0} for col in feature_cols}

    for col in feature_cols:
        col_mean = log_df[col].mean()
        col_std = log_df[col].std(ddof=0)  # Population std for consistency, or ddof=1

        # Handle zero standard deviation
        if col_std is None or col_std == 0.0 or np.isnan(col_std):
            col_std = 1.0

        if col_mean is None or np.isnan(col_mean):
            col_mean = 0.0

        params[col] = {
            "mean": float(col_mean),
            "std": float(col_std)
        }

    return params

def transform_features(df: pl.DataFrame, params: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Apply log1p and z-score scaling using pre-fit parameters."""
    if len(df) == 0:
        return df

    feature_cols = _get_feature_cols(df)

    exprs = []
    for col in feature_cols:
        if col in params:
            col_mean = params[col]["mean"]
            col_std = params[col]["std"]

            # log1p(x) - mean / std
            expr = ((pl.col(col).log1p() - col_mean) / col_std).alias(col)
            exprs.append(expr)

    # Apply transformations and keep all other columns intact
    if exprs:
        return df.with_columns(exprs)

    return df

def save_scaler_params(params: dict[str, dict[str, float]], output_path: Path) -> None:
    """Save normalization parameters to JSON file."""
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

def load_scaler_params(input_path: Path) -> dict[str, dict[str, float]]:
    """Load normalization parameters from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)

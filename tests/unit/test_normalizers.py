"""Unit tests for normalizers module."""

from pathlib import Path

import numpy as np
import polars as pl

from src.data.normalizers import (
    create_time_series_splits,
    fit_scalers,
    load_scaler_params,
    save_scaler_params,
    transform_features,
)


def test_create_time_series_splits():
    """Test forward-chaining splits."""
    df = pl.DataFrame({
        "WINDOW_START_DAY": [0, 7, 14, 140, 147, 210, 217],
        "DATA": [1, 2, 3, 4, 5, 6, 7]
    })

    # Train 20 weeks (0 - 139)
    # Val 10 weeks (140 - 209)
    # Test remaining (210+)

    train, val, test = create_time_series_splits(df, train_weeks=20, val_weeks=10)

    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 2

    assert train["WINDOW_START_DAY"].max() < 140
    assert val["WINDOW_START_DAY"].min() >= 140
    assert val["WINDOW_START_DAY"].max() < 210
    assert test["WINDOW_START_DAY"].min() >= 210

def test_fit_and_transform_features():
    """Test log1p and z-score scaling logic."""
    df = pl.DataFrame({
        "HOUSEHOLD_KEY": ["H1", "H2", "H3"],
        "COMMODITY_A_SPEND": [0.0, 10.0, 100.0],
        "COMMODITY_B_QTY": [1.0, 5.0, 20.0],
        "IGNORE_COL": [1, 2, 3]
    })

    params = fit_scalers(df)

    assert "COMMODITY_A_SPEND" in params
    assert "COMMODITY_B_QTY" in params
    assert "IGNORE_COL" not in params

    transformed = transform_features(df, params)

    # Original should have IGNORE_COL unchanged
    assert transformed["IGNORE_COL"].to_list() == [1, 2, 3]

    # Mean of transformed features should be approximately 0
    # Standard deviation should be approximately 1
    np.testing.assert_almost_equal(transformed["COMMODITY_A_SPEND"].mean(), 0.0)
    np.testing.assert_almost_equal(transformed["COMMODITY_A_SPEND"].std(ddof=0), 1.0)
    np.testing.assert_almost_equal(transformed["COMMODITY_B_QTY"].mean(), 0.0)
    np.testing.assert_almost_equal(transformed["COMMODITY_B_QTY"].std(ddof=0), 1.0)

def test_save_and_load_params(tmp_path: Path):
    """Test saving and loading normalization parameters."""
    params = {
        "COMMODITY_A_SPEND": {"mean": 1.5, "std": 0.5},
        "COMMODITY_B_QTY": {"mean": 0.0, "std": 1.0}
    }

    file_path = tmp_path / "scaler_params.json"
    save_scaler_params(params, file_path)

    assert file_path.exists()

    loaded = load_scaler_params(file_path)
    assert loaded == params

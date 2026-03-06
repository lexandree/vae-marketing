import numpy as np
import polars as pl

from src.data.extractors import (
    add_temporal_encodings,
    extract_features,
    filter_transactions,
    map_hierarchical_features,
)


def test_filter_transactions() -> None:
    """Test filtering of non-purchase transactions."""
    df = pl.DataFrame({
        "SALES_VALUE": [10.0, -5.0, 0.0, 20.0],
        "QUANTITY": [1, 1, 0, 2]
    })
    filtered = filter_transactions(df)
    assert len(filtered) == 2
    assert filtered["SALES_VALUE"][0] == 10.0


def test_map_hierarchical_features() -> None:
    """Test mapping of products to transactions."""
    transactions = pl.DataFrame({
        "PRODUCT_ID": ["A", "B", "C"],
        "SALES_VALUE": [10.0, 20.0, 30.0]
    })
    products = pl.DataFrame({
        "PRODUCT_ID": ["A", "B"],
        "COMMODITY_DESC": ["CAT_A", "CAT_B"]
    })
    mapped = map_hierarchical_features(transactions, products)
    assert mapped.filter(pl.col("PRODUCT_ID") == "A")["COMMODITY_DESC"][0] == "CAT_A"
    assert mapped.filter(pl.col("PRODUCT_ID") == "C")["COMMODITY_DESC"][0] is None


def test_add_temporal_encodings() -> None:
    """Test cyclical encoding of temporal features."""
    df = pl.DataFrame({"DAY": [0, 7, 365]})
    encoded = add_temporal_encodings(df)
    assert "TEMPORAL_WEEK_SIN" in encoded.columns
    np.testing.assert_almost_equal(encoded["TEMPORAL_WEEK_SIN"][0], 0.0)
    np.testing.assert_almost_equal(encoded["TEMPORAL_WEEK_COS"][0], 1.0)


def test_extract_features_rolling_window() -> None:
    """Test full extraction including 7-day rolling window."""
    transactions = pl.DataFrame({
        "HOUSEHOLD_KEY": ["h1", "h1"],
        "PRODUCT_ID": ["A", "B"],
        "DAY": [1, 2],
        "SALES_VALUE": [10.0, 20.0],
        "QUANTITY": [1, 2]
    })
    products = pl.DataFrame({
        "PRODUCT_ID": ["A", "B"],
        "COMMODITY_DESC": ["CAT_A", "CAT_B"]
    })

    features = extract_features(transactions, products)
    assert len(features) > 0
    assert "COMMODITY_CAT_A_SPEND" in features.columns
    assert "TEMPORAL_WEEK_SIN" in features.columns

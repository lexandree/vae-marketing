"""Unit tests for data preparation module."""

import numpy as np
import polars as pl

from src.data.extractors import (
    add_temporal_encodings,
    extract_features,
    filter_transactions,
    map_hierarchical_features,
)

def test_filter_transactions():
    """Test filtering of non-purchase transactions."""
    df = pl.DataFrame({
        "SALES_VALUE": [10.0, -5.0, 0.0, 15.0],
        "QUANTITY": [2.0, 1.0, 0.0, -1.0]
    })
    
    filtered = filter_transactions(df)
    assert len(filtered) == 1
    assert filtered["SALES_VALUE"][0] == 10.0

def test_map_hierarchical_features():
    """Test mapping of products to transactions."""
    transactions = pl.DataFrame({
        "PRODUCT_ID": ["A", "B", "C"],
        "SALES_VALUE": [10.0, 20.0, 30.0]
    })
    
    products = pl.DataFrame({
        "PRODUCT_ID": ["A", "B"],
        "COMMODITY_DESC": ["MEAT", "DAIRY"]
    })
    
    mapped = map_hierarchical_features(transactions, products)
    assert len(mapped) == 3
    assert "COMMODITY_DESC" in mapped.columns
    assert mapped.filter(pl.col("PRODUCT_ID") == "A")["COMMODITY_DESC"][0] == "MEAT"
    assert mapped.filter(pl.col("PRODUCT_ID") == "C")["COMMODITY_DESC"][0] is None

def test_add_temporal_encodings():
    """Test cyclical encoding of temporal features."""
    df = pl.DataFrame({"DAY": [0, 7, 365]})
    encoded = add_temporal_encodings(df, day_col="DAY")
    
    assert "TEMPORAL_WEEK_SIN" in encoded.columns
    assert "TEMPORAL_WEEK_COS" in encoded.columns
    assert "TEMPORAL_DAY_SIN" in encoded.columns
    assert "TEMPORAL_DAY_COS" in encoded.columns
    assert "TEMPORAL_MONTH_SIN" in encoded.columns
    assert "TEMPORAL_MONTH_COS" in encoded.columns
    
    # Check week cyclical nature
    # Day 0 and 7 should be close/exact depending on week definition. 
    # Wait, DAY=0 -> week 0, DAY=7 -> week 1
    # Day 0 is week 0 (sin(0) = 0, cos(0) = 1)
    np.testing.assert_almost_equal(encoded["TEMPORAL_WEEK_SIN"][0], 0.0)
    np.testing.assert_almost_equal(encoded["TEMPORAL_WEEK_COS"][0], 1.0)

def test_extract_features_rolling_window():
    """Test full extraction including 7-day rolling window."""
    transactions = pl.DataFrame({
        "BASKET_ID": ["1", "2", "3", "4"],
        "HOUSEHOLD_KEY": ["H1", "H1", "H2", "H1"],
        "DAY": [1, 2, 8, 15],  # Weeks 0, 1, 2
        "PRODUCT_ID": ["P1", "P2", "P1", "P2"],
        "QUANTITY": [1.0, 2.0, 1.0, 3.0],
        "SALES_VALUE": [10.0, 20.0, 15.0, 30.0],
        "STORE_ID": ["S1", "S1", "S1", "S1"],
        "TRANS_TIME": [1000, 1100, 1200, 1300]
    })
    
    products = pl.DataFrame({
        "PRODUCT_ID": ["P1", "P2"],
        "COMMODITY_DESC": ["C1", "C2"],
        "SUB_COMMODITY_DESC": ["SC1", "SC2"]
    })
    
    features = extract_features(transactions, products)
    
    # Windows should be 0, 7, 14
    # H1 has purchases in week 0 (days 1, 2) and week 2 (day 15). Gap in week 1.
    # H2 has purchase in week 1 (day 8). Gap in week 0 and week 2.
    
    assert "COMMODITY_C1_SPEND" in features.columns
    assert "COMMODITY_C2_QTY" in features.columns
    
    # H1, Week 0 (WINDOW_START_DAY=0)
    h1_w0 = features.filter((pl.col("HOUSEHOLD_KEY") == "H1") & (pl.col("WINDOW_START_DAY") == 0))
    assert len(h1_w0) == 1
    assert h1_w0["COMMODITY_C1_SPEND"][0] == 10.0
    assert h1_w0["COMMODITY_C2_SPEND"][0] == 20.0
    
    # H1, Week 1 (WINDOW_START_DAY=7) - This should be imputed as 0
    h1_w1 = features.filter((pl.col("HOUSEHOLD_KEY") == "H1") & (pl.col("WINDOW_START_DAY") == 7))
    assert len(h1_w1) == 1
    assert h1_w1["COMMODITY_C1_SPEND"][0] == 0.0
    
    # H2, Week 0 - This should be imputed as 0
    h2_w0 = features.filter((pl.col("HOUSEHOLD_KEY") == "H2") & (pl.col("WINDOW_START_DAY") == 0))
    assert len(h2_w0) == 1
    assert h2_w0["COMMODITY_C1_SPEND"][0] == 0.0

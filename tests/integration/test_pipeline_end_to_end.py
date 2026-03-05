"""End-to-end integration test for data preparation pipeline."""

import json
from pathlib import Path

import polars as pl
import pytest

from src.data.prepare import main

def test_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test full data preparation pipeline execution."""
    
    # 1. Create mock data
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    transactions_path = raw_dir / "transaction_data.csv"
    products_path = raw_dir / "product.csv"
    
    transactions = pl.DataFrame({
        "BASKET_ID": ["1", "2", "3", "4", "5", "6"],
        "HOUSEHOLD_KEY": ["H1", "H2", "H1", "H3", "H1", "H2"],
        "DAY": [1, 15, 150, 200, 215, 300], # Weeks: 0, 2, 21, 28, 30, 42
        "PRODUCT_ID": ["P1", "P2", "P1", "P3", "P2", "P1"],
        "QUANTITY": [1.0, 2.0, 1.0, 5.0, 1.0, 2.0],
        "SALES_VALUE": [10.0, 20.0, 15.0, 50.0, 10.0, 25.0],
        "STORE_ID": ["S1"] * 6,
        "TRANS_TIME": [1000] * 6
    })
    
    products = pl.DataFrame({
        "PRODUCT_ID": ["P1", "P2", "P3"],
        "COMMODITY_DESC": ["C1", "C2", "C1"],
        "SUB_COMMODITY_DESC": ["SC1", "SC2", "SC3"]
    })
    
    transactions.write_csv(transactions_path)
    products.write_csv(products_path)
    
    out_dir = tmp_path / "processed"
    
    # 2. Mock CLI arguments
    test_args = [
        "prepare.py",
        "--input-transactions", str(transactions_path),
        "--input-products", str(products_path),
        "--output-dir", str(out_dir),
        "--train-weeks", "20",
        "--val-weeks", "10",
        "--seed", "42"
    ]
    
    monkeypatch.setattr("sys.argv", test_args)
    
    # 3. Run pipeline
    main()
    
    # 4. Verify outputs
    assert out_dir.exists()
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "val.parquet").exists()
    assert (out_dir / "test.parquet").exists()
    assert (out_dir / "scaler_params.json").exists()
    
    # Check data content
    train_df = pl.read_parquet(out_dir / "train.parquet")
    val_df = pl.read_parquet(out_dir / "val.parquet")
    test_df = pl.read_parquet(out_dir / "test.parquet")
    
    # Train: Days < 140 (Weeks 0-19) -> Day 1 (Week 0), Day 15 (Week 2) -> H1 and H2
    # Val: Days 140-209 (Weeks 20-29) -> Day 150 (Week 21), Day 200 (Week 28) -> H1, H3
    # Test: Days >= 210 (Weeks 30+) -> Day 215 (Week 30), Day 300 (Week 42) -> H1, H2
    
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    
    # Ensure all data has the right features
    for df in [train_df, val_df, test_df]:
        assert "HOUSEHOLD_KEY" in df.columns
        assert "WINDOW_START_DAY" in df.columns
        assert "COMMODITY_C1_SPEND" in df.columns
        assert "TEMPORAL_WEEK_SIN" in df.columns
        
    # Verify params loaded correctly
    with open(out_dir / "scaler_params.json", "r") as f:
        params = json.load(f)
        assert "COMMODITY_C1_SPEND" in params

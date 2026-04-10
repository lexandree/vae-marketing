from pathlib import Path

import polars as pl
import pytest

from main import main


def test_build_window_attributes_cli_writes_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transactions = tmp_path / "transaction_data.csv"
    products = tmp_path / "product.csv"
    prepared = tmp_path / "prepared.parquet"
    output = tmp_path / "attributes.parquet"

    pl.DataFrame(
        {
            "BASKET_ID": [1, 2],
            "HOUSEHOLD_KEY": [1, 1],
            "DAY": [1, 2],
            "PRODUCT_ID": [101, 102],
            "QUANTITY": [1, 2],
            "SALES_VALUE": [10.0, 20.0],
            "STORE_ID": [1, 1],
            "RETAIL_DISC": [0.0, 0.0],
            "TRANS_TIME": [1000, 1000],
            "WEEK_NO": [0, 0],
            "COUPON_DISC": [0.0, 0.0],
            "COUPON_MATCH_DISC": [0.0, 0.0],
        }
    ).write_csv(transactions)
    pl.DataFrame(
        {
            "PRODUCT_ID": [101, 102],
            "COMMODITY_DESC": ["C1", "C2"],
            "SUB_COMMODITY_DESC": ["SC1", "SC2"],
        }
    ).write_csv(products)
    pl.DataFrame(
        {
            "HOUSEHOLD_KEY": ["1"],
            "WINDOW_START_DAY": [0],
        }
    ).write_parquet(prepared)

    test_args = [
        "main.py",
        "build-window-attributes",
        "--transactions",
        str(transactions),
        "--products",
        str(products),
        "--prepared-data",
        str(prepared),
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert output.exists()
    out_df = pl.read_parquet(output)
    assert out_df.height == 1
    assert "total_spend" in out_df.columns

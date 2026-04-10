import json
from pathlib import Path

import polars as pl
import pytest

from main import main


def test_build_validation_data_cli_generates_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    transactions_path = raw_dir / "transaction_data.csv"
    products_path = raw_dir / "product.csv"
    campaign_table_path = raw_dir / "campaign_table.csv"
    campaign_desc_path = raw_dir / "campaign_desc.csv"
    coupon_path = raw_dir / "coupon.csv"
    coupon_redempt_path = raw_dir / "coupon_redempt.csv"
    demographics_path = raw_dir / "hh_demographic.csv"
    causal_path = raw_dir / "causal_data.csv"

    pl.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 1, 1, 2, 2, 2],
            "BASKET_ID": [10, 11, 12, 20, 21, 22],
            "DAY": [90, 105, 130, 90, 105, 130],
            "PRODUCT_ID": [101, 101, 101, 102, 102, 102],
            "QUANTITY": [1, 2, 1, 1, 1, 1],
            "SALES_VALUE": [10.0, 18.0, 11.0, 8.0, 9.0, 7.0],
            "STORE_ID": [1, 1, 1, 1, 1, 1],
            "WEEK_NO": [13, 15, 19, 13, 15, 19],
        }
    ).write_csv(transactions_path)
    pl.DataFrame(
        {
            "PRODUCT_ID": [101, 102],
            "COMMODITY_DESC": ["A", "B"],
        }
    ).write_csv(products_path)
    pl.DataFrame(
        {
            "DESCRIPTION": ["TypeA"],
            "household_key": [1],
            "CAMPAIGN": [18],
        }
    ).write_csv(campaign_table_path)
    pl.DataFrame(
        {
            "DESCRIPTION": ["TypeA"],
            "CAMPAIGN": [18],
            "START_DAY": [100],
            "END_DAY": [120],
        }
    ).write_csv(campaign_desc_path)
    pl.DataFrame(
        {
            "COUPON_UPC": [5001],
            "PRODUCT_ID": [101],
            "CAMPAIGN": [18],
        }
    ).write_csv(coupon_path)
    pl.DataFrame(
        {
            "household_key": [1],
            "DAY": [105],
            "COUPON_UPC": [5001],
            "CAMPAIGN": [18],
        }
    ).write_csv(coupon_redempt_path)
    pl.DataFrame(
        {
            "household_key": [1, 2],
            "AGE_DESC": ["35-44", "45-54"],
        }
    ).write_csv(demographics_path)
    pl.DataFrame(
        {
            "PRODUCT_ID": [101, 102],
            "STORE_ID": [1, 1],
            "WEEK_NO": [15, 15],
            "display": [1, 0],
            "mailer": ["A", "A"],
        }
    ).write_csv(causal_path)

    out_dir = tmp_path / "validation"
    test_args = [
        "main.py",
        "build-validation-data",
        "--transactions",
        str(transactions_path),
        "--products",
        str(products_path),
        "--campaign-table",
        str(campaign_table_path),
        "--campaign-desc",
        str(campaign_desc_path),
        "--coupon",
        str(coupon_path),
        "--coupon-redempt",
        str(coupon_redempt_path),
        "--demographics",
        str(demographics_path),
        "--causal-data",
        str(causal_path),
        "--campaign-ids",
        "18",
        "--output-dir",
        str(out_dir),
        "--pre-weeks",
        "2",
        "--post-weeks",
        "2",
        "--seed",
        "42",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert (out_dir / "campaign_analysis.parquet").exists()
    assert (out_dir / "comparison_pool.parquet").exists()
    assert (out_dir / "validation_attributes.parquet").exists()
    assert (out_dir / "dataset_summary.json").exists()

    analysis_df = pl.read_parquet(out_dir / "campaign_analysis.parquet")
    comparison_df = pl.read_parquet(out_dir / "comparison_pool.parquet")
    attributes_df = pl.read_parquet(out_dir / "validation_attributes.parquet")
    with open(out_dir / "dataset_summary.json", "r") as f:
        summary = json.load(f)

    assert analysis_df.height == 2
    assert comparison_df.height == 1
    assert attributes_df.height > 0
    assert summary["selected_campaigns"] == [18]

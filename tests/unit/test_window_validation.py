from pathlib import Path

import polars as pl

from src.data.window_validation import build_window_attribute_dataset


def test_build_window_attribute_dataset_aligns_to_prepared_keys() -> None:
    transactions = pl.DataFrame(
        {
            "BASKET_ID": ["1", "2", "3"],
            "HOUSEHOLD_KEY": ["H1", "H1", "H2"],
            "DAY": [1, 3, 8],
            "PRODUCT_ID": ["P1", "P2", "P1"],
            "QUANTITY": [1.0, 2.0, 1.0],
            "SALES_VALUE": [10.0, 20.0, 5.0],
            "STORE_ID": ["S1", "S1", "S1"],
            "TRANS_TIME": [1000, 1000, 1000],
        }
    )
    products = pl.DataFrame(
        {
            "PRODUCT_ID": ["P1", "P2"],
            "COMMODITY_DESC": ["C1", "C2"],
        }
    )
    prepared = pl.DataFrame(
        {
            "HOUSEHOLD_KEY": ["H1", "H2", "H3"],
            "WINDOW_START_DAY": [0, 7, 0],
        }
    )

    attributes = build_window_attribute_dataset(
        transactions=transactions,
        products=products,
        prepared_frame=prepared,
    )

    assert attributes.height == 3
    rows = attributes.sort(["HOUSEHOLD_KEY", "WINDOW_START_DAY"]).to_dicts()
    assert rows[0]["total_spend"] == 30.0
    assert rows[0]["trip_count"] == 2.0
    assert rows[1]["total_spend"] == 5.0
    assert rows[2]["total_spend"] == 0.0

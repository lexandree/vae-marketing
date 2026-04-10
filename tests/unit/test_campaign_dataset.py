import polars as pl

from src.data.campaign_dataset import build_campaign_analysis_dataset
from src.data.campaign_windows import build_campaign_windows


def test_build_campaign_windows() -> None:
    campaign_desc = pl.DataFrame(
        {
            "CAMPAIGN": [18],
            "START_DAY": [100],
            "END_DAY": [120],
        }
    )

    windows = build_campaign_windows(campaign_desc, pre_weeks=2, post_weeks=3)

    assert windows["pre_window_start_day"][0] == 86
    assert windows["pre_window_end_day"][0] == 99
    assert windows["campaign_start_day"][0] == 100
    assert windows["campaign_end_day"][0] == 120
    assert windows["post_window_start_day"][0] == 121
    assert windows["post_window_end_day"][0] == 141


def test_build_campaign_analysis_dataset_outputs_treated_and_comparison() -> None:
    transactions = pl.DataFrame(
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
    )
    products = pl.DataFrame(
        {
            "PRODUCT_ID": [101, 102],
            "COMMODITY_DESC": ["A", "B"],
        }
    )
    campaign_table = pl.DataFrame(
        {
            "DESCRIPTION": ["TypeA"],
            "household_key": [1],
            "CAMPAIGN": [18],
        }
    )
    campaign_desc = pl.DataFrame(
        {
            "DESCRIPTION": ["TypeA"],
            "CAMPAIGN": [18],
            "START_DAY": [100],
            "END_DAY": [120],
        }
    )
    coupon = pl.DataFrame(
        {
            "COUPON_UPC": [5001],
            "PRODUCT_ID": [101],
            "CAMPAIGN": [18],
        }
    )
    coupon_redempt = pl.DataFrame(
        {
            "household_key": [1],
            "DAY": [105],
            "COUPON_UPC": [5001],
            "CAMPAIGN": [18],
        }
    )
    demographics = pl.DataFrame({"household_key": [1, 2], "AGE_DESC": ["35-44", "45-54"]})
    causal_data = pl.DataFrame(
        {
            "PRODUCT_ID": [101, 102],
            "STORE_ID": [1, 1],
            "WEEK_NO": [15, 15],
            "display": [1, 0],
            "mailer": ["A", "A"],
        }
    )

    analysis_df, comparison_df, attributes_df, summary = build_campaign_analysis_dataset(
        transactions=transactions,
        products=products,
        campaign_table=campaign_table,
        campaign_desc=campaign_desc,
        coupon=coupon,
        coupon_redempt=coupon_redempt,
        demographics=demographics,
        causal_data=causal_data,
        campaign_ids=[18],
        pre_weeks=2,
        post_weeks=2,
    )

    assert analysis_df.height == 2
    assert set(analysis_df["treatment_status"].to_list()) == {"treated", "comparison"}
    assert comparison_df.height == 1
    assert attributes_df.height > 0
    assert "total_spend" in analysis_df.columns
    assert summary["selected_campaigns"] == [18]
    assert summary["treated_records"] == 1
    assert summary["comparison_records"] == 1

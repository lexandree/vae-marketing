"""Schema definitions for raw and validation-stage datasets."""

import polars as pl


RAW_TRANSACTION_SCHEMA = {
    "HOUSEHOLD_KEY": pl.Int64,
    "BASKET_ID": pl.Int64,
    "DAY": pl.Int64,
    "PRODUCT_ID": pl.Int64,
    "QUANTITY": pl.Int64,
    "SALES_VALUE": pl.Float64,
    "STORE_ID": pl.Int64,
    "RETAIL_DISC": pl.Float64,
    "TRANS_TIME": pl.Int64,
    "WEEK_NO": pl.Int64,
    "COUPON_DISC": pl.Float64,
    "COUPON_MATCH_DISC": pl.Float64,
}

PRODUCT_SCHEMA = {
    "PRODUCT_ID": pl.Int64,
    "MANUFACTURER": pl.Int64,
    "DEPARTMENT": pl.Utf8,
    "BRAND": pl.Utf8,
    "COMMODITY_DESC": pl.Utf8,
    "SUB_COMMODITY_DESC": pl.Utf8,
}

CAMPAIGN_TABLE_SCHEMA = {
    "DESCRIPTION": pl.Utf8,
    "household_key": pl.Int64,
    "CAMPAIGN": pl.Int64,
}

CAMPAIGN_DESC_SCHEMA = {
    "DESCRIPTION": pl.Utf8,
    "CAMPAIGN": pl.Int64,
    "START_DAY": pl.Int64,
    "END_DAY": pl.Int64,
}

COUPON_SCHEMA = {
    "COUPON_UPC": pl.Int64,
    "PRODUCT_ID": pl.Int64,
    "CAMPAIGN": pl.Int64,
}

COUPON_REDEMPTION_SCHEMA = {
    "household_key": pl.Int64,
    "DAY": pl.Int64,
    "COUPON_UPC": pl.Int64,
    "CAMPAIGN": pl.Int64,
}

DEMOGRAPHIC_SCHEMA = {
    "AGE_DESC": pl.Utf8,
    "MARITAL_STATUS_CODE": pl.Utf8,
    "INCOME_DESC": pl.Utf8,
    "HOMEOWNER_DESC": pl.Utf8,
    "HH_COMP_DESC": pl.Utf8,
    "HOUSEHOLD_SIZE_DESC": pl.Utf8,
    "KID_CATEGORY_DESC": pl.Utf8,
    "household_key": pl.Int64,
}

CAUSAL_DATA_SCHEMA = {
    "PRODUCT_ID": pl.Int64,
    "STORE_ID": pl.Int64,
    "WEEK_NO": pl.Int64,
    "display": pl.Int64,
    "mailer": pl.Utf8,
}

PREPARED_VAE_SCHEMA = {
    "HOUSEHOLD_KEY": pl.Int64,
    "time_window": pl.Int64,
    "week_of_year": pl.Int64,
    "day_of_week": pl.Int64,
    "month_of_year": pl.Int64,
    "is_holiday_period": pl.Boolean,
    "household_total_sales": pl.Float64,
    "sparsity_indicator": pl.Int64,
}

CAMPAIGN_ANALYSIS_SCHEMA = {
    "household_key": pl.Int64,
    "CAMPAIGN": pl.Int64,
    "treatment_status": pl.Utf8,
    "pre_window_start_day": pl.Int64,
    "pre_window_end_day": pl.Int64,
    "campaign_start_day": pl.Int64,
    "campaign_end_day": pl.Int64,
    "post_window_start_day": pl.Int64,
    "post_window_end_day": pl.Int64,
    "eligibility_status": pl.Utf8,
}

VALIDATION_ATTRIBUTE_SCHEMA = {
    "household_key": pl.Int64,
    "CAMPAIGN": pl.Int64,
    "window_type": pl.Utf8,
    "attribute_name": pl.Utf8,
    "attribute_value": pl.Float64,
    "attribute_family": pl.Utf8,
}

CRITICAL_COLUMNS = ["HOUSEHOLD_KEY", "PRODUCT_ID", "DAY"]

import polars as pl

# Schema for raw Dunnhumby transaction data
RAW_TRANSACTION_SCHEMA = {
    "household_key": pl.Int64,
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

# Schema for prepared VAE input data (per household per time window)
# This is a representative schema, actual features might vary based on engineering
PREPARED_VAE_SCHEMA = {
    "household_key": pl.Int64,
    "time_window": pl.Int64,
    "week_of_year": pl.Int64,
    "day_of_week": pl.Int64,
    "month_of_year": pl.Int64,
    "is_holiday_period": pl.Boolean,
    "household_total_sales": pl.Float64,
    "sparsity_indicator": pl.Int64,
}

# Columns that must not have null values for valid processing
CRITICAL_COLUMNS = ["household_key", "PRODUCT_ID", "DAY"]

import logging
import os
from typing import Optional

import pandas as pd
import polars as pl
import psutil

from src.data.schema import RAW_TRANSACTION_SCHEMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def log_memory_usage(message: Optional[str] = None) -> None:
    """Logs the current memory usage of the process.

    Args:
        message: Optional message to include in the log.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    log_msg = f"Memory Usage: {rss_mb:.2f} MB"
    if message:
        log_msg = f"{message} | {log_msg}"
    logger.info(log_msg)


def load_and_preprocess_transaction_data(file_path: str) -> pl.DataFrame:
    """Loads transaction data from a CSV file, performs basic preprocessing using Polars.

    Args:
        file_path: The path to the CSV data file.

    Returns:
        The preprocessed Polars DataFrame.
    """
    logger.info("Starting data preparation pipeline...")
    log_memory_usage("Before loading data")

    logger.info("Loading transaction data from %s...", file_path)
    # Using Polars lazy scan with schema for memory efficiency as per T005
    try:
        # scan_csv is lazy, it won't load the whole file into memory
        lazy_df = pl.scan_csv(file_path, schema=RAW_TRANSACTION_SCHEMA)

        # We perform operations on the lazy frame to keep memory usage low
        logger.info("Performing basic preprocessing on transaction data (lazy)...")

        # Convert 'DAY' to a more usable date format (assuming day 1 is '2023-01-01')
        base_date = pd.to_datetime("2023-01-01")

        processed_lazy_df = lazy_df.with_columns(
            (pl.lit(base_date) + pl.duration(days=pl.col("DAY") - 1)).alias(
                "transaction_date"
            )
        )

        # Numerical columns handling is mostly covered by schema casting in scan_csv
        # but we handle nulls explicitly here
        for col, dtype in RAW_TRANSACTION_SCHEMA.items():
            if dtype in [pl.Float64, pl.Int64]:
                processed_lazy_df = processed_lazy_df.with_columns(
                    pl.col(col).fill_null(0.0 if dtype == pl.Float64 else 0)
                )

        # Drop rows where essential identifiers might still be missing
        processed_lazy_df = processed_lazy_df.filter(
            (pl.col("household_key").is_not_null()) &
            (pl.col("household_key") != 0) &
            (pl.col("PRODUCT_ID").is_not_null()) &
            (pl.col("PRODUCT_ID") != 0)
        )

        # Example feature creation: total sales value per household
        # This is a window function, which Polars handles efficiently
        processed_lazy_df = processed_lazy_df.with_columns(
            pl.col("SALES_VALUE").sum().over("household_key").alias("household_total_sales")
        )

        # Collect the results. For 1TB, we would use sink_parquet or similar
        # but for now we collect to return a DataFrame as per existing signature.
        # In a real 1TB scenario, this function would return a LazyFrame or sink to disk.
        df_pl = processed_lazy_df.collect()

    except Exception as e:
        logger.error("Failed to process data: %s", e)
        raise

    log_memory_usage("After collecting data")
    logger.info("Transaction data preprocessing complete.")

    return df_pl


if __name__ == "__main__":
    # Example usage
    data_file_name = "data/transaction_data.csv"
    try:
        processed_transaction_data = load_and_preprocess_transaction_data(data_file_name)
        logger.info("Preprocessed transaction data (first 5 rows):")
        print(processed_transaction_data.head())
        logger.info("Shape of processed data: %s", processed_transaction_data.shape)
    except FileNotFoundError:
        logger.error(
            "Data file '%s' not found. Please ensure the file exists.", data_file_name
        )
    except Exception as e:
        logger.error("An error occurred during data processing: %s", e)

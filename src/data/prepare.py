"""Main entrypoint for data preparation pipeline.

This module provides the CLI interface to run the data preparation
pipeline for the VAE Marketing Analysis project.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

from src.data.extractors import extract_features, extract_vocabulary
from src.data.normalizers import (
    create_time_series_splits,
    fit_scalers,
    save_scaler_params,
    transform_features,
)
from src.utils.seed import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Schema Definitions for Input Validation
TRANSACTION_SCHEMA = {
    "BASKET_ID": pl.Utf8,
    "HOUSEHOLD_KEY": pl.Utf8,
    "DAY": pl.Int64,
    "PRODUCT_ID": pl.Utf8,
    "QUANTITY": pl.Float64,
    "SALES_VALUE": pl.Float64,
    "STORE_ID": pl.Utf8,
    "TRANS_TIME": pl.Int64,
}

PRODUCT_SCHEMA = {
    "PRODUCT_ID": pl.Utf8,
    "COMMODITY_DESC": pl.Utf8,
    "SUB_COMMODITY_DESC": pl.Utf8,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Marketing Analysis Data Preparation")

    parser.add_argument(
        "--input-transactions",
        type=Path,
        required=True,
        help="Path to the raw Dunnhumby transaction CSV file",
    )
    parser.add_argument(
        "--input-products",
        type=Path,
        required=True,
        help="Path to the product hierarchy CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where output Parquet files will be saved",
    )
    parser.add_argument(
        "--train-weeks",
        type=int,
        default=20,
        help="Number of weeks to include in the training split",
    )
    parser.add_argument(
        "--val-weeks",
        type=int,
        default=10,
        help="Number of weeks to include in the validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def validate_and_load_data(
    transactions_path: Path,
    products_path: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load data and validate schema strictly."""
    logger.info(f"Loading transactions from {transactions_path}")

    try:
        # Some datasets have household_key in lowercase
        transactions = pl.read_csv(
            transactions_path,
            infer_schema_length=10000,
            ignore_errors=False
        )
        if "household_key" in transactions.columns and "HOUSEHOLD_KEY" not in transactions.columns:
            transactions = transactions.rename({"household_key": "HOUSEHOLD_KEY"})

        # Ensure schema types for expected columns
        for col, dtype in TRANSACTION_SCHEMA.items():
            if col in transactions.columns:
                transactions = transactions.with_columns(pl.col(col).cast(dtype))
    except FileNotFoundError:
        logger.error(f"Transactions file not found: {transactions_path}")
        raise

    logger.info(f"Loading products from {products_path}")
    try:
        products = pl.read_csv(
            products_path,
            schema_overrides=PRODUCT_SCHEMA,
            infer_schema_length=10000,
            ignore_errors=False
        )
    except FileNotFoundError:
        logger.error(f"Products file not found: {products_path}")
        raise

    return transactions, products


def main() -> None:
    """Run the data preparation pipeline."""
    args = parse_args()

    set_seed(args.seed)

    logger.info("Starting data preparation pipeline...")
    try:
        # Create output dir if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate
        transactions, products = validate_and_load_data(
            args.input_transactions,
            args.input_products
        )

        logger.info(f"Loaded {len(transactions)} transactions and {len(products)} products.")

        logger.info("Extracting vocabulary from products catalog...")
        vocabulary = extract_vocabulary(products)
        logger.info(f"Found {len(vocabulary)} valid unique commodity categories.")

        # Save vocabulary
        vocab_path = args.output_dir / "vocabulary.json"
        with open(vocab_path, "w") as f:
            json.dump(vocabulary, f, indent=2)
        logger.info(f"Saved vocabulary to {vocab_path}")

        logger.info("Extracting features with 7-day rolling window...")
        features_df = extract_features(transactions, products, vocabulary=vocabulary)
        logger.info(f"Extracted features shape: {features_df.shape}")

        if len(features_df) == 0:
            logger.warning("No features extracted. Exiting.")
            sys.exit(0)

        logger.info(f"Splitting dataset (train={args.train_weeks}w, val={args.val_weeks}w)...")
        train_df, val_df, test_df = create_time_series_splits(
            features_df,
            train_weeks=args.train_weeks,
            val_weeks=args.val_weeks
        )

        logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

        logger.info("Fitting scalers on training data...")
        scaler_params = fit_scalers(train_df)

        logger.info("Applying transformations...")
        train_scaled = transform_features(train_df, scaler_params)
        val_scaled = transform_features(val_df, scaler_params)
        test_scaled = transform_features(test_df, scaler_params)

        logger.info(f"Saving artifacts to {args.output_dir}...")

        train_scaled.write_parquet(args.output_dir / "train.parquet")
        val_scaled.write_parquet(args.output_dir / "val.parquet")
        test_scaled.write_parquet(args.output_dir / "test.parquet")

        save_scaler_params(scaler_params, args.output_dir / "scaler_params.json")

        logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

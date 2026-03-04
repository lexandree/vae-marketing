# Quickstart: Data Preparation Pipeline

This guide explains how to run the data preparation pipeline for the VAE Marketing Analysis project.

## Prerequisites

Ensure you have the required dependencies installed.

```bash
pip install pandas polars scikit-learn pyarrow
```

You must also have the raw Dunnhumby dataset files available locally:
- `transaction_data.csv`
- `product.csv`

## Running the Pipeline

To execute the data preparation pipeline and generate the Parquet files for VAE training, run the following command from the project root:

```bash
python -m src.data.prepare \
    --input-transactions data/raw/transaction_data.csv \
    --input-products data/raw/product.csv \
    --output-dir data/processed/ \
    --train-weeks 20 \
    --val-weeks 10
```

## Verifying the Output

Check the output directory (`data/processed/`) to ensure the following files were created:
- `train.parquet`
- `val.parquet`
- `test.parquet`
- `scaler_params.json`

You can use Polars or Pandas to inspect the shape and features of the generated data:

```python
import polars as pl

df = pl.read_parquet("data/processed/train.parquet")
print(f"Dataset shape: {df.shape}")
print(df.head())
```
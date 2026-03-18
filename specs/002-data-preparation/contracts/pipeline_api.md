# Pipeline API Contract

## Command Line Interface (CLI)

The data preparation pipeline will be exposed as a Python script with the following CLI arguments:

```bash
python -m src.data.prepare \
    --input-transactions <path_to_raw_transactions.csv> \
    --input-products <path_to_product_hierarchy.csv> \
    --output-dir <path_to_output_directory> \
    --train-weeks <int> \
    --val-weeks <int>
```

### Arguments:
- `--input-transactions` (Required): Path to the raw Dunnhumby transaction CSV file.
- `--input-products` (Required): Path to the product hierarchy CSV file.
- `--output-dir` (Required): Directory where the output Parquet files (`train.parquet`, `val.parquet`, `test.parquet`, and `scaler_params.json`) will be saved.
- `--train-weeks` (Optional, Default: 20): Number of weeks to include in the training split, starting from week 0.
- `--val-weeks` (Optional, Default: 10): Number of weeks to include in the validation split, immediately following the training split.

## Output Artifacts

Upon successful execution, the pipeline will produce:
1. `<output-dir>/train.parquet`: The training dataset.
2. `<output-dir>/val.parquet`: The validation dataset.
3. `<output-dir>/test.parquet`: The test dataset (all remaining weeks).
4. `<output-dir>/scaler_params.json`: The normalization parameters (mean, std) fit on the training cohort, for reproducibility.
# Quickstart Guide

## Prerequisites
- Python 3.11+
- Recommended: A virtual environment (e.g., `venv` or `conda`)

## Installation
1. Clone the repository.
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ds
   ```

## Running the Pipeline
To run the analysis pipeline locally on a provided dataset:

```python
import polars as pl
from src.services.baseline import train_baseline_vae
from src.services.impact_analysis import calculate_deviation

# 1. Load Data
transactions = pl.read_csv("data/transactions.csv").to_pandas()
stimuli = pl.read_csv("data/stimuli.csv").to_pandas()

# 2. Train Baseline Model (using transactions without stimuli)
# Filter data, initialize model, and train...
model = train_baseline_vae(transactions, model, epochs=50)

# 3. Analyze Impact
# For a specific household and stimulus:
deviation = calculate_deviation(baseline_profile, post_stimulus_data, model)
print(f"Calculated Deviation: {deviation}")
```
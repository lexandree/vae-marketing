# Quickstart: Data Preparation & Model Pipeline

This guide explains how to prepare the Dunnhumby dataset, train the baseline VAE model, and run marketing impact inference.

## 1. Data Preparation

The preparation script processes raw CSV files into aggregated, normalized Parquet windows. It also establishes the **Vocabulary Contract** used by the neural network.

```bash
python -m src.data.prepare \
    --input-transactions data/transaction_data.csv \
    --input-products data/product.csv \
    --output-dir data/processed/ \
    --train-weeks 20 \
    --val-weeks 10 \
    --seed 42
```

### Key Parameters:
- `--input-transactions`: Path to the raw Dunnhumby transaction CSV.
- `--input-products`: Path to the product hierarchy CSV (used to build the vocabulary).
- `--output-dir`: Where to save Parquet files, `vocabulary.json`, and `scaler_params.json`.
- `--train-weeks`: Number of initial weeks to use for the baseline "normal" behavior (default: 20).
- `--val-weeks`: Number of subsequent weeks to use for the "stimulus period" (default: 10).

---

## 2. Model Training

Train the VAE on the baseline training data. The model architecture (input/output layers) is automatically configured based on the `vocabulary.json`.

```bash
python main.py train \
    --data data/processed/train.parquet \
    --vocab data/processed/vocabulary.json \
    --model-out data/vae_baseline.pt \
    --epochs 50
```

### Key Parameters:
- `--data`: Path to the `train.parquet` file.
- `--vocab`: Path to the `vocabulary.json` file (defines the input dimension).
- `--model-out`: Filename for the saved PyTorch model weights.
- `--epochs`: Number of training iterations.

---

## 3. Impact Analysis (Inference)

Run the inference command to compare the "stimulus period" behavior against the learned baseline.

```bash
python main.py infer \
    --data data/processed/val.parquet \
    --train-data data/processed/train.parquet \
    --vocab data/processed/vocabulary.json \
    --model-in data/vae_baseline.pt
```

### Key Parameters:
- `--data`: The period to analyze for deviations (e.g., `val.parquet`).
- `--train-data`: Used to re-generate the baseline latent profiles for each household.
- `--vocab`: Ensures dimensions match the trained model.
- `--model-in`: Path to the saved weights from the training step.

---

## 4. Understanding the Report

The inference command outputs a **Pipeline Summary Report**.

### What we have now:
1. **Total Households Analyzed**: Scale of the affected audience.
2. **Average Latent Deviation**: A mathematical measure of how much shopping patterns changed.
3. **Average Persistence**: How many days it took for behavior to return to the baseline (based on latent distance threshold).
4. **Segment Distribution**: Basic clustering of households into groups based on their reaction intensity and persistence.
5. **Top Sensitive Categories**: Ranking of product categories (`COMMODITY_DESC`) that contributed most to the volume of transactions during the deviation.

### What is missing (Planned for Phase 3/4):
- **Interpretability**: The "Latent Deviation" is currently a single abstract number. We cannot yet say *why* it changed (e.g., price sensitivity vs. quantity stockpiling).
- **Semantic Segmentation**: Clusters are currently numbered (`Cluster 0`). We need to map these to business personas (e.g., "Impulse Buyers").
- **Directional Analysis**: Knowing that behavior shifted is useful, but we need to know if it shifted "up" (trading up to premium brands) or "down".
- **Causal Linking**: Direct correlation between specific campaign IDs from `campaign_table.csv` and the detected shifts.

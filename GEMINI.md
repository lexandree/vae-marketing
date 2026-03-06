# vae_marketing Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-06

## Active Technologies
- Python 3.11+ + Polars (fast processing of up to 1TB data), Pandas, Scikit-learn, PyArrow (Parquet)
- PyTorch (Beta-VAE, Disentangled factors)
- Weights & Biases (WandB) for tracking and HPO (Sweeps)
- Experiments stored in `experiments/[run_id]/` including `inference_report.json`

## Commands

### Data Prep
PYTHONPATH=. python3 src/data/prepare.py --input-transactions data/transaction_data.csv --input-products data/product.csv --output-dir data/processed_full --train-weeks 72 --val-weeks 14

### Training
PYTHONPATH=. python3 main.py train --arch beta_vae --data data/processed_full/train.parquet --vocab data/processed_full/vocabulary.json --wandb

### Inference
PYTHONPATH=. python3 main.py infer --run-id [RUN_ID] --data [TARGET] --baseline [BASELINE] --limit 100

### Tests
cd src && pytest && ruff check .

## Recent Changes
- 003-disentangled-vae-wandb: Implemented Beta-VAE with WandB integration, HPO Sweeps support, and polymorphic reporting (factor breakdown).
- 002-data-preparation: Finalized full-scale data pipeline with 72/14/15 week time-series splits.

<!-- MANUAL ADDITIONS START -->


<!-- MANUAL ADDITIONS END -->

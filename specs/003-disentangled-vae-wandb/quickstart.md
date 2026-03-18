# Quickstart: Stage 003 (Disentangled VAE)

## Prerequisites
1. Ensure `wandb` is installed: `pip install wandb`.
2. Login to your account: `wandb login [YOUR_API_KEY]`.
3. Prepared data from Stage 002 must be in `data/processed/`.

## 1. Train a Disentangled Model
Train a Beta-VAE with β=4.0 and 20 epochs of annealing to learn interpretable latent factors.

```bash
python main.py train \
  --arch beta_vae \
  --beta 4.0 \
  --anneal-end 20 \
  --latent-dim 32 \
  --run-id my-first-beta-vae \
  --wandb
```

## 2. Monitor Experiments
Open your browser and navigate to the link provided in the terminal output to see:
- Real-time training loss.
- KL divergence vs. MSE trade-off.
- Disentanglement metrics (MIG/SAP) as training progresses.

## 3. Generate Disentangled Report
Analyze a campaign's impact using the learned factors.

```bash
python main.py infer \
  --run-id my-first-beta-vae \
  --data data/processed/campaign_data.parquet
```

The system will automatically detect the `beta_vae` architecture and provide a factor-level breakdown of behavior changes.

## 4. Compare Models
Compare your disentangled model's performance against the baseline.

```bash
# Baseline check
python main.py train --arch baseline --run-id base-comparison

# Compare in WandB UI or locally
cat experiments/my-first-beta-vae/metrics.json
cat experiments/base-comparison/metrics.json
```

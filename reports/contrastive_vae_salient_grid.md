# Contrastive VAE Salient Grid

This note records the first leakage-controlled `Contrastive VAE` comparison pass using `salient-only` latent validation on the held-out household protocol.

## Protocol

- Training data: `data/contrastive_no_leak_train`
- Evaluation data: `data/processed_no_leak_eval/val.parquet`
- Evaluation attributes: `data/processed_no_leak_eval/val_attributes.parquet`
- Eval mode: `--latent-mode salient`
- Metrics:
  - `MIG` via `binned` estimation with `32 quantile bins`
  - `SAP` via `vectorized` approximation

The point of this pass is not to prove that `Contrastive VAE` already beats the leakage-controlled `beta-VAE`, but to identify which latent split and salient regularization regime deserve deeper tuning.

## Reference Baseline

`beta-VAE` reference run:

- Run: `noleak-beta-vae-32d`
- Metrics:
  - `MIG = 0.05678`
  - `SAP = 0.03013`
- Campaign-relevant mappings:
  - `total_spend -> latent 23` with `0.3783`
  - `category_diversity -> latent 14` with `0.8812`

## Contrastive Results

| Run | Shared/Salient | Salient weight | Epochs | MIG | SAP | Notes |
|---|---:|---:|---:|---:|---:|---|
| `contrastive-vae-16s16q` | `16 / 16` | `1.0` | `10` | `0.03549` | `0.01543` | Best current contrastive metrics |
| `contrastive-vae-24s8q-w10-e5` | `24 / 8` | `1.0` | `5` | `0.02580` | `0.01232` | Compact salient space, clean top mappings |
| `contrastive-vae-8s24q-w10-e5` | `8 / 24` | `1.0` | `5` | `0.03066` | `0.01110` | Strongest campaign-relevant salient factors |
| `contrastive-vae-16s16q-w05-e5` | `16 / 16` | `0.5` | `5` | `0.02637` | `0.01097` | Strong `category_diversity`, weaker spend signal |
| `contrastive-vae-16s16q-w20-e5` | `16 / 16` | `2.0` | `5` | `0.02493` | `0.00937` | Stronger salient suppression, lower global metrics |

## Campaign-Relevant Salient Mappings

### `contrastive-vae-16s16q`

- `total_spend -> latent 3` with `0.4146`
- `category_diversity -> latent 10` with `0.7131`

### `contrastive-vae-24s8q-w10-e5`

- `total_spend -> latent 6` with `0.4731`
- `category_diversity -> latent 7` with `0.5536`

### `contrastive-vae-8s24q-w10-e5`

- `total_spend -> latent 19` with `0.7417`
- `category_diversity -> latent 18` with `0.7961`

### `contrastive-vae-16s16q-w05-e5`

- `category_diversity -> latent 10` with `0.8157`
- No comparably strong `total_spend` mapping in the top salient factors

### `contrastive-vae-16s16q-w20-e5`

- `total_spend -> latent 9` with `0.4806`
- `category_diversity -> latent 10` with `0.7192`

## Interpretation

`salient-only` validation is the correct comparison lens for `Contrastive VAE`. In every contrastive run, salient-only metrics are more meaningful than the earlier combined-latent evaluation because the architecture is supposed to isolate campaign-salient variation into the salient subspace rather than spread it across shared and salient dimensions.

The current leakage-controlled `beta-VAE` still wins on global `MIG` and `SAP`. That means `Contrastive VAE` has not yet demonstrated better overall disentanglement or better generic alignment with the observable validation attributes.

At the same time, the contrastive runs already show a different strength: several salient configurations produce very strong direct mappings for the two campaign-relevant business attributes that matter most in the restricted case studies, namely `total_spend` and `category_diversity`.

## Recommended Zones For The Next Pass

These are the three configurations worth carrying forward:

1. `contrastive-vae-16s16q`
   - Best current contrastive `MIG/SAP`
   - Best balanced starting point
   - Should remain the main reference run

2. `contrastive-vae-8s24q-w10-e5`
   - Best campaign-relevant salient factors
   - Best `total_spend` and `category_diversity` associations
   - Strong candidate if the goal is campaign-salient interpretability rather than global disentanglement

3. `contrastive-vae-24s8q-w10-e5`
   - Small salient bottleneck
   - Cleaner and easier-to-explain salient subspace
   - Good low-capacity benchmark

The `16/16` runs with `0.5` and `2.0` salient weights were useful sensitivity checks, but neither beats the base `16/16, weight=1.0` regime on the current held-out metrics.

## Proposed Next Search Space

If the next step is a broader search, it should stay local to the three promising zones above rather than reopen the whole space.

- `shared_dim / salient_dim`
  - `16 / 16`
  - `12 / 20`
  - `8 / 24`
  - `20 / 12`
  - `24 / 8`
- `salient_background_weight`
  - `0.75`
  - `1.0`
  - `1.5`
- `beta_salient`
  - `0.5`
  - `1.0`
  - `1.5`
- `lr`
  - `5e-4`
  - `1e-3`

The search objective should not be raw training loss. It should favor held-out `salient-only` quality, ideally with a composite criterion that keeps campaign-relevant attributes central:

- primary:
  - `salient_only_mig`
  - `salient_only_sap`
- secondary:
  - top `total_spend` association strength
  - top `category_diversity` association strength

## Reproduction Commands

Base contrastive run:

```bash
PYTHONPATH=. python main.py train-contrastive \
  --target-data data/contrastive_no_leak_train/target.parquet \
  --background-data data/contrastive_no_leak_train/background.parquet \
  --run-id contrastive-vae-16s16q \
  --shared-dim 16 \
  --salient-dim 16 \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.001 \
  --beta-shared 1.0 \
  --beta-salient 1.0 \
  --salient-background-weight 1.0 \
  --verbosity 1
```

Salient-only evaluation:

```bash
PYTHONPATH=. python main.py validate-latents \
  --analysis-data data/processed_no_leak_eval/val.parquet \
  --attributes data/processed_no_leak_eval/val_attributes.parquet \
  --run-ids contrastive-vae-16s16q \
  --output-dir experiments/latent_validation_contrastive_salient_eval \
  --latent-mode salient \
  --mig-method binned \
  --mig-bins 32 \
  --mig-binning quantile \
  --sap-method vectorized
```

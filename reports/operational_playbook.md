# Operational Playbook

This playbook is the shortest practical path for using the current environment
on real data.

It assumes the goal is:

1. validate a campaign effect
2. interpret the effect with a leakage-controlled latent model

Use this file as the operating procedure. Use
`model_selection_and_usage_guide.md` when you need the reasoning behind the
procedure.

## Stage 0: Decide The Question

Pick the question before picking the model.

- "Is there a credible campaign effect?"
  - use the quasi-causal validation layer first
- "What kind of behavior shifted?"
  - use a latent model after campaign validation
- "Which latent factors best explain the campaign shift?"
  - use the bridge after latent validation

## Stage 1: Build Leakage-Controlled Splits

Use this whenever the campaign case studies are fixed in advance.

```bash
PYTHONPATH=. python3 main.py build-household-splits \
  --campaign-table data/campaign_table.csv \
  --eval-campaign-ids 26 30 \
  --output data/splits/household_splits.json \
  --seed 42
```

Output:

- `data/splits/household_splits.json`

Practical meaning:

- evaluation campaigns are separated from representation training
- final latent claims can be defended as held-out

## Stage 2: Build No-Leak Representation Data

Train split:

```bash
PYTHONPATH=. python3 src/data/prepare.py \
  --input-transactions data/transaction_data.csv \
  --input-products data/product.csv \
  --output-dir data/processed_no_leak_train \
  --train-weeks 72 \
  --val-weeks 14 \
  --household-splits data/splits/household_splits.json \
  --split-role train
```

Eval split:

```bash
PYTHONPATH=. python3 src/data/prepare.py \
  --input-transactions data/transaction_data.csv \
  --input-products data/product.csv \
  --output-dir data/processed_no_leak_eval \
  --train-weeks 72 \
  --val-weeks 14 \
  --household-splits data/splits/household_splits.json \
  --split-role eval
```

Build aligned observable attributes for held-out latent validation:

```bash
PYTHONPATH=. python3 main.py build-window-attributes \
  --transactions data/transaction_data.csv \
  --products data/product.csv \
  --prepared-data data/processed_no_leak_eval/val.parquet \
  --output data/processed_no_leak_eval/val_attributes.parquet
```

## Stage 3: Build The Campaign Validation Dataset

This step creates the treated/comparison panel and observable window
attributes.

```bash
PYTHONPATH=. python3 main.py build-validation-data \
  --transactions data/transaction_data.csv \
  --products data/product.csv \
  --campaign-table data/campaign_table.csv \
  --campaign-desc data/campaign_desc.csv \
  --coupon data/coupon.csv \
  --coupon-redempt data/coupon_redempt.csv \
  --demographics data/hh_demographic.csv \
  --causal-data data/causal_data.csv \
  --campaign-ids 26 30 \
  --output-dir data/validation_26_30 \
  --pre-weeks 4 \
  --post-weeks 4
```

Main outputs:

- `campaign_analysis.parquet`
- `comparison_pool.parquet`
- `validation_attributes.parquet`
- `dataset_summary.json`

## Stage 4: Validate Campaign Effects

Restricted quasi-causal validation:

```bash
PYTHONPATH=. python3 main.py validate-campaigns \
  --analysis-data data/validation_26_30/campaign_analysis.parquet \
  --campaign-ids 26 30 \
  --method matched-did \
  --output-dir data/validation_26_30/restricted \
  --matching-method propensity \
  --propensity-caliper 0.02
```

Main outputs:

- `campaign_effects.json`
- `campaign_diagnostics.json`
- `campaign_event_study.parquet`
- `campaign_balance_details.json`
- `campaign_cohort_summary.json`

Interpretation:

- `supported` means the restricted design survived the current diagnostics
- `weak` means the result is usable as suggestive evidence, not as a strong headline
- `unsupported` means do not write causal-style claims

## Stage 5: Run Sensitivity Analysis

Use this before finalizing a narrative for a campaign.

```bash
PYTHONPATH=. python3 main.py analyze-campaign-sensitivity \
  --transactions data/transaction_data.csv \
  --products data/product.csv \
  --campaign-table data/campaign_table.csv \
  --campaign-desc data/campaign_desc.csv \
  --coupon data/coupon.csv \
  --coupon-redempt data/coupon_redempt.csv \
  --demographics data/hh_demographic.csv \
  --campaign-ids 26 30 \
  --weeks-grid 2 3 4 5 \
  --output-dir data/campaign_sensitivity \
  --matching-methods propensity \
  --propensity-calipers 0.02
```

Use this stage to answer:

- does the finding survive nearby window sizes?
- does matching quality remain acceptable?
- which campaign is stable enough to center in the write-up?

## Stage 6: Train The General Latent Baseline

Default baseline:

```bash
PYTHONPATH=. python3 main.py train \
  --arch beta_vae \
  --run-id noleak-beta-vae-32d \
  --data data/processed_no_leak_train/train.parquet \
  --vocab data/processed_no_leak_train/vocabulary.json \
  --latent-dim 32 \
  --beta 2.0 \
  --anneal-end 5 \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.001
```

Use this model when:

- you need the safest overall latent baseline
- you want the strongest general held-out semantics

## Stage 7: Validate The General Latent Baseline

```bash
PYTHONPATH=. python3 main.py validate-latents \
  --analysis-data data/processed_no_leak_eval/val.parquet \
  --attributes data/processed_no_leak_eval/val_attributes.parquet \
  --run-ids noleak-beta-vae-32d \
  --output-dir experiments/latent_validation_noleak_eval \
  --mig-method binned \
  --mig-bins 32 \
  --mig-binning quantile \
  --sap-method vectorized
```

Main outputs:

- `latent_metrics.json`
- `factor_mappings.json`
- `latent_stability.json`

## Stage 8: Train The Best Contrastive VAE

Use this when the question is explicitly target-vs-background.

Build the contrastive dataset:

```bash
PYTHONPATH=. python3 main.py build-contrastive-data \
  --prepared-data data/processed_no_leak_train/train.parquet \
  --campaign-table data/campaign_table.csv \
  --campaign-desc data/campaign_desc.csv \
  --output-dir data/contrastive_no_leak_train \
  --exclude-campaign-ids 26 30 \
  --background-ratio 1.0 \
  --seed 42
```

Train the best current contrastive configuration:

```bash
PYTHONPATH=. python3 main.py train-contrastive \
  --target-data data/contrastive_no_leak_train/target.parquet \
  --background-data data/contrastive_no_leak_train/background.parquet \
  --run-id contrastive-vae-best-noleak \
  --latent-split 16_16 \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.0012063261039997576 \
  --beta-shared 1.0 \
  --beta-salient 0.5 \
  --salient-background-weight 1.415544161258955 \
  --eval-analysis-data data/processed_no_leak_eval/val.parquet \
  --eval-attributes data/processed_no_leak_eval/val_attributes.parquet \
  --eval-latent-mode salient \
  --eval-mig-method binned \
  --eval-mig-bins 32 \
  --eval-mig-binning quantile \
  --eval-sap-method vectorized
```

Use this model when:

- you want campaign-salient latent structure
- you care more about target-vs-background interpretation than about being the best global latent baseline

## Stage 9: Train The Bridge-Oriented Beta-TCVAE

Use this when the question is:

- which latent factors best align with campaign-visible shifts?

```bash
PYTHONPATH=. python3 main.py train \
  --arch beta_tcvae \
  --run-id beta-tcvae-32d-noleak \
  --data data/processed_no_leak_train/train.parquet \
  --vocab data/processed_no_leak_train/vocabulary.json \
  --latent-dim 32 \
  --beta 2.0 \
  --tc-alpha 1.0 \
  --tc-lambda 1.0 \
  --anneal-end 5 \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.001
```

Then validate it:

```bash
PYTHONPATH=. python3 main.py validate-latents \
  --analysis-data data/processed_no_leak_eval/val.parquet \
  --attributes data/processed_no_leak_eval/val_attributes.parquet \
  --run-ids beta-tcvae-32d-noleak \
  --output-dir experiments/latent_validation_beta_tcvae_noleak_eval \
  --mig-method binned \
  --mig-bins 32 \
  --mig-binning quantile \
  --sap-method vectorized
```

Use this model when:

- the bridge is the main deliverable
- `total_spend` and `category_diversity` interpretation matters more than top global `MIG/SAP`

## Stage 10: Build Campaign Bridges

General bridge from no-leak `beta-VAE`:

```bash
PYTHONPATH=. python3 main.py build-campaign-latent-bridge \
  --campaign-results data/restricted_26_w2/restricted/campaign_effects.json \
  --factor-mappings experiments/latent_validation_noleak_eval/factor_mappings.json \
  --attributes data/restricted_26_w2/validation_attributes.parquet \
  --output-dir data/restricted_26_w2/latent_bridge_noleak \
  --top-k-attributes 5
```

Bridge from `beta-TCVAE`:

```bash
PYTHONPATH=. python3 main.py build-campaign-latent-bridge \
  --campaign-results data/restricted_26_w2/restricted/campaign_effects.json \
  --factor-mappings experiments/latent_validation_beta_tcvae_noleak_eval/factor_mappings.json \
  --attributes data/restricted_26_w2/validation_attributes.parquet \
  --output-dir data/restricted_26_w2/latent_bridge_beta_tcvae \
  --top-k-attributes 5
```

Repeat the same for campaign `30`.

## Which Model To Use

Use this rule of thumb:

- `beta-VAE`
  - best overall latent baseline
- `Contrastive VAE`
  - best for target-vs-background campaign interpretation
- `beta-TCVAE`
  - best for campaign-to-latent bridge strength

## What Not To Do

- do not use non-held-out latent results as final evidence
- do not present VAE outputs as causal proof
- do not interpret every latent dimension as meaningful
- do not skip sensitivity analysis before writing campaign claims

## Minimum Deliverable Set

If time is short, the minimum defensible output is:

1. restricted `validate-campaigns` result
2. held-out `beta-VAE` latent validation
3. one bridge artifact for the campaign you want to discuss

If time allows, add:

4. `Contrastive VAE`
5. `beta-TCVAE`
6. side-by-side bridge comparison

# Quickstart: Campaign Impact and Latent Validation

## Prerequisites

1. Ensure the Dunnhumby Complete Journey source files are present in `data/`.
2. Ensure the existing environment is installed with project dependencies.
3. Confirm that trained model artifacts or known-good run IDs are available for baseline VAE and Beta-VAE comparison.
4. Use a reproducible seed for all validation runs.

## 1. Build the Campaign Validation Dataset

Generate a campaign-linked dataset for the strongest restricted quasi-causal
candidates found so far.

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
  --output-dir data/validation \
  --pre-weeks 4 \
  --post-weeks 4 \
  --seed 42
```

Review `dataset_summary.json` to confirm:
- treated and untreated households were created for each selected campaign
- pre-period, in-period, and post-period windows are populated
- exclusions are documented rather than silently dropped

## 2. Validate Campaign Effects

Run the restricted quasi-causal design with explicit diagnostics.

```bash
PYTHONPATH=. python3 main.py validate-campaigns \
  --analysis-data data/validation/campaign_analysis.parquet \
  --campaign-ids 26 30 \
  --method matched-did \
  --output-dir experiments/campaign_validation \
  --matching-method propensity \
  --propensity-caliper 0.02 \
  --seed 42
```

Inspect:
- `campaign_effects.json` for effect direction, magnitude, and evidence classification
- `campaign_diagnostics.json` for balance checks, placebo checks, and pre-trend status
- `campaign_event_study.parquet` for compact pre/campaign/post mean comparisons
- `campaign_balance_details.json` for covariate-level standardized mean differences
- `campaign_cohort_summary.json` for treated/comparison retention after matching

The workflow is expected to emit `supported`, `weak`, `unsupported`, or `insufficient_data` findings.

## 2b. Run Design Sensitivity Checks

Verify that the findings survive nearby window choices rather than a single
hand-picked design.

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
  --output-dir experiments/campaign_sensitivity \
  --matching-methods propensity \
  --propensity-calipers 0.02
```

Inspect:
- `campaign_sensitivity.json`
- `campaign_sensitivity.parquet`

Focus on:
- evidence label stability across nearby windows
- `worst_pre_smd`
- `matched_retention_rate`

## 3. Validate Latent Factor Semantics

Use the same analytic dataset and observable attributes to test factor stability.

Use experiment directory names under `experiments/` or pass explicit run paths.

```bash
PYTHONPATH=. python3 main.py validate-latents \
  --analysis-data data/validation/campaign_analysis.parquet \
  --attributes data/validation/validation_attributes.parquet \
  --run-ids baseline-best beta-best \
  --output-dir experiments/latent_validation
```

Inspect:
- `latent_metrics.json` for reconstruction and disentanglement metrics
- `factor_mappings.json` for candidate and final mapping decisions
- `latent_stability.json` for holdout and cross-run stability results

Reject factor names that fail holdout or stability checks.

## 3b. Build A Campaign-Latent Bridge

Connect campaign-visible attribute shifts to latent dimensions that were already
validated.

```bash
PYTHONPATH=. python3 main.py build-campaign-latent-bridge \
  --campaign-results experiments/campaign_validation/campaign_effects.json \
  --factor-mappings experiments/latent_validation/factor_mappings.json \
  --attributes data/validation/validation_attributes.parquet \
  --output-dir experiments/campaign_latent_bridge \
  --top-k-attributes 5
```

Inspect:
- `campaign_latent_bridge.json`
- `campaign_latent_bridge.parquet`

## 4. Generate the Final Validation Report

Compile campaign findings and latent validation results into one research artifact.

```bash
PYTHONPATH=. python3 main.py generate-validation-report \
  --campaign-results experiments/campaign_validation/campaign_effects.json \
  --campaign-diagnostics experiments/campaign_validation/campaign_diagnostics.json \
  --latent-results experiments/latent_validation/factor_mappings.json \
  --latent-metrics experiments/latent_validation/latent_metrics.json \
  --output reports/validation_report.md
```

The final report should answer:
- which campaign-effect claims are supported, weak, or unsupported
- which latent-factor mappings are validated, unstable, or unassessed
- which project narratives should be kept, softened, or withdrawn

The reporting step also writes `reports/claim_recommendations.json` for downstream tooling.

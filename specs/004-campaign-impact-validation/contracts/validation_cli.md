# CLI Contract: Campaign Impact and Latent Validation

## Purpose

Define the command-line interface expected for the campaign validation workflow. The CLI contract extends the existing project style and provides reproducible entry points for dataset construction, campaign validation, latent validation, and final report generation.

## Command 1: Build Validation Dataset

### Example

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
  --campaign-ids 18 13 8 \
  --output-dir data/validation
```

### Required Inputs

- `--transactions`: Raw transaction source
- `--products`: Product hierarchy source
- `--campaign-table`: Household campaign assignment source
- `--campaign-desc`: Campaign timing source
- `--coupon`: Coupon issuance source
- `--coupon-redempt`: Coupon redemption source
- `--campaign-ids`: One or more campaign identifiers to analyze
- `--output-dir`: Output directory for validation datasets

### Optional Inputs

- `--demographics`: Household demographic source
- `--causal-data`: Promotional exposure proxy source
- `--pre-weeks`: Length of pre-period window
- `--post-weeks`: Length of post-period window
- `--seed`: Reproducibility seed

### Outputs

- `campaign_analysis.parquet`: Household-campaign analysis dataset
- `comparison_pool.parquet`: Untreated comparison pool
- `validation_attributes.parquet`: Observable outcome and factor-validation attributes
- `dataset_summary.json`: Counts, missingness summary, and exclusions

## Command 2: Validate Campaign Effects

### Example

```bash
PYTHONPATH=. python3 main.py validate-campaigns \
  --analysis-data data/validation/campaign_analysis.parquet \
  --campaign-ids 18 13 8 \
  --method matched_did \
  --output-dir experiments/campaign_validation
```

### Required Inputs

- `--analysis-data`: Campaign-linked analysis dataset
- `--campaign-ids`: One or more campaign identifiers
- `--method`: Validation method identifier
- `--output-dir`: Output directory for campaign validation artifacts

### Optional Inputs

- `--outcomes`: Explicit list of behavioral outcomes to evaluate
- `--min-treated`: Minimum treated cohort size
- `--min-comparison`: Minimum comparison cohort size
- `--seed`: Reproducibility seed

### Outputs

- `campaign_effects.json`: Effect estimates and evidence classifications
- `campaign_diagnostics.json`: Balance, placebo, and pre-trend diagnostics
- `campaign_event_study.parquet`: Weekly aligned campaign trend output

## Command 3: Validate Latent Semantics

### Example

```bash
PYTHONPATH=. python3 main.py validate-latents \
  --analysis-data data/validation/campaign_analysis.parquet \
  --attributes data/validation/validation_attributes.parquet \
  --run-ids baseline-best beta-best \
  --output-dir experiments/latent_validation
```

### Required Inputs

- `--analysis-data`: Campaign-linked analysis dataset or aligned household-window dataset
- `--attributes`: Observable validation attributes
- `--run-ids`: One or more trained model identifiers
- `--output-dir`: Output directory for latent validation artifacts

### Optional Inputs

- `--model-types`: Explicit model-variant labels
- `--holdout-split`: Holdout selection rule
- `--seeds`: Additional evaluation seeds
- `--top-k-attributes`: Limit for reported candidate mappings

### Outputs

- `latent_metrics.json`: MIG, SAP, reconstruction, and related metrics
- `factor_mappings.json`: Candidate and final mapping decisions
- `latent_stability.json`: Cross-run and holdout stability outputs

## Command 4: Generate Research Report

### Example

```bash
PYTHONPATH=. python3 main.py generate-validation-report \
  --campaign-results experiments/campaign_validation/campaign_effects.json \
  --campaign-diagnostics experiments/campaign_validation/campaign_diagnostics.json \
  --latent-results experiments/latent_validation/factor_mappings.json \
  --latent-metrics experiments/latent_validation/latent_metrics.json \
  --output reports/validation_report.md
```

### Required Inputs

- `--campaign-results`: Campaign validation output
- `--campaign-diagnostics`: Campaign diagnostics output
- `--latent-results`: Latent mapping assessment output
- `--latent-metrics`: Latent metric output
- `--output`: Final report path

### Outputs

- `validation_report.md`: Reproducible research summary
- `claim_recommendations.json`: Structured recommendation on which project claims to keep, soften, or reject

## Error Handling Rules

- Commands must fail loudly when required inputs are missing.
- Campaigns with insufficient treated or comparison sample size must be marked as `insufficient_data`.
- Validation commands must preserve failed diagnostics in outputs instead of suppressing them.
- Final report generation must include negative and unsupported findings whenever present.

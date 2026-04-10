# Validation Computation Guide

This note explains the validation workflow as a sequence of reproducible data
transformations and statistical checks. It is written for an engineer who needs
to understand the logic quickly, not for a marketing specialist.

## Workflow Map

The current validation workflow has four layers:

1. build a campaign-linked analysis table
2. estimate campaign effects with a restrained quasi-causal design
3. test sensitivity to window size and matching restrictions
4. connect campaign-visible shifts to validated latent factors

The corresponding CLI commands are:

```bash
python main.py build-validation-data ...
python main.py validate-campaigns ...
python main.py analyze-campaign-sensitivity ...
python main.py build-campaign-latent-bridge ...
```

## Step 1: Build The Campaign Analysis Table

`build-validation-data` joins several source tables:

- `transaction_data.csv`
- `product.csv`
- `campaign_table.csv`
- `campaign_desc.csv`
- `coupon.csv`
- `coupon_redempt.csv`
- optionally `hh_demographic.csv`
- optionally `causal_data.csv`

The output is a household-campaign panel. Each row represents one
`HOUSEHOLD_KEY x CAMPAIGN` pair.

### Main Concepts

- `treated household`: a household listed in `campaign_table.csv` for a campaign
- `comparison household`: a household not assigned to the campaign but active in
  the same time span
- `pre window`: period before campaign start
- `campaign window`: period between `START_DAY` and `END_DAY`
- `post window`: period after campaign end

### Main Outputs

- `campaign_analysis.parquet`
- `comparison_pool.parquet`
- `validation_attributes.parquet`
- `dataset_summary.json`

### Key Derived Metrics

For each window, the pipeline aggregates behavior into metrics such as:

- `total_spend`
- `trip_count`
- `category_diversity`
- `promo_share`
- `avg_price_per_unit`
- `coupon_redemption_count`
- `spend_concentration`
- `target_product_share`

These metrics are later reused for both campaign validation and latent-factor
validation.

## Step 2: Estimate Campaign Effects

`validate-campaigns` reads `campaign_analysis.parquet` and computes
campaign-level effect summaries.

### Core Idea: Difference-In-Differences

The main effect is a simple **difference-in-differences** (`DiD`) estimate:

```text
DiD =
  (treated_campaign - treated_pre)
  - (comparison_campaign - comparison_pre)
```

Interpretation:

- positive `DiD`: treated households increased more than comparison households
- negative `DiD`: treated households decreased more than comparison households
- near zero: both cohorts moved similarly

Important:

- this is **quasi-causal**, not randomized causal proof
- it only becomes defensible if the cohorts look comparable before treatment

### Why Matching Is Used

Raw treated and untreated households are often very different before the
campaign starts. The pipeline therefore supports **propensity score matching**.

High-level process:

1. take only `eligible` records with complete `pre / campaign / post` data
2. estimate a **propensity score**
3. match treated and comparison households with similar scores
4. optionally reject poor matches with a **caliper**

### Key Terms To Learn Separately

- `difference-in-differences`
- `propensity score`
- `caliper matching`
- `common support`

## Step 3: Balance And Placebo Diagnostics

The validator does not trust an effect estimate by itself. It also checks
whether the design still looks credible.

### Balance Check

For each pre-period covariate, the pipeline computes a
**standardized mean difference** (`SMD`):

```text
SMD = (mean_treated - mean_comparison) / pooled_std
```

Interpretation:

- small absolute value means cohorts look similar
- large absolute value means treated and comparison are still far apart

The new `campaign_balance_details.json` file stores balance rows for each
campaign and outcome.

Key term:

- `standardized mean difference`

### Placebo Check

The workflow uses a simple post-period gap as a **placebo-style diagnostic**:

```text
placebo_effect = mean(post_treated) - mean(post_comparison)
```

This is not a classical placebo design, but a practical guardrail:

- if large post gaps remain after matching, the cohorts may still be drifting
- if that happens, the workflow weakens or rejects the evidence label

### Evidence Labels

Each outcome gets one label:

- `supported`
- `weak`
- `unsupported`
- `insufficient_data`

The label depends on:

- treated/comparison sample sizes
- `balance_pass`
- `placebo_pass`

The new `campaign_cohort_summary.json` file shows how many households were kept
after matching and what fraction of the treated cohort survived the restriction.

## Step 4: Sensitivity Analysis

`analyze-campaign-sensitivity` reruns the same design over a grid of window
sizes and matching settings.

Why this matters:

- a result that only appears at one arbitrary window size is fragile
- a result that survives `2`, `3`, and `4` week windows is more credible

Main outputs:

- `campaign_sensitivity.json`
- `campaign_sensitivity.parquet`

Important fields:

- `weeks`
- `matching_method`
- `propensity_caliper`
- `evidence_classification`
- `worst_pre_smd`
- `matched_retention_rate`

Interpretation rule:

- prefer designs with small `worst_pre_smd`
- prefer designs with high retention
- prefer designs whose evidence label remains stable across nearby windows

## Step 5: Latent-Factor Validation

`validate-latents` is a separate workflow. It does not prove campaign
causality. It tests whether latent dimensions from the VAE/Beta-VAE correspond
to stable, measurable behavioral attributes.

### Main Metrics

- `MIG`: **Mutual Information Gap**
- `SAP`: **Separated Attribute Predictability**

Practical interpretation:

- higher values suggest clearer separation between latent dimensions and
  observed attributes
- but these metrics are only meaningful when they are stable across holdout
  data and runs

### Fast Approximation Modes

The repository now supports faster alternatives:

- `--sap-method vectorized`
- `--mig-method binned`

These are useful for iteration and sensitivity checks. They should be treated as
approximate scoring modes, not as automatic replacements for every final run.

Key terms:

- `mutual information`
- `disentanglement`
- `holdout validation`

## Step 6: Campaign-Latent Bridge

`build-campaign-latent-bridge` links campaign-visible attribute shifts to
latent dimensions that were already validated.

This step is interpretive, not causal.

It asks:

- which observed attributes move most during a campaign?
- do any of those attributes already have validated latent dimensions?

Outputs:

- `campaign_latent_bridge.json`
- `campaign_latent_bridge.parquet`

This is useful for a final story like:

`Campaign 26 increased spend and category diversity; the largest moving
attributes overlap with validated latent dimensions tied to basket breadth or
promotion intensity.`

## Current Recommended Restricted Designs

Based on the current exploration:

- `Campaign 26`: start with `2` or `4` week windows
- `Campaign 30`: start with `3` or `5` week windows, then compare against `4`
- use `--matching-method propensity`
- use `--propensity-caliper 0.02`

These are not universal truths. They are currently the best-performing
configurations under the implemented diagnostics.

## How To Read The Main Artifacts

### `campaign_effects.json`

Contains one row per campaign-outcome pair:

- effect size
- direction
- evidence label
- treated/comparison sample sizes

### `campaign_diagnostics.json`

Contains the pass/fail diagnostics:

- `balance_pass`
- `placebo_pass`
- `placebo_effect`
- matching method used

### `campaign_balance_details.json`

Contains covariate-level balance diagnostics:

- covariate name
- treated mean
- comparison mean
- `standardized_mean_difference`
- `balance_status`

### `campaign_cohort_summary.json`

Contains cohort retention information:

- raw treated/comparison sizes
- matched treated/comparison sizes
- matched retention rate

### `campaign_sensitivity.json`

Contains the same findings repeated over a design grid so you can ask:

- which window sizes are stable?
- how much balance worsens when the window expands?
- does evidence collapse when retention becomes too low?

## Recommended Reading Order

If you are short on time, learn these in order:

1. `difference-in-differences`
2. `standardized mean difference`
3. `propensity score`
4. `caliper matching`
5. `common support`
6. `event study`
7. `mutual information`
8. `disentanglement`

That set is enough to understand almost every calculation in the current
validation pipeline.

# User Guide: Behavioral Impact Analysis Tool

**Target Audience:** Marketing Analysts, CRM Managers, Strategy Teams.

## What is this tool?
Traditional marketing analytics tells you *what* happened (e.g., "Sales went up 15% during the promo"). This tool is designed to tell you *how customer behavior shifted*, and how credible that shift looks under a restricted observational validation design.

Using VAE-family latent models together with campaign validation diagnostics, it helps compare whether customers simply bought more of the same or showed a broader behavioral shift.

## Why use it over standard A/B testing?
1. **Useful when experimentation is limited:** The workflow can support post-campaign diagnostics even when no formal holdout was used, but it should be treated as quasi-causal evidence rather than a replacement for randomized testing.
2. **Behavior-focused diagnostics:** It focuses on structural changes in basket composition, promotional activity, and category breadth rather than only top-line sales.
3. **Evidence labels included:** Campaign findings are expected to be labeled as supported, weak, unsupported, or insufficient data.
4. **Latent interpretation is optional:** If a campaign survives the validation layer, latent factors can help interpret the type of behavioral shift. If the mappings are unstable, they should not be used in a business-facing story.

---

## How to Run an Analysis

You don't need to be a data scientist to get answers. You just need your transaction data.

### 1. Prepare your data
Ensure you have your historical data (e.g., the last 12 months) and your post-campaign data (e.g., the 3 months following your initiative) in standard Parquet format.

### 2. Run the Command
Ask your engineering team to build the validation dataset, run campaign validation, and then review the generated report. For final latent claims, use the leakage-controlled path rather than exploratory runs.

```bash
python main.py build-validation-data --transactions data/transaction_data.csv --products data/product.csv --campaign-table data/campaign_table.csv --campaign-desc data/campaign_desc.csv --coupon data/coupon.csv --coupon-redempt data/coupon_redempt.csv --campaign-ids 26 30 --output-dir data/validation

python main.py validate-campaigns --analysis-data data/validation/campaign_analysis.parquet --campaign-ids 26 30 --method matched-did --output-dir data/validation/restricted --matching-method propensity --propensity-caliper 0.02

python main.py generate-validation-report --campaign-results data/validation/restricted/campaign_effects.json --campaign-diagnostics data/validation/restricted/campaign_diagnostics.json --latent-results experiments/latent_validation_noleak_eval/factor_mappings.json --latent-metrics experiments/latent_validation_noleak_eval/latent_metrics.json --output reports/validation_report.md
```

### 3. Read the Report
The tool will generate `reports/validation_report.md` and `reports/claim_recommendations.json`. You can read the markdown summary directly or move the structured outputs into BI tooling.

---

## How to Interpret the Metrics

When you receive the report, focus on these core business metrics:

### 1. Evidence Classification
* **What it is:** The final label for each campaign outcome.
* **How to read it:**
  * `supported`: strongest current evidence under the restricted design
  * `weak`: direction is interesting, but diagnostics are not strong enough for a hard claim
  * `unsupported`: do not write a campaign-effect claim from this result
  * `insufficient_data`: too little data for a defensible conclusion

### 2. Balance Diagnostics
* **What it is:** Checks whether treated and comparison households look similar before the campaign.
* **How to read it:** If balance fails, a positive effect estimate is not enough. The cohorts may simply be too different for a credible comparison.

### 3. Placebo-Style Diagnostics
* **What it is:** A guardrail that checks whether problematic gaps remain after matching.
* **How to read it:** If placebo diagnostics fail, keep the conclusion cautious even if the headline metric looks attractive.

### 4. Factor Breakdown
* **What it is:** A latent interpretation layer that attempts to map behavior shifts to validated latent factors.
* **How to read it:**
  * Use factors only when the mapping is marked as `validated`.
  * Treat them as interpretation aids, not as proof of causality.

### 5. Top Sensitive Attributes Or Categories
* **What it is:** A ranked list of observable attributes or categories associated with the shift.
* **How to use it:** Use them as hypotheses for future targeting, experimentation, or merchandising. They are not proof by themselves.

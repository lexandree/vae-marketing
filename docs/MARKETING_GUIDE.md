# User Guide: Behavioral Impact Analysis Tool

**Target Audience:** Marketing Analysts, CRM Managers, Strategy Teams.

## What is this tool?
Traditional marketing analytics tells you *what* happened (e.g., "Sales went up 15% during the promo"). This tool tells you *why* and *how deep* the change was. 

Using advanced Machine Learning (Disentangled Variational Autoencoders), it learns a historical behavioral baseline for your customers from months of transaction data. When you run a campaign, it helps compare whether customers simply bought more of the same or showed a broader behavioral shift.

## Why use it over standard A/B testing?
1. **Useful when experimentation is limited:** The workflow can support post-campaign diagnostics even when no formal holdout was used, but it should be treated as quasi-causal evidence rather than a replacement for randomized testing.
2. **Behavior-focused diagnostics:** It focuses on structural changes in basket composition, promotional activity, and category breadth rather than only top-line sales.
3. **Evidence labels included:** Campaign findings are expected to be labeled as supported, weak, unsupported, or insufficient data.

---

## How to Run an Analysis

You don't need to be a data scientist to get answers. You just need your transaction data.

### 1. Prepare your data
Ensure you have your historical data (e.g., the last 12 months) and your post-campaign data (e.g., the 3 months following your initiative) in standard Parquet format.

### 2. Run the Command
Ask your engineering team to build the validation dataset, run campaign validation, and then review the generated report.

```bash
python main.py build-validation-data --transactions data/transaction_data.csv --products data/product.csv --campaign-table data/campaign_table.csv --campaign-desc data/campaign_desc.csv --coupon data/coupon.csv --coupon-redempt data/coupon_redempt.csv --campaign-ids 18 13 8 --output-dir data/validation

python main.py validate-campaigns --analysis-data data/validation/campaign_analysis.parquet --campaign-ids 18 13 8 --method matched_did --output-dir experiments/campaign_validation

python main.py generate-validation-report --campaign-results experiments/campaign_validation/campaign_effects.json --campaign-diagnostics experiments/campaign_validation/campaign_diagnostics.json --latent-results experiments/latent_validation/factor_mappings.json --latent-metrics experiments/latent_validation/latent_metrics.json --output reports/validation_report.md
```

### 3. Read the Report
The tool will generate `reports/validation_report.md` and `reports/claim_recommendations.json`. You can read the markdown summary directly or move the structured outputs into BI tooling.

---

## How to Interpret the Metrics

When you receive the report, focus on these core business metrics:

### 1. Average Latent Deviation (The "Impact Score")
* **What it is:** A mathematical measure of how far the customer moved from their historical baseline.
* **How to read it:** 
  * `~ 0.0 - 0.5`: **Business as Usual.** The campaign generated sales, but didn't change habits. People just stocked up.
  * `0.5 - 2.0`: **Moderate Shift.** You successfully introduced customers to new categories or price tiers.
  * `> 2.0`: **Transformational.** You completely changed how these households interact with your brand.

### 2. Average Persistence Days (The "Stickiness")
* **What it is:** How many days the customer maintained the new behavior before reverting to their old baseline.
* **How to read it:** If a campaign yields a high Impact Score but 0 Persistence Days, it means customers "gamed" the promotion and immediately churned back. Look for campaigns that yield **30+ days** of persistence.

### 3. Factor Breakdown (The "Nature of Change")
* **What it is:** The tool attempts to break the total shift down into latent factors and then validate those factors against observable attributes.
* **How to read it:** 
  * If the dominant factor maps to *Volume* and the mapping is marked as validated, your campaign may have triggered stockpiling.
  * If the dominant factor mapping is rejected or unstable, do not use it as a business-facing explanation.

### 4. Top Sensitive Categories
* **What it is:** A ranked list of product categories that drove the biggest behavioral shifts.
* **How to use it:** Use these categories as hypotheses for future testing. They are not, by themselves, proof of causal marketing impact.

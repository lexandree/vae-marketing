# User Guide: Behavioral Impact Analysis Tool

**Target Audience:** Marketing Analysts, CRM Managers, Strategy Teams.

## What is this tool?
Traditional marketing analytics tells you *what* happened (e.g., "Sales went up 15% during the promo"). This tool tells you *why* and *how deep* the change was. 

Using advanced Machine Learning (Disentangled Variational Autoencoders), it learns the "baseline behavioral DNA" of your customers from months of historical data. When you run a campaign, it measures if the customer simply bought more of the same, or if you successfully **altered their fundamental shopping habits**.

## Why use it over standard A/B testing?
1. **No Control Group Needed:** The model uses the customer's own deep historical profile as the control. Perfect for mass campaigns where holdout groups are impossible.
2. **Filters out Noise:** It inherently ignores seasonal spikes and focuses on structural changes in basket composition (price sensitivity, brand loyalty, category exploration).
3. **Measures Stickiness:** It doesn't just look at the campaign week; it calculates how many days the new behavior persisted after the discount ended.

---

## How to Run an Analysis

You don't need to be a data scientist to get answers. You just need your transaction data.

### 1. Prepare your data
Ensure you have your historical data (e.g., the last 12 months) and your post-campaign data (e.g., the 3 months following your initiative) in standard Parquet format.

### 2. Run the Command
Ask your engineering team to execute the `infer` command, pointing it to your targeted customer dataset.

```bash
# Example command for the terminal
python main.py infer --run-id campaign_summer_26 --data post_campaign.parquet --baseline history.parquet
```

### 3. Read the Report
The tool will generate a comprehensive `inference_report.json`. You can load this into your BI tool (Tableau, PowerBI) or read the summary.

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
* **What it is:** The tool breaks the total shift down into independent factors.
* **How to read it:** 
  * If the dominant factor maps to *Volume*, your campaign triggered stockpiling.
  * If the dominant factor maps to *Category Diversity*, your cross-sell initiative worked.

### 4. Top Sensitive Categories
* **What it is:** A ranked list of product categories that drove the biggest behavioral shifts.
* **How to use it:** Use these categories as "Gateway Products" in your next campaign. Discounting these specific items is statistically proven to alter customer profiles more effectively than discounting random inventory.

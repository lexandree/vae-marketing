# Phase 0: Research — Data Preparation & Feature Engineering

## Decision 1: Filtering non-purchase transactions
- **Decision**: Exclude all returns/refunds and coupon redemptions from the raw data before aggregation.
- **Rationale**: Returns and coupon redemptions can distort true baseline consumption patterns. Filtering them out early improves VAE stability.
- **Alternatives considered**: Including all transactions and letting the model learn, which could confuse the anomaly detection, or keeping only coupons.

## Decision 2: Rolling Window Size
- **Decision**: Use a 7-day (weekly) rolling window for aggregating household transactions.
- **Rationale**: A weekly rolling window provides a good balance between capturing seasonal patterns (day-of-week) without excessive smoothing that obscures short-term marketing anomalies.
- **Alternatives considered**: Daily (too noisy/sparse), Bi-weekly, Monthly (too smooth).

## Decision 3: Handling Households with Zero Transactions
- **Decision**: Filter out households with zero transactions over the entire historical training period, but retain households with sparse data (imputing temporary gaps as zero-vectors).
- **Rationale**: Households with absolutely no history cannot provide a baseline for anomaly detection. However, temporary gaps are informative and should be retained as zero-vectors.
- **Alternatives considered**: Dropping all households with any gaps, which would bias the model towards only high-frequency shoppers.

## Decision 4: Output Format
- **Decision**: Save the processed dataset in Parquet format.
- **Rationale**: Parquet natively supports columnar storage and compression, aligning with the requirement to handle heavy-tailed data and scale to 1TB efficiently via vertical scaling. It is also highly optimized for loading into PyTorch via Polars or Pandas.
- **Alternatives considered**: CSV (too slow, large footprint), SQLite.

## Decision 5: Temporal Feature Encoding
- **Decision**: Use cyclical continuous encoding (sine/cosine transformations) for temporal features (week-of-year, month-of-year, day-of-week).
- **Rationale**: Cyclical encoding preserves the semantic proximity of temporal boundaries (e.g., December 31st and January 1st are close, not far apart), which is critical for continuous models like VAEs.
- **Alternatives considered**: Ordinal encoding (creates artificial jumps at year boundaries), One-hot encoding (increases dimensionality unnecessarily).
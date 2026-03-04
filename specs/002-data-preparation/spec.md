# Feature Specification: Data Preparation for VAE Marketing Analysis

**Feature Branch**: `002-data-preparation`
**Created**: 2026-03-04
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare Raw Transaction Data with Semantic & Temporal Features (Priority: P1)

As an ML Engineer, I need to prepare the raw Dunnhumby transaction data by applying hierarchical aggregations, temporal feature extraction, and rolling windows so that the data correctly models seasonal trends and product relationships without sparse one-hot explosions.

**Why this priority**: Without properly encoding the semantics of products and temporal context, the VAE model will treat all purchases as isolated anomalies and fail to learn true baseline household behavior.

**Independent Test**: Execute the pipeline on raw Dunnhumby data and verify that the output contains hierarchical product groupings, correct week/day seasonality flags, and dense aggregated vectors instead of raw one-hot IDs.

**Acceptance Scenarios**:

1. **Given** raw transaction data, **When** the pipeline processes product IDs, **Then** it encodes them as hierarchical aggregates (e.g., COMMODITY > SUB_COMMODITY) or embeddings instead of one-hot vectors.
2. **Given** transaction timestamps, **When** temporal extraction is applied, **Then** explicit week-of-year, day-of-week, and month-of-year features are included in the output vector.
3. **Given** highly sparse transaction history, **When** aggregation runs, **Then** transactions are aggregated over a rolling window and missing periods/products are explicitly encoded as zero-vectors (not removed).

---

### User Story 2 - Robust Normalization and Time-Series Splits (Priority: P2)

As an ML Engineer, I need to properly scale heavy-tailed purchase data and split datasets using forward-chaining so that the VAE baseline evaluation is free of data leakage.

**Why this priority**: Kaggle references and standard retail practices show that random shuffling introduces temporal leakage, invalidating any anomaly detection testing. Log-scaling prevents large promotional purchases from dominating the loss function.

**Independent Test**: Verify that the generated train/validation splits respect temporal order (e.g., Train: weeks 0-20, Val: weeks 21-30) and that normalization parameters are fit purely on the training window.

**Acceptance Scenarios**:

1. **Given** heavy-tailed numerical features (quantities, sales), **When** normalization is applied, **Then** the pipeline applies log-scale or Box-Cox transformation followed by z-score scaling fit on the entire training split distribution.
2. **Given** the need for model evaluation, **When** the pipeline generates dataset splits, **Then** it uses a forward-chaining (expanding window) strategy, ensuring no future data leaks into the training set.

---

### Edge Cases

- If the raw input data schema changes unexpectedly (e.g., new columns, missing expected columns), the pipeline MUST fail fast with detailed error reporting.
- If corrupt or malformed entries in the raw transaction data are detected (e.g., non-numeric sales figures), the pipeline MUST fail fast with detailed error reporting.
- Handling gaps: Households with gaps in purchase history must not be dropped (imputed as zero-vectors), but households with zero total purchases over the entire historical training period MUST be filtered out.

## Clarifications
- Q: What encoding strategy should be used for temporal features (week-of-year, month-of-year, etc.) to ensure the model understands time cycles? → A: Cyclical continuous encoding (sine/cosine transformations).

- Q: What format should be used to save the processed dataset? → A: Parquet.

- Q: How should households that have zero transactions within the entire historical training period be handled? → A: Filter them out entirely before modeling.

- Q: What is the primary rolling window size (e.g., daily, weekly, monthly) for aggregating household transactions? → A: Weekly (7 days).


### Session 2026-03-04

- Q: Which transaction types (e.g., refunds, coupon redemptions) should be included or excluded from the raw data before modeling? → A: Exclude all returns/refunds and coupon redemptions.


## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST load raw transaction data from specified input paths.
- **FR-002**: The system MUST filter out households with zero transactions over the entire historical training period, but retain households with sparse data (imputing temporary gaps as zero-vectors).
- **FR-003**: The system MUST transform high-cardinality product IDs using hierarchical product features (e.g., COMMODITY → SUB_COMMODITY) or learned embeddings, strictly avoiding raw one-hot encoding.
- **FR-004**: The system MUST extract and include explicit temporal features (week-of-year, day-of-week, month-of-year) for each transaction window.
- **FR-005**: The system MUST aggregate household transactions over a 7-day (weekly) rolling window to create fixed-size, dense feature vectors.
- **FR-005.1**: The system MUST filter out all non-purchase transaction types (e.g., returns, refunds, coupon redemptions) before aggregation.
- **FR-006**: The system MUST apply log-scale or Box-Cox transformations to heavy-tailed features (e.g., sales values) and normalize them (e.g., z-score) using parameters fit on the entire training split distribution (formerly "household cohort") to avoid leakage.
- **FR-007**: The system MUST partition the data using a forward-chaining (expanding window) time-series validation strategy.
- **FR-008**: The system MUST save the processed dataset to a specified output path in Parquet format to optimize for PyTorch and scale.

### Key Entities *(include if feature involves data)*

- **Raw Transaction Data**: Unprocessed retail transaction records (customer identifiers, product details, purchase amounts, timestamps).
- **Prepared VAE Data**: Aggregated, hierarchical, and scaled dense vectors ready for feedforward VAE consumption.
- **Time-Series Splits**: Temporally ordered datasets (Train, Validation, Test) generated via forward-chaining.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of raw transaction data files are successfully loaded and processed without unhandled errors.
- **SC-002**: The data preparation pipeline completes processing of a standard-sized dataset (e.g., 1GB) within 15 minutes, and scales to handle up to 1TB of raw data via vertical scaling.
- **SC-003**: The resulting dataset has a bounded dimensionality suitable for a feedforward VAE (e.g., < 2000 features per vector), successfully avoiding the sparse explosion of raw product IDs.
- **SC-004**: Data splits strictly preserve temporal order (0% future data leakage in the training set).
- **SC-005**: The prepared data, when used for VAE training, enables model convergence (reconstruction loss decreases by at least 50% within 50 epochs).

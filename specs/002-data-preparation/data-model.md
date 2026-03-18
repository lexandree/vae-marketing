# Data Model

## 1. Raw Transaction Data Schema (Input)
Represents the unprocessed retail transaction records (Dunnhumby data format).

| Field | Type | Description |
|-------|------|-------------|
| `BASKET_ID` | String | Unique identifier for a shopping basket/transaction. |
| `HOUSEHOLD_KEY` | String | Unique identifier for the customer/household. |
| `DAY` | Integer | Day of the transaction (can be converted to timestamp). |
| `PRODUCT_ID` | String | Unique identifier for the product. |
| `QUANTITY` | Integer/Float | Number of items purchased. |
| `SALES_VALUE` | Float | Total amount paid for the product. |
| `STORE_ID` | String | Identifier for the store. |
| `TRANS_TIME` | Integer | Time of transaction. |

## 2. Product Hierarchy Schema (Auxiliary Input)
Represents the metadata for products to enable hierarchical aggregation.

| Field | Type | Description |
|-------|------|-------------|
| `PRODUCT_ID` | String | Unique identifier for the product. |
| `COMMODITY_DESC` | String | High-level category (e.g., MEAT, GROCERY). |
| `SUB_COMMODITY_DESC`| String | Detailed category (e.g., BEEF - GROUND). |

## 3. Prepared VAE Data Schema (Output)
Aggregated, hierarchical, and scaled dense vectors ready for feedforward VAE consumption. Saved as Parquet.

| Field | Type | Description |
|-------|------|-------------|
| `HOUSEHOLD_KEY` | String | Unique identifier for the customer/household. |
| `WINDOW_START_DAY`| Integer | The starting day of the 7-day rolling window. |
| `COMMODITY_*_SPEND` | Float | Log-scaled and z-score normalized spend for a specific commodity group. |
| `COMMODITY_*_QTY` | Float | Log-scaled and z-score normalized quantity for a specific commodity group. |
| `TEMPORAL_WEEK_SIN` | Float | Cyclical sine encoding of the week-of-year. |
| `TEMPORAL_WEEK_COS` | Float | Cyclical cosine encoding of the week-of-year. |
| `TEMPORAL_DAY_SIN` | Float | Cyclical sine encoding of the day-of-week. |
| `TEMPORAL_DAY_COS` | Float | Cyclical cosine encoding of the day-of-week. |
| `TEMPORAL_MONTH_SIN`| Float | Cyclical sine encoding of the month-of-year. |
| `TEMPORAL_MONTH_COS`| Float | Cyclical cosine encoding of the month-of-year. |

**Constraints & Validation**:
- Dimensionality must be < 2000 features per vector.
- No `NaN` values permitted (imputed to 0 or appropriate scaled value).

## 4. Time-Series Splits
- **Train Split**: Transactions from the first `N` weeks (e.g., weeks 0-20). Used to fit normalization parameters (mean, std).
- **Validation Split**: Transactions from the subsequent `M` weeks (e.g., weeks 21-30). Scaled using Train parameters.
- **Test Split**: Transactions from the remaining weeks. Scaled using Train parameters.
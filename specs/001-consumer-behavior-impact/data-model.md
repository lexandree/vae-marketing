# Data Model

## Entities

### Household
Represents the purchasing unit.
- `household_id` (String/Int): Unique identifier.
- `baseline_profile` (Vector/Array): The mean (mu) vector in the VAE latent space representing normal behavior.

### Transaction (Input Data)
A single purchase event.
- `transaction_id` (String/Int)
- `household_id` (String/Int)
- `timestamp` (DateTime)
- `product_category` (String)
- `quantity` (Float/Int)
- `price` (Float)
- `month_of_year` (Int): Derived for seasonality context.
- `week_of_year` (Int): Derived for seasonality context.

### External Stimulus
A marketing activity or event.
- `stimulus_id` (String)
- `stimulus_type` (String): e.g., 'Discount', 'Ad Campaign'
- `start_time` (DateTime)
- `end_time` (DateTime)
- `affected_categories` (List[String])

### Behavioral Shift (Output)
The measured deviation from the baseline.
- `household_id` (String)
- `stimulus_id` (String)
- `quantitative_magnitude` (Float): Distance measured in latent space.
- `qualitative_nature` (String): Enum `['Stockpiling', 'Trading Up', 'Brand Switching', 'No Change']`.
- `persistence_duration_days` (Int): Time for behavior to return to baseline.
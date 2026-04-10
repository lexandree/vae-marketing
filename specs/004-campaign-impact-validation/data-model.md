# Data Model: Campaign Impact and Latent Validation

## Entities

### Campaign Definition
Represents the calendar and identity of a marketing campaign.
- **campaign_id**: Integer campaign identifier
- **campaign_type**: Descriptive campaign category
- **start_day**: First active day of the campaign
- **end_day**: Last active day of the campaign
- **duration_days**: Campaign length derived from start and end day

### Household Campaign Assignment
Represents whether a household was assigned to a specific campaign.
- **household_id**: Household identifier
- **campaign_id**: Linked campaign identifier
- **treatment_status**: Assigned or not assigned
- **campaign_type**: Inherited campaign category for reporting and cohort filtering

### Campaign Analysis Record
Core analytic unit used for campaign validation.
- **record_id**: Unique household-campaign key
- **household_id**: Household identifier
- **campaign_id**: Campaign identifier
- **treatment_status**: Treated or untreated
- **pre_window_start_day**: First day of pre-period observation
- **pre_window_end_day**: Last day of pre-period observation
- **campaign_start_day**: Campaign start day
- **campaign_end_day**: Campaign end day
- **post_window_start_day**: First day of post-period observation
- **post_window_end_day**: Last day of post-period observation
- **eligibility_status**: Eligible, insufficient history, overlapping campaign, or excluded

### Behavioral Outcome Summary
Observable purchase-behavior aggregates computed for a campaign analysis record within a specific window.
- **record_id**: Linked campaign analysis record
- **window_type**: Pre-period, in-period, or post-period
- **total_spend**: Aggregate spending level
- **total_quantity**: Aggregate purchased quantity
- **trip_count**: Number of shopping baskets or visits
- **avg_price_per_unit**: Ratio of spend to quantity
- **promo_share**: Share of observed spending tied to promoted items or conditions
- **coupon_redemption_count**: Number of redeemed coupons in the window
- **category_diversity**: Breadth of category participation
- **spend_concentration**: Degree to which spend is concentrated in a small set of categories
- **target_product_share**: Share linked to campaign coupon products when available

### Household Baseline Profile
Pre-period descriptive summary used for matching and balance checks.
- **household_id**: Household identifier
- **baseline_spend_level**: Pre-period spending summary
- **baseline_trip_level**: Pre-period visit intensity
- **baseline_category_mix**: Pre-period category composition summary
- **baseline_coupon_activity**: Historical coupon usage summary
- **baseline_promo_share**: Historical promotional participation
- **demographic_segment**: Household demographic summary

### Comparison Cohort Membership
Represents how untreated households are selected for campaign evaluation.
- **campaign_id**: Campaign identifier
- **household_id**: Household identifier
- **cohort_role**: Treated, matched comparison, unmatched comparison, or excluded
- **matching_group**: Identifier for matched set or weighting group
- **balance_status**: Balanced, weak_balance, or failed_balance

### Latent Representation Snapshot
Latent summary extracted from an existing model for a household-window observation.
- **household_id**: Household identifier
- **window_type**: Period aligned to the validation dataset
- **model_variant**: Baseline VAE or Beta-VAE
- **run_id**: Source experiment identifier
- **latent_vector**: Ordered latent dimension values

### Validation Attribute
Observable behavior-derived attribute used to test latent semantics.
- **household_id**: Household identifier
- **window_type**: Aligned evaluation period
- **attribute_name**: Human-readable attribute label
- **attribute_value**: Numeric value used in validation
- **attribute_family**: Spending, diversity, promotion, pricing, concentration, or frequency

### Campaign Effect Assessment
Result of quasi-causal validation for one campaign and outcome.
- **campaign_id**: Campaign identifier
- **outcome_name**: Evaluated behavioral outcome
- **treated_sample_size**: Number of treated households evaluated
- **comparison_sample_size**: Number of comparison households evaluated
- **effect_direction**: Increase, decrease, or neutral
- **effect_size**: Estimated campaign effect magnitude
- **diagnostic_status**: Passed, weak, failed, or insufficient data
- **evidence_classification**: Supported, weak, unsupported, or insufficient data

### Factor Mapping Assessment
Result of testing whether a latent dimension has stable semantic meaning.
- **model_variant**: Baseline VAE or Beta-VAE
- **run_id**: Source experiment identifier
- **latent_dimension**: Index of evaluated latent factor
- **candidate_attribute**: Best-aligned observable attribute
- **association_strength**: Quantitative strength of the relationship
- **holdout_status**: Confirmed, weakened, failed, or not assessed
- **stability_status**: Stable, unstable, or inconclusive
- **mapping_decision**: Validated, rejected, or deferred

### Research Report Artifact
Reproducible summary of campaign and latent validation findings.
- **report_id**: Unique report identifier
- **campaign_scope**: Included campaigns
- **model_scope**: Included model variants and runs
- **generated_at**: Report creation time
- **finding_inventory**: Set of supported, weak, unsupported, and insufficient-data findings
- **claim_recommendations**: Which repository claims remain defensible, should be softened, or should be withdrawn

## Relationships

- **Campaign Definition** links to many **Household Campaign Assignments**.
- **Household Campaign Assignment** produces one or more **Campaign Analysis Records** after window construction.
- **Campaign Analysis Record** owns multiple **Behavioral Outcome Summaries**, one per window.
- **Household Baseline Profile** informs **Comparison Cohort Membership** and campaign matching.
- **Comparison Cohort Membership** feeds **Campaign Effect Assessment**.
- **Campaign Analysis Record** aligns with **Latent Representation Snapshot** and **Validation Attribute** on household and window.
- **Latent Representation Snapshot** and **Validation Attribute** produce **Factor Mapping Assessments**.
- **Campaign Effect Assessments** and **Factor Mapping Assessments** are aggregated into a **Research Report Artifact**.

## Validation Rules

- Campaign analysis records require valid campaign timing and non-null household identifiers.
- A treated household must be assigned to the evaluated campaign in the assignment table.
- Untreated comparison records must not be assigned to the evaluated campaign and must share the same campaign calendar alignment.
- Records lacking sufficient pre-period or post-period observations must be marked ineligible rather than silently included.
- Campaign effect assessments cannot be labeled `supported` unless at least one diagnostic passes and no critical diagnostic fails.
- Factor mappings cannot be labeled `validated` unless the mapping is observed on holdout data and passes stability checks across repeated evaluations.

## State Transitions

1. **Campaign Definition Imported**: Campaign metadata loaded and validated.
2. **Analytic Dataset Built**: Household-campaign windows and behavioral summaries constructed.
3. **Comparison Cohorts Assigned**: Treated and untreated records aligned and matched.
4. **Campaign Effects Evaluated**: Effect estimates and diagnostics computed.
5. **Latent Semantics Evaluated**: Latent factors compared against validation attributes.
6. **Report Generated**: Findings and claim recommendations written to a reproducible artifact.

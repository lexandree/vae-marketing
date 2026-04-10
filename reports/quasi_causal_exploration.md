# Quasi-Causal Exploration Notes

## Scope

This note documents exploratory work on quasi-causal campaign validation for the
Dunnhumby Complete Journey data using the new validation workflow. The goal was
to test whether campaign-level treated vs. non-treated comparisons could support
credible quasi-causal claims before elevating any findings into the main report.

Campaigns screened first:

- `18`
- `13`
- `8`
- `30`
- `26`

Primary outcomes:

- `total_spend`
- `trip_count`
- `category_diversity`
- `promo_share`

Important interpretation rule used throughout:

- Passing balance alone is not enough.
- Passing balance and still showing large post-period differences is not enough.
- Strong quasi-causal claims should not be made unless both cohort comparability
  and placebo-style diagnostics look reasonable.

## Data And Workflow

Exploration used:

- `data/transaction_data.csv`
- `data/product.csv`
- `data/campaign_table.csv`
- `data/campaign_desc.csv`
- `data/coupon.csv`
- `data/coupon_redempt.csv`
- `data/hh_demographic.csv`

For each campaign:

1. Construct an event window from `START_DAY/END_DAY` with a `60` day lead and lag.
2. Define treated households from `campaign_table.csv`.
3. Build a wide untreated candidate pool from active households in the same window.
4. Assemble a household-campaign validation dataset.
5. Evaluate simple quasi-causal diagnostics.
6. For the most promising campaigns, test propensity-style `1:1` matching and
   caliper matching on pre-period covariates.

## Screening Results

Source artifact:

- `data/campaign_screening_results.json`

### Campaign 18

Summary:

- `1133 treated`
- `1000 comparison`

Outcome status:

- `total_spend`: `unsupported`
- `trip_count`: `unsupported`
- `category_diversity`: `unsupported`
- `promo_share`: `supported`

Diagnostics:

- `balance_pass = false` for spend, trips, diversity
- `placebo_pass = false` for spend, trips, diversity

Interpretation:

- Not suitable for quasi-causal claims in the current design.

### Campaign 13

Summary:

- `1077 treated`
- `1000 comparison`

Outcome status:

- `total_spend`: `unsupported`
- `trip_count`: `unsupported`
- `category_diversity`: `unsupported`
- `promo_share`: `supported`

Diagnostics:

- strong pre-period imbalance
- placebo diagnostics fail

Interpretation:

- Worse than a weak result; the comparison design is not credible here.

### Campaign 8

Summary:

- `1076 treated`
- `1000 comparison`

Outcome status:

- `total_spend`: `unsupported`
- `trip_count`: `unsupported`
- `category_diversity`: `unsupported`
- `promo_share`: `supported`

Diagnostics:

- strong pre-period imbalance
- placebo diagnostics fail

Interpretation:

- Not a viable candidate for quasi-causal claims in the current setup.

### Campaign 30

Summary:

- `361 treated`
- `1000 comparison`

Outcome status:

- `total_spend`: `weak`
- `trip_count`: `weak`
- `category_diversity`: `unsupported`
- `promo_share`: `supported`

Diagnostics:

- balance improved relative to `18/13/8`
- placebo still fails for the business-relevant outcomes

Interpretation:

- One of the two best candidates for deeper identification work.

### Campaign 26

Summary:

- `332 treated`
- `1000 comparison`

Outcome status:

- `total_spend`: `weak`
- `trip_count`: `weak`
- `category_diversity`: `weak`
- `promo_share`: `supported`

Diagnostics:

- better cohort comparability than `18/13/8`
- placebo still fails

Interpretation:

- Best candidate from the first pass, but still not strong enough for a causal-style claim.

## Matching Experiments

### Propensity-Style Matching Setup

Matching used pre-period covariates:

- `pre_total_spend`
- `pre_trip_count`
- `pre_category_diversity`
- `pre_avg_price_per_unit`
- `pre_promo_share`
- `pre_coupon_redemption_count`
- `pre_spend_concentration`
- `pre_target_product_share`

The practical goal was to test whether improving pre-period balance would also
reduce post-period placebo-style gaps.

### Campaign 26 After `1:1` Matching

Matched sample:

- `310 treated`
- `310 comparison`

Result:

- pre-period balance became good across almost all tracked covariates
- placebo-style post differences remained material

Selected values:

- `total_spend`: `did = -2.88`, `post_diff = -62.54`
- `trip_count`: `did = -0.51`, `post_diff = -1.05`
- `category_diversity`: `did = -2.58`, `post_diff = -5.19`

Interpretation:

- Balance improved, but the broader identification problem did not disappear.

### Campaign 30 After `1:1` Matching

Matched sample:

- `336 treated`
- `336 comparison`

Result:

- pre-period balance became good
- placebo-style post differences became smaller than campaign `26`
- but they remained too large for a strong quasi-causal interpretation

Selected values:

- `total_spend`: `did = -16.55`, `post_diff = -30.76`
- `trip_count`: `did = -0.40`, `post_diff = -1.11`
- `category_diversity`: `did = -1.62`, `post_diff = -2.97`

Interpretation:

- Better candidate than `26`, but still weak rather than compelling.

### Campaigns 13 And 8 After Matching

Result:

- matching did not fix balance
- post-period gaps remained large

Campaign `13`:

- matched `846 / 846`
- still large standardized mean differences
- `total_spend`: `did = -37.69`, `post_diff = 222.12`

Campaign `8`:

- matched `825 / 825`
- still large standardized mean differences
- `total_spend`: `did = -41.66`, `post_diff = 195.42`

Interpretation:

- `13` and `8` should be dropped from the quasi-causal path in the current design.

## Caliper Matching For The Best Candidates

Additional caliper matching was tested for campaigns `26` and `30`.

### Campaign 26

Best observed caliper region:

- around `0.02`

Result:

- matched sample stayed around `307`
- worst pre-period SMD improved to about `0.078`
- post-period gaps still remained material

Selected values at `0.02`:

- `total_spend`: `did = -6.62`, `post_diff = -52.93`
- `trip_count`: `did = -0.51`, `post_diff = -0.71`
- `category_diversity`: `did = -2.85`, `post_diff = -4.34`

### Campaign 30

Best observed caliper region:

- around `0.01` to `0.05`

Result:

- matched sample stayed around `330`
- worst pre-period SMD dropped to about `0.025`
- post-period gaps still did not collapse

Selected values at `0.01`:

- `total_spend`: `did = -18.17`, `post_diff = -31.62`
- `trip_count`: `did = -0.46`, `post_diff = -1.05`
- `category_diversity`: `did = -1.18`, `post_diff = -2.62`

Interpretation:

- Caliper matching improves common support, especially for campaign `30`.
- It does not eliminate the remaining post-period differences.

## What This Supports

Supported:

- the validation workflow is able to reject weak designs instead of producing
  automatic positive claims
- campaigns `26` and `30` are better candidates than `18`, `13`, and `8`
- propensity-style matching and calipers improve pre-period balance

Not supported:

- a strong causal claim for any of the screened campaigns
- the claim that the current treated-vs-untreated design is sufficient by itself

## Recommended Next Steps

Two research paths remain valid.

### Path A: Deeper Quasi-Causal Design

Focus only on `26` and `30`.

Candidate directions:

- tighter event-time cohort restrictions
- alternative control construction
- campaign-specific coupon or exposure-driven cohorts
- explicit pre-trend screening before effect estimation

### Path B: Observational Campaign Response Framing

If deeper identification still fails, keep the work valuable by reframing it as:

- observational campaign-response analysis
- behavior-shift analysis
- latent-factor interpretation layered on top of observed campaign windows

## Current Recommendation

Yes, both paths can be explored.

Best sequence:

1. Continue quasi-causal work only for `26` and `30`.
2. In parallel, preserve the observational framing as the fallback narrative.
3. Drop `18`, `13`, and `8` from the strong-claim track unless new identifying
   information becomes available.

## Restricted Event-Time Design

After the first screening and matching rounds, a narrower event-time design was
tested for the best candidates. The key idea was:

- shorten the pre and post windows
- keep propensity-based matching
- keep a strict caliper
- see whether post-period gaps shrink enough to support a more credible
  quasi-causal interpretation

### Campaign 26 Restricted Run

Configuration:

- `pre_weeks = 2`
- `post_weeks = 2`
- `matching_method = propensity`
- `propensity_caliper = 0.02`

Artifacts:

- `data/restricted_26_w2/campaign_analysis.parquet`
- `data/restricted_26_w2/restricted/campaign_effects.json`
- `data/restricted_26_w2/restricted/campaign_diagnostics.json`

Observed result:

- `total_spend`: `supported`
- `trip_count`: `supported`
- `category_diversity`: `supported`
- `promo_share`: `supported`

Selected values:

- `total_spend`: `effect_size = 68.77`, `placebo_effect = 17.70`
- `trip_count`: `effect_size = 1.46`, `placebo_effect = 0.23`
- `category_diversity`: `effect_size = 5.19`, `placebo_effect = 2.84`

Interpretation:

- Narrowing the event window substantially improved diagnostics.
- This is the strongest quasi-causal candidate found so far.
- It still needs cautious language, but it is no longer just a weak or negative result.

### Campaign 30 Restricted Run

Configuration:

- `pre_weeks = 4`
- `post_weeks = 4`
- `matching_method = propensity`
- `propensity_caliper = 0.02`

Artifacts:

- `data/restricted_30_w4/campaign_analysis.parquet`
- `data/restricted_30_w4/restricted/campaign_effects.json`
- `data/restricted_30_w4/restricted/campaign_diagnostics.json`

Observed result:

- `total_spend`: `supported`
- `trip_count`: `weak`
- `category_diversity`: `supported`
- `promo_share`: `supported`

Selected values:

- `total_spend`: `effect_size = 22.42`, `placebo_effect = 17.34`
- `trip_count`: `effect_size = 0.41`, `placebo_effect = 0.67`
- `category_diversity`: `effect_size = 3.52`, `placebo_effect = 2.55`

Interpretation:

- Restricted windows improved the campaign substantially.
- `trip_count` still looks weaker than the other outcomes.
- Campaign `30` is now a plausible secondary quasi-causal example behind campaign `26`.

## Updated Recommendation

The restricted design changes the recommendation materially:

- `26` should be the primary campaign for the quasi-causal track.
- `30` should be the secondary campaign for the quasi-causal track.
- `18`, `13`, and `8` should remain outside the strong-claim path for now.

The most defensible current story is:

- use `26` as the main quasi-causal case study
- use `30` as a supporting case
- keep observational framing available as fallback for the broader project narrative

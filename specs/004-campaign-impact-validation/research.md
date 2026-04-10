# Phase 0: Research - Campaign Impact and Latent Validation

## Decision 1: Prioritize High-Coverage Campaigns for First-Pass Validation
- **Decision**: Start the validation workflow with campaigns `18`, `13`, and `8`, and keep campaigns `30` and `26` as secondary candidates for expansion.
- **Rationale**: Local dataset inspection shows that campaigns `18`, `13`, and `8` have the strongest household coverage among campaign assignments, while also showing meaningful coupon redemption activity. Starting with the largest campaigns maximizes statistical power and reduces the risk of spending early effort on underpowered cohorts.
- **Alternatives considered**: Evaluate all campaigns equally from the start. Rejected because early analysis would be dominated by sparse campaigns, increasing the chance of inconclusive results and slowing down the first credible validation pass.

## Decision 2: Use Household-Campaign Windows as the Core Analytic Unit
- **Decision**: Build the analysis dataset at the household-campaign level, with three window types per record: an 8-week pre-period, the actual campaign period from campaign metadata, and an 8-week post-period after campaign end.
- **Rationale**: This unit aligns naturally with the business question, supports both campaign evaluation and latent validation, and avoids conflating unrelated calendar periods. Fixed pre/post spans keep cross-campaign comparisons interpretable while preserving the real campaign duration during treatment.
- **Alternatives considered**: Use only global train/validation/test windows or pure household-week panels. Rejected because they weaken campaign attribution and make campaign-aligned diagnostics harder to interpret.

## Decision 3: Define Untreated Cohorts from Non-Assigned Households with Baseline Matching
- **Decision**: For each selected campaign, construct untreated cohorts from households not assigned to that campaign, then match or weight them using pre-period household behavior and demographic summaries.
- **Rationale**: The project does not have randomized control groups. Matching on observed baseline characteristics is the most defensible way to move from descriptive comparison toward quasi-causal evidence while staying within the available data.
- **Alternatives considered**: Use raw non-treated averages without adjustment or limit analysis to treated households only. Rejected because both alternatives leave the workflow overly vulnerable to selection bias and make campaign-effect claims much weaker.

## Decision 4: Use Matched Difference-in-Differences as the Primary Quasi-Causal Baseline
- **Decision**: Use matched treated-versus-untreated difference-in-differences as the primary campaign evaluation method, supplemented by event-study style weekly plots and at least one placebo or pre-trend diagnostic.
- **Rationale**: Difference-in-differences is interpretable, compatible with campaign timing, and explicitly tests whether post-period changes differ from pre-period patterns in comparable cohorts. Event-study and placebo checks prevent overclaiming when assumptions fail.
- **Alternatives considered**: Propensity-only adjustment, synthetic controls, or relying on latent shift magnitude as the main impact metric. Rejected because they are either less transparent for first-pass validation, too fragile for small campaign cohorts, or insufficiently causal on their own.

## Decision 5: Bound Causal Claims to Quasi-Causal Evidence
- **Decision**: Treat all campaign-impact outputs from this feature as quasi-causal evidence, not strict causal proof.
- **Rationale**: The available data contains campaign assignment, campaign timing, coupon activity, and promotional proxies, but it does not provide randomized treatment assignment or complete household-level exposure logs. That makes strong causal language unsafe even if diagnostics look favorable.
- **Alternatives considered**: Preserve the existing project language around proven marketing impact. Rejected because it would overstate what the dataset and identification strategy can support.

## Decision 6: Use a Small Set of Primary Outcomes for the First Validation Pass
- **Decision**: Use `total_spend`, `trip_count`, `category_diversity`, and `promo_share` as the primary campaign outcomes for the first validation pass. Treat quantity, average price per unit, concentration measures, and coupon usage as secondary outcomes and diagnostics.
- **Rationale**: A small set of primary outcomes keeps the first report interpretable and reduces the risk of drowning the analysis in loosely connected metrics. The chosen outcomes cover purchase intensity, shopping frequency, basket breadth, and promotional behavior.
- **Alternatives considered**: Treat every engineered outcome as primary. Rejected because it would make the first pass hard to interpret and easier to overfit to noisy campaign-specific movement.

## Decision 7: Measure Observable Outcomes Before Interpreting Latent Space
- **Decision**: Define a standard set of campaign outcomes from observed behavior before using latent factors for interpretation. Initial outcomes will include spend, quantity, trips, average price per unit, category diversity, concentration, coupon usage, and promoted-item share.
- **Rationale**: Campaign-effect validation should first establish whether there is any credible movement in observed behavior. Latent analysis is more defensible when it explains an already measured behavioral change instead of standing in for the causal claim.
- **Alternatives considered**: Treat Euclidean latent deviation as the primary campaign outcome. Rejected because the current project needs a more defensible baseline before model-driven summaries can be trusted.

## Decision 8: Treat `causal_data.csv` as a Promotional Proxy Source, Not Direct Exposure Truth
- **Decision**: Use `causal_data.csv` only as a store-week-product promotional proxy and explicitly avoid treating it as a direct household-level exposure record.
- **Rationale**: Local inspection shows that `causal_data.csv` is keyed by product, store, and week with display and mailer indicators. It can enrich promoted-item features, but it does not prove that a specific household saw or responded to a promotion.
- **Alternatives considered**: Promote `causal_data.csv` to a true exposure table. Rejected because that would overstate the granularity and reliability of the source.

## Decision 9: Validate Latent Semantics with Holdout Attributes and Stability Checks
- **Decision**: Validate Beta-VAE latent factors against holdout observable attributes using MIG, SAP, rank-based association measures, simple probe models, and cross-run stability checks.
- **Rationale**: A latent factor should only receive a business label if the relationship to observed behavior is repeatable across holdout data, model variants, and seeds. MIG and SAP alone are not enough; stability is part of the validation standard.
- **Alternatives considered**: Use only MIG/SAP or rely on narrative naming from a single run. Rejected because that would repeat the current weakness of the repository: attractive factor labels without enough evidence that they are real.

## Decision 10: Compare Baseline VAE and Beta-VAE Under the Same Validation Protocol
- **Decision**: Evaluate baseline VAE and Beta-VAE with the same analytic dataset, the same observable attributes, and the same holdout rules.
- **Rationale**: The question is not whether Beta-VAE is interesting in isolation, but whether it improves semantic stability without unacceptable reconstruction loss or interpretability drift.
- **Alternatives considered**: Validate only Beta-VAE. Rejected because there would be no credible reference point for whether the added disentanglement objective actually helps.

## Decision 11: Treat Negative Findings as First-Class Outputs
- **Decision**: The report and workflow will explicitly support labels such as `unsupported`, `weak`, and `insufficient_data` for campaigns, and `unstable` or `unvalidated` for latent factor mappings.
- **Rationale**: The project is currently exposed to overstated claims. The validation workflow must be capable of disproving those claims, not just polishing them.
- **Alternatives considered**: Restrict outputs to positive summaries or omit failed analyses from the final report. Rejected because that would undermine the scientific value of the feature and recreate the same credibility problem.

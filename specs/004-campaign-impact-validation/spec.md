# Feature Specification: Validate Campaign Impact Claims and Latent Factor Semantics

**Feature Branch**: `004-campaign-impact-validation`  
**Created**: 2026-04-09  
**Status**: Draft  
**Input**: User description: "Validate campaign impact claims and latent factor semantics on the Dunnhumby Complete Journey dataset."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Campaign Validation Dataset (Priority: P1)

As a data scientist, I want a unified analysis dataset that links campaign assignments, campaign timing, purchase behavior, coupon activity, and household context so that I can evaluate campaign effects on a consistent basis.

**Why this priority**: Without a campaign-linked dataset, neither campaign impact analysis nor latent-factor validation can be performed credibly.

**Independent Test**: Can be fully tested by generating an analysis-ready dataset for selected campaigns and verifying that each household-campaign record contains valid pre-period, in-period, and post-period behavioral summaries.

**Acceptance Scenarios**:

1. **Given** the available campaign, transaction, coupon, and demographic source tables, **When** the validation dataset is generated, **Then** each selected campaign participant is linked to campaign timing and observable behavioral summaries for defined pre-period, in-period, and post-period windows.
2. **Given** households not assigned to a selected campaign, **When** the validation dataset is generated, **Then** the dataset includes untreated comparison records that can be aligned to the same campaign calendar.

---

### User Story 2 - Evaluate Campaign Effect Credibility (Priority: P1)

As a reviewer of the project, I want the workflow to estimate campaign effects with explicit assumption checks so that I can judge whether campaign impact claims are supported, weak, or unsupported.

**Why this priority**: The current project makes campaign-oriented claims, but those claims are not credible until they are tested against an explicit quasi-causal baseline.

**Independent Test**: Can be fully tested by running the workflow on one or more campaigns and confirming that the output includes an effect estimate, supporting diagnostics, and a clear evidence classification.

**Acceptance Scenarios**:

1. **Given** a selected campaign with treated and untreated households, **When** the campaign validation workflow is run, **Then** it produces effect estimates for defined behavioral outcomes and labels the evidence strength for that campaign.
2. **Given** a campaign where pre-period diagnostics fail or placebo checks contradict the claimed effect, **When** the workflow completes, **Then** the result explicitly marks the campaign claim as weak or unsupported instead of overstating confidence.

---

### User Story 3 - Validate Latent Factor Semantics (Priority: P2)

As a data scientist, I want to compare latent dimensions against observable household behavior attributes so that I can determine whether factor names are stable, evidence-based interpretations rather than narrative labels.

**Why this priority**: Interpretable latent factors are a major value claim of the Beta-VAE approach, but they are secondary to first establishing a credible campaign analysis baseline.

**Independent Test**: Can be fully tested by computing validation attributes on holdout data, comparing latent factors against those attributes, and reporting whether factor mappings are stable or unstable.

**Acceptance Scenarios**:

1. **Given** holdout household behavior data and model latent representations, **When** latent-factor validation is run, **Then** the workflow reports the strength and stability of the relationship between latent dimensions and observable attributes.
2. **Given** a factor mapping that changes meaning across runs, holdout splits, or model variants, **When** the workflow summarizes results, **Then** that mapping is rejected as unstable rather than presented as a validated business factor.

---

### User Story 4 - Produce an Evidence-Based Research Report (Priority: P3)

As a project maintainer, I want a reproducible report that summarizes both campaign-effect credibility and latent-factor validation so that the repository narrative is based on measured evidence rather than assumptions.

**Why this priority**: The project needs a defensible output artifact that can be used in interviews, reviews, and future iteration planning.

**Independent Test**: Can be fully tested by generating a report for selected campaigns and verifying that it documents methodology, results, evidence strength, and negative findings in a reproducible format.

**Acceptance Scenarios**:

1. **Given** completed campaign and latent-factor validation runs, **When** the report is generated, **Then** it summarizes key findings, diagnostics, and limits for each selected campaign.
2. **Given** mixed or negative results, **When** the report is generated, **Then** it documents unsupported claims and unstable factor mappings with the same visibility as positive findings.

### Edge Cases

- What happens when a selected campaign has too few treated households, too few untreated comparisons, or too little observed purchase activity to support a credible estimate?
- How does the workflow handle households assigned to overlapping campaigns or campaigns whose measurement windows collide?
- What happens when treated and untreated cohorts cannot be balanced well enough to support a credible comparison?
- How does the workflow behave when coupon issuance exists but coupon redemption is sparse or absent?
- What happens when latent factors show inconsistent mappings across seeds, time splits, or campaigns?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create a campaign-linked analysis dataset from available campaign, transaction, product, coupon, coupon redemption, demographic, and promotional exposure sources.
- **FR-002**: System MUST represent campaign timing explicitly and assign pre-period, in-period, and post-period observation windows to each household-campaign record.
- **FR-003**: System MUST include untreated comparison records aligned to the same campaign calendar for each selected campaign analysis.
- **FR-004**: System MUST compute observable behavioral outcomes for each observation window, including purchase intensity, category mix, price or value orientation, promotional activity, and concentration or diversity measures.
- **FR-005**: System MUST run at least one quasi-causal campaign evaluation method that compares treated and untreated households for selected campaigns.
- **FR-006**: System MUST perform assumption checks for campaign evaluation, including at least one diagnostic that can invalidate or weaken a campaign-effect claim.
- **FR-007**: System MUST classify campaign evidence strength using explicit categories such as supported, weak, or unsupported.
- **FR-008**: System MUST derive observable validation attributes for latent-factor analysis from household purchase behavior.
- **FR-009**: System MUST evaluate latent-factor quality on holdout data using disentanglement-oriented metrics and association-based validation against observable attributes.
- **FR-010**: System MUST compare at least two model variants on reconstruction fidelity, factor quality, and semantic stability.
- **FR-011**: System MUST allow the workflow to conclude that a claimed campaign effect is unsupported when diagnostics do not justify confidence.
- **FR-012**: System MUST allow the workflow to reject latent-factor names when mappings are unstable or not repeatable.
- **FR-013**: System MUST generate a reproducible research report for selected campaigns that documents methodology, findings, diagnostics, limitations, and negative results.

### Key Entities *(include if feature involves data)*

- **Campaign Analysis Record**: A household-campaign observation that combines campaign identity, campaign timing, treatment status, and behavioral summaries across analysis windows.
- **Comparison Cohort**: A set of untreated household records aligned to the same campaign calendar and used to evaluate campaign-effect credibility.
- **Behavioral Outcome**: A measurable summary of household purchase behavior, such as spending level, shopping intensity, category diversity, promotional participation, or concentration.
- **Validation Attribute**: An observable household behavior characteristic used to test whether a latent dimension has a stable semantic interpretation.
- **Evidence Classification**: A result label that states whether a campaign effect claim is supported, weak, or unsupported after diagnostics.
- **Factor Mapping Assessment**: A result describing whether a latent dimension has a repeatable relationship to an observable behavioral attribute.
- **Research Report**: A reproducible summary artifact that records campaign findings, latent-factor validation results, diagnostics, and limitations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Analysts can generate an analysis-ready dataset for at least 3 selected campaigns, with treated and untreated records and complete observation windows, without manual table joins.
- **SC-002**: For every selected campaign included in the report, the workflow produces both an effect estimate and at least one assumption-check result before assigning an evidence classification.
- **SC-003**: At least 90% of campaign findings in the final report include an explicit evidence label of supported, weak, unsupported, or insufficient data.
- **SC-004**: Latent-factor validation results are reported for at least 80% of evaluated latent dimensions or the workflow explicitly states why those dimensions could not be assessed.
- **SC-005**: The final report documents at least one positive finding and one negative, weak, or unsupported finding, unless all evaluated results fall into a single category and that absence is explicitly stated.
- **SC-006**: A reviewer can inspect the final report and determine, without reading source code, which project claims remain defensible and which claims should be withdrawn or softened.

## Assumptions

- The available campaign assignment and timing tables are sufficiently reliable to define campaign-specific treatment windows.
- Untreated comparison households can be constructed from households not assigned to a given campaign, even if the resulting analysis remains quasi-causal rather than fully causal.
- Observable purchase behavior provides enough signal to define validation attributes for latent-factor analysis, even if some candidate factor names must ultimately be rejected.
- The workflow is expected to support strong negative findings; disproving a claim is considered a valid and valuable outcome.

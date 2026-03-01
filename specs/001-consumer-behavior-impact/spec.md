# Feature Specification: Impact of External Stimuli on Consumer Purchase Flows

> **Novelty:** this project applies a Variational Autoencoder to retail transaction data in order to quantify deviations from a learned baseline due to marketing campaigns. While VAEs are commonly used for anomaly detection and representation learning, their use in measuring marketing‑driven behaviour shifts is rare, making this work a valuable portfolio piece.


**Feature Branch**: `001-consumer-behavior-impact`  
**Created**: 2026-02-25  
**Status**: Draft  

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish Baseline Behavior (Priority: P1)

As a data analyst, I want the system to establish a "normal" purchase profile for each household in the absence of external influence so that I have a foundation to measure deviations against.

**Why this priority**: Without a baseline, any measurement of deviation or impact is impossible. This is the core foundation of the system.

**Independent Test**: Can be fully tested by providing historical purchase data without marketing triggers and verifying that the system generates a stable, consistent purchase profile for the households.

**Acceptance Scenarios**:

1. **Given** a dataset of household purchases with no recorded external triggers, **When** the baseline model is run, **Then** it produces a defined purchase profile with expected volume and category preferences.

---

### User Story 2 - Measure Impact of External Stimuli (Priority: P1)

As a data analyst, I want the system to detect moments of external intervention and measure the degree of deviation from the established baseline so that I can quantify the impact of marketing activities.

**Why this priority**: Quantifying the deviation is the primary goal of the feature.

**Independent Test**: Can be tested by introducing a dataset with known marketing interventions and verifying the system accurately flags the intervention and calculates the magnitude of deviation.

**Acceptance Scenarios**:

1. **Given** an established baseline and a new dataset containing an external trigger, **When** the impact measurement is run, **Then** the system quantifies the deviation from the baseline.

---

### User Story 3 - Describe Nature of Change (Priority: P2)

As a marketer, I want to understand the character of the shift (e.g., trading up to premium versions vs. stockpiling routine items) rather than just volume changes, so that I can understand how consumer preferences are evolving.

**Why this priority**: Adds qualitative depth to the quantitative impact measurement, which is critical for actionable marketing insights.

**Independent Test**: Can be tested by providing data where a household buys higher-priced items than usual after a trigger, verifying the system flags this as "trading up".

**Acceptance Scenarios**:

1. **Given** a measured deviation involving higher-priced items in the same category, **When** the nature of change is analyzed, **Then** the system categorizes the shift as "trading up".
2. **Given** a measured deviation involving unusually large quantities of regular items, **When** the nature of change is analyzed, **Then** the system categorizes the shift as "stockpiling".

---

### User Story 4 - Analyze Persistence of Change (Priority: P2)

As a marketer, I want to determine how quickly a consumer’s behavior reverts to their original baseline once the external stimulus is removed, so that I can evaluate the long-term effectiveness of the stimulus.

**Why this priority**: Essential for understanding if marketing efforts create lasting habits or temporary spikes.

**Independent Test**: Can be tested using longitudinal data following an intervention to verify the system calculates the time taken to return to baseline metrics.

**Acceptance Scenarios**:

1. **Given** a quantified deviation and subsequent historical data, **When** the persistence analysis is run, **Then** the system outputs the duration until behavior matched the baseline again.

---

### User Story 5 - Aggregate Reporting & Segmentation (Priority: P3)

As a data analyst, I want to group consumers into segments based on similarity in their reaction patterns to identical triggers and rank product categories based on their sensitivity, so that I can generate aggregate insights for marketing campaigns.

**Why this priority**: Required for SC-001, SC-002, and SC-004. Groups individual behavioral shifts into strategic insights.

**Independent Test**: Provide multiple shift records and verify accurate cluster assignment and ranked sensitive categories.

**Acceptance Scenarios**:

1. **Given** a set of behavioral shifts from multiple households, **When** the segmentation is run, **Then** the system assigns each household to a reaction pattern cluster.
2. **Given** a set of behavioral shifts, **When** the category ranking is run, **Then** the system outputs the top sensitive product categories.

---

### Edge Cases

- What happens when a household has insufficient historical data to establish a reliable baseline?
- How does the system handle overlapping external stimuli (e.g., two marketing campaigns running simultaneously)?
- What happens if a consumer's baseline naturally drifts over a long period (seasonality or life changes) independent of targeted marketing?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST process historical household purchase data and output a quantified baseline purchase profile.
- **FR-002**: System MUST process data regarding external marketing activities and map them temporally to purchase data.
- **FR-003**: System MUST calculate the quantitative deviation in purchase behavior during and after an external stimulus compared to the baseline.
- **FR-004**: System MUST categorize the qualitative nature of behavioral shifts into defined categories (e.g., stockpiling, trading up, brand switching).
- **FR-005**: System MUST calculate the time duration for a household's purchase behavior to return to the established baseline after a stimulus ends.
- **FR-006**: System MUST group consumers into segments based on similarity in their reaction patterns to identical triggers.
- **FR-007**: System MUST identify and rank product categories based on their sensitivity (degree of deviation) to external shifts.
- **FR-008**: System MUST account for natural seasonality in the baseline calculation by using temporal context features (like month/week-of-year) provided as auxiliary inputs to the model.

### Key Entities

- **Household**: Represents the purchasing unit, characterized by a baseline purchase profile and a history of transactions.
- **External Stimulus**: Represents a marketing activity or event, characterized by its type, start time, and end time.
- **Purchase Profile**: A quantified representation of a household's expected buying behavior (volume, categories, price tiers).
- **Behavioral Shift**: The measured deviation from the baseline, containing both quantitative magnitude and qualitative character (e.g., stockpiling).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system can successfully group at least 80% of exposed consumers into distinct reaction pattern segments.
- **SC-002**: The system accurately identifies the top 5 most sensitive product categories for any given external shift type.
- **SC-003**: The system establishes a baseline profile for a household using historical data in under 5 seconds per household.
- **SC-004**: Analysts can generate a complete impact report (baseline, deviation, nature of change, persistence) for a marketing campaign within 10 minutes of providing the data.

## Assumptions

- **Data Availability**: It is assumed that clean, timestamped historical purchase data and marketing activity logs are available.
- **Granularity**: It is assumed that purchase data includes item-level details (price, category, quantity) necessary to detect "trading up" or "stockpiling".
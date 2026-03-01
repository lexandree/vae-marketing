---
description: "Task list for feature implementation"
---

# Tasks: Impact of External Stimuli on Consumer Purchase Flows

**Input**: Design documents from `/specs/001-consumer-behavior-impact/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included as explicitly requested implicitly by the testing framework requirement in `plan.md` and standard development practices.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure (`src/models/`, `src/services/`, `src/data/`, `src/utils/`, `tests/integration/`, `tests/unit/`)
- [x] T002 Initialize Python environment with PyTorch, Pandas, Polars, Scikit-learn, Plotly, Seaborn
- [x] T003 [P] Configure pytest, ruff, and formatting tools
- [x] T004 [P] Configure linting via ruff to automatically check and enforce type hints and Google-style docstrings

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T005 Create foundational dataset loading utilities in `src/data/dataset.py`
- [x] T006 [P] Setup base logging and metrics utilities in `src/utils/metrics.py`
- [x] T007 [P] Implement global random seed setup for Torch and NumPy reproducibility in `src/utils/seed.py`

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Establish Baseline Behavior (Priority: P1) 🎯 MVP

**Goal**: Establish a "normal" purchase profile for each household in the absence of external influence using a VAE.

**Independent Test**: Provide historical purchase data without triggers and verify the system generates a stable, consistent purchase profile.

### Tests for User Story 1

- [x] T008 [P] [US1] Create unit tests for VAE model architecture in `tests/unit/test_vae.py`
- [x] T009 [P] [US1] Create unit tests for baseline training and profiling in `tests/unit/test_baseline.py`

### Implementation for User Story 1

- [x] T010 [P] [US1] Define Household and Transaction data schemas/models enforcing memory-efficient Pandas/Polars dtypes in `src/data/dataset.py`
- [x] T011 [US1] Implement VAE model architecture (`build_vae_model`) including wiring of temporal auxiliary inputs in `src/models/vae.py` (depends on T010)
- [x] T012 [US1] Implement `train_baseline_vae` with reparameterization trick, KL + Recon loss, and temporal features passing in `src/services/baseline.py`. Include validation to handle/reject households with insufficient historical data.
- [x] T013 [US1] Implement `get_household_profile` to extract latent representation in `src/services/baseline.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently.

---

## Phase 4: User Story 2 - Measure Impact of External Stimuli (Priority: P1)

**Goal**: Detect moments of external intervention and measure the degree of deviation from the established baseline.

**Independent Test**: Introduce dataset with known marketing interventions and verify accurate calculation of magnitude of deviation.

### Tests for User Story 2

- [x] T014 [P] [US2] Create unit tests for deviation calculation in `tests/unit/test_impact_analysis.py`

### Implementation for User Story 2

- [x] T015 [P] [US2] Update data schemas for External Stimulus in `src/data/dataset.py`, enforcing memory-efficient Pandas/Polars dtypes.
- [x] T016 [US2] Implement `calculate_deviation` (distance in latent space) in `src/services/impact_analysis.py`. Include logic to handle overlapping external stimuli.
- [x] T017 [US2] Define Quantitative Behavioral Shift output structure in `src/data/dataset.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently.

---

## Phase 5: User Story 3 - Describe Nature of Change (Priority: P2)

**Goal**: Understand the character of the shift (e.g., trading up vs. stockpiling) rather than just volume changes.

**Independent Test**: Provide data with higher-priced items after trigger, verify it flags as "trading up".

### Tests for User Story 3

- [x] T018 [P] [US3] Add unit tests for shift categorization rules in `tests/unit/test_impact_analysis.py`

### Implementation for User Story 3

- [x] T019 [US3] Implement `categorize_shift` applying heuristic rules for stock piling/trading up in `src/services/impact_analysis.py`
- [x] T020 [US3] Update Behavioral Shift with qualitative enum in `src/data/dataset.py`

**Checkpoint**: Qualitative shift analysis is functional.

---

## Phase 6: User Story 4 - Analyze Persistence of Change (Priority: P2)

**Goal**: Determine how quickly a consumer’s behavior reverts to their original baseline once the external stimulus is removed.

**Independent Test**: Use longitudinal data following an intervention to verify calculation of time taken to return to baseline metrics.

### Tests for User Story 4

- [x] T021 [P] [US4] Add unit tests for persistence sliding window analysis in `tests/unit/test_impact_analysis.py`

### Implementation for User Story 4

- [x] T022 [US4] Implement `analyze_persistence` using a sliding window approach in `src/services/impact_analysis.py`
- [x] T023 [US4] Update Behavioral Shift output with persistence duration in `src/data/dataset.py`

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 7: Aggregate Reporting & Segmentation (Priority: P3)

**Goal**: Group consumers into segments by reaction patterns and rank product categories based on sensitivity to generate an aggregate impact report.

**Independent Test**: Provide multiple shift records and verify accurate cluster assignment and ranked sensitive categories.

### Tests for Reporting

- [x] T024 [P] [US5] Add unit tests for segmentation and ranking logic in `tests/unit/test_reporting.py`

### Implementation for Reporting

- [x] T025 [US5] Implement `segment_consumers` using clustering algorithms in `src/services/reporting.py`
- [x] T026 [US5] Implement `rank_sensitive_categories` in `src/services/reporting.py`
- [x] T027 [US5] Implement function to generate complete aggregate impact report combining baseline, deviation, nature, and persistence

**Checkpoint**: System can generate final 10-minute aggregate report with clustering and ranking.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T028 [P] Create full pipeline integration test in `tests/integration/test_pipeline.py`
- [x] T029 Refactor and optimize to ensure < 5s per household baseline establishment
- [x] T030 [P] Finalize documentation and validate scripts in `quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - Phase 3 (US1) unblocks Phase 4 (US2) since impact needs a baseline.
  - US3 and US4 depend on US1 and US2 being conceptually complete, but can be implemented modularly.
  - Phase 7 (Reporting) depends on output structures from US2, US3, and US4.
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### Parallel Opportunities

- Tests within each user story can be written in parallel with model schema definitions.
- E.g., for US1: T008, T009, and T010 can be executed in parallel.
- E.g., for US2: T014 and T015 can be executed in parallel.

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together
Task: "T008 [P] [US1] Create unit tests for VAE model architecture in tests/unit/test_vae.py"
Task: "T009 [P] [US1] Create unit tests for baseline training and profiling in tests/unit/test_baseline.py"

# Data schemas can be done concurrently
Task: "T010 [P] [US1] Define Household and Transaction data schemas/models enforcing memory-efficient Pandas/Polars dtypes in src/data/dataset.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 & 2)

1. Complete Phase 1 & 2: Setup & Foundational
2. Complete Phase 3: User Story 1 (Baseline)
3. Complete Phase 4: User Story 2 (Impact Measurement)
4. **STOP and VALIDATE**: Verify we can train a baseline and measure deviation on sample data.

### Incremental Delivery

1. Baseline creation (US1) provides the core VAE.
2. Impact measurement (US2) provides quantitative results.
3. Shift categorization (US3) enriches the results qualitatively.
4. Persistence analysis (US4) adds a temporal dimension to the results.
5. Aggregate reporting (US5) produces final grouped insights.

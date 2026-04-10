# Tasks: Campaign Impact and Latent Validation

**Input**: Design documents from `/specs/004-campaign-impact-validation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include targeted unit and integration tests because this feature introduces new dataset joins, quasi-causal diagnostics, and evidence classification logic that must remain reproducible.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g. `US1`, `US2`, `US3`)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the repository for campaign validation and latent validation work.

- [X] T001 Update project packaging and test import path setup in `/home/admin2/vae_marketing/pyproject.toml` and `/home/admin2/vae_marketing/tests/conftest.py`
- [X] T002 [P] Add validation workflow documentation stubs and report output paths in `/home/admin2/vae_marketing/README.md` and `/home/admin2/vae_marketing/docs/`
- [X] T003 [P] Create module stubs for planned validation components in `/home/admin2/vae_marketing/src/data/campaign_dataset.py`, `/home/admin2/vae_marketing/src/data/campaign_windows.py`, `/home/admin2/vae_marketing/src/data/validation_attributes.py`, `/home/admin2/vae_marketing/src/services/campaign_validation.py`, `/home/admin2/vae_marketing/src/services/latent_validation.py`, `/home/admin2/vae_marketing/src/services/validation_reporting.py`, `/home/admin2/vae_marketing/src/utils/matching.py`, and `/home/admin2/vae_marketing/src/utils/statistics.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared campaign-validation infrastructure required by all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Extend shared dataset schemas and source loading helpers in `/home/admin2/vae_marketing/src/data/dataset.py` and `/home/admin2/vae_marketing/src/data/schema.py`
- [X] T005 [P] Implement campaign calendar and window construction helpers in `/home/admin2/vae_marketing/src/data/campaign_windows.py`
- [X] T006 [P] Implement matching and balance utility functions in `/home/admin2/vae_marketing/src/utils/matching.py`
- [X] T007 [P] Implement statistical diagnostics and evidence-label helpers in `/home/admin2/vae_marketing/src/utils/statistics.py`
- [X] T008 Add reusable latent extraction and holdout alignment helpers in `/home/admin2/vae_marketing/src/services/latent_validation.py` and `/home/admin2/vae_marketing/src/models/factory.py`
- [X] T009 Extend CLI argument parsing and command dispatch scaffolding for validation workflows in `/home/admin2/vae_marketing/main.py`

**Checkpoint**: Foundation ready. User story implementation can now proceed.

---

## Phase 3: User Story 1 - Build Campaign Validation Dataset (Priority: P1) 🎯 MVP

**Goal**: Produce an analysis-ready household-campaign dataset with treated and untreated records, aligned windows, and observable outcomes.

**Independent Test**: Generate `campaign_analysis.parquet`, `comparison_pool.parquet`, `validation_attributes.parquet`, and `dataset_summary.json` for campaigns `18`, `13`, and `8`, then verify that each included household-campaign record has valid pre-period, in-period, and post-period summaries plus documented exclusions.

### Tests for User Story 1

- [X] T010 [P] [US1] Add unit tests for campaign window assignment and eligibility rules in `/home/admin2/vae_marketing/tests/unit/test_campaign_dataset.py`
- [X] T011 [P] [US1] Add integration coverage for generation of `campaign_analysis.parquet`, `comparison_pool.parquet`, `validation_attributes.parquet`, and `dataset_summary.json` in `/home/admin2/vae_marketing/tests/integration/test_campaign_validation_flow.py`

### Implementation for User Story 1

- [X] T012 [P] [US1] Implement campaign source joins and household-campaign record assembly in `/home/admin2/vae_marketing/src/data/campaign_dataset.py`
- [X] T013 [P] [US1] Implement observable behavioral outcome and validation attribute aggregation in `/home/admin2/vae_marketing/src/data/validation_attributes.py`
- [X] T014 [US1] Implement untreated comparison pool construction and exclusion tracking in `/home/admin2/vae_marketing/src/data/campaign_dataset.py`
- [X] T015 [US1] Add the `build-validation-data` CLI command and explicit writing of `campaign_analysis.parquet`, `comparison_pool.parquet`, `validation_attributes.parquet`, and `dataset_summary.json` in `/home/admin2/vae_marketing/main.py`
- [X] T016 [US1] Document validation dataset generation and expected artifacts in `/home/admin2/vae_marketing/specs/004-campaign-impact-validation/quickstart.md` and `/home/admin2/vae_marketing/README.md`

**Checkpoint**: User Story 1 should generate a reproducible campaign-linked dataset without manual joins.

---

## Phase 4: User Story 2 - Evaluate Campaign Effect Credibility (Priority: P1)

**Goal**: Estimate campaign effects with explicit diagnostics and evidence classifications.

**Independent Test**: Run campaign validation on selected campaigns and confirm that each result includes effect estimates, diagnostic outputs, and a `supported`, `weak`, `unsupported`, or `insufficient_data` label.

### Tests for User Story 2

- [X] T017 [P] [US2] Add unit tests for cohort matching, balance checks, placebo checks, and evidence classification in `/home/admin2/vae_marketing/tests/unit/test_campaign_validation.py`
- [X] T018 [P] [US2] Extend integration coverage for `validate-campaigns` outputs and diagnostics in `/home/admin2/vae_marketing/tests/integration/test_campaign_validation_flow.py`
- [X] T019 [P] [US2] Add unit tests for primary outcome selection and secondary diagnostic handling in `/home/admin2/vae_marketing/tests/unit/test_campaign_validation.py`
- [X] T020 [P] [US2] Add tests that enforce explicit evidence labels on campaign findings and fail when label coverage drops below the required threshold in `/home/admin2/vae_marketing/tests/unit/test_campaign_validation.py` and `/home/admin2/vae_marketing/tests/integration/test_campaign_validation_flow.py`

### Implementation for User Story 2

- [X] T021 [P] [US2] Implement matched treated-vs-untreated cohort selection and balance reporting in `/home/admin2/vae_marketing/src/services/campaign_validation.py`
- [X] T022 [P] [US2] Implement difference-in-differences estimation, placebo checks, and event-study summaries in `/home/admin2/vae_marketing/src/services/campaign_validation.py`
- [X] T023 [US2] Encode `total_spend`, `trip_count`, `category_diversity`, and `promo_share` as primary outcomes with all other metrics treated as secondary diagnostics in `/home/admin2/vae_marketing/src/services/campaign_validation.py`
- [X] T024 [US2] Implement campaign evidence classification, label-completeness enforcement, and JSON artifact generation in `/home/admin2/vae_marketing/src/services/campaign_validation.py`
- [X] T025 [US2] Add the `validate-campaigns` CLI command and output handling in `/home/admin2/vae_marketing/main.py`
- [X] T026 [US2] Add report-ready campaign diagnostics serialization in `/home/admin2/vae_marketing/src/services/validation_reporting.py`

**Checkpoint**: User Story 2 should produce defensible quasi-causal campaign results independent of latent analysis.

---

## Phase 5: User Story 3 - Validate Latent Factor Semantics (Priority: P2)

**Goal**: Test whether latent dimensions have stable, observable semantic mappings on holdout data.

**Independent Test**: Run latent validation for baseline VAE and Beta-VAE on holdout data and verify that the workflow reports reconstruction, disentanglement, association strength, and mapping stability for evaluated dimensions.

### Tests for User Story 3

- [X] T027 [P] [US3] Add unit tests for latent attribute alignment, MIG/SAP evaluation, and mapping decisions in `/home/admin2/vae_marketing/tests/unit/test_latent_validation.py`
- [X] T028 [P] [US3] Add integration coverage for `validate-latents` outputs across multiple run IDs in `/home/admin2/vae_marketing/tests/integration/test_latent_validation_flow.py`

### Implementation for User Story 3

- [X] T029 [P] [US3] Implement holdout attribute extraction and model-aligned latent snapshot loading in `/home/admin2/vae_marketing/src/services/latent_validation.py`
- [X] T030 [P] [US3] Implement MIG, SAP, association, and stability evaluation pipeline in `/home/admin2/vae_marketing/src/services/latent_validation.py` and `/home/admin2/vae_marketing/src/utils/metrics.py`
- [X] T031 [US3] Implement final factor mapping decisions and unstable-mapping rejection logic in `/home/admin2/vae_marketing/src/services/latent_validation.py`
- [X] T032 [US3] Add the `validate-latents` CLI command and artifact writing flow in `/home/admin2/vae_marketing/main.py`
- [X] T033 [US3] Update model comparison outputs to include latent validation context in `/home/admin2/vae_marketing/main.py` and `/home/admin2/vae_marketing/src/services/reporting.py`

**Checkpoint**: User Story 3 should validate or reject latent-factor semantics without depending on the final report generator.

---

## Phase 6: User Story 4 - Produce an Evidence-Based Research Report (Priority: P3)

**Goal**: Generate a reproducible report that summarizes campaign evidence and latent-factor validation findings, including negative results.

**Independent Test**: Generate the final report from campaign and latent artifacts and verify that it records methodology, evidence classifications, unstable mappings, and claim recommendations.

### Tests for User Story 4

- [X] T034 [P] [US4] Add unit tests for report aggregation, mixed-findings preservation, and claim recommendation rules in `/home/admin2/vae_marketing/tests/unit/test_validation_reporting.py`
- [X] T035 [P] [US4] Add integration coverage for end-to-end report generation in `/home/admin2/vae_marketing/tests/integration/test_validation_reporting_flow.py`

### Implementation for User Story 4

- [X] T036 [P] [US4] Implement report assembly, mixed-findings preservation, and claim recommendation logic in `/home/admin2/vae_marketing/src/services/validation_reporting.py`
- [X] T037 [US4] Add the `generate-validation-report` CLI command and markdown/JSON output flow in `/home/admin2/vae_marketing/main.py`
- [X] T038 [US4] Create the validation report template and artifact conventions in `/home/admin2/vae_marketing/reports/` and `/home/admin2/vae_marketing/src/services/validation_reporting.py`
- [X] T039 [US4] Update public project narrative to reflect evidence-based outcomes in `/home/admin2/vae_marketing/README.md`, `/home/admin2/vae_marketing/docs/MARKETING_GUIDE.md`, and `/home/admin2/vae_marketing/portfolio-export.md`

**Checkpoint**: User Story 4 should produce a reviewer-ready report that can support, weaken, or reject current project claims.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improve reliability, documentation, and end-to-end consistency across all stories.

- [X] T040 [P] Run and fix the full validation test suite in `/home/admin2/vae_marketing/tests/`
- [X] T041 [P] Validate the quickstart workflow against the implemented CLI contract in `/home/admin2/vae_marketing/specs/004-campaign-impact-validation/quickstart.md`
- [X] T042 Calibrate performance and sample-size thresholds for selected campaigns in `/home/admin2/vae_marketing/src/services/campaign_validation.py` and `/home/admin2/vae_marketing/src/utils/statistics.py`
- [X] T043 Perform documentation cleanup for validation artifacts and reproducibility notes in `/home/admin2/vae_marketing/README.md` and `/home/admin2/vae_marketing/docs/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on User Story 1 because campaign validation requires the generated analysis dataset
- **User Story 3 (Phase 5)**: Depends on User Story 1 because latent validation requires aligned validation attributes and household windows
- **User Story 4 (Phase 6)**: Depends on User Story 2 and User Story 3
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: First MVP slice and prerequisite for the rest of the workflow
- **User Story 2 (P1)**: Depends on User Story 1 outputs, but not on User Story 3
- **User Story 3 (P2)**: Depends on User Story 1 outputs, but not on User Story 2
- **User Story 4 (P3)**: Depends on both campaign validation and latent validation artifacts

### Within Each User Story

- Tests should be added before or alongside implementation for the relevant workflow
- Data construction before CLI wiring for each story
- Core service logic before report-facing serialization
- Story checkpoint should pass before starting dependent stories

## Parallel Opportunities

- `T002` and `T003` can run in parallel after setup starts
- `T005`, `T006`, and `T007` can run in parallel in Foundational
- In User Story 1, `T012` and `T013` can run in parallel before `T014`
- In User Story 2, `T021` and `T022` can run in parallel before `T023`
- In User Story 3, `T029` and `T030` can run in parallel before `T031`
- In User Story 4, `T034` and `T036` can run in parallel before final CLI wiring

## Parallel Example: User Story 1

```bash
# Parallel implementation for the campaign-linked dataset
Task: "Implement campaign source joins and household-campaign record assembly in /home/admin2/vae_marketing/src/data/campaign_dataset.py"
Task: "Implement observable behavioral outcome and validation attribute aggregation in /home/admin2/vae_marketing/src/data/validation_attributes.py"
```

## Implementation Strategy

### MVP First

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate dataset outputs for campaigns `18`, `13`, and `8`
5. Proceed to campaign and latent validation only after the dataset is stable

### Incremental Delivery

1. Deliver campaign-linked dataset generation
2. Add quasi-causal campaign validation
3. Add latent-factor validation
4. Add final report synthesis
5. Finish with documentation and threshold calibration

### Suggested MVP Scope

User Story 1 only. It unlocks both campaign-effect estimation and latent-factor validation while producing a standalone, demonstrable artifact.

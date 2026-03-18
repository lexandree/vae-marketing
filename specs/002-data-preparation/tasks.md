# Tasks: Data Preparation for VAE Marketing Analysis

**Input**: Design documents from `/specs/002-data-preparation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/pipeline_api.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create missing directories and basic files in `src/data/` and `tests/` based on plan structure
- [x] T002 Add required libraries (polars, pyarrow, scikit-learn) to project dependencies
- [x] T003 [P] Setup logging and error handling boilerplate in `src/data/prepare.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T000 [P] Initialize deterministic global random seeds (numpy, torch, python) in `src/utils/seed.py`
- [x] T004 Define schemas and data structures for raw transactions and product hierarchy
- [x] T005 Implement CLI argument parsing structure in `src/data/prepare.py` based on `pipeline_api.md` contract
- [x] T005.1 Implement strict Polars input schema validation (check dtypes for sales and IDs) before processing in `src/data/prepare.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Prepare Raw Transaction Data with Semantic & Temporal Features (Priority: P1) 🎯 MVP

**Goal**: Prepare the raw Dunnhumby transaction data by applying hierarchical aggregations, temporal feature extraction, and rolling windows so that the data correctly models seasonal trends and product relationships without sparse one-hot explosions.

**Independent Test**: Execute the pipeline on raw Dunnhumby data and verify that the output contains hierarchical product groupings, correct week/day seasonality flags, and dense aggregated vectors instead of raw one-hot IDs.

### Tests for User Story 1

- [x] T006 [P] [US1] Write unit tests for hierarchical mapping and temporal encoding logic in `tests/unit/test_data_prep.py`
- [x] T007 [P] [US1] Write unit tests for 7-day rolling window aggregation in `tests/unit/test_data_prep.py`

### Implementation for User Story 1

- [x] T008 [P] [US1] Implement hierarchical product features mapping (COMMODITY -> SUB_COMMODITY) in `src/data/extractors.py`
- [x] T009 [P] [US1] Implement cyclical continuous encoding (sine/cosine) for temporal features (week/month/day) in `src/data/extractors.py`
- [x] T010 [US1] Implement 7-day rolling window aggregation (imputing temporary gaps as zero-vectors and filtering zero-history households) in `src/data/extractors.py`
- [x] T011 [US1] Implement filtering of non-purchase transactions (returns/refunds/coupons) in `src/data/extractors.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Robust Normalization and Time-Series Splits (Priority: P2)

**Goal**: Properly scale heavy-tailed purchase data and split datasets using forward-chaining so that the VAE baseline evaluation is free of data leakage.

**Independent Test**: Verify that the generated train/validation splits respect temporal order (e.g., Train: weeks 0-20, Val: weeks 21-30) and that normalization parameters are fit purely on the training window.

### Tests for User Story 2

- [x] T012 [P] [US2] Write unit tests for log-scale/z-score normalization in `tests/unit/test_data_prep.py`
- [x] T013 [P] [US2] Write unit tests for forward-chaining time-series splits in `tests/unit/test_data_prep.py`

### Implementation for User Story 2

- [x] T014 [P] [US2] Implement log-scale/Box-Cox transformation and z-score scaling per cohort in `src/data/normalizers.py`
- [x] T015 [P] [US2] Implement forward-chaining time-series data split logic in `src/data/normalizers.py`
- [x] T016 [US2] Implement parameter saving/loading for normalizer metrics (`scaler_params.json`) in `src/data/normalizers.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T017 Integrate extractors and normalizers into the main CLI execution flow in `src/data/prepare.py`
- [x] T018 Implement output saving to Parquet format (`train.parquet`, `val.parquet`, `test.parquet`) in `src/data/prepare.py`
- [x] T019 Write end-to-end integration test in `tests/integration/test_pipeline_end_to_end.py`
- [x] T020 Run quickstart.md validation locally to verify the pipeline executes correctly

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write unit tests for hierarchical mapping and temporal encoding logic in tests/unit/test_data_prep.py"
Task: "Write unit tests for 7-day rolling window aggregation in tests/unit/test_data_prep.py"

# Implement extraction components together:
Task: "Implement hierarchical product features mapping in src/data/extractors.py"
Task: "Implement cyclical continuous encoding in src/data/extractors.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → MVP!
3. Add User Story 2 → Test independently
4. Complete Polish phase for integration.
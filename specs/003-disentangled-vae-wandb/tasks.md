# Tasks: Disentangled VAE (Stage 003)

**Input**: Design documents from `/specs/003-disentangled-vae-wandb/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are requested in the feature specification (Acceptance Scenarios and SC-002, SC-003).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and experiment structure

- [x] T001 Create `experiments/` base directory for Run-ID storage
- [x] T002 Initialize `wandb` in `src/utils/wandb_logger.py` with `.env` support for API key (FR-007)
- [x] T003 [P] Configure `.gitignore` to ignore large `weights.pt`, `experiments/`, and `.env` while allowing `config.json` templates

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure for polymorphic models and metrics

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Refactor `BaselineVAE` from `src/models/vae.py` into its own file `src/models/baseline_vae.py`
- [x] T004.1 Rename `src/services/reporting.py` to `src/services/reporting_baseline.py`
- [x] T005 [P] Implement `ModelFactory` in `src/models/factory.py` to load architectures from `config.json`
- [x] T006 [P] Implement `MIG` (Mutual Information Gap) metric in `src/utils/metrics.py`
- [x] T007 [P] Implement `SAP` (Separated Attribute Predictability) metric in `src/utils/metrics.py`
- [x] T008 [P] Update `src/data/dataset.py` to include dynamic attribute extraction for metrics calculation
- [x] T009 Create `src/models/__init__.py` to expose factory and model classes

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Model Training with WandB (Priority: P1) 🎯 MVP

**Goal**: Train a Beta-VAE and log its progress/artifacts to WandB with a unique Run-ID.

**Independent Test**: Run `python main.py train --arch beta_vae --run-id test-run --wandb` and verify metrics in WandB dashboard and local files in `experiments/test-run/`.

### Tests for User Story 1

- [x] T010 [P] [US1] Integration test for Beta-VAE training loop in `tests/integration/test_beta_trainer.py`
- [x] T011 [P] [US1] Unit test for Beta-VAE loss function (MSE + β*KL) in `tests/unit/test_beta_vae.py`

### Implementation for User Story 1

- [x] T012 [P] [US1] Implement `BetaVAE` architecture in `src/models/beta_vae.py`
- [x] T013 [US1] Implement β-annealing schedule logic in `src/models/beta_vae.py`
- [x] T014 [US1] Implement hybrid checkpointing (best + periodic) in `src/utils/wandb_logger.py`
- [x] T015 [US1] Update `main.py` `train` command to support `--arch beta_vae`, `--beta`, and `--anneal-end`
- [x] T016 [US1] Implement `config.json` and `metrics.json` saving logic in `src/models/factory.py`
- [x] T017 [US1] Add logging for training progress and metrics in `src/main.py`

**Checkpoint**: User Story 1 is functional: Beta-VAE models can be trained and tracked.

---

## Phase 4: User Story 2 - Disentangled Factor Analysis (Priority: P1)

**Goal**: Provide a breakdown of behavior changes into independent factors for marketing analysis.

**Independent Test**: Run `python main.py infer --run-id test-run --data data/processed/test.parquet` and verify factor-level breakdown in the generated report.

### Tests for User Story 2

- [x] T018 [P] [US2] Unit test for factor shift calculation in `tests/unit/test_impact_analysis.py`
- [x] T019 [P] [US2] Integration test for polymorphic reporting in `tests/integration/test_reporting_beta.py`

### Implementation for User Story 2

- [x] T020 [P] [US2] Create `src/services/reporting_beta.py` for factor-level impact breakdown
- [x] T021 [US2] Implement `src/services/reporting.py` as a dispatcher (facade) that delegates to `reporting_baseline` or `reporting_beta`
- [x] T022 [US2] Implement factor shift categorization logic (e.g., "Trading Up") based on latent dimension deltas in `src/services/impact_analysis.py`
- [x] T023 [US2] Update `main.py` `infer` command to load model via `--run-id` using the factory

**Checkpoint**: User Story 2 is functional: Insights are now decomposed into interpretable factors.

---

## Phase 5: User Story 3 - Model Architecture Comparison (Priority: P2)

**Goal**: Quantify the trade-off between reconstruction accuracy and disentanglement quality.

**Independent Test**: Run comparison utility on two Run-IDs and verify output table with MSE, KL, and MIG scores.

### Tests for User Story 3

- [x] T024 [P] [US3] Unit test for MIG calculation accuracy in `tests/unit/test_metrics.py`
- [x] T025 [P] [US3] Unit test for SAP calculation accuracy in `tests/unit/test_metrics.py`

### Implementation for User Story 3

- [x] T026 [P] [US3] Implement experimental **Generalized KL (GKL)** loss in `src/models/beta_vae.py`
- [x] T027 [US3] Create comparison CLI command in `main.py` to aggregate metrics across Run-IDs
- [x] T028 [US3] Implement validation against SC-002 and SC-003 thresholds in `src/utils/metrics.py`

**Checkpoint**: Stage 003 is fully functional with scientific validation.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final cleanup

- [x] T029 [P] Update `README.md` with new CLI commands and Run-ID documentation
- [x] T030 [P] Update `specs/003-disentangled-vae-wandb/quickstart.md` with final parameters
- [x] T031 Final code cleanup, type hints validation, and Google-style docstrings check across `src/models/` and `src/utils/`
- [x] T032 Run `pytest tests/integration/test_pipeline_end_to_end.py` to ensure no regressions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Can start immediately.
- **Foundational (Phase 2)**: Depends on T001-T003 completion.
- **User Stories (Phase 3+)**: All depend on Phase 2 completion.
  - US1 (Phase 3) is a prerequisite for US2 (Phase 4) because analysis requires trained models.
  - US3 (Phase 5) can run in parallel with US2 after US1 is complete.

### User Story Dependencies

- **US1 (P1)**: Prerequisite for all downstream analysis.
- **US2 (P1)**: Core business requirement.
- **US3 (P2)**: Scientific validation.

### Parallel Opportunities

- T004, T005, T006, T007 can be developed in parallel (Foundational).
- T010 and T011 can be developed in parallel (US1 Tests).
- T018 and T019 can be developed in parallel (US2 Tests).

---

## Parallel Example: Foundational Phase

```bash
# Implement metrics and factory in parallel
Task: "Implement MIG metric in src/utils/metrics.py"
Task: "Implement ModelFactory in src/models/factory.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup + Foundational.
2. Complete US1 (Beta-VAE Training + WandB).
3. **STOP and VALIDATE**: Verify that we can train a model and see it in WandB.

### Incremental Delivery

1. Foundation ready.
2. US1 added → Training functionality.
3. US2 added → Interpretation functionality (Factor analysis).
4. US3 added → Validation functionality (Metrics comparison).

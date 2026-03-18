# Feature Specification: Disentangled VAE for Marketing Impact Analysis (Stage 003)

**Feature Branch**: `003-disentangled-vae-wandb`  
**Created**: 2026-03-05  
**Status**: Draft  
**Input**: User description: "Implement Stage 003: Disentangled VAE for Marketing Impact Analysis with WandB integration for experiment tracking and model versioning."

## Clarifications

### Session 2026-03-05
- Q: Should GKL loss modification be mandatory or optional? → A: Optional / Experimental
- Q: Should validation attributes for MIG/SAP be pre-computed or dynamic? → A: Dynamic Calculation (calculated from validation data during training)
- Q: Should the Beta-VAE annealing schedule be fixed or configurable? → A: Configurable Epoch Range (specify start and end epochs for β increase)
- Q: What is the preferred model checkpointing strategy? → A: Hybrid (save best validation loss model AND periodic snapshots)
- Q: Should the latent dimension size be fixed or configurable? → A: Configurable Hyperparameter (set via CLI/config)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Model Training with WandB Integration (Priority: P1)

As a data scientist, I want to train a Beta-VAE and log its progress to Weights & Biases (WandB), so that I can track experiments, visualize losses, and compare different hyperparameters (like $\beta$ and annealing schedules) in real-time.

**Why this priority**: Core technical foundation for the stage. Without experiment tracking and comparison, we cannot prove the value of disentanglement.

**Independent Test**: Verify that a training run with the `--wandb` flag successfully initializes a project, logs MSE/KL losses, and stores artifacts.

**Acceptance Scenarios**:

1. **Given** a prepared dataset in `.parquet` format, **When** the training command is executed with `--arch beta_vae --wandb`, **Then** training metrics (MSE, KL, Total Loss) are visible in the WandB dashboard.
2. **Given** a completed training run, **When** checking the `experiments/[run_id]` directory, **Then** it must contain `config.json`, `weights.pt`, and `metrics.json`.

---

### User Story 2 - Disentangled Factor Analysis for Marketing (Priority: P1)

As a marketing analyst, I want to see a breakdown of behavior changes into independent factors (e.g., spending volume vs. price sensitivity), so that I can understand the specific nature of a campaign's impact.

**Why this priority**: Primary business value of this stage. Moves from "unexplained shift" to "actionable insight."

**Independent Test**: Run the inference command on a campaign period and verify the generated report identifies specific latent factor shifts.

**Acceptance Scenarios**:

1. **Given** a trained Beta-VAE model, **When** the inference command is run on a target household segment, **Then** the report lists the top factors that contributed to the behavior shift.
2. **Given** a specific campaign dataset, **When** analyzing the "nature of shift," **Then** the report categorizes it based on factor-level magnitude (e.g., "Trading Up" vs. "Stockpiling").

---

### User Story 3 - Model Architecture Comparison (Priority: P2)

As a researcher, I want to compare a Beta-VAE model against a Baseline VAE to quantify the trade-off between reconstruction accuracy and disentanglement quality.

**Why this priority**: Essential for scientific validation of the approach. Ensures we don't "over-disentangle" at the cost of data fidelity.

**Independent Test**: Use a comparison utility to analyze `metrics.json` from two different runs.

**Acceptance Scenarios**:

1. **Given** a Baseline VAE run and a Beta-VAE run, **When** comparing their metrics, **Then** the Beta-VAE must show higher Mutual Information Gap (MIG) scores.
2. **Given** both models, **When** evaluating reconstruction quality, **Then** the Beta-VAE's MSE should be no more than 15% higher than the Baseline VAE's MSE.

---

### Edge Cases

- **WandB Connection Failure**: The system should handle offline training or connection timeouts gracefully, falling back to local logging.
- **KL Collapse**: How does the system handle runs where KL divergence drops to zero? (Resolved via $\beta$-annealing schedule).
- **Missing Run-ID**: The system should provide a clear error message if an inference command refers to a non-existent `run-id`.
- **Large Vocabulary**: Handle potential memory issues when logging feature-level correlations to WandB for datasets with many categories.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support training a **Beta-VAE** with adjustable $\beta$, **configurable annealing schedule**, and **configurable latent dimension**.
- **FR-001.1**: System SHOULD support an experimental **Generalized KL (GKL)** divergence loss variant for handling long-tail transaction data.
- **FR-002**: System MUST integrate with **Weights & Biases (WandB)** for real-time logging and MUST implement a **hybrid checkpointing strategy** (best-model + periodic).
- **FR-003**: System MUST implement a **"Run-ID" directory convention**, where all experiment artifacts (config, weights, metrics) are stored together.
- **FR-004**: System MUST calculate and log **MIG (Mutual Information Gap)** and **SAP (Separated Attribute Predictability)** scores using attributes calculated dynamically from validation data.
- **FR-005**: System MUST provide a **Model Factory** to load the appropriate architecture (Baseline vs. Beta) based on the `config.json` in the Run-ID folder.
- **FR-006**: The reporting module MUST be **polymorphic**, providing factor-level decomposition when a disentangled model is used.
- **FR-007**: System MUST support **WandB API key** authentication via a `.env` file or environment variables.

### Key Entities *(include if feature involves data)*

- **Experiment (Run-ID)**: A persistent record of a single training run, including hyperparameters, weight tensors, and performance metrics.
- **Latent Factor**: An independent dimension in the VAE's bottleneck layer that represents a specific, interpretable aspect of customer behavior.
- **Disentanglement Score**: A quantitative measure (MIG/SAP) of how well the latent factors represent distinct attributes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can trigger a full training run with WandB logging using a single CLI command with fewer than 5 arguments.
- **SC-002**: Beta-VAE achieves a **MIG score > 0.15** higher than the Baseline VAE on synthetic validation attributes.
- **SC-003**: Reconstruction error (MSE) for Beta-VAE is **within 15%** of the Baseline VAE's error on the same dataset.
- **SC-004**: The inference report provides a **ranked list of factor shifts** for at least 80% of analyzed campaign participants.

# Implementation Plan: Disentangled VAE (Stage 003)

**Branch**: `003-disentangled-vae-wandb` | **Date**: 2026-03-05 | **Spec**: [/specs/003-disentangled-vae-wandb/spec.md](spec.md)
**Input**: Feature specification from `/specs/003-disentangled-vae-wandb/spec.md`

## Summary

Implement a disentangled VAE (Beta-VAE) to decompose marketing behavior shifts into interpretable independent factors (e.g., price sensitivity vs. purchase volume). The system will utilize Weights & Biases (WandB) for experiment tracking and implement a "Run-ID" convention for model management and versioning.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: PyTorch, Polars, Pandas, Scikit-learn, WandB, PyArrow
**Storage**: Local filesystem (Parquet files, model artifacts in `experiments/[run_id]/`)
**Testing**: pytest
**Target Platform**: Linux
**Project Type**: ML Pipeline / CLI
**Performance Goals**: Maintain reconstruction MSE within 15% of baseline; efficient processing of transaction data via Polars.
**Constraints**: Latent space disentanglement measured via MIG > 0.15 improvement over baseline.
**Scale/Scope**: ~32-64 latent dimensions, 2500+ households.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] PyTorch used for Deep Learning
- [x] Polars/Pandas used for data handling
- [x] Type hints and Google-style docstrings required
- [x] Reparameterization trick explicitly implemented
- [x] Loss includes Reconstruction (MSE) + KL Divergence (with β)
- [x] Latent dimension size is configurable (FR-001)
- [x] Modularity (data, models, services separated)

## Project Structure

### Documentation (this feature)

```text
specs/003-disentangled-vae-wandb/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── data/
│   └── (existing preparation logic)
├── models/
│   ├── __init__.py
│   ├── baseline_vae.py  # Refactored baseline
│   ├── beta_vae.py      # New disentangled architecture
│   └── factory.py       # Model loading factory (FR-005)
├── services/
│   ├── impact_analysis.py
│   ├── reporting_baseline.py
│   └── reporting_beta.py # Polymorphic reporting (FR-006)
├── utils/
│   ├── wandb_logger.py  # WandB integration (FR-002)
│   └── metrics.py       # MIG/SAP calculation (FR-004)
└── main.py              # CLI with --run-id support (FR-003)

tests/
├── unit/
│   ├── test_beta_vae.py
│   └── test_metrics.py
└── integration/
    └── test_experiment_flow.py
```

**Structure Decision**: Single project structure, following existing modularity while adding experiment-specific isolation.

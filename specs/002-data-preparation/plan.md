# Implementation Plan: Data Preparation

**Branch**: `002-data-preparation` | **Date**: 2026-03-04 | **Spec**: [specs/002-data-preparation/spec.md](spec.md)
**Input**: Feature specification from `/specs/002-data-preparation/spec.md`

## Summary

Prepare the raw Dunnhumby transaction data by applying hierarchical aggregations, cyclical temporal feature extraction, and 7-day rolling windows. The data will be log-scaled, z-score normalized per cohort (to handle heavy-tailed purchase data), and split using forward-chaining to prevent temporal leakage. The pipeline will output dense feature vectors in Parquet format, optimized for PyTorch VAE ingestion.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Polars (for fast processing of up to 1TB data), Pandas, Scikit-learn, PyArrow (for Parquet)
**Storage**: Parquet files on local filesystem
**Testing**: pytest
**Target Platform**: Linux server
**Project Type**: ML Data Pipeline (CLI)
**Performance Goals**: Process 1GB in < 15 minutes; scale to 1TB via vertical scaling
**Constraints**: Output dimensionality < 2000 features; memory efficient aggregation
**Scale/Scope**: Up to 1TB raw transaction data

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Type Hinting**: Required for all pipeline functions.
- **Documentation**: Google-style docstrings required.
- **Memory Efficiency**: Addressed by using Polars and Parquet format, and aggregating over rolling windows instead of raw one-hot encoding.
- **Reproducibility**: Required to set random seeds and save scaling parameters.

## Project Structure

### Documentation (this feature)

```text
specs/002-data-preparation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── pipeline_api.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── data/
│   ├── __init__.py
│   ├── prepare.py          # Main CLI entrypoint for pipeline
│   ├── extractors.py       # Temporal and hierarchical feature extraction
│   └── normalizers.py      # Log-scaling, z-score, and parameter saving
tests/
├── unit/
│   └── test_data_prep.py   # Unit tests for extractors and normalizers
└── integration/
    └── test_pipeline_end_to_end.py
```

**Structure Decision**: Utilizing the existing `src/data/` structure for the data pipeline modules.

## Complexity Tracking

*(No violations of constitution identified. Memory efficiency strictly adhered to by choosing Polars and avoiding massive one-hot encodings).*
# Implementation Plan: Campaign Impact and Latent Validation

**Branch**: `004-campaign-impact-validation` | **Date**: 2026-04-09 | **Spec**: [/specs/004-campaign-impact-validation/spec.md](spec.md)
**Input**: Feature specification from `/specs/004-campaign-impact-validation/spec.md`

## Summary

Implement a research-grade validation workflow that links Dunnhumby campaign data to household purchase behavior, estimates campaign effects with explicit quasi-causal diagnostics, and validates whether Beta-VAE latent factors have stable, evidence-based semantics. The workflow will produce an analysis-ready campaign dataset, campaign-level evidence classifications, latent-factor validation outputs, and a reproducible research report that can confirm, weaken, or reject current project claims.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch, Pandas, Polars, NumPy, Scikit-learn, PyArrow, Seaborn/Plotly  
**Storage**: Local filesystem using CSV inputs, Parquet intermediate datasets, JSON summaries, and markdown report artifacts  
**Testing**: pytest  
**Target Platform**: Linux CLI environment  
**Project Type**: ML research pipeline / CLI  
**Performance Goals**: Build campaign-linked datasets for at least 3 selected campaigns in a single reproducible run; generate campaign evidence and latent validation outputs within a single analyst session on local data; preserve current model comparison workflow while adding validation outputs  
**Constraints**: Must support negative findings, reuse existing trained models and known-good VAE configurations, avoid unsupported causal claims, acknowledge the absence of randomized assignment and household-level exposure logs, and remain reproducible with fixed seeds and deterministic cohort definitions where possible  
**Scale/Scope**: 2500 households, 30 campaigns, 1584 assigned campaign households, 3-5 priority campaigns for the first validation pass, baseline and Beta-VAE comparison on selected holdout cohorts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Python 3.11+ remains the implementation language
- [x] PyTorch remains the deep learning framework for model-side evaluation
- [x] Pandas/Polars remain the primary data-handling tools
- [x] Type hints and Google-style docstrings remain required for new Python modules
- [x] Reproducibility is preserved through explicit seeds and deterministic cohort construction where feasible
- [x] Modularity is preserved by separating data integration, campaign validation, latent validation, and reporting responsibilities
- [x] VAE-specific rules remain intact because this feature validates existing latent representations rather than replacing the current architecture
- [x] No constitution violations identified before research

## Project Structure

### Documentation (this feature)

```text
specs/004-campaign-impact-validation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── validation_cli.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── data/
│   ├── prepare.py
│   ├── dataset.py
│   ├── extractors.py
│   ├── normalizers.py
│   ├── campaign_dataset.py        # New campaign-linked dataset builder
│   ├── campaign_windows.py        # New window assignment and cohort calendar logic
│   └── validation_attributes.py   # New observable outcomes and factor attributes
├── models/
│   ├── baseline_vae.py
│   ├── beta_vae.py
│   └── factory.py
├── services/
│   ├── baseline.py
│   ├── impact_analysis.py
│   ├── reporting.py
│   ├── reporting_baseline.py
│   ├── reporting_beta.py
│   ├── campaign_validation.py     # New quasi-causal evaluation workflow
│   ├── latent_validation.py       # New latent semantics evaluation workflow
│   └── validation_reporting.py    # New evidence-based report generation
├── utils/
│   ├── metrics.py
│   ├── seed.py
│   ├── wandb_logger.py
│   ├── matching.py                # New cohort matching helpers
│   └── statistics.py              # New diagnostics and summary helpers
└── main.py                        # Extended CLI for validation workflows

tests/
├── unit/
│   ├── test_data_prep.py
│   ├── test_metrics.py
│   ├── test_campaign_dataset.py
│   ├── test_campaign_validation.py
│   ├── test_latent_validation.py
│   └── test_validation_reporting.py
└── integration/
    ├── test_pipeline.py
    ├── test_pipeline_end_to_end.py
    ├── test_campaign_validation_flow.py
    ├── test_latent_validation_flow.py
    └── test_validation_reporting_flow.py
```

**Structure Decision**: Keep the existing single-project CLI layout. Add new modules under `src/data` for campaign-linked dataset assembly, `src/services` for campaign and latent validation workflows, and `src/utils` for matching and statistical diagnostics. Extend `main.py` instead of introducing a separate application boundary.

## Phase 0: Research Summary

Phase 0 decisions are captured in [research.md](research.md). The key outcomes are:
- prioritize campaigns `18`, `13`, and `8` for first-pass validation because they have the strongest treated-household coverage and redemption signal
- use household-campaign windows with fixed pre/post spans and campaign-native in-period spans
- use matched treated-vs-untreated difference-in-differences as the first quasi-causal baseline, with event-study and placebo diagnostics
- treat campaign evidence as quasi-causal rather than strictly causal because assignment is not randomized and exposure is only partially observed
- use `total_spend`, `trip_count`, `category_diversity`, and `promo_share` as the primary first-pass outcomes, with other metrics treated as secondary diagnostics
- validate latent semantics against observable household attributes on holdout data using both disentanglement metrics and stability checks

## Phase 1: Design Summary

Phase 1 artifacts define:
- a campaign-linked household analysis model in [data-model.md](data-model.md)
- a CLI contract for building data, validating campaigns, validating latents, and generating final reports in [contracts/validation_cli.md](contracts/validation_cli.md)
- an operator workflow in [quickstart.md](quickstart.md)

## Post-Design Constitution Check

- [x] Design keeps PyTorch-centered model evaluation intact
- [x] Design keeps data preparation, modeling, services, and reporting modular
- [x] Design adds reproducible research artifacts rather than ad hoc notebook-only outputs
- [x] Design does not introduce a second application architecture or non-approved framework
- [x] Design remains consistent with type hinting, docstring, and memory-efficiency expectations

## Complexity Tracking

No constitution exceptions or justified complexity escalations are required at planning time.

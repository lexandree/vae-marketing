---
title: "VAE Marketing Impact Engine"
description: "A research-oriented machine learning pipeline for campaign diagnostics on retail transactions, combining restricted quasi-causal validation with VAE-family latent models."
pubDate: 2026-03-18
tags: ["PyTorch", "VAE", "Causal Inference", "Polars", "WandB", "Data Science"]
githubUrl: "https://github.com/lexandree/vae-marketing"
featured: true
order: 1
---

## Problem Statement
Traditional marketing analytics often answers *what changed* but not *how customer behavior shifted structurally*. In retail campaign data, another problem appears immediately: campaign assignment is observational, so it is easy to overclaim causal impact from noisy treated-vs-untreated comparisons.

The project goal is therefore split into two parts:

1. build a restricted quasi-causal validation workflow for campaign effects
2. interpret validated behavioral shifts with VAE-family latent models

## Approach / Architecture
This project uses the **Dunnhumby Complete Journey** dataset and combines a campaign-validation layer with several VAE-family models.

- **Campaign Validation Layer**: Builds campaign-linked household panels, applies restricted matching, and assigns evidence labels such as `supported`, `weak`, or `unsupported` instead of assuming campaign effects are real by default.
- **Leakage-Controlled Representation Learning**: Holds out case-study campaigns from latent-model training and validates latent mappings only on eval households.
- **Latent Modeling Stack**:
  - `Beta-VAE` as the main held-out semantic baseline
  - `Contrastive VAE` for target-vs-background campaign-salient variation
  - `Beta-TCVAE` for stronger campaign-to-latent bridges
- **Data Engineering**: Uses `Polars`, `Pandas`, and `PyArrow` for rolling-window household panels, campaign joins, and validation artifacts.
- **Experiment Tracking**: Uses **Weights & Biases** for hyperparameter search and model comparison.

## Results / Metrics
- **Campaign Diagnostics**: The restricted quasi-causal workflow identified campaigns `26` and `30` as the two strongest case studies in the current dataset. Larger campaigns such as `18`, `13`, and `8` looked weaker once balance and placebo-style diagnostics were enforced.
- **Held-Out Latent Validation**: Final latent claims are based on leakage-controlled splits rather than in-sample representations.
- **Model Comparison**:
  - `Beta-VAE` is the strongest overall held-out latent baseline
  - `Contrastive VAE` is the best fit for campaign-salient interpretation
  - `Beta-TCVAE` produces the strongest bridge from campaign-visible shifts to validated latent factors
- **Interpretation Layer**: For campaigns `26` and `30`, the bridge artifacts consistently align campaign shifts with validated factors related to `total_spend` and `category_diversity`.

### Training Dynamics
The original `Beta-VAE` baseline still serves as the cleanest global latent benchmark and remains useful for showing the reconstruction-vs-regularization trade-off during training.

![Training Curves](/schaufenster/assets/vae-marketing/training_curves.svg)  
![Beta Annealing](/schaufenster/assets/vae-marketing/beta_annealing.svg)  

### Hyperparameter Search
The following interactive chart shows the original Bayesian search over `Beta-VAE` hyperparameters. It is still useful as the baseline search landscape for the project.

<div class="w-full border border-skin-line rounded-lg overflow-hidden bg-white mt-4" style="height: 800px;">
  <iframe 
    src="https://api.wandb.ai/links/andreev-al/kap3b0vl" 
    style="width: 100%; height: 100%; border: none;"
    title="WandB Sweep Parallel Coordinates"
  ></iframe>
</div>  

*(Note: If the interactive chart above doesn't load, view the static version below)*

![Hyperparameter Search (Static)](/schaufenster/assets/vae-marketing/sweep_parallel_coords.svg)

## Current Positioning

This is best presented as a **research and validation framework**, not as a production marketing decision engine.

What the project can currently claim:

- it builds a reproducible campaign-validation pipeline on real retail data
- it supports negative findings instead of forcing positive campaign stories
- it shows how different VAE-family models behave under leakage-controlled evaluation
- it connects supported campaign shifts to validated latent factors

What it should not currently claim:

- strict causal proof
- replacement of randomized experimentation
- production-readiness at enterprise scale

## Ongoing Research

The next reasonable research direction is not "add every disentanglement paper at once", but to evaluate a small number of stronger successors against the current no-leak baselines.

Planned directions now include:

- better `beta-TCVAE` tuning
- `FactorVAE` as a stronger total-correlation benchmark
- later causal-latent variants only if they clearly improve on the current stack

The living roadmap is tracked in `TODO.md`.

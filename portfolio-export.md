---
title: "VAE Marketing Impact Engine"
description: "A research-oriented machine learning pipeline based on Variational Autoencoders (Beta-VAE) for campaign diagnostics, quasi-causal validation, and latent-factor analysis on retail behavior."
pubDate: 2026-03-18
tags: ["PyTorch", "Beta-VAE", "Polars", "WandB", "Data Science"]
githubUrl: "https://github.com/lexandree/vae-marketing"
featured: true
order: 1
---

## Problem Statement
Traditional marketing analytics (like A/B testing) typically tells you *what* happened (e.g., "Sales increased by 15%"), but fails to explain *how* customer habits structurally changed. Furthermore, in mass marketing campaigns where holdout/control groups are impossible, measuring true impact becomes extremely difficult. There is a need for a system that can learn a baseline "behavioral DNA" for each customer and measure the depth and persistence of behavioral shifts caused by a campaign.

## Approach / Architecture
This project applies a **Disentangled Variational Autoencoder (Beta-VAE)** to raw retail transaction data (Dunnhumby dataset) to learn latent representations of household shopping habits and then validate those representations against observable behavior.
- **High-Performance Data Engineering**: Utilizes `Polars` and `PyArrow` to process large-scale, high-cardinality transaction data (scalable to 1TB+) using rolling windows and cyclical temporal encodings.
- **Deep Generative Modeling**: Implements a PyTorch-based Beta-VAE. The annealing of the $\beta$ parameter forces the model to learn *disentangled* factors (e.g., price sensitivity vs. category exploration vs. volume stockpiling).
- **Campaign Validation Workflow**: Adds campaign-linked datasets, treated-vs-comparison analysis, and evidence labels so campaign claims can be supported, weakened, or rejected.
- **Experiment Tracking**: Full integration with **Weights & Biases (WandB)** for metric logging (MIG, SAP, MSE), model versioning, and hyperparameter optimization (WandB Sweeps).

## Results / Metrics
- **Quantifiable Behavioral Shifts**: Measures observed behavioral movement and labels campaign evidence strength as supported, weak, unsupported, or insufficient data.
- **Latent Validation**: Compares latent dimensions against observable attributes and explicitly rejects unstable mappings instead of relying only on post-hoc naming.
- **Actionable Research Outputs**: Produces campaign diagnostics, latent validation artifacts, and a final evidence-based validation report.

### Training Dynamics
The model effectively balances reconstruction fidelity (MSE Loss) with latent space regularization (KL Divergence) over a carefully tuned annealing schedule.

![Training Curves](/schaufenster/assets/vae-marketing/training_curves.svg)  
![Beta Annealing](/schaufenster/assets/vae-marketing/beta_annealing.svg)  

### Hyperparameter Search
The following interactive chart shows the results of the Bayesian hyperparameter search. You can filter and reorder axes to explore the relationships between Latent Dimension, Beta Regularization, and Learning Rate.

<div class="w-full border border-skin-line rounded-lg overflow-hidden bg-white mt-4" style="height: 800px;">
  <iframe 
    src="https://api.wandb.ai/links/andreev-al/kap3b0vl" 
    style="width: 100%; height: 100%; border: none;"
    title="WandB Sweep Parallel Coordinates"
  ></iframe>
</div>  

*(Note: If the interactive chart above doesn't load, view the static version below)*

![Hyperparameter Search (Static)](/schaufenster/assets/vae-marketing/sweep_parallel_coords.svg)

## 🔬 Ongoing Research

We are actively expanding this framework to establish a new state-of-the-art (SOTA) for the reconstruction-disentanglement trade-off in behavioral modeling. 

Our current research focuses on a **Combinatorial Synergy Matrix**, systematically evaluating and combining advanced regularizations from $\beta$-TCVAE, FactorVAE, DIP-VAE, and InfoVAE to isolate the optimal mathematical constraints for marketing analytics. You can view our detailed Research Roadmap in the repository's `TODO.md`.

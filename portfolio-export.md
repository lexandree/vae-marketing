---
title: "VAE Marketing Impact Engine"
description: "A production-ready machine learning pipeline based on Variational Autoencoders (Beta-VAE) to quantify the impact of marketing stimuli on consumer behavior, bypassing traditional A/B testing limitations."
pubDate: 2026-03-18
tags: ["PyTorch", "Beta-VAE", "Polars", "WandB", "Data Science"]
githubUrl: "https://github.com/lexandree/vae-marketing"
featured: true
order: 1
---

## Problem Statement
Traditional marketing analytics (like A/B testing) typically tells you *what* happened (e.g., "Sales increased by 15%"), but fails to explain *how* customer habits structurally changed. Furthermore, in mass marketing campaigns where holdout/control groups are impossible, measuring true impact becomes extremely difficult. There is a need for a system that can learn a baseline "behavioral DNA" for each customer and measure the depth and persistence of behavioral shifts caused by a campaign.

## Approach / Architecture
This project applies a **Disentangled Variational Autoencoder (Beta-VAE)** to raw retail transaction data (Dunnhumby dataset) to learn deep, independent latent representations of household shopping habits.
- **High-Performance Data Engineering**: Utilizes `Polars` and `PyArrow` to process large-scale, high-cardinality transaction data (scalable to 1TB+) using rolling windows and cyclical temporal encodings.
- **Deep Generative Modeling**: Implements a PyTorch-based Beta-VAE. The annealing of the $\beta$ parameter forces the model to learn *disentangled* factors (e.g., price sensitivity vs. category exploration vs. volume stockpiling).
- **Unsupervised Control Groups**: By using the customer's own deep historical profile as the baseline, the system measures campaign impact via Euclidean distance in the latent space, eliminating the need for a physical control group.
- **Experiment Tracking**: Full integration with **Weights & Biases (WandB)** for metric logging (MIG, SAP, MSE), model versioning, and hyperparameter optimization (WandB Sweeps).

## Results / Metrics
- **Quantifiable Behavioral Shifts**: Successfully isolated distinct consumer reactions into independent factors, proving that certain campaigns caused *price-tier upgrades* rather than just *volume stockpiling*.
- **High-Fidelity Tracking**: Automated HPO Sweeps identified optimal latent dimensions (`latent_dim=32`) and regularization strengths (`beta=2.0`), achieving high Mutual Information Gap (MIG) scores, indicating strong factor disentanglement.
- **Actionable Business Insights**: Outputs a detailed `inference_report.json` segmenting audiences based on their reaction depth and identifying "gateway categories" that drive long-term habit changes.

### Training Dynamics
The model effectively balances reconstruction fidelity (MSE Loss) with latent space regularization (KL Divergence) over a carefully tuned annealing schedule.

![Training Curves](/schaufenster/assets/vae-marketing/training_curves.svg)  
![Beta Annealing](/schaufenster/assets/vae-marketing/beta_annealing.svg)  

### Hyperparameter Search
The following interactive chart shows the results of the Bayesian hyperparameter search. You can filter and reorder axes to explore the relationships between Latent Dimension, Beta Regularization, and Learning Rate.

<div class="w-full h-[400px] border border-skin-line rounded-lg overflow-hidden bg-white mt-4">
  <iframe 
    src="https://wandb.ai/andreev-al/vae_marketing/sweeps/idn2al3o/panel/xrqe8320z?nw=nwuserandreeval" 
    class="w-full h-full border-none"
    title="WandB Sweep Parallel Coordinates"
  ></iframe>
</div>

*(Note: If the interactive chart above doesn't load, view the static version below)*

![Hyperparameter Search (Static)](/schaufenster/assets/vae-marketing/sweep_parallel_coords.svg)

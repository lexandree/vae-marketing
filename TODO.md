# VAE Marketing Impact Engine - TODO & Future Research

## Active Tasks

- [ ] **Document the current model stack and final public narrative**
  *Context: The repository now contains a much richer story than the original Beta-VAE prototype. Public-facing descriptions must be kept aligned with the actual evidence hierarchy: quasi-causal validation first, then no-leak latent interpretation.*

  **Current facts to preserve:**
  1. `beta-VAE` is still the strongest general held-out latent baseline.
  2. `Contrastive VAE` is the best campaign-salient representation model so far.
  3. `beta-TCVAE` currently looks better for campaign-to-latent bridge strength than for global latent quality.
  4. Campaigns `26` and `30` are the current case studies; `18`, `13`, and `8` are weaker under diagnostics.

- [ ] **Tune Beta-TCVAE Against The No-Leak Baselines**
  *Context: Beta-TCVAE has already been implemented and trained. It currently underperforms `beta-VAE` on held-out global latent metrics, but it produces stronger bridges for `total_spend` and `category_diversity`. That makes it a promising interpretation model, but not yet a better universal baseline.*

  **Questions to answer:**
  1. Can `beta-TCVAE` close the `MIG/SAP` gap without losing bridge quality?
  2. Which coefficients matter most: `beta`, `tc_alpha`, or `tc_lambda`?
  3. Does a milder TC penalty recover better reconstruction while preserving campaign-relevant mappings?

  **Candidates to Evaluate:**
  1. **`beta-TCVAE` coefficient sweeps**
     - *Why:* cheapest next step, already implemented
     - *Expected Impact:* improve held-out `MIG/SAP` without losing bridge quality
  2. **FactorVAE** [[Paper](https://arxiv.org/abs/1802.05983)]
     - *Why:* strongest natural next TC-style benchmark after `beta-TCVAE`
     - *Expected Impact:* may sharpen disentanglement, but introduces adversarial instability
  3. **DIP-VAE** [[Paper](https://arxiv.org/abs/1711.00848)]
     - *Why:* covariance-based alternative if TC estimation remains noisy
     - *Expected Impact:* potentially cleaner global latent structure with a simpler training loop than FactorVAE
  4. **InfoVAE** [[Paper](https://arxiv.org/abs/1706.02262)]
     - *Why:* may help preserve informative latent usage if stronger TC penalties start collapsing useful signal
     - *Expected Impact:* possible recovery of semantic richness at similar reconstruction quality

- [ ] **Investigate FactorVAE As The Next Real Benchmark**
  *Context: FactorVAE is the next most natural model after `beta-TCVAE`, but it requires a discriminator and an adversarial training loop. It should only be added if we are ready to support the additional tuning and instability cost.*

  **Questions to answer:**
  - Does FactorVAE beat `beta-VAE` or `beta-TCVAE` on held-out global metrics?
  - Does it improve the campaign bridge for `26/30`?
  - Is the extra adversarial complexity worth it in this project?

- [ ] **Review More Recent Latent-Variable Literature Before Adding New Families**
  *Context: The classic disentanglement papers remain useful baselines, but the research space has moved. Before adding more architectures, we should review newer work to avoid spending time on stale variants that no longer represent the strongest practical direction.*

  **Focus for the literature review:**
  - stronger disentanglement baselines after the original `beta-TCVAE / FactorVAE / DIP-VAE / InfoVAE` wave
  - newer contrastive or intervention-aware latent models
  - methods that better align with observational retail behavior rather than synthetic disentanglement benchmarks

  **Next Steps:**
  - [x] Implement `beta_tcvae` support in `src/models/`, `main.py`, and `factory.py`.
  - [x] Train and validate a no-leak `beta-TCVAE` baseline.
  - [ ] Add a comparable `FactorVAE` implementation in `src/models/`.
  - [ ] Decide whether `DIP-VAE` is a better low-complexity next step than `FactorVAE`.
  - [ ] Run a targeted WandB sweep for `beta-TCVAE` coefficients instead of broad architecture sprawl.
  - [ ] Only after that, revisit whether combination-style objectives are still worth the complexity.

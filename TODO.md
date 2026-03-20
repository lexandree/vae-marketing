# VAE Marketing Impact Engine - TODO & Future Research

## Active Tasks

- [ ] **Explore Advanced VAE Architectures for Better Accuracy-Disentanglement Trade-off**
  *Context: While Beta-VAE successfully disentangles factors by penalizing the KL divergence, it often suffers from the well-known "disentanglement-reconstruction trade-off". Increasing $\beta$ forces independence but degrades the model's ability to accurately reconstruct the original data (MSE loss increases). Now that we have established the $\beta$-VAE optimum baseline and HPO sweep landscape, we should test newer variants that resolve this trade-off.*

  **Candidates to Evaluate:**
  1. **$\beta$-TCVAE (Total Correlation VAE)** [[Paper](https://arxiv.org/abs/1802.04942)]
     - *Why:* Decomposes the KL divergence and only heavily penalizes the Total Correlation (TC) term (which forces independence) without overly penalizing the mutual information between the latent space and the input.
     - *Expected Impact:* Same or better MIG/SAP scores as Beta-VAE, but with significantly lower MSE (better accuracy). Very stable to train.
  2. **FactorVAE** [[Paper](https://arxiv.org/abs/1802.05983)]
     - *Why:* Uses an adversarial discriminator to penalize Total Correlation.
     - *Expected Impact:* Often yields very sharp disentanglement, though the adversarial training loop can be slightly harder to tune than $\beta$-TCVAE.
  3. **DIP-VAE (Disentangled Inferred Prior VAE)** [[Paper](https://arxiv.org/abs/1711.00848)]
     - *Why:* Pushes the covariance matrix of the aggregated posterior to be diagonal.
     - *Expected Impact:* Decorrelates latent dimensions effectively, good alternative if TC estimation is too noisy.
  4. **InfoVAE** [[Paper](https://arxiv.org/abs/1706.02262)]
     - *Why:* Solves the "information preference" problem by explicitly maximizing mutual information, ensuring the latent codes are actually used by the decoder.

- [ ] **Investigate Combinatorial Synergy (The "Matrix of Combinations")**
  *Context: The regularizations proposed in the papers above target different parts of the objective function (e.g., Total Correlation vs. Mutual Information vs. Covariance). It is highly probable that combining some of these approaches will yield synergistic improvements, while others might conflict and degrade performance.*
  
  **Matrix Experiments to Run:**
  - **$\beta$-TCVAE + InfoVAE:** Can we heavily penalize Total Correlation for disentanglement ($\beta$-TCVAE) while simultaneously forcing the decoder to maximize Mutual Information with the latent codes (InfoVAE) to prevent posterior collapse and maintain high accuracy?
  - **FactorVAE + DIP-VAE:** Does combining adversarial TC penalty (FactorVAE) with explicit covariance diagonalization (DIP-VAE) over-regularize the latent space, or does it create the "ultimate" disentangled representation?
  - **Ablation Studies:** Systematically turn on/off individual loss components across the combinations to isolate which mathematical constraint contributes most to the target metrics (MIG/SAP vs. MSE).

  **Next Steps:**
  - [ ] Implement `beta_tc_vae.py` in `src/models/`.
  - [ ] Implement `factor_vae.py` (with auxiliary discriminator) in `src/models/`.
  - [ ] Implement `dip_vae.py` and `info_vae.py` in `src/models/`.
  - [ ] Update `main.py` and `factory.py` to support `--arch tc_vae`, `--arch factor_vae`, `--arch dip_vae`, and `--arch info_vae`.
  - [ ] **Implement a flexible `--loss-components` argument** in the trainer to dynamically toggle TC-penalty, MI-maximization, and Covariance-diagonalization within a single model run.
  - [ ] Run a WandB sweep comparing all variants **and their combinations** across identical latent dimensions to measure the Pareto frontier improvement (MSE vs MIG).

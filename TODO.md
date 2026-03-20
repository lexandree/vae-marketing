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

  **Next Steps:**
  - [ ] Implement `beta_tc_vae.py` in `src/models/`.
  - [ ] Implement `factor_vae.py` (with auxiliary discriminator) in `src/models/`.
  - [ ] Implement `dip_vae.py` and `info_vae.py` in `src/models/`.
  - [ ] Update `main.py` and `factory.py` to support `--arch tc_vae`, `--arch factor_vae`, `--arch dip_vae`, and `--arch info_vae`.
  - [ ] Run a WandB sweep comparing all variants across identical latent dimensions to measure the Pareto frontier improvement (MSE vs MIG).

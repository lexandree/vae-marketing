# Phase 0: Research - Disentangled VAE & Experiment Tracking

## Decision 1: β-VAE Implementation Strategy
- **Decision**: Implement the standard β-VAE formulation with a linear annealing schedule for the β parameter.
- **Rationale**: β-VAE (Higgins et al., 2017) is a proven baseline for unsupervised disentanglement. By penalizing the KL divergence term (β > 1), we force the model to learn independent factors. Annealing from β=0 to the target β prevents "KL collapse" where the latent space fails to learn any meaningful information.
- **Alternatives considered**: Factor-VAE or β-TCVAE. These are more complex and require additional discriminator networks. We will start with β-VAE and only pivot if disentanglement quality is insufficient.

## Decision 2: Experiment Tracking with Weights & Biases (WandB)
- **Decision**: Use WandB for logging scalar metrics (losses), histograms (latent distributions), and artifact versioning (saving `weights.pt` and `config.json`).
- **Rationale**: WandB is the industry standard for ML experiment tracking. It allows for easy comparison of different β values and latent dimensions across multiple runs. artifact support simplifies the "Run-ID" convention.
- **Alternatives considered**: MLflow (more complex local setup required) or simple local logging (no visualization or easy comparison).

## Decision 3: Disentanglement Metrics (MIG & SAP)
- **Decision**: Implement **Mutual Information Gap (MIG)** and **Separated Attribute Predictability (SAP)** as the primary quantitative metrics for disentanglement.
- **Rationale**:
    - **MIG** (Chen et al., 2018) measures the difference in mutual information between a latent factor and its most correlated attribute vs. the second most correlated. A high gap indicates a one-to-one mapping.
    - **SAP** (Kumar et al., 2017) trains a simple linear classifier to predict attributes from individual factors.
- **Alternatives considered**: DCI (Disentanglement, Completeness, Informativeness). DCI is more comprehensive but significantly more complex to implement and interpret.

## Decision 4: Experimental Generalized KL (GKL)
- **Decision**: Implement GKL as a toggleable loss variant based on arXiv:2503.08038v1.
- **Rationale**: Traditional KL divergence can struggle with "long-tail" data (rare purchase categories). GKL allows for weighted penalties that can potentially preserve signals from rare but important transactions.
- **Alternatives considered**: Standard KL only. We keep GKL as an experiment to potentially improve results on sparse marketing data.

## Decision 5: "Run-ID" File Convention
- **Decision**: Each run will be stored in `experiments/[arch]/[timestamp_or_name]/`.
- **Rationale**: Ensures complete isolation of weights, configurations, and reports. Prevents accidental overwriting of baseline models.
- **Alternatives considered**: Flat directory with naming prefixes. Harder to manage as the number of experiments grows.

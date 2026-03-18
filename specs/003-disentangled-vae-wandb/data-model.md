# Data Model: Disentangled VAE (Stage 003)

## Entities

### Experiment Configuration (config.json)
Represents the hyperparameters and metadata for a specific training run.
- **run_id**: String (unique identifier)
- **architecture**: Enum ("baseline", "beta_vae")
- **latent_dim**: Integer (default: 32)
- **beta**: Float (KL penalty weight)
- **anneal_epochs**: Integer (number of epochs for β to reach target)
- **learning_rate**: Float
- **batch_size**: Integer
- **vocabulary_path**: String (path to vocabulary.json used)
- **timestamp**: ISO8601 String

### Latent Factors
The disentangled representation of a household's behavior.
- **household_id**: String
- **factors**: Vector[Float] (size = latent_dim)
- **factor_interpretations**: Map[Integer, String] (post-hoc mapping of factor index to business meaning)

### Metrics (metrics.json)
Quantitative assessment of model performance and disentanglement.
- **mse_loss**: Float (reconstruction error)
- **kl_divergence**: Float
- **mig_score**: Float (Mutual Information Gap)
- **sap_score**: Float (Separated Attribute Predictability)

## Relationships
- **Experiment** owns 1 **Model Weight file** (`weights.pt`).
- **Experiment** owns 1 **Config file** (`config.json`).
- **Inference Run** uses 1 **Experiment** to produce **Latent Factors**.
- **Report** uses **Latent Factors** and **Metrics** to generate insights.

## State Transitions
1. **DRAFT**: Parameters defined via CLI.
2. **TRAINING**: Model weights being updated; metrics logged to WandB.
3. **SAVED**: Artifacts persisted in `experiments/[run_id]/`.
4. **VALIDATED**: MIG/SAP metrics calculated and added to `metrics.json`.
5. **READY**: Model available for polymorphic reporting and campaign analysis.

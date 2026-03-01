# Phase 0: Research

## Decision 1: VAE Architecture for Transaction Data
- **Decision**: Use a feedforward Variational Autoencoder (VAE) implemented in PyTorch with explicit Reparameterization Trick, incorporating temporal features (month/week-of-year) as auxiliary inputs to the encoder and decoder.
- **Rationale**: The constitution mandates PyTorch and explicit reparameterization. VAEs are excellent for learning the underlying latent representation (baseline) of household purchase behavior. Adding temporal context features fulfills FR-008.
- **Alternatives considered**: Recurrent Neural Networks (RNN) or LSTMs were considered but a feedforward VAE with temporal auxiliary inputs is simpler, faster (meeting the < 5s/household constraint), and aligns with the deep autoencoder focus of the project.

## Decision 2: Measuring Behavioral Shift (Deviation)
- **Decision**: Calculate the deviation as the distance in the latent space (e.g., Euclidean or Cosine) before and after the external stimulus, combined with analyzing the reconstruction loss for anomalies. The reconstructed output will be categorized (e.g., stockpiling, trading up) using predefined heuristic rules based on price and volume changes.
- **Rationale**: Latent space shifts directly quantify the change in the learned underlying preferences, satisfying FR-003 and SC-001. Reconstruction loss spikes can identify moments of intervention (FR-002).
- **Alternatives considered**: Raw volume tracking was rejected as the spec requires qualitative understanding (trading up vs. stockpiling) rather than just volume changes.

## Decision 3: Persistence Analysis
- **Decision**: Use a sliding window approach over the post-stimulus data, projecting each window into the latent space and measuring the distance to the pre-stimulus baseline profile. The persistence is defined as the time until this distance falls below a defined threshold (e.g., 1 standard deviation of baseline variance).
- **Rationale**: This provides a robust, mathematically sound way to satisfy FR-005.
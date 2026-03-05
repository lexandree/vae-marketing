# VAE Marketing Analysis

This repository contains a machine learning pipeline built around a Variational Autoencoder (VAE) trained on retail transaction data from the Dunnhumby dataset. The core purpose is to reconstruct behaviour profiles for households and quantify deviations caused by marketing campaigns.

## Why this project?

A machine learning approach to quantifying the impact of marketing campaigns on consumer behaviour. Dunnhumby provides real-world customer purchases coupled with campaign metadata, making it an ideal testbed for demonstration.

## Novelty and value

- **Unsupervised deviation measurement.** The project learns a baseline behaviour profile in an unsupervised manner and quantifies campaign impact by measuring shifts in the latent space—enabling analysis without labeled intervention data.
- **VAE in a new domain.** VAEs are popular for anomaly detection in images or sensor data; their application to marketing behaviour, especially to observe the impact of external stimuli, is a niche and powerful demonstration.

## Project structure

```
src/
 tests/
 specs/
 data/
 README.md
```

## Tech stack

- Python 3.11+
- PyTorch (deep learning)
- Pandas/Polars (data handling)
- Scikit-learn
- Plotly/Seaborn (visualization)

## Getting started

For a complete step-by-step guide on data preparation, model training, and running the impact analysis report, please refer to the **[Quickstart Guide](specs/002-data-preparation/quickstart.md)**.

### High-level workflow:

1. **Prepare Data**: Run `src.data.prepare` to transform raw CSVs into Parquet.
2. **Train Model**: Run `main.py train` to establish a behavioral baseline.
3. **Analyze Impact**: Run `main.py infer` to detect deviations in consumer behavior.

This document serves as a high-level introduction; further instructions exist in `specs/002-data-preparation/quickstart.md`.
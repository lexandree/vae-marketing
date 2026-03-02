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

1. Create and activate the `ds` conda environment:
   ```bash
   conda env create -f environment.yml  # or manually install packages
   conda activate ds
   ```
2. Place Dunnhumby CSVs in `data/` or adapt `src/data/dataset.py` accordingly. Data files are intentionally **not** tracked (see `.gitignore`).
3. Run the unit test suite from the repository root. The `src` package must be importable; you can either install it in editable mode or set `PYTHONPATH`:
   ```bash
   # option A: install package locally
   pip install -e src
   pytest

   # option B: export PYTHONPATH
   export PYTHONPATH="$PWD/src"
   pytest
   ```
   This ensures `import src` works and avoids ``ModuleNotFoundError``.

This document serves as a high-level introduction; further instructions exist in `specs/001-consumer-behavior-impact/quickstart.md` once implemented.
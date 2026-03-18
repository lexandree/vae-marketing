import numpy as np
import pandas as pd
import pytest
import torch

from src.models.baseline_vae import build_vae_model
from src.services.baseline import get_household_profile, train_baseline_vae, vae_loss


def test_vae_loss() -> None:
    """Test the VAE loss calculation."""
    recon_x = torch.tensor([[0.5, 0.5]])
    x = torch.tensor([[0.5, 0.5]])
    mu = torch.tensor([[0.0, 0.0]])
    logvar = torch.tensor([[0.0, 0.0]])

    loss = vae_loss(recon_x, x, mu, logvar)
    assert loss.item() == 0.0


def test_train_baseline_insufficient_data() -> None:
    """Test training with insufficient data (no category columns)."""
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    df = pd.DataFrame({
        'household_id': ['h1', 'h1'],
        'cat_1': [1.0, 2.0],
        'cat_2': [0.5, 0.5],
        'month_of_year': [1, 2],
        'week_of_year': [1, 5]
    })

    with pytest.raises(ValueError, match="No spend/qty features found in data for training."):
        train_baseline_vae(df, model)


def test_train_baseline_success() -> None:
    """Test successful baseline VAE training."""
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    h_ids = ['h1'] * 6 + ['h2'] * 6
    df = pd.DataFrame({
        'HOUSEHOLD_KEY': h_ids,
        'cat_1_SPEND': np.random.rand(12),
        'cat_1_QTY': np.random.rand(12),
        'TEMPORAL_X': np.random.rand(12),
        'TEMPORAL_Y': np.random.rand(12)
    })

    trained_model = train_baseline_vae(df, model, epochs=2)
    assert isinstance(trained_model, torch.nn.Module)


def test_get_household_profile() -> None:
    """Test extracting household latent profile."""
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    df = pd.DataFrame({
        'cat_1_SPEND': [0.5, 0.6],
        'cat_1_QTY': [1.0, 1.1],
        'TEMPORAL_1': [0.1, 0.2],
        'TEMPORAL_2': [0.3, 0.4]
    })

    profile = get_household_profile(model, df)
    assert profile.shape == (4,)

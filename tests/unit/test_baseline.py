import numpy as np
import pandas as pd
import pytest
import torch

from src.models.vae import build_vae_model
from src.services.baseline import get_household_profile, train_baseline_vae, vae_loss


def test_vae_loss() -> None:
    recon_x = torch.tensor([[0.5, 0.5]])
    x = torch.tensor([[0.5, 0.5]])
    mu = torch.tensor([[0.0, 0.0]])
    logvar = torch.tensor([[0.0, 0.0]])

    loss = vae_loss(recon_x, x, mu, logvar)
    assert loss.item() == 0.0

def test_train_baseline_insufficient_data() -> None:
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    df = pd.DataFrame({
        'household_id': ['h1', 'h1'],
        'cat_1': [1.0, 2.0],
        'cat_2': [0.5, 0.5],
        'month_of_year': [1, 2],
        'week_of_year': [1, 5]
    })

    with pytest.raises(ValueError, match="Insufficient historical data"):
        train_baseline_vae(df, model, min_transactions=5)

def test_train_baseline_success() -> None:
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    h_ids = ['h1'] * 6 + ['h2'] * 6
    df = pd.DataFrame({
        'household_id': h_ids,
        'cat_1': np.random.rand(12),
        'cat_2': np.random.rand(12),
        'month_of_year': np.random.randint(1, 12, 12),
        'week_of_year': np.random.randint(1, 52, 12)
    })

    trained_model = train_baseline_vae(df, model, epochs=2, min_transactions=5)
    assert isinstance(trained_model, torch.nn.Module)

def test_get_household_profile() -> None:
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    df = pd.DataFrame({
        'household_id': ['h1'],
        'cat_1': [1.0],
        'cat_2': [0.5],
        'month_of_year': [1],
        'week_of_year': [1]
    })

    profile = get_household_profile(model, df)
    assert profile.shape == (4,)

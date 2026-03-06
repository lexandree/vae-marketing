import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    """Calculates the combined Reconstruction + KL divergence loss."""
    # Using MSE for reconstruction, can be configured based on data scaling
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss + kl_loss


def train_baseline_vae(
    data: pd.DataFrame,
    model: nn.Module,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3
) -> nn.Module:
    """Trains the VAE on historical purchase data without stimuli.

    Args:
        data: DataFrame containing transactions (needs to be grouped/aggregated per household).
        model: The VAE model instance.
        epochs: Number of training epochs.
        batch_size: Size of training batches.
        lr: Learning rate.

    Returns:
        The trained VAE model.
    """
    category_cols = [c for c in data.columns if c.endswith("_SPEND") or c.endswith("_QTY")]
    temporal_cols = [c for c in data.columns if c.startswith("TEMPORAL_")]

    if not category_cols:
        raise ValueError("No spend/qty features found in data for training.")

    x_tensor = torch.tensor(data[category_cols].values, dtype=torch.float32)
    t_tensor = (
        torch.tensor(data[temporal_cols].values, dtype=torch.float32)
        if temporal_cols
        else torch.zeros((len(data), 1))
    )

    dataset = TensorDataset(x_tensor, t_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_t in dataloader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_t)
            loss = vae_loss(recon_x, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    return model


def get_household_profile(model: nn.Module, household_data: pd.DataFrame) -> np.ndarray:
    """Extracts the mean latent representation (mu) for a given household's history.

    Args:
        model: The trained VAE model.
        household_data: DataFrame containing history for a single household.

    Returns:
        Numpy array representing the latent profile.
    """
    model.eval()

    category_cols = [
        c for c in household_data.columns if c.endswith("_SPEND") or c.endswith("_QTY")
    ]
    temporal_cols = [c for c in household_data.columns if c.startswith("TEMPORAL_")]

    x_tensor = torch.tensor(household_data[category_cols].values, dtype=torch.float32)
    t_tensor = (
        torch.tensor(household_data[temporal_cols].values, dtype=torch.float32)
        if temporal_cols
        else torch.zeros((len(household_data), 1))
    )

    with torch.no_grad():
        mu, _ = model.encode(x_tensor, t_tensor)

    # Average the latent vectors across all time points to get a stable profile
    return mu.mean(dim=0).numpy()

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Calculates the combined Reconstruction + KL divergence loss."""
    # Using MSE for reconstruction, can be configured based on data scaling
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

def train_baseline_vae(
    data: pd.DataFrame, 
    model: nn.Module, 
    epochs: int = 50, 
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """Trains the VAE on historical purchase data without stimuli.
    
    Args:
        data: DataFrame containing transactions (needs to be grouped/aggregated per household).
        model: The VAE model.
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        learning_rate: Optimizer learning rate.

    Returns:
        The trained VAE model.
    """
    logger.info(f"Training on {len(data)} windows.")

    category_cols = [c for c in data.columns if c.endswith('_SPEND') or c.endswith('_QTY')]
    temporal_cols = [c for c in data.columns if c.startswith('TEMPORAL_')]

    if not category_cols:
        raise ValueError("No category columns found (expected suffix '_SPEND' or '_QTY').")

    x_tensor = torch.tensor(data[category_cols].values, dtype=torch.float32)
    t_tensor = torch.tensor(data[temporal_cols].values, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, t_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataset):.4f}")

    return model

def get_household_profile(model: nn.Module, household_data: pd.DataFrame) -> np.ndarray:
    """Returns the latent representation (mu vector) for a given household."""
    model.eval()

    category_cols = [c for c in household_data.columns if c.endswith('_SPEND') or c.endswith('_QTY')]
    temporal_cols = [c for c in household_data.columns if c.startswith('TEMPORAL_')]

    if not category_cols or not temporal_cols:
        raise ValueError("Household data missing required category or temporal columns.")

    x_tensor = torch.tensor(household_data[category_cols].values, dtype=torch.float32)
    t_tensor = torch.tensor(household_data[temporal_cols].values, dtype=torch.float32)

    with torch.no_grad():
        mu, _ = model.encode(x_tensor, t_tensor)

    # Return the mean latent representation if multiple rows
    return mu.mean(dim=0).numpy()

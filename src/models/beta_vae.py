from typing import Tuple

import torch
from torch import nn


class BetaVAE(nn.Module):
    """Feedforward Beta-Variational Autoencoder."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_categories: int = 10,
        num_temporal_features: int = 6
    ) -> None:
        """Initializes the Beta-VAE model."""
        super().__init__()

        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.num_temporal_features = num_temporal_features

        input_dim = num_categories + num_temporal_features

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder (receives latent z + temporal features)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_temporal_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_categories)
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes inputs into latent space parameters."""
        inputs = torch.cat([x, t], dim=-1)
        h1 = self.encoder(inputs)
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector into reconstruction."""
        inputs = torch.cat([z, t], dim=-1)
        return self.decoder(inputs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass."""
        mu, logvar = self.encode(x, t)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, t)
        return recon_x, mu, logvar

    @staticmethod
    def get_beta(current_epoch: int, total_anneal_epochs: int, target_beta: float) -> float:
        """Calculate current beta for linear annealing schedule."""
        if total_anneal_epochs <= 0:
            return target_beta
        if current_epoch >= total_anneal_epochs:
            return target_beta
        return target_beta * (current_epoch / total_anneal_epochs)


def beta_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    use_gkl: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Beta-VAE loss (MSE + β * KL)."""
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    if use_gkl:
        # Experimental Generalized KL divergence proxy for long-tail transaction data
        epsilon = 1e-4
        gkl_div = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - (logvar.exp() + epsilon).pow(0.8), dim=1
        )
        kl_loss = torch.mean(gkl_div)
    else:
        # Standard KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_div)

    total_loss = mse_loss + beta * kl_loss
    return total_loss, mse_loss, kl_loss

def build_beta_vae_model(
    latent_dim: int = 32,
    num_categories: int = 10,
    num_temporal_features: int = 6
) -> nn.Module:
    """Constructs the feedforward Beta-Variational Autoencoder."""
    return BetaVAE(latent_dim, num_categories, num_temporal_features)

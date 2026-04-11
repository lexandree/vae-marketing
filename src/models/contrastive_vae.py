"""Contrastive VAE for campaign-salient behavioral variation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ContrastiveVAE(nn.Module):
    """Feedforward Contrastive VAE with shared and salient latent spaces.

    `shared` captures general household behavior.
    `salient` captures target-specific variation that should separate treated
    campaign behavior from the broader background cohort.
    """

    def __init__(
        self,
        shared_dim: int = 16,
        salient_dim: int = 16,
        num_categories: int = 10,
        num_temporal_features: int = 6,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.salient_dim = salient_dim
        self.latent_dim = shared_dim + salient_dim
        self.num_categories = num_categories
        self.num_temporal_features = num_temporal_features

        input_dim = num_categories + num_temporal_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.shared_mu = nn.Linear(128, shared_dim)
        self.shared_logvar = nn.Linear(128, shared_dim)
        self.salient_mu = nn.Linear(128, salient_dim)
        self.salient_logvar = nn.Linear(128, salient_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + num_temporal_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_categories),
        )

    def encode_components(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode inputs into shared/salient Gaussian parameters."""
        inputs = torch.cat([x, t], dim=-1)
        hidden = self.encoder(inputs)
        return (
            self.shared_mu(hidden),
            self.shared_logvar(hidden),
            self.salient_mu(hidden),
            self.salient_logvar(hidden),
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility encoder returning concatenated shared+salient latents."""
        shared_mu, shared_logvar, salient_mu, salient_logvar = self.encode_components(x, t)
        return (
            torch.cat([shared_mu, salient_mu], dim=-1),
            torch.cat([shared_logvar, salient_logvar], dim=-1),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors from Gaussian parameters."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, shared_z: torch.Tensor, salient_z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Decode the concatenated shared and salient representation."""
        inputs = torch.cat([shared_z, salient_z, t], dim=-1)
        return self.decoder(inputs)

    def forward_target(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for target examples using both shared and salient codes."""
        shared_mu, shared_logvar, salient_mu, salient_logvar = self.encode_components(x, t)
        shared_z = self.reparameterize(shared_mu, shared_logvar)
        salient_z = self.reparameterize(salient_mu, salient_logvar)
        recon_x = self.decode(shared_z, salient_z, t)
        return recon_x, shared_mu, shared_logvar, salient_mu, salient_logvar

    def forward_background(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for background examples with salient code suppressed."""
        shared_mu, shared_logvar, salient_mu, salient_logvar = self.encode_components(x, t)
        shared_z = self.reparameterize(shared_mu, shared_logvar)
        salient_z = torch.zeros_like(salient_mu)
        recon_x = self.decode(shared_z, salient_z, t)
        return recon_x, shared_mu, shared_logvar, salient_mu, salient_logvar


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-sample KL divergence between q(z|x) and the unit Gaussian prior."""
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def contrastive_vae_loss(
    *,
    target_recon: torch.Tensor,
    target_x: torch.Tensor,
    target_shared_mu: torch.Tensor,
    target_shared_logvar: torch.Tensor,
    target_salient_mu: torch.Tensor,
    target_salient_logvar: torch.Tensor,
    background_recon: torch.Tensor,
    background_x: torch.Tensor,
    background_shared_mu: torch.Tensor,
    background_shared_logvar: torch.Tensor,
    background_salient_mu: torch.Tensor,
    background_salient_logvar: torch.Tensor,
    beta_shared: float = 1.0,
    beta_salient: float = 1.0,
    salient_background_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute contrastive-VAE loss with salient suppression on background data."""
    recon_target = nn.functional.mse_loss(target_recon, target_x, reduction="sum") / target_x.shape[0]
    recon_background = (
        nn.functional.mse_loss(background_recon, background_x, reduction="sum") / background_x.shape[0]
    )

    kl_target_shared = _kl_divergence(target_shared_mu, target_shared_logvar)
    kl_target_salient = _kl_divergence(target_salient_mu, target_salient_logvar)
    kl_background_shared = _kl_divergence(background_shared_mu, background_shared_logvar)

    # Penalize salient activity on background examples to keep this subspace
    # focused on target-specific variation.
    salient_background_penalty = torch.mean(
        background_salient_mu.pow(2) + (background_salient_logvar.exp() - 1.0).pow(2)
    )

    total_loss = (
        recon_target
        + recon_background
        + beta_shared * (kl_target_shared + kl_background_shared)
        + beta_salient * kl_target_salient
        + salient_background_weight * salient_background_penalty
    )
    metrics = {
        "recon_target": float(recon_target.detach().cpu()),
        "recon_background": float(recon_background.detach().cpu()),
        "kl_target_shared": float(kl_target_shared.detach().cpu()),
        "kl_target_salient": float(kl_target_salient.detach().cpu()),
        "kl_background_shared": float(kl_background_shared.detach().cpu()),
        "salient_background_penalty": float(salient_background_penalty.detach().cpu()),
    }
    return total_loss, metrics


def build_contrastive_vae_model(
    shared_dim: int = 16,
    salient_dim: int = 16,
    num_categories: int = 10,
    num_temporal_features: int = 6,
) -> nn.Module:
    """Construct the feedforward Contrastive VAE."""
    return ContrastiveVAE(
        shared_dim=shared_dim,
        salient_dim=salient_dim,
        num_categories=num_categories,
        num_temporal_features=num_temporal_features,
    )

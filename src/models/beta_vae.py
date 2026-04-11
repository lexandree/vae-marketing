from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


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
    """Calculate Beta-VAE loss (MSE + β * KL).
    
    Uses per-sample normalization to maintain balance regardless of feature count.
    """
    # MSE sum per sample, then averaged over batch
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]

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


def _log_density_gaussian(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Elementwise log-density of a diagonal Gaussian.

    This helper is used by the Beta-TCVAE objective to estimate:
    - `q(z|x)` for the sampled latent,
    - the aggregated posterior `q(z)`,
    - the factorized marginals `prod_j q(z_j)`.
    """
    normalization = -0.5 * (torch.log(torch.tensor(2.0 * torch.pi, device=z.device)) + logvar)
    inv_var = torch.exp(-logvar)
    return normalization - 0.5 * ((z - mu) ** 2) * inv_var


def beta_tcvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    dataset_size: int,
    alpha: float = 1.0,
    lambda_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the Beta-TCVAE loss via KL decomposition.

    The KL term is decomposed into three interpretable pieces:
    - mutual information `I(x; z)`
    - total correlation `TC(z)`
    - dimension-wise KL to the prior

    We keep the backbone identical to `BetaVAE` and only swap the objective,
    which makes comparison against `beta_vae` substantially cleaner.
    """
    batch_size = x.shape[0]
    if batch_size <= 1:
        raise ValueError("Beta-TCVAE requires batch_size > 1 for KL decomposition estimates.")

    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

    log_q_zx = _log_density_gaussian(z, mu, logvar).sum(dim=1)

    z_expanded = z.unsqueeze(1)
    mu_expanded = mu.unsqueeze(0)
    logvar_expanded = logvar.unsqueeze(0)
    mat_log_q_z = _log_density_gaussian(z_expanded, mu_expanded, logvar_expanded)

    normalizer = torch.log(torch.tensor(float(max(dataset_size, batch_size)), device=z.device))
    log_q_z = torch.logsumexp(mat_log_q_z.sum(dim=2), dim=1) - normalizer
    log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1).sum(dim=1) - z.shape[1] * normalizer

    log_p_z = _log_density_gaussian(
        z,
        torch.zeros_like(z),
        torch.zeros_like(z),
    ).sum(dim=1)

    mutual_information = torch.mean(log_q_zx - log_q_z)
    total_correlation = torch.mean(log_q_z - log_prod_q_z)
    dimension_wise_kl = torch.mean(log_prod_q_z - log_p_z)

    total_loss = (
        recon_loss
        + alpha * mutual_information
        + beta * total_correlation
        + lambda_weight * dimension_wise_kl
    )
    kl_loss = mutual_information + total_correlation + dimension_wise_kl
    return (
        total_loss,
        recon_loss,
        kl_loss,
        mutual_information,
        total_correlation,
        dimension_wise_kl,
    )

def build_beta_vae_model(
    latent_dim: int = 32,
    num_categories: int = 10,
    num_temporal_features: int = 6
) -> nn.Module:
    """Constructs the feedforward Beta-Variational Autoencoder."""
    return BetaVAE(latent_dim, num_categories, num_temporal_features)

import torch
from torch import nn


class VAE(nn.Module):
    """Feedforward Variational Autoencoder with temporal auxiliary inputs."""

    def __init__(self, latent_dim: int = 16, num_categories: int = 10, num_temporal_features: int = 2) -> None:
        """Initializes the VAE.

        Args:
            latent_dim: Dimension of the latent space (mu and logvar).
            num_categories: Number of product categories (input features).
            num_temporal_features: Number of temporal context features (e.g., month, week).
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.num_temporal_features = num_temporal_features

        input_dim = num_categories + num_temporal_features

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder (receives latent z + temporal features)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_temporal_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_categories),
            # Output is scaled/normalized values per category, could use Sigmoid if inputs are [0,1]
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input data and temporal features into latent parameters."""
        inputs = torch.cat([x, t], dim=-1)
        h1 = self.encoder(inputs)
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Decodes the latent vector and temporal features into reconstruction."""
        inputs = torch.cat([z, t], dim=-1)
        return self.decoder(inputs)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass of the VAE."""
        mu, logvar = self.encode(x, t)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, t)
        return recon_x, mu, logvar

def build_vae_model(latent_dim: int = 16, num_categories: int = 10, num_temporal_features: int = 2) -> nn.Module:
    """Constructs the feedforward Variational Autoencoder."""
    return VAE(latent_dim, num_categories, num_temporal_features)

import torch

from src.models.baseline_vae import build_vae_model


def test_vae_model_forward() -> None:
    """Test standard VAE model forward pass."""
    latent_dim = 8
    num_categories = 5
    num_temporal = 2
    model = build_vae_model(
        latent_dim=latent_dim,
        num_categories=num_categories,
        num_temporal_features=num_temporal
    )

    batch_size = 4
    x = torch.randn(batch_size, num_categories)
    t = torch.randn(batch_size, num_temporal)

    recon, mu, logvar = model(x, t)

    assert recon.shape == (batch_size, num_categories)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

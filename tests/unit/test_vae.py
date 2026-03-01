import torch

from src.models.vae import build_vae_model


def test_vae_model_forward() -> None:
    latent_dim = 8
    num_categories = 5
    num_temporal = 2
    model = build_vae_model(latent_dim=latent_dim, num_categories=num_categories, num_temporal_features=num_temporal)

    batch_size = 4
    x = torch.rand(batch_size, num_categories)
    t = torch.rand(batch_size, num_temporal)

    recon_x, mu, logvar = model(x, t)

    assert recon_x.shape == (batch_size, num_categories)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

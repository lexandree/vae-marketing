import torch

from src.models.beta_vae import BetaVAE, beta_tcvae_loss, beta_vae_loss, build_beta_vae_model
from src.models.factory import ModelFactory


def test_beta_vae_forward() -> None:
    """Test Beta-VAE forward pass."""
    model = BetaVAE(latent_dim=4, num_categories=2, num_temporal_features=2)
    x = torch.rand(4, 2)
    t = torch.rand(4, 2)
    recon_x, mu, logvar = model(x, t)

    assert recon_x.shape == (4, 2)
    assert mu.shape == (4, 4)
    assert logvar.shape == (4, 4)


def test_beta_vae_loss() -> None:
    """Test Beta-VAE loss calculation."""
    recon_x = torch.tensor([[0.5, 0.5]])
    x = torch.tensor([[0.5, 0.5]])
    mu = torch.tensor([[0.0, 0.0]])
    logvar = torch.tensor([[0.0, 0.0]])

    loss, mse, kl = beta_vae_loss(recon_x, x, mu, logvar, beta=1.0)
    assert loss.item() == 0.0
    assert mse.item() == 0.0
    assert kl.item() == 0.0


def test_beta_annealing() -> None:
    """Test Beta annealing schedule."""
    model = BetaVAE(latent_dim=4, num_categories=2, num_temporal_features=2)
    beta = model.get_beta(current_epoch=5, total_anneal_epochs=10, target_beta=1.0)
    assert beta == 0.5

    beta_final = model.get_beta(current_epoch=15, total_anneal_epochs=10, target_beta=1.0)
    assert beta_final == 1.0


def test_build_beta_vae_model() -> None:
    """Test Beta-VAE model builder."""
    model = build_beta_vae_model(latent_dim=8, num_categories=10, num_temporal_features=4)
    assert isinstance(model, torch.nn.Module)
    assert model.latent_dim == 8


def test_beta_tcvae_loss_returns_finite_components() -> None:
    """Beta-TCVAE loss should return finite KL decomposition terms."""
    x = torch.rand(4, 3)
    recon_x = torch.rand(4, 3)
    mu = torch.zeros(4, 2)
    logvar = torch.zeros(4, 2)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    loss, mse, kl, mi, tc, dw_kl = beta_tcvae_loss(
        recon_x=recon_x,
        x=x,
        z=z,
        mu=mu,
        logvar=logvar,
        beta=2.0,
        dataset_size=16,
    )

    for value in (loss, mse, kl, mi, tc, dw_kl):
        assert torch.isfinite(value)


def test_model_factory_supports_beta_tcvae() -> None:
    """Factory should create a BetaVAE backbone for beta_tcvae architecture."""
    model = ModelFactory.create_model(
        {
            "arch": "beta_tcvae",
            "latent_dim": 6,
            "num_categories": 4,
            "num_temporal_features": 2,
        }
    )
    assert isinstance(model, BetaVAE)
    assert model.latent_dim == 6

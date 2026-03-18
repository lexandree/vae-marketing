import torch

from src.models.beta_vae import BetaVAE, beta_vae_loss, build_beta_vae_model


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

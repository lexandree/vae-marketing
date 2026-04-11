from pathlib import Path

import polars as pl
import torch

from src.data.contrastive_dataset import build_contrastive_dataset
from src.models.contrastive_vae import ContrastiveVAE, contrastive_vae_loss
from src.models.factory import ModelFactory


def test_contrastive_vae_encode_and_forward_shapes() -> None:
    model = ContrastiveVAE(shared_dim=3, salient_dim=2, num_categories=4, num_temporal_features=2)
    x = torch.randn(5, 4)
    t = torch.randn(5, 2)

    mu, logvar = model.encode(x, t)
    assert mu.shape == (5, 5)
    assert logvar.shape == (5, 5)

    target_recon, shared_mu, shared_logvar, salient_mu, salient_logvar = model.forward_target(x, t)
    assert target_recon.shape == (5, 4)
    assert shared_mu.shape == (5, 3)
    assert salient_mu.shape == (5, 2)

    background_recon, *_ = model.forward_background(x, t)
    assert background_recon.shape == (5, 4)

    loss, metrics = contrastive_vae_loss(
        target_recon=target_recon,
        target_x=x,
        target_shared_mu=shared_mu,
        target_shared_logvar=shared_logvar,
        target_salient_mu=salient_mu,
        target_salient_logvar=salient_logvar,
        background_recon=background_recon,
        background_x=x,
        background_shared_mu=shared_mu,
        background_shared_logvar=shared_logvar,
        background_salient_mu=salient_mu,
        background_salient_logvar=salient_logvar,
    )
    assert loss.ndim == 0
    assert "recon_target" in metrics


def test_factory_creates_contrastive_model() -> None:
    model = ModelFactory.create_model(
        {
            "arch": "contrastive_vae",
            "latent_dim": 6,
            "shared_dim": 4,
            "salient_dim": 2,
            "num_categories": 8,
            "num_temporal_features": 6,
        }
    )
    assert isinstance(model, ContrastiveVAE)
    assert model.latent_dim == 6


def test_build_contrastive_dataset_writes_target_and_background(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.parquet"
    campaign_table = tmp_path / "campaign_table.csv"
    campaign_desc = tmp_path / "campaign_desc.csv"
    output_dir = tmp_path / "contrastive"

    pl.DataFrame(
        {
            "HOUSEHOLD_KEY": ["1", "2", "3"],
            "WINDOW_START_DAY": [14, 14, 14],
            "COMMODITY_A_SPEND": [1.0, 0.5, 0.0],
            "COMMODITY_A_QTY": [1.0, 1.0, 0.0],
            "TEMPORAL_WEEK_SIN": [0.0, 0.0, 0.0],
            "TEMPORAL_WEEK_COS": [1.0, 1.0, 1.0],
        }
    ).write_parquet(prepared)
    pl.DataFrame(
        {
            "DESCRIPTION": ["X"],
            "household_key": [1],
            "CAMPAIGN": [8],
        }
    ).write_csv(campaign_table)
    pl.DataFrame(
        {
            "DESCRIPTION": ["X"],
            "CAMPAIGN": [8],
            "START_DAY": [14],
            "END_DAY": [20],
        }
    ).write_csv(campaign_desc)

    build_contrastive_dataset(
        prepared_data=prepared,
        campaign_table_path=campaign_table,
        campaign_desc_path=campaign_desc,
        output_dir=output_dir,
        background_ratio=1.0,
        seed=42,
    )

    target = pl.read_parquet(output_dir / "target.parquet")
    background = pl.read_parquet(output_dir / "background.parquet")
    assert target.height == 1
    assert background.height >= 1
    assert target["contrastive_role"].to_list() == ["target"]

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from main import main
from src.models.baseline_vae import build_vae_model
from src.models.contrastive_vae import ContrastiveVAE
from src.models.factory import ModelFactory


def test_validate_latents_cli_generates_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = tmp_path / "analysis.parquet"
    attributes_path = tmp_path / "attributes.parquet"

    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "WINDOW_START_DAY": [7, 14, 21, 28],
            "COMMODITY_A_SPEND": [1.0, 0.5, -0.5, -1.0],
            "COMMODITY_A_QTY": [1.0, 0.5, -0.5, -1.0],
            "TEMPORAL_WEEK_SIN": [0.0, 0.5, -0.5, 0.0],
            "TEMPORAL_WEEK_COS": [1.0, 0.5, 0.5, 1.0],
        }
    ).to_parquet(analysis_path, index=False)
    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "WINDOW_START_DAY": [7, 14, 21, 28],
            "promo_share": [1.0, 0.5, -0.5, -1.0],
            "trip_count": [2.0, 1.5, 0.5, 0.0],
        }
    ).to_parquet(attributes_path, index=False)

    run_id = "latent-cli-test"
    run_dir = tmp_path / "experiments" / run_id
    run_dir.mkdir(parents=True)
    config = {
        "arch": "baseline",
        "latent_dim": 2,
        "num_categories": 2,
        "num_temporal_features": 2,
    }
    ModelFactory.save_config(config, run_dir)
    model = build_vae_model(latent_dim=2, num_categories=2, num_temporal_features=2)
    torch.save(model.state_dict(), run_dir / "best_model.pth")

    out_dir = tmp_path / "latent_validation"
    test_args = [
        "main.py",
        "validate-latents",
        "--analysis-data",
        str(analysis_path),
        "--attributes",
        str(attributes_path),
        "--run-ids",
        str(run_dir),
        "--output-dir",
        str(out_dir),
        "--mig-method",
        "binned",
        "--mig-bins",
        "8",
        "--mig-binning",
        "quantile",
        "--sap-method",
        "vectorized",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert (out_dir / "latent_metrics.json").exists()
    assert (out_dir / "factor_mappings.json").exists()
    assert (out_dir / "latent_stability.json").exists()

    with open(out_dir / "latent_metrics.json", "r") as f:
        metrics = json.load(f)
    with open(out_dir / "factor_mappings.json", "r") as f:
        mappings = json.load(f)
    with open(out_dir / "latent_stability.json", "r") as f:
        stability = json.load(f)

    assert metrics
    assert mappings
    assert stability


def test_validate_latents_cli_supports_salient_mode_for_contrastive_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = tmp_path / "analysis.parquet"
    attributes_path = tmp_path / "attributes.parquet"

    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "WINDOW_START_DAY": [7, 14, 21, 28],
            "COMMODITY_A_SPEND": [1.0, 0.5, -0.5, -1.0],
            "COMMODITY_A_QTY": [1.0, 0.5, -0.5, -1.0],
            "TEMPORAL_WEEK_SIN": [0.0, 0.5, -0.5, 0.0],
            "TEMPORAL_WEEK_COS": [1.0, 0.5, 0.5, 1.0],
        }
    ).to_parquet(analysis_path, index=False)
    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "WINDOW_START_DAY": [7, 14, 21, 28],
            "promo_share": [1.0, 0.5, -0.5, -1.0],
            "trip_count": [2.0, 1.5, 0.5, 0.0],
        }
    ).to_parquet(attributes_path, index=False)

    run_id = "contrastive-cli-test"
    run_dir = tmp_path / "experiments" / run_id
    run_dir.mkdir(parents=True)
    config = {
        "arch": "contrastive_vae",
        "latent_dim": 4,
        "shared_dim": 2,
        "salient_dim": 2,
        "num_categories": 2,
        "num_temporal_features": 2,
    }
    ModelFactory.save_config(config, run_dir)
    model = ContrastiveVAE(shared_dim=2, salient_dim=2, num_categories=2, num_temporal_features=2)
    torch.save(model.state_dict(), run_dir / "best_model.pth")

    out_dir = tmp_path / "latent_validation_salient"
    test_args = [
        "main.py",
        "validate-latents",
        "--analysis-data",
        str(analysis_path),
        "--attributes",
        str(attributes_path),
        "--run-ids",
        str(run_dir),
        "--output-dir",
        str(out_dir),
        "--latent-mode",
        "salient",
        "--mig-method",
        "binned",
        "--mig-bins",
        "8",
        "--mig-binning",
        "quantile",
        "--sap-method",
        "vectorized",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    with open(out_dir / "latent_metrics.json", "r") as f:
        metrics = json.load(f)
    with open(out_dir / "factor_mappings.json", "r") as f:
        mappings = json.load(f)

    assert metrics[0]["latent_mode"] == "salient"
    assert all(row["latent_mode"] == "salient" for row in mappings)

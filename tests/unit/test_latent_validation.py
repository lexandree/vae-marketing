import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.models.baseline_vae import build_vae_model
from src.models.contrastive_vae import ContrastiveVAE
from src.models.factory import ModelFactory
from src.services.latent_validation import (
    align_holdout_frames,
    extract_latent_snapshots,
    pivot_validation_attributes,
    validate_latent_factors,
    validate_latent_runs,
)


def test_pivot_validation_attributes() -> None:
    attributes_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 1, 2, 2],
            "CAMPAIGN": [18, 18, 18, 18],
            "window_type": ["campaign", "campaign", "campaign", "campaign"],
            "attribute_name": ["promo_share", "trip_count", "promo_share", "trip_count"],
            "attribute_value": [0.5, 2.0, 0.1, 1.0],
            "attribute_family": ["promotion", "frequency", "promotion", "frequency"],
        }
    )

    wide_df = pivot_validation_attributes(attributes_df)
    assert set(["HOUSEHOLD_KEY", "CAMPAIGN", "window_type", "promo_share", "trip_count"]).issubset(wide_df.columns)
    assert len(wide_df) == 2


def test_align_holdout_frames_on_available_keys() -> None:
    analysis_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2],
            "WINDOW_START_DAY": [7, 14],
            "COMMODITY_A_SPEND": [1.0, 2.0],
            "COMMODITY_A_QTY": [1.0, 1.0],
        }
    )
    attrs_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2],
            "WINDOW_START_DAY": [7, 14],
            "promo_share": [0.5, 0.1],
        }
    )

    merged = align_holdout_frames(analysis_df, attrs_df)
    assert len(merged) == 2
    assert "promo_share" in merged.columns


def test_validate_latent_runs_outputs_metrics_and_mappings(tmp_path: Path) -> None:
    analysis_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "WINDOW_START_DAY": [7, 14, 21, 28],
            "COMMODITY_A_SPEND": [1.0, 0.5, -0.5, -1.0],
            "COMMODITY_A_QTY": [1.0, 0.5, -0.5, -1.0],
            "TEMPORAL_WEEK_SIN": [0.0, 0.5, -0.5, 0.0],
            "TEMPORAL_WEEK_COS": [1.0, 0.5, 0.5, 1.0],
        }
    )
    attributes_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "WINDOW_START_DAY": [7, 14, 21, 28],
            "promo_share": [1.0, 0.5, -0.5, -1.0],
            "trip_count": [2.0, 1.5, 0.5, 0.0],
        }
    )

    run_dir = tmp_path / "baseline-test"
    run_dir.mkdir()
    config = {
        "arch": "baseline",
        "latent_dim": 2,
        "num_categories": 2,
        "num_temporal_features": 2,
    }
    ModelFactory.save_config(config, run_dir)
    model = build_vae_model(latent_dim=2, num_categories=2, num_temporal_features=2)
    torch.save(model.state_dict(), run_dir / "best_model.pth")

    metrics, mappings, stability = validate_latent_runs(
        analysis_df=analysis_df,
        attributes_df=attributes_df,
        run_dirs=[run_dir],
        mig_method="binned",
        mig_bins=8,
        mig_binning="quantile",
        sap_method="vectorized",
    )

    assert metrics
    assert mappings
    assert stability
    assert "mig_score" in metrics[0]
    assert "candidate_attribute" in mappings[0]


def test_validate_latent_factors_filters_to_requested_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analysis_path = tmp_path / "analysis.parquet"
    attributes_path = tmp_path / "attributes.parquet"
    split_path = tmp_path / "splits.json"

    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": ["H1", "H2", "H3"],
            "WINDOW_START_DAY": [7, 14, 21],
            "COMMODITY_A_SPEND": [1.0, 0.5, -0.5],
            "COMMODITY_A_QTY": [1.0, 0.5, -0.5],
            "TEMPORAL_WEEK_SIN": [0.0, 0.5, -0.5],
            "TEMPORAL_WEEK_COS": [1.0, 0.5, 0.5],
        }
    ).to_parquet(analysis_path, index=False)
    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": ["H1", "H2", "H3"],
            "WINDOW_START_DAY": [7, 14, 21],
            "promo_share": [1.0, 0.5, -0.5],
            "trip_count": [2.0, 1.5, 0.5],
        }
    ).to_parquet(attributes_path, index=False)
    split_path.write_text(
        json.dumps(
            {
                "eval_campaign_ids": [26, 30],
                "eval_households": ["H2"],
                "train_households": ["H1"],
                "validation_households": ["H3"],
                "metadata": {"split_seed": 42},
            }
        )
    )

    captured = {}

    def fake_validate_latent_runs(**kwargs):
        captured["analysis_households"] = kwargs["analysis_df"]["HOUSEHOLD_KEY"].tolist()
        captured["attribute_households"] = kwargs["attributes_df"]["HOUSEHOLD_KEY"].tolist()
        return [], [], []

    def fake_serialize(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "src.services.latent_validation.validate_latent_runs",
        fake_validate_latent_runs,
    )
    monkeypatch.setattr(
        "src.services.latent_validation.serialize_latent_validation_outputs",
        fake_serialize,
    )

    args = SimpleNamespace(
        analysis_data=analysis_path,
        attributes=attributes_path,
        run_ids=[],
        output_dir=tmp_path / "out",
        household_splits=split_path,
        split_role="eval",
        mig_method="binned",
        mig_bins=8,
        mig_binning="quantile",
        sap_method="vectorized",
    )

    validate_latent_factors(args=args)

    assert captured["analysis_households"] == ["H2"]
    assert captured["attribute_households"] == ["H2"]


def test_extract_latent_snapshots_supports_salient_mode() -> None:
    model = ContrastiveVAE(shared_dim=2, salient_dim=3, num_categories=2, num_temporal_features=2)
    analysis_df = pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2],
            "WINDOW_START_DAY": [7, 14],
            "COMMODITY_A_SPEND": [1.0, 0.5],
            "COMMODITY_A_QTY": [1.0, 0.5],
            "TEMPORAL_WEEK_SIN": [0.0, 0.5],
            "TEMPORAL_WEEK_COS": [1.0, 0.5],
        }
    )

    shared_df = extract_latent_snapshots(model, analysis_df, latent_mode="shared")
    salient_df = extract_latent_snapshots(model, analysis_df, latent_mode="salient")
    combined_df = extract_latent_snapshots(model, analysis_df, latent_mode="combined")

    assert len([c for c in shared_df.columns if c.startswith("latent_")]) == 2
    assert len([c for c in salient_df.columns if c.startswith("latent_")]) == 3
    assert len([c for c in combined_df.columns if c.startswith("latent_")]) == 5

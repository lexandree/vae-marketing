import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.services.campaign_sensitivity import run_campaign_sensitivity


def test_run_campaign_sensitivity_writes_grid_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_load_validation_source(path: Path, dataset_type: str) -> pd.DataFrame:
        return pd.DataFrame({"stub": [1]})

    def fake_build_campaign_analysis_dataset(**kwargs):
        analysis_df = pd.DataFrame(
            {
                "HOUSEHOLD_KEY": [1, 2],
                "CAMPAIGN": [26, 26],
                "treatment_status": ["treated", "comparison"],
                "eligibility_status": ["eligible", "eligible"],
                "pre_total_spend": [10.0, 10.1],
                "campaign_total_spend": [15.0, 10.5],
                "post_total_spend": [11.0, 10.2],
            }
        )
        return analysis_df, pd.DataFrame(), pd.DataFrame(), {
            "analysis_records": 2,
            "excluded_records": 0,
        }

    def fake_compute_campaign_effects(**kwargs):
        effects_df = pd.DataFrame(
            [
                {
                    "CAMPAIGN": 26,
                    "outcome_name": "total_spend",
                    "effect_size": 4.0,
                    "effect_direction": "increase",
                    "evidence_classification": "supported",
                }
            ]
        )
        diagnostics_df = pd.DataFrame(
            [
                {
                    "CAMPAIGN": 26,
                    "outcome_name": "total_spend",
                    "balance_pass": True,
                    "placebo_pass": True,
                    "placebo_effect": 0.5,
                }
            ]
        )
        balance_df = pd.DataFrame(
            [
                {
                    "CAMPAIGN": 26,
                    "outcome_name": "total_spend",
                    "covariate_name": "pre_total_spend",
                    "standardized_mean_difference": 0.03,
                }
            ]
        )
        cohort_df = pd.DataFrame(
            [
                {
                    "CAMPAIGN": 26,
                    "matched_treated_size": 10,
                    "matched_comparison_size": 10,
                    "matched_retention_rate": 0.8,
                }
            ]
        )
        return effects_df, diagnostics_df, pd.DataFrame(), balance_df, cohort_df

    monkeypatch.setattr(
        "src.services.campaign_sensitivity.load_validation_source",
        fake_load_validation_source,
    )
    monkeypatch.setattr(
        "src.services.campaign_sensitivity.build_campaign_analysis_dataset",
        fake_build_campaign_analysis_dataset,
    )
    monkeypatch.setattr(
        "src.services.campaign_sensitivity.compute_campaign_effects",
        fake_compute_campaign_effects,
    )

    args = SimpleNamespace(
        transactions=tmp_path / "transactions.csv",
        products=tmp_path / "products.csv",
        campaign_table=tmp_path / "campaign_table.csv",
        campaign_desc=tmp_path / "campaign_desc.csv",
        coupon=tmp_path / "coupon.csv",
        coupon_redempt=tmp_path / "coupon_redempt.csv",
        demographics=None,
        campaign_ids=[26],
        weeks_grid=[2, 4],
        output_dir=tmp_path / "out",
        outcomes=["total_spend"],
        matching_methods=["propensity"],
        propensity_calipers=[0.02],
        min_treated=1,
        min_comparison=1,
    )

    run_campaign_sensitivity(args=args)

    json_path = args.output_dir / "campaign_sensitivity.json"
    parquet_path = args.output_dir / "campaign_sensitivity.parquet"
    assert json_path.exists()
    assert parquet_path.exists()

    payload = json.loads(json_path.read_text())
    assert len(payload) == 2
    assert {row["weeks"] for row in payload} == {2, 4}


import json
from pathlib import Path

import polars as pl
import pytest

from main import main


def test_validate_campaigns_cli_generates_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_data = tmp_path / "campaign_analysis.parquet"
    pl.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 2],
            "CAMPAIGN": [18, 18],
            "treatment_status": ["treated", "comparison"],
            "pre_total_spend": [10.0, 8.0],
            "campaign_total_spend": [18.0, 9.0],
            "post_total_spend": [11.0, 7.0],
            "pre_trip_count": [1.0, 1.0],
            "campaign_trip_count": [2.0, 1.0],
            "post_trip_count": [1.0, 1.0],
            "pre_category_diversity": [1.0, 1.0],
            "campaign_category_diversity": [2.0, 1.0],
            "post_category_diversity": [1.0, 1.0],
            "pre_promo_share": [0.0, 0.0],
            "campaign_promo_share": [0.5, 0.0],
            "post_promo_share": [0.0, 0.0],
        }
    ).write_parquet(analysis_data)

    out_dir = tmp_path / "campaign_validation"
    test_args = [
        "main.py",
        "validate-campaigns",
        "--analysis-data",
        str(analysis_data),
        "--campaign-ids",
        "18",
        "--method",
        "matched_did",
        "--output-dir",
        str(out_dir),
        "--min-treated",
        "1",
        "--min-comparison",
        "1",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert (out_dir / "campaign_effects.json").exists()
    assert (out_dir / "campaign_diagnostics.json").exists()
    assert (out_dir / "campaign_event_study.parquet").exists()

    with open(out_dir / "campaign_effects.json", "r") as f:
        effects = json.load(f)
    with open(out_dir / "campaign_diagnostics.json", "r") as f:
        diagnostics = json.load(f)

    event_study_df = pl.read_parquet(out_dir / "campaign_event_study.parquet")
    assert effects
    assert diagnostics
    assert event_study_df.height > 0


def test_generate_validation_report_cli_writes_report_and_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_results = tmp_path / "campaign_effects.json"
    campaign_diagnostics = tmp_path / "campaign_diagnostics.json"
    latent_results = tmp_path / "factor_mappings.json"
    latent_metrics = tmp_path / "latent_metrics.json"
    report_path = tmp_path / "reports" / "validation_report.md"

    campaign_results.write_text(
        json.dumps(
            [
                {"CAMPAIGN": 18, "outcome_name": "total_spend", "effect_size": 1.2, "evidence_classification": "supported"},
                {"CAMPAIGN": 13, "outcome_name": "promo_share", "effect_size": -0.3, "evidence_classification": "unsupported"},
            ]
        )
    )
    campaign_diagnostics.write_text(
        json.dumps(
            [
                {"CAMPAIGN": 18, "outcome_name": "total_spend", "balance_pass": True, "placebo_pass": True},
            ]
        )
    )
    latent_results.write_text(
        json.dumps(
            [
                {"run_id": "beta-best", "latent_dimension": 0, "candidate_attribute": "promo_share", "mapping_decision": "validated"},
                {"run_id": "beta-best", "latent_dimension": 1, "candidate_attribute": "trip_count", "mapping_decision": "rejected"},
            ]
        )
    )
    latent_metrics.write_text(
        json.dumps(
            [
                {"run_id": "beta-best", "model_type": "beta_vae", "mig_score": 0.2, "sap_score": 0.3},
            ]
        )
    )

    test_args = [
        "main.py",
        "generate-validation-report",
        "--campaign-results",
        str(campaign_results),
        "--campaign-diagnostics",
        str(campaign_diagnostics),
        "--latent-results",
        str(latent_results),
        "--latent-metrics",
        str(latent_metrics),
        "--output",
        str(report_path),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert report_path.exists()
    assert (report_path.parent / "claim_recommendations.json").exists()
    report_text = report_path.read_text()
    assert "supported" in report_text
    assert "unsupported" in report_text

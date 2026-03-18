import pandas as pd

from src.services.reporting import generate_aggregate_report


def test_polymorphic_reporting() -> None:
    """Test that the reporting dispatcher correctly handles different model types."""
    profiles_df = pd.DataFrame([
        {"household_id": "h1", "baseline_profile": [0.0, 0.0]},
        {"household_id": "h2", "baseline_profile": [0.1, 0.2]}
    ])

    shifts_df = pd.DataFrame([
        {
            "household_id": "h1", "stimulus_id": "s1",
            "quantitative_magnitude": 0.5, "qualitative_nature": "Trading Up",
            "persistence_duration_days": 10
        },
        {
            "household_id": "h2", "stimulus_id": "s1",
            "quantitative_magnitude": 0.1, "qualitative_nature": "No Change",
            "persistence_duration_days": 0
        }
    ])

    post_data = pd.DataFrame({
        "HOUSEHOLD_KEY": ["h1", "h2"],
        "COMMODITY_A_SPEND": [10.0, 5.0],
        "COMMODITY_A_QTY": [1, 1],
        "TEMPORAL_1": [0.1, 0.2]
    })

    # Test baseline reporting
    report_baseline = generate_aggregate_report(
        profiles_df, shifts_df, post_data, model_type="baseline"
    )
    assert "total_households_analyzed" in report_baseline

    # Test Beta-VAE reporting
    report_beta = generate_aggregate_report(
        profiles_df, shifts_df, post_data, model_type="beta_vae"
    )
    assert "total_households_analyzed" in report_beta
    assert "factor_breakdown" in report_beta

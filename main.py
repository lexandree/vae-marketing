"""Main entry point for the VAE Marketing Impact Analysis pipeline."""

import logging

import numpy as np
import pandas as pd

from src.models.vae import build_vae_model
from src.services.baseline import get_household_profile, train_baseline_vae
from src.services.impact_analysis import (
    analyze_persistence,
    calculate_deviation,
    categorize_shift,
)
from src.services.reporting import generate_aggregate_report
from src.utils.metrics import setup_logger
from src.utils.seed import set_global_seed

logger = setup_logger(__name__)


def run_pipeline() -> None:
    """Executes the full impact analysis pipeline on synthetic data for demonstration."""
    set_global_seed(42)

    # 1. Setup / Load Data
    logger.info("Generating demonstration dataset...")
    num_households = 20
    num_transactions = 200

    h_ids = [f"h_{i}" for i in range(num_households)] * (num_transactions // num_households)

    transactions = pd.DataFrame(
        {
            "household_id": h_ids,
            "cat_1": np.random.rand(num_transactions),
            "cat_2": np.random.rand(num_transactions),
            "price_tier": np.random.rand(num_transactions) * 5,
            "quantity": np.random.randint(1, 10, num_transactions),
            "month_of_year": np.random.randint(1, 12, num_transactions),
            "week_of_year": np.random.randint(1, 52, num_transactions),
            "timestamp": pd.date_range(start="2023-01-01", periods=num_transactions, freq="D"),
        }
    )

    # 2. Train Baseline VAE
    logger.info("Establishing behavioral baseline using VAE...")
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    trained_model = train_baseline_vae(transactions, model, epochs=20, min_transactions=5)

    # 3. Get Profiles
    logger.info("Extracting household latent profiles...")
    baseline_profiles = []
    for h_id in transactions["household_id"].unique():
        h_data = transactions[transactions["household_id"] == h_id]
        profile = get_household_profile(trained_model, h_data)
        baseline_profiles.append({"household_id": h_id, "baseline_profile": profile})
    profiles_df = pd.DataFrame(baseline_profiles)

    # 4. Generate Stimulus & Post-Stimulus Data
    logger.info("Simulating external marketing stimulus impact...")
    stimulus_end = pd.to_datetime("2023-04-01")
    post_data = transactions.copy()
    post_data["timestamp"] = post_data["timestamp"] + pd.Timedelta(days=100)
    # Ensure quantity is float to avoid lossy setitem error
    post_data["quantity"] = post_data["quantity"].astype(float)
    # Introduce a shift for half of the households
    affected = [f"h_{i}" for i in range(10)]
    post_data.loc[post_data["household_id"].isin(affected), "cat_1"] *= 4.0
    post_data.loc[post_data["household_id"].isin(affected), "quantity"] *= 2.5

    # 5. Measure Impact
    logger.info("Measuring behavioral deviation and persistence...")
    shift_results = []
    for h_id in transactions["household_id"].unique():
        base_prof = profiles_df[profiles_df["household_id"] == h_id]["baseline_profile"].iloc[0]
        h_post = post_data[post_data["household_id"] == h_id]
        h_base = transactions[transactions["household_id"] == h_id]

        dev = calculate_deviation(base_prof, h_post, trained_model)
        cat = categorize_shift(h_base, h_post)
        pers = analyze_persistence(h_post, stimulus_end, trained_model, base_prof)

        shift_results.append(
            {
                "household_id": h_id,
                "stimulus_id": "STIM_001",
                "quantitative_magnitude": dev,
                "qualitative_nature": cat,
                "persistence_duration_days": pers,
            }
        )

    shifts_df = pd.DataFrame(shift_results)

    # 6. Aggregate Reporting
    logger.info("Generating aggregate impact report...")
    report = generate_aggregate_report(profiles_df, shifts_df, post_data)

    # Output results
    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Total Households: {report['total_households_analyzed']}")
    print(f"Average Latent Deviation: {report['average_magnitude']:.4f}")
    print(f"Average Persistence: {report['average_persistence_days']:.1f} days")
    print("\nSegment Distribution:")
    for cluster, count in report["segments"].items():
        print(f"  Cluster {cluster}: {count} households")
    print("\nTop Sensitive Categories:")
    for cat in report["top_sensitive_categories"]:
        print(f"  {cat['product_category']}: Score {cat['sensitivity_score']:.2f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_pipeline()

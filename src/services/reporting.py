import pandas as pd

from src.services.reporting_baseline import (
    generate_aggregate_report as generate_baseline_report,
)
from src.services.reporting_beta import generate_aggregate_report as generate_beta_report


def generate_aggregate_report(
    profiles_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    transactions: pd.DataFrame,
    model_type: str = "baseline",
) -> dict:
    """Dispatcher for generating aggregate impact reports based on model architecture.

    Args:
        profiles_df: DataFrame of baseline profiles.
        shifts_df: DataFrame of calculated shifts.
        transactions: Post-stimulus transactions.
        model_type: The architecture used ('baseline' or 'beta_vae').

    Returns:
        A dictionary containing the report metrics.
    """
    if model_type == "beta_vae":
        return generate_beta_report(profiles_df, shifts_df, transactions)
    return generate_baseline_report(profiles_df, shifts_df, transactions)

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
        model_type: The architecture used ('baseline', 'beta_vae', or 'beta_tcvae').

    Returns:
        A dictionary containing the report metrics.
    """
    if model_type in {"beta_vae", "beta_tcvae"}:
        return generate_beta_report(profiles_df, shifts_df, transactions)
    return generate_baseline_report(profiles_df, shifts_df, transactions)

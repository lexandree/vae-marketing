import numpy as np
import pandas as pd
import torch
from torch import nn


def calculate_deviation(baseline_profile: np.ndarray, post_stimulus_data: pd.DataFrame, model: nn.Module) -> float:
    """Calculates the quantitative magnitude of shift (e.g., Euclidean distance in latent space).

    Args:
        baseline_profile: The baseline latent representation (mu vector).
        post_stimulus_data: DataFrame containing transactions during/after the stimulus.
        model: The trained VAE model.

    Returns:
        The Euclidean distance between the baseline profile and the post-stimulus profile.
    """
    model.eval()

    category_cols = [c for c in post_stimulus_data.columns if c.endswith('_SPEND') or c.endswith('_QTY')]
    temporal_cols = [c for c in post_stimulus_data.columns if c.startswith('TEMPORAL_')]

    if not category_cols or not temporal_cols:
        raise ValueError("Post-stimulus data missing required columns (categories or temporal).")

    x_tensor = torch.tensor(post_stimulus_data[category_cols].values, dtype=torch.float32)
    t_tensor = torch.tensor(post_stimulus_data[temporal_cols].values, dtype=torch.float32)

    with torch.no_grad():
        mu, _ = model.encode(x_tensor, t_tensor)

    post_profile = mu.mean(dim=0).numpy()

    # Calculate Euclidean distance
    distance = np.linalg.norm(baseline_profile - post_profile)
    return float(distance)

def categorize_shift(baseline_data: pd.DataFrame, post_stimulus_data: pd.DataFrame) -> str:
    """Categorizes the qualitative nature of shift into predefined categories based on rules.

    Args:
        baseline_data: DataFrame of household transactions before the stimulus.
        post_stimulus_data: DataFrame of household transactions during/after the stimulus.

    Returns:
        A string representing the category of shift (e.g., 'Trading Up', 'Stockpiling', 'No Change').
    """
    # TODO: Update this logic to use the new COMMODITY_*_SPEND/QTY features in Phase 3/4.
    # For now, return a placeholder based on simple magnitude heuristics or 'Unknown'
    return 'Unknown (Pending Phase 3/4)'

def analyze_persistence(household_data: pd.DataFrame, stimulus_end_day: int, model: nn.Module, baseline_profile: np.ndarray, threshold: float = 0.5) -> int:
    """Returns the number of days until the household behavior returns to the baseline.

    Args:
        household_data: DataFrame of household transactions after the stimulus.
        stimulus_end_day: Day index of when the stimulus ended.
        model: The trained VAE model.
        baseline_profile: The baseline latent representation.
        threshold: The distance threshold to consider behavior "returned to normal".

    Returns:
        Number of days for persistence.
    """
    if 'WINDOW_START_DAY' not in household_data.columns:
        return 0

    # Sort data chronologically
    df = household_data.sort_values('WINDOW_START_DAY')

    for _i, (_, row) in enumerate(df.iterrows()):
        row_df = pd.DataFrame([row])
        try:
            dist = calculate_deviation(baseline_profile, row_df, model)
            if dist < threshold:
                return int(row['WINDOW_START_DAY'] - stimulus_end_day)
        except ValueError:
            continue

    if not df.empty:
        return int(df.iloc[-1]['WINDOW_START_DAY'] - stimulus_end_day)

    return 0


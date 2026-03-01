
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

    category_cols = [c for c in post_stimulus_data.columns if c.startswith('cat_')]
    temporal_cols = ['month_of_year', 'week_of_year']

    if not category_cols or not temporal_cols or not all(col in post_stimulus_data.columns for col in temporal_cols):
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
    if 'price_tier' not in baseline_data.columns or 'quantity' not in baseline_data.columns:
        # Fallback or need these columns for the heuristic
        return 'Unknown'

    base_price = baseline_data['price_tier'].mean()
    base_qty = baseline_data['quantity'].mean()

    post_price = post_stimulus_data['price_tier'].mean()
    post_qty = post_stimulus_data['quantity'].mean()

    if post_price > base_price * 1.5:
        return 'Trading Up'
    elif post_qty > base_qty * 2.0:
        return 'Stockpiling'
    else:
        return 'No Change'

def analyze_persistence(household_data: pd.DataFrame, stimulus_end: pd.Timestamp, model: nn.Module, baseline_profile: np.ndarray, threshold: float = 0.5) -> int:
    """Returns the number of days until the household behavior returns to the baseline.

    Args:
        household_data: DataFrame of household transactions after the stimulus.
        stimulus_end: Timestamp of when the stimulus ended.
        model: The trained VAE model.
        baseline_profile: The baseline latent representation.
        threshold: The distance threshold to consider behavior "returned to normal".

    Returns:
        Number of days for persistence.
    """
    # Sort data chronologically
    df = household_data.sort_values('timestamp')

    # Analyze in windows (e.g., weekly)
    # For simplicity, calculate distance row by row or rolling window.
    # We will compute distance sequentially until it drops below threshold.

    # We need to compute deviation for subsets.
    # Group by week or use individual rows. Let's process row by row for the example.
    for _i, (_, row) in enumerate(df.iterrows()):
        row_df = pd.DataFrame([row])
        # Try to calculate distance if required cols exist
        try:
            dist = calculate_deviation(baseline_profile, row_df, model)
            if dist < threshold:
                return (row['timestamp'] - stimulus_end).days
        except ValueError:
            continue

    # If it never returns below threshold in the provided data
    if not df.empty:
        return (df.iloc[-1]['timestamp'] - stimulus_end).days

    return 0

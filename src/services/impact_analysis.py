import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Any


def calculate_deviation(
    baseline_profile: np.ndarray,
    post_stimulus_data: pd.DataFrame,
    model: nn.Module,
    return_vector: bool = False
) -> Any:
    """Calculates the quantitative magnitude of shift.

    Args:
        baseline_profile: The baseline latent representation (mu vector).
        post_stimulus_data: DataFrame containing transactions during/after the stimulus.
        model: The trained VAE model.
        return_vector: If True, returns raw difference vector instead of distance.

    Returns:
        Euclidean distance (float) or difference vector (np.ndarray).
    """
    model.eval()

    category_cols = [
        c for c in post_stimulus_data.columns
        if c.endswith("_SPEND") or c.endswith("_QTY")
    ]
    temporal_cols = [c for c in post_stimulus_data.columns if c.startswith("TEMPORAL_")]

    if not category_cols or not temporal_cols:
        raise ValueError("Post-stimulus data missing required columns (categories or temporal).")

    # Get device from model
    model_device = next(model.parameters()).device

    x_tensor = torch.tensor(post_stimulus_data[category_cols].values, dtype=torch.float32).to(model_device)
    t_tensor = torch.tensor(post_stimulus_data[temporal_cols].values, dtype=torch.float32).to(model_device)

    with torch.no_grad():
        mu, _ = model.encode(x_tensor, t_tensor)

    post_profile = mu.mean(dim=0).cpu().numpy()
    diff_vector = post_profile - baseline_profile

    if return_vector:
        return diff_vector
        
    return float(np.linalg.norm(diff_vector))


def _categorize_from_latent(delta: np.ndarray) -> str:
    """Helper to categorize shift from latent delta."""
    if len(delta) < 2:
        return "Unknown"

    # Price sensitivity is usually one of the top factors in Beta-VAE
    # We use raw indices until mapping is done via MIG/SAP
    price_shift = delta[0]
    volume_shift = delta[1]

    if price_shift > 0.5 and volume_shift > -0.2:
        return "Trading Up"
    if volume_shift > 0.5 and price_shift < 0.2:
        return "Stockpiling"
    if volume_shift < -0.5:
        return "Reduced Consumption"
    if np.linalg.norm(delta) < 0.2:
        return "No Change"
    return "Mixed Shift"


def _categorize_from_data(baseline_data: pd.DataFrame, post_stimulus_data: pd.DataFrame) -> str:
    """Helper to categorize shift from raw data heuristics."""
    if "price_tier" in baseline_data.columns and "price_tier" in post_stimulus_data.columns:
        base_price = baseline_data["price_tier"].mean()
        post_price = post_stimulus_data["price_tier"].mean()
        if post_price > base_price * 1.5:
            return "Trading Up"

    if "quantity" in baseline_data.columns and "quantity" in post_stimulus_data.columns:
        base_qty = baseline_data["quantity"].mean()
        post_qty = post_stimulus_data["quantity"].mean()
        if post_qty > base_qty * 2.0:
            return "Stockpiling"

    return "Unknown"


def categorize_shift(
    baseline_data: pd.DataFrame = None,
    post_stimulus_data: pd.DataFrame = None,
    baseline_profile: np.ndarray = None,
    post_profile: np.ndarray = None
) -> str:
    """Categorizes the qualitative nature of shift into predefined categories."""
    if baseline_profile is not None and post_profile is not None:
        return _categorize_from_latent(post_profile - baseline_profile)

    if baseline_data is not None and post_stimulus_data is not None:
        return _categorize_from_data(baseline_data, post_stimulus_data)

    return "Unknown"


def analyze_persistence(
    household_data: pd.DataFrame,
    stimulus_end_day: int,
    model: nn.Module,
    baseline_profile: np.ndarray,
    threshold: float = 0.5
) -> int:
    """Returns the number of days until the household behavior returns to the baseline."""
    if "WINDOW_START_DAY" not in household_data.columns:
        return 0

    # Sort data chronologically
    df = household_data.sort_values("WINDOW_START_DAY")
    
    # Get device from model
    model_device = next(model.parameters()).device

    for _i, (_, row) in enumerate(df.iterrows()):
        row_df = pd.DataFrame([row])
        try:
            dist = calculate_deviation(baseline_profile, row_df, model)
            if dist < threshold:
                return int(row["WINDOW_START_DAY"] - stimulus_end_day)
        except ValueError:
            continue

    if not df.empty:
        return int(df.iloc[-1]["WINDOW_START_DAY"] - stimulus_end_day)

    return 0

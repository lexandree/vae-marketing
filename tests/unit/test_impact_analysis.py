import numpy as np
import pandas as pd

from src.models.baseline_vae import build_vae_model
from src.services.impact_analysis import (
    analyze_persistence,
    calculate_deviation,
    categorize_shift,
)


def test_calculate_deviation_success() -> None:
    """Test successful deviation calculation."""
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    baseline_profile = np.array([0.0, 0.0, 0.0, 0.0])

    df = pd.DataFrame({
        'cat_1_SPEND': [0.1],
        'cat_1_QTY': [0.2],
        'TEMPORAL_1': [0.5],
        'TEMPORAL_2': [0.6]
    })

    deviation = calculate_deviation(baseline_profile, df, model)
    assert deviation >= 0.0


def test_calculate_deviation_invalid_data() -> None:
    """Test deviation calculation with invalid data."""
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    baseline_profile = np.array([0.0, 0.0, 0.0, 0.0])

    df = pd.DataFrame({'invalid_col': [0.1]})

    import pytest
    with pytest.raises(ValueError, match="Post-stimulus data missing required columns"):
        calculate_deviation(baseline_profile, df, model)


def test_categorize_shift_trading_up() -> None:
    """Test shift categorization as 'Trading Up'."""
    baseline_prof = np.array([0.0, 0.0])
    post_prof = np.array([0.6, 0.0])

    assert categorize_shift(
        baseline_profile=baseline_prof, post_profile=post_prof
    ) == 'Trading Up'


def test_categorize_shift_stockpiling() -> None:
    """Test shift categorization as 'Stockpiling'."""
    baseline_prof = np.array([0.0, 0.0])
    post_prof = np.array([0.0, 0.6])

    assert categorize_shift(
        baseline_profile=baseline_prof, post_profile=post_prof
    ) == 'Stockpiling'


def test_analyze_persistence() -> None:
    """Test behavioral shift persistence analysis."""
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    baseline_profile = np.array([0.0, 0.0, 0.0, 0.0])

    df = pd.DataFrame({
        'WINDOW_START_DAY': [10, 17, 24],
        'cat_1_SPEND': [1.0, 0.5, 0.1],
        'cat_1_QTY': [2.0, 1.0, 0.2],
        'TEMPORAL_1': [0.1, 0.2, 0.3],
        'TEMPORAL_2': [0.4, 0.5, 0.6]
    })

    persistence = analyze_persistence(df, stimulus_end_day=0, model=model,
                                     baseline_profile=baseline_profile, threshold=5.0)
    assert persistence >= 0

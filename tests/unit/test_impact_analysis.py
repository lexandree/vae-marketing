import numpy as np
import pandas as pd
import pytest

from src.models.vae import build_vae_model
from src.services.impact_analysis import analyze_persistence, calculate_deviation, categorize_shift


def test_calculate_deviation_success() -> None:
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    baseline_profile = np.array([0.0, 0.0, 0.0, 0.0])

    df = pd.DataFrame({
        'household_id': ['h1'],
        'cat_1': [1.0],
        'cat_2': [0.5],
        'month_of_year': [2],
        'week_of_year': [6]
    })

    deviation = calculate_deviation(baseline_profile, df, model)
    assert isinstance(deviation, float)
    assert deviation >= 0.0

def test_calculate_deviation_invalid_data() -> None:
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    baseline_profile = np.array([0.0, 0.0, 0.0, 0.0])

    df = pd.DataFrame({'invalid_col': [1.0]})

    with pytest.raises(ValueError, match="Post-stimulus data missing required columns"):
        calculate_deviation(baseline_profile, df, model)

def test_categorize_shift_trading_up() -> None:
    baseline = pd.DataFrame({
        'price_tier': [1.0],
        'quantity': [2.0]
    })
    post = pd.DataFrame({
        'price_tier': [3.0],
        'quantity': [2.0]
    })
    assert categorize_shift(baseline, post) == 'Trading Up'

def test_categorize_shift_stockpiling() -> None:
    baseline = pd.DataFrame({
        'price_tier': [1.0],
        'quantity': [2.0]
    })
    post = pd.DataFrame({
        'price_tier': [1.0],
        'quantity': [10.0]
    })
    assert categorize_shift(baseline, post) == 'Stockpiling'

def test_analyze_persistence() -> None:
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    baseline_profile = np.array([0.0, 0.0, 0.0, 0.0])

    # 3 weeks of data
    df = pd.DataFrame({
        'household_id': ['h1'] * 3,
        'cat_1': [1.0, 0.5, 0.1], # returning to normal
        'cat_2': [0.5, 0.2, 0.0],
        'month_of_year': [2, 2, 2],
        'week_of_year': [6, 7, 8],
        'timestamp': pd.to_datetime(['2023-02-01', '2023-02-08', '2023-02-15'])
    })

    stimulus_end = pd.to_datetime('2023-01-31')

    duration = analyze_persistence(df, stimulus_end, model, baseline_profile)
    assert isinstance(duration, int)
    assert duration >= 0

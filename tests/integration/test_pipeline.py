import numpy as np
import pandas as pd

from src.models.baseline_vae import build_vae_model
from src.services.baseline import get_household_profile, train_baseline_vae
from src.services.impact_analysis import calculate_deviation


def test_end_to_end_impact_analysis_logic() -> None:
    """Tests the logic of impact analysis on a synthetic but realistic pipeline."""
    # 1. Setup
    num_households = 10
    num_days = 100
    h_ids = [f"h_{i}" for i in range(num_households)]

    # Create synthetic pre-processed data
    # (assuming it has been through extract_features and normalizers)
    data = []
    for h_id in h_ids:
        for day in range(0, num_days, 7):
            data.append({
                'HOUSEHOLD_KEY': h_id,
                'WINDOW_START_DAY': day,
                'cat_1_SPEND': np.random.rand() * 2.0,
                'cat_1_QTY': np.random.rand() * 5.0,
                'cat_2_SPEND': np.random.rand() * 1.0,
                'cat_2_QTY': np.random.rand() * 3.0,
                'TEMPORAL_W_SIN': np.sin(2 * np.pi * day / 365),
                'TEMPORAL_W_COS': np.cos(2 * np.pi * day / 365)
            })

    df = pd.DataFrame(data)

    # 2. Train Model
    model = build_vae_model(latent_dim=4, num_categories=4, num_temporal_features=2)
    trained_model = train_baseline_vae(df, model, epochs=2)

    # 3. Create Baseline Profile
    h_test_id = "h_0"
    h_data = df[df["HOUSEHOLD_KEY"] == h_test_id]
    baseline_profile = get_household_profile(trained_model, h_data)

    # 4. Create Post-Stimulus Data
    # Copy baseline but change behavior
    post_data = h_data.copy()
    post_data['WINDOW_START_DAY'] = post_data['WINDOW_START_DAY'] + 100
    # Introduce a shift for first 5 households
    h_list = [f"h_{i}" for i in range(5)]
    post_data.loc[post_data['HOUSEHOLD_KEY'].isin(h_list), 'cat_1_SPEND'] *= 5.0

    # 5. Measure Impact
    deviation = calculate_deviation(baseline_profile, post_data, trained_model)

    assert deviation >= 0.0
    assert isinstance(deviation, float)

import numpy as np
import pandas as pd

from src.models.vae import build_vae_model
from src.services.baseline import get_household_profile, train_baseline_vae
from src.services.impact_analysis import analyze_persistence, calculate_deviation, categorize_shift
from src.services.reporting import generate_aggregate_report
from src.utils.seed import set_global_seed


def test_full_pipeline_integration() -> None:
    """Test the full VAE impact analysis pipeline from data load to reporting."""
    set_global_seed(42)

    # 1. Generate Synthetic Data
    np.random.seed(42)
    num_households = 10
    num_transactions = 100

    h_ids = [f"h_{i}" for i in range(num_households)] * (num_transactions // num_households)

    transactions = pd.DataFrame({
        'household_id': h_ids,
        'cat_1': np.random.rand(num_transactions),
        'cat_2': np.random.rand(num_transactions),
        'price_tier': np.random.rand(num_transactions) * 5,
        'quantity': np.random.randint(1, 10, num_transactions),
        'month_of_year': np.random.randint(1, 12, num_transactions),
        'week_of_year': np.random.randint(1, 52, num_transactions),
        'timestamp': pd.date_range(start='2023-01-01', periods=num_transactions, freq='D')
    })

    # 2. Train Baseline VAE
    model = build_vae_model(latent_dim=4, num_categories=2, num_temporal_features=2)
    trained_model = train_baseline_vae(transactions, model, epochs=2, min_transactions=5)

    # 3. Get Profiles
    baseline_profiles = []
    for h_id in transactions['household_id'].unique():
        h_data = transactions[transactions['household_id'] == h_id]
        profile = get_household_profile(trained_model, h_data)
        baseline_profiles.append({
            'household_id': h_id,
            'baseline_profile': profile
        })
    profiles_df = pd.DataFrame(baseline_profiles)

    # 4. Generate Stimulus & Post-Stimulus Data
    stimulus_end = pd.to_datetime('2023-04-01')
    post_data = transactions.copy()
    post_data['timestamp'] = post_data['timestamp'] + pd.Timedelta(days=100) # shift to after stimulus
    # Introduce a shift for first 5 households
    post_data.loc[post_data['household_id'].isin([f"h_{i}" for i in range(5)]), 'cat_1'] *= 5.0
    post_data.loc[post_data['household_id'].isin([f"h_{i}" for i in range(5)]), 'quantity'] *= 3.0

    # 5. Measure Impact
    shift_results = []
    for h_id in transactions['household_id'].unique():
        base_prof = profiles_df[profiles_df['household_id'] == h_id]['baseline_profile'].iloc[0]
        h_post = post_data[post_data['household_id'] == h_id]
        h_base = transactions[transactions['household_id'] == h_id]

        dev = calculate_deviation(base_prof, h_post, trained_model)
        cat = categorize_shift(h_base, h_post)
        pers = analyze_persistence(h_post, stimulus_end, trained_model, base_prof)

        shift_results.append({
            'household_id': h_id,
            'stimulus_id': 's1',
            'quantitative_magnitude': dev,
            'qualitative_nature': cat,
            'persistence_duration_days': pers
        })

    shifts_df = pd.DataFrame(shift_results)

    # 6. Aggregate Reporting
    report = generate_aggregate_report(profiles_df, shifts_df, post_data)

    assert report['total_households_analyzed'] == 10
    assert report['total_shifts_detected'] == 10
    assert report['average_magnitude'] >= 0
    assert len(report['segments']) > 0
    assert isinstance(report['top_sensitive_categories'], list)

import pandas as pd

from src.services.reporting import rank_sensitive_categories, segment_consumers


def test_segment_consumers() -> None:
    shifts = pd.DataFrame({
        'household_id': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
        'quantitative_magnitude': [1.0, 1.1, 5.0, 5.2, 0.1, 0.2],
        'persistence_duration_days': [5, 6, 20, 21, 1, 2]
    })

    # Needs at least enough samples for the default n_clusters (e.g. 3)
    segmented = segment_consumers(shifts, n_clusters=3)
    assert 'cluster' in segmented.columns
    assert len(segmented['cluster'].unique()) == 3

def test_rank_sensitive_categories() -> None:
    shifts = pd.DataFrame({
        'household_id': ['h1', 'h2'],
        'stimulus_id': ['s1', 's1'],
        'quantitative_magnitude': [5.0, 10.0]
    })

    transactions = pd.DataFrame({
        'household_id': ['h1', 'h2', 'h1', 'h2'],
        'product_category': ['A', 'B', 'B', 'B'],
        # Assuming we track volume/value affected
    })

    ranked = rank_sensitive_categories(shifts, transactions, top_k=2)
    assert isinstance(ranked, pd.DataFrame)
    assert len(ranked) <= 2
    assert 'sensitivity_score' in ranked.columns

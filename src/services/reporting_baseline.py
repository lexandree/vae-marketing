import pandas as pd
from sklearn.cluster import KMeans


def segment_consumers(shifts: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Segments consumers based on the magnitude and persistence of behavioral shifts."""
    if shifts.empty:
        return shifts

    # Use magnitude and persistence for clustering
    features = shifts[['quantitative_magnitude', 'persistence_duration_days']].fillna(0)

    # Simplified clustering for demonstration
    kmeans = KMeans(n_clusters=min(n_clusters, len(shifts)), random_state=42, n_init='auto')
    shifts['cluster'] = kmeans.fit_predict(features)

    return shifts


def rank_sensitive_categories(
    shifts: pd.DataFrame,
    transactions: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """Identifies and ranks the top product categories based on sensitivity.

    Heuristic: Categories where households with large latent shifts spend most.
    """
    if shifts.empty or transactions.empty:
        return pd.DataFrame(columns=['product_category', 'sensitivity_score'])

    # Join transactions with magnitude of shift per household_id
    merged = transactions.merge(
        shifts[['household_id', 'quantitative_magnitude']],
        left_on='HOUSEHOLD_KEY',
        right_on='household_id',
        how='inner'
    )

    # Check if 'product_category' exists, if not, try to unpivot 'COMMODITY_' columns
    if 'product_category' not in merged.columns:
        cat_cols = [
            c for c in merged.columns
            if c.startswith('COMMODITY_') and c.endswith('_SPEND')
        ]
        if not cat_cols:
            return pd.DataFrame(columns=['product_category', 'sensitivity_score'])

        # Unpivot category columns
        merged = merged.melt(
            id_vars=['HOUSEHOLD_KEY', 'quantitative_magnitude'],
            value_vars=cat_cols,
            var_name='product_category',
            value_name='value'
        )
        # Only keep rows where household actually bought something (>0)
        merged = merged[merged['value'] > 0]

    # Weighted score: average magnitude for each category
    ranked = merged.groupby('product_category')['quantitative_magnitude'].mean().reset_index()
    ranked.columns = ['product_category', 'sensitivity_score']

    ranked = ranked.sort_values(by='sensitivity_score', ascending=False).head(top_k)
    return ranked


def generate_aggregate_report(
    baseline_profiles: pd.DataFrame,
    shifts: pd.DataFrame,
    transactions: pd.DataFrame
) -> dict:
    """Generates the comprehensive aggregate impact report."""
    segmented = segment_consumers(shifts)
    ranked_cats = rank_sensitive_categories(shifts, transactions)

    avg_magnitude = shifts['quantitative_magnitude'].mean() if not shifts.empty else 0
    avg_persistence = shifts['persistence_duration_days'].mean() if not shifts.empty else 0

    return {
        "total_households_analyzed": len(baseline_profiles),
        "total_shifts_detected": len(shifts),
        "average_magnitude": avg_magnitude,
        "average_persistence_days": avg_persistence,
        "segments": (
            segmented['cluster'].value_counts().to_dict() if not segmented.empty else {}
        ),
        "top_sensitive_categories": ranked_cats.to_dict('records')
    }

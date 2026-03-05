import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def segment_consumers(shifts: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Groups consumers into segments based on similarity in their reaction patterns.

    Args:
        shifts: DataFrame containing behavioral shift records.
        n_clusters: Number of clusters for K-Means.

    Returns:
        DataFrame with an added 'cluster' column.
    """
    if shifts.empty or len(shifts) < n_clusters:
        shifts['cluster'] = 0
        return shifts

    features = shifts[['quantitative_magnitude', 'persistence_duration_days']].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    shifts['cluster'] = kmeans.fit_predict(scaled_features)

    return shifts

def rank_sensitive_categories(shifts: pd.DataFrame, transactions: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Identifies and ranks the top product categories based on sensitivity.

    Args:
        shifts: DataFrame of calculated shifts.
        transactions: DataFrame of transactions during the affected period.
        top_k: Number of top categories to return.

    Returns:
        DataFrame of ranked categories and their sensitivity scores.
    """
    if shifts.empty or transactions.empty:
        return pd.DataFrame(columns=['product_category', 'sensitivity_score'])

    # Simplified heuristic: Group transactions by household, merge with shift magnitude,
    # and aggregate magnitude by category. A more rigorous approach would isolate
    # the exact deviation per category.
    # Updated column names to use HOUSEHOLD_KEY instead of household_id
    merged = transactions.merge(shifts[['household_id', 'quantitative_magnitude']], left_on='HOUSEHOLD_KEY', right_on='household_id', how='inner')

    # Check if 'product_category' exists, if not, try to unpivot 'COMMODITY_' columns for the heuristic
    if 'product_category' not in merged.columns:
        cat_cols = [c for c in merged.columns if c.startswith('COMMODITY_') and c.endswith('_SPEND')]
        if not cat_cols:
            return pd.DataFrame(columns=['product_category', 'sensitivity_score'])

        # Melt the dataframe to have a 'product_category' column
        merged = merged.melt(id_vars=['household_id', 'quantitative_magnitude'], value_vars=cat_cols, var_name='product_category', value_name='value')
        # Only consider categories where the user actually bought something (>0 or > threshold)
        merged = merged[merged['value'] > 0]

    # Score = sum of shift magnitudes for households that bought the category
    # (assuming categories bought during shift are the sensitive ones)
    ranked = merged.groupby('product_category')['quantitative_magnitude'].sum().reset_index()
    ranked.rename(columns={'quantitative_magnitude': 'sensitivity_score'}, inplace=True)

    ranked = ranked.sort_values(by='sensitivity_score', ascending=False).head(top_k)
    return ranked
def generate_aggregate_report(baseline_profiles: pd.DataFrame, shifts: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """Generates the comprehensive aggregate impact report."""
    segmented = segment_consumers(shifts)
    ranked_cats = rank_sensitive_categories(shifts, transactions)

    return {
        "total_households_analyzed": len(baseline_profiles),
        "total_shifts_detected": len(shifts),
        "average_magnitude": shifts['quantitative_magnitude'].mean() if not shifts.empty else 0,
        "average_persistence_days": shifts['persistence_duration_days'].mean() if not shifts.empty else 0,
        "segments": segmented['cluster'].value_counts().to_dict() if not segmented.empty else {},
        "top_sensitive_categories": ranked_cats.to_dict('records')
    }

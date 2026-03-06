import pandas as pd
import numpy as np
from src.services.reporting_baseline import rank_sensitive_categories, segment_consumers


def generate_aggregate_report(
    profiles_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    transactions: pd.DataFrame
) -> dict:
    """Generates the comprehensive aggregate impact report for Beta-VAE.

    Includes real factor-level breakdowns based on latent space deviations.
    """
    segmented = segment_consumers(shifts_df)
    ranked_cats = rank_sensitive_categories(shifts_df, transactions)

    # Group by qualitative nature to see what types of shifts were most common
    nature_counts = {}
    if not shifts_df.empty:
        nature_counts = shifts_df['qualitative_nature'].value_counts().to_dict()

    avg_magnitude = 0.0
    avg_persistence = 0.0
    factor_impacts = {}

    if not shifts_df.empty and 'latent_vector' in shifts_df.columns:
        avg_magnitude = shifts_df['quantitative_magnitude'].mean()
        avg_persistence = shifts_df['persistence_duration_days'].mean()

        # Calculate average absolute deviation per latent dimension
        # shifts_df['latent_vector'] contains lists of floats
        all_vectors = np.array(shifts_df['latent_vector'].tolist())
        mean_abs_dev = np.mean(np.abs(all_vectors), axis=0)
        
        # Identify top moving factors
        top_indices = np.argsort(mean_abs_dev)[::-1]
        
        for idx in top_indices:
            # For now we use generic names, but MIG/SAP scores will soon map these to real concepts
            factor_impacts[f"latent_factor_{idx}"] = float(mean_abs_dev[idx])

    return {
        "total_households_analyzed": len(profiles_df),
        "total_shifts_detected": len(shifts_df),
        "average_magnitude": avg_magnitude,
        "average_persistence_days": avg_persistence,
        "segments": segmented['cluster'].value_counts().to_dict() if not segmented.empty else {},
        "top_sensitive_categories": ranked_cats.to_dict('records'),
        "nature_distribution": nature_counts,
        "factor_breakdown": factor_impacts
    }

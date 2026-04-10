"""Matching utilities for treated and comparison cohorts.

These helpers sit between raw campaign cohorts and quasi-causal estimation.
The main idea is simple:

- treated households receive a campaign assignment
- comparison households do not
- before comparing outcomes, we try to make both groups look similar in the
  pre-period so that post-period differences are less likely to be driven by
  baseline behavior alone
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_standardized_mean_difference(
    treated: pd.Series,
    comparison: pd.Series,
) -> float:
    """Compute the standardized mean difference for a covariate.

    Args:
        treated: Covariate values for treated households.
        comparison: Covariate values for comparison households.

    Returns:
        The standardized mean difference. Returns `0.0` when both groups are empty
        or have zero pooled variance.
    """
    if treated.empty or comparison.empty:
        return 0.0

    treated_mean = float(treated.mean())
    comparison_mean = float(comparison.mean())
    treated_var = float(treated.var(ddof=1)) if len(treated) > 1 else 0.0
    comparison_var = float(comparison.var(ddof=1)) if len(comparison) > 1 else 0.0
    pooled_std = np.sqrt((treated_var + comparison_var) / 2.0)
    if pooled_std == 0.0:
        return 0.0
    return (treated_mean - comparison_mean) / pooled_std


def build_matched_cohort(
    treated_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    score_column: str,
    max_matches_per_treated: int = 1,
) -> pd.DataFrame:
    """Construct a deterministic nearest-score matched cohort.

    Args:
        treated_df: Treated cohort containing the score column.
        comparison_df: Untreated comparison cohort containing the score column.
        score_column: Name of the numeric baseline score used for nearest matching.
        max_matches_per_treated: Maximum number of comparison rows per treated row.

    Returns:
        A DataFrame with treated rows and their matched comparison rows.
    """
    if score_column not in treated_df.columns or score_column not in comparison_df.columns:
        raise ValueError(f"score column '{score_column}' must exist in both input frames")

    if max_matches_per_treated < 1:
        raise ValueError("max_matches_per_treated must be >= 1")

    if treated_df.empty or comparison_df.empty:
        return pd.DataFrame(columns=list(treated_df.columns) + ["match_group", "cohort_role"])

    treated = treated_df.reset_index(drop=True).copy()
    comparison = comparison_df.reset_index(drop=True).copy()
    used_indices: set[int] = set()
    matched_frames: list[pd.DataFrame] = []

    for idx, row in treated.iterrows():
        available = comparison.loc[~comparison.index.isin(used_indices)].copy()
        if available.empty:
            break

        available["distance"] = (available[score_column] - row[score_column]).abs()
        selected = available.nsmallest(max_matches_per_treated, "distance").drop(columns=["distance"])
        used_indices.update(selected.index.tolist())

        treated_row = row.to_frame().T.copy()
        treated_row["match_group"] = idx
        treated_row["cohort_role"] = "treated"

        selected = selected.copy()
        selected["match_group"] = idx
        selected["cohort_role"] = "comparison"

        matched_frames.extend([treated_row, selected])

    if not matched_frames:
        return pd.DataFrame(columns=list(treated_df.columns) + ["match_group", "cohort_role"])

    return pd.concat(matched_frames, ignore_index=True)


def attach_propensity_scores(
    cohort_df: pd.DataFrame,
    covariates: Sequence[str],
    treatment_column: str = "treatment_status",
) -> pd.DataFrame:
    """Estimate a propensity score from pre-period covariates.

    The propensity score is the model-estimated probability of being assigned to
    treatment given observed pre-period behavior. It is not a causal guarantee;
    it is just a compact summary used to find households with similar baseline
    profiles before effect estimation.
    """
    if cohort_df.empty:
        return cohort_df.copy()

    design = cohort_df.copy()
    available_covariates = [column for column in covariates if column in design.columns]
    if not available_covariates:
        raise ValueError("No matching covariates available to estimate propensity scores.")

    X = design[available_covariates]
    y = (design[treatment_column] == "treated").astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X, y)
    design["propensity_score"] = model.predict_proba(X)[:, 1]
    return design


def build_caliper_matched_cohort(
    treated_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    score_column: str,
    caliper: float,
) -> pd.DataFrame:
    """Construct a nearest-neighbor matched cohort with a distance threshold.

    The caliper is a hard maximum allowed distance on the matching score. It
    narrows common support: if no untreated household is close enough to a
    treated household, that treated household is dropped from the matched cohort.
    """
    if score_column not in treated_df.columns or score_column not in comparison_df.columns:
        raise ValueError(f"score column '{score_column}' must exist in both input frames")
    if caliper <= 0:
        raise ValueError("caliper must be > 0")
    if treated_df.empty or comparison_df.empty:
        return pd.DataFrame(columns=list(treated_df.columns) + ["match_group", "cohort_role"])

    treated = treated_df.reset_index(drop=True).copy()
    comparison = comparison_df.reset_index(drop=True).copy()
    used_indices: set[int] = set()
    matched_frames: list[pd.DataFrame] = []

    for idx, row in treated.sort_values(score_column).iterrows():
        available = comparison.loc[~comparison.index.isin(used_indices)].copy()
        if available.empty:
            break
        available["distance"] = (available[score_column] - row[score_column]).abs()
        available = available[available["distance"] <= caliper]
        if available.empty:
            continue

        selected = available.nsmallest(1, "distance").drop(columns=["distance"])
        used_indices.update(selected.index.tolist())

        treated_row = row.to_frame().T.copy()
        treated_row["match_group"] = idx
        treated_row["cohort_role"] = "treated"

        selected = selected.copy()
        selected["match_group"] = idx
        selected["cohort_role"] = "comparison"
        matched_frames.extend([treated_row, selected])

    if not matched_frames:
        return pd.DataFrame(columns=list(treated_df.columns) + ["match_group", "cohort_role"])

    return pd.concat(matched_frames, ignore_index=True)


def summarize_balance(
    treated_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Summarize balance using standardized mean differences for covariates.

    Args:
        treated_df: Treated cohort.
        comparison_df: Comparison cohort.
        covariates: Covariate column names to evaluate.

    Returns:
        A DataFrame with one row per covariate and its balance statistic.
    """
    rows = []
    for covariate in covariates:
        if covariate not in treated_df.columns or covariate not in comparison_df.columns:
            continue
        smd = compute_standardized_mean_difference(
            treated_df[covariate],
            comparison_df[covariate],
        )
        rows.append(
            {
                "covariate": covariate,
                "standardized_mean_difference": smd,
                "balance_status": "balanced" if abs(smd) <= 0.1 else "weak_balance",
            }
        )
    return pd.DataFrame(rows)

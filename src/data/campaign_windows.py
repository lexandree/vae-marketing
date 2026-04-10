"""Campaign window assignment and alignment helpers."""

from __future__ import annotations

import polars as pl


def build_campaign_windows(
    campaign_desc: pl.DataFrame,
    pre_weeks: int = 8,
    post_weeks: int = 8,
) -> pl.DataFrame:
    """Construct campaign-aligned pre, in, and post windows.

    Args:
        campaign_desc: Campaign metadata with `CAMPAIGN`, `START_DAY`, and `END_DAY`.
        pre_weeks: Number of weeks to include before campaign start.
        post_weeks: Number of weeks to include after campaign end.

    Returns:
        A Polars DataFrame containing explicit campaign window boundaries.
    """
    required_columns = {"CAMPAIGN", "START_DAY", "END_DAY"}
    missing = required_columns.difference(campaign_desc.columns)
    if missing:
        raise ValueError(f"campaign_desc is missing required columns: {sorted(missing)}")

    pre_days = pre_weeks * 7
    post_days = post_weeks * 7

    return campaign_desc.with_columns(
        [
            (pl.col("START_DAY") - pre_days).alias("pre_window_start_day"),
            (pl.col("START_DAY") - 1).alias("pre_window_end_day"),
            (pl.col("START_DAY")).alias("campaign_start_day"),
            (pl.col("END_DAY")).alias("campaign_end_day"),
            (pl.col("END_DAY") + 1).alias("post_window_start_day"),
            (pl.col("END_DAY") + post_days).alias("post_window_end_day"),
        ]
    )

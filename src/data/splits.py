"""Household split helpers for leakage-resistant training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


def build_household_splits(
    campaign_table: pl.DataFrame,
    eval_campaign_ids: list[int],
    seed: int = 42,
) -> dict[str, Any]:
    """Build train/eval household splits with campaign-level holdout protection.

    Policy:
    - households participating in `eval_campaign_ids` are excluded from model training
    - remaining households are split into train/validation buckets deterministically
    """
    eval_campaign_set = set(eval_campaign_ids)
    if "household_key" in campaign_table.columns and "HOUSEHOLD_KEY" not in campaign_table.columns:
        campaign_table = campaign_table.rename({"household_key": "HOUSEHOLD_KEY"})
    if "campaign" in campaign_table.columns and "CAMPAIGN" not in campaign_table.columns:
        campaign_table = campaign_table.rename({"campaign": "CAMPAIGN"})

    if "CAMPAIGN" not in campaign_table.columns or "HOUSEHOLD_KEY" not in campaign_table.columns:
        raise ValueError("campaign_table must contain CAMPAIGN and HOUSEHOLD_KEY columns")

    campaign_df = campaign_table.select(["HOUSEHOLD_KEY", "CAMPAIGN"]).unique()
    eval_households = (
        campaign_df.filter(pl.col("CAMPAIGN").is_in(list(eval_campaign_set)))
        .select("HOUSEHOLD_KEY")
        .unique()
        .to_series()
        .to_list()
    )
    all_households = (
        campaign_df.select("HOUSEHOLD_KEY").unique().to_series().to_list()
    )
    train_universe = sorted(h for h in all_households if h not in set(eval_households))

    # Deterministic split without introducing another dependency.
    rng = __import__("random").Random(seed)
    shuffled = list(train_universe)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * 0.2))) if shuffled else 0
    validation_households = sorted(shuffled[:val_size])
    train_households = sorted(shuffled[val_size:])

    return {
        "eval_campaign_ids": sorted(eval_campaign_ids),
        "eval_households": sorted(eval_households),
        "train_households": train_households,
        "validation_households": validation_households,
        "metadata": {
            "split_seed": seed,
            "policy": "exclude_all_households_participating_in_eval_campaigns",
        },
    }


def write_household_splits(splits: dict[str, Any], output_path: Path) -> None:
    """Write household split metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(splits, indent=2))


def load_household_splits(path: Path | str) -> dict[str, Any]:
    """Load household split metadata from JSON."""
    return json.loads(Path(path).read_text())


def household_ids_for_role(splits: dict[str, Any], split_role: str) -> list[str]:
    """Return the household ids configured for the requested split role."""
    if split_role == "train":
        return list(splits.get("train_households", []))
    if split_role == "validation":
        return list(splits.get("validation_households", []))
    if split_role == "eval":
        return list(splits.get("eval_households", []))
    if split_role == "all":
        return sorted(
            set(splits.get("train_households", []))
            | set(splits.get("validation_households", []))
            | set(splits.get("eval_households", []))
        )
    raise ValueError(f"Unsupported split_role: {split_role}")


def filter_households(
    df: pl.DataFrame | pd.DataFrame,
    allowed_households: list[str],
    household_column: str = "HOUSEHOLD_KEY",
) -> pl.DataFrame | pd.DataFrame:
    """Filter a frame to the requested households while preserving frame type."""
    if household_column not in df.columns:
        return df
    if isinstance(df, pl.DataFrame):
        column_dtype = df.schema.get(household_column)
        normalized_ids = (
            [str(value) for value in allowed_households]
            if column_dtype == pl.Utf8
            else allowed_households
        )
        return df.filter(pl.col(household_column).is_in(normalized_ids))
    series = df[household_column]
    normalized_ids = (
        [str(value) for value in allowed_households]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        else allowed_households
    )
    return df[df[household_column].isin(normalized_ids)].copy()

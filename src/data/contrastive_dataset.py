"""Contrastive dataset builders for campaign-salient VAE training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.data.campaign_windows import build_campaign_windows
from src.data.dataset import load_validation_source


def _campaign_window_rows(
    prepared_df: pl.DataFrame,
    campaign_windows: pl.DataFrame,
    campaign_table: pl.DataFrame,
    background_ratio: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Select treated target rows and sampled background rows in campaign windows."""
    prepared_df = prepared_df.with_columns(pl.col("HOUSEHOLD_KEY").cast(pl.Utf8))
    campaign_table = campaign_table.with_columns(pl.col("household_key").cast(pl.Utf8))

    all_households = prepared_df["HOUSEHOLD_KEY"].unique().to_list()
    target_frames: list[pl.DataFrame] = []
    background_frames: list[pl.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for row in campaign_windows.iter_rows(named=True):
        campaign_id = int(row["CAMPAIGN"])
        treated = set(
            campaign_table.filter(pl.col("CAMPAIGN") == campaign_id)["household_key"].unique().to_list()
        )
        if not treated:
            continue
        comparison = [household for household in all_households if household not in treated]
        campaign_slice = prepared_df.filter(
            (pl.col("WINDOW_START_DAY") <= int(row["campaign_end_day"]))
            & ((pl.col("WINDOW_START_DAY") + 6) >= int(row["campaign_start_day"]))
        )
        target = campaign_slice.filter(pl.col("HOUSEHOLD_KEY").is_in(list(treated))).with_columns(
            pl.lit(campaign_id).alias("CAMPAIGN"),
            pl.lit("target").alias("contrastive_role"),
        )
        background = campaign_slice.filter(pl.col("HOUSEHOLD_KEY").is_in(comparison)).with_columns(
            pl.lit(campaign_id).alias("CAMPAIGN"),
            pl.lit("background").alias("contrastive_role"),
        )
        if target.is_empty() or background.is_empty():
            continue
        max_background = max(1, int(round(target.height * background_ratio)))
        if background.height > max_background:
            background = background.sample(n=max_background, seed=seed, shuffle=True)
        target_frames.append(target)
        background_frames.append(background)
        summaries.append(
            {
                "campaign_id": campaign_id,
                "target_rows": target.height,
                "background_rows": background.height,
            }
        )

    target_df = pl.concat(target_frames) if target_frames else pl.DataFrame()
    background_df = pl.concat(background_frames) if background_frames else pl.DataFrame()
    summary = {
        "campaigns_used": [row["campaign_id"] for row in summaries],
        "per_campaign": summaries,
        "target_rows": target_df.height,
        "background_rows": background_df.height,
        "background_ratio": background_ratio,
    }
    return (
        target_df.sort(["CAMPAIGN", "HOUSEHOLD_KEY", "WINDOW_START_DAY"]) if target_df.height else target_df,
        background_df.sort(["CAMPAIGN", "HOUSEHOLD_KEY", "WINDOW_START_DAY"]) if background_df.height else background_df,
        summary,
    )


def build_contrastive_dataset(
    *,
    prepared_data: Path,
    campaign_table_path: Path,
    campaign_desc_path: Path,
    output_dir: Path,
    campaign_ids: list[int] | None = None,
    exclude_campaign_ids: list[int] | None = None,
    background_ratio: float = 1.0,
    seed: int = 42,
) -> None:
    """Build target/background training frames from prepared weekly features."""
    prepared_df = pl.read_parquet(prepared_data)
    campaign_table = load_validation_source(campaign_table_path, "campaign_table")
    campaign_desc = load_validation_source(campaign_desc_path, "campaign_desc")

    selected = campaign_desc
    if campaign_ids:
        selected = selected.filter(pl.col("CAMPAIGN").is_in(campaign_ids))
    if exclude_campaign_ids:
        selected = selected.filter(~pl.col("CAMPAIGN").is_in(exclude_campaign_ids))
    campaign_windows = build_campaign_windows(selected, pre_weeks=0, post_weeks=0)

    target_df, background_df, summary = _campaign_window_rows(
        prepared_df=prepared_df,
        campaign_windows=campaign_windows,
        campaign_table=campaign_table,
        background_ratio=background_ratio,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_df.write_parquet(output_dir / "target.parquet")
    background_df.write_parquet(output_dir / "background.parquet")
    with open(output_dir / "contrastive_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)


def build_contrastive_dataset_from_args(*, args: Any) -> None:
    """CLI entrypoint for contrastive dataset creation."""
    build_contrastive_dataset(
        prepared_data=args.prepared_data,
        campaign_table_path=args.campaign_table,
        campaign_desc_path=args.campaign_desc,
        output_dir=args.output_dir,
        campaign_ids=args.campaign_ids,
        exclude_campaign_ids=args.exclude_campaign_ids,
        background_ratio=args.background_ratio,
        seed=args.seed,
    )

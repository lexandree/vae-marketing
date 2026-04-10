"""Latent factor validation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from src.data.splits import filter_households, household_ids_for_role, load_household_splits
from src.models.factory import ModelFactory
from src.services.validation_reporting import serialize_latent_validation_outputs
from src.utils.metrics import calculate_mig_with_method, calculate_sap_with_method


def _select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select category and temporal feature columns from a prepared frame."""
    category_cols = [c for c in df.columns if c.endswith("_SPEND") or c.endswith("_QTY")]
    temporal_cols = [c for c in df.columns if c.startswith("TEMPORAL_")]
    if not category_cols:
        raise ValueError("No spend/qty feature columns found for latent extraction.")
    return category_cols, temporal_cols


def pivot_validation_attributes(attributes_df: pd.DataFrame) -> pd.DataFrame:
    """Convert row-oriented validation attributes into a wide frame when needed."""
    required_columns = {"attribute_name", "attribute_value"}
    if not required_columns.issubset(attributes_df.columns):
        return attributes_df.copy()

    index_columns = [
        column
        for column in ("HOUSEHOLD_KEY", "CAMPAIGN", "WINDOW_START_DAY", "window_type")
        if column in attributes_df.columns
    ]
    if not index_columns:
        raise ValueError("Validation attributes need join keys for pivoting.")

    wide_df = (
        attributes_df.pivot_table(
            index=index_columns,
            columns="attribute_name",
            values="attribute_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_df.columns.name = None
    return wide_df


def align_holdout_frames(
    analysis_df: pd.DataFrame,
    attributes_df: pd.DataFrame,
    join_keys: Iterable[str] = ("HOUSEHOLD_KEY", "WINDOW_START_DAY"),
) -> pd.DataFrame:
    """Align analysis and attribute frames on shared holdout keys."""
    keys = [key for key in join_keys if key in analysis_df.columns and key in attributes_df.columns]
    if not keys:
        shared = [key for key in ("HOUSEHOLD_KEY", "CAMPAIGN") if key in analysis_df.columns and key in attributes_df.columns]
        keys = shared
    if not keys:
        raise ValueError("No shared join keys available for holdout alignment.")
    return analysis_df.merge(attributes_df, on=keys, how="inner")


def extract_latent_snapshots(
    model: torch.nn.Module,
    analysis_df: pd.DataFrame,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Extract latent means for each aligned analysis row."""
    category_cols, temporal_cols = _select_feature_columns(analysis_df)
    target_device = device or torch.device("cpu")

    x_tensor = torch.tensor(analysis_df[category_cols].values, dtype=torch.float32, device=target_device)
    if temporal_cols:
        t_tensor = torch.tensor(
            analysis_df[temporal_cols].values,
            dtype=torch.float32,
            device=target_device,
        )
    else:
        temporal_width = getattr(model, "num_temporal_features", 1)
        t_tensor = torch.zeros((len(analysis_df), temporal_width), dtype=torch.float32, device=target_device)

    model = model.to(target_device)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_tensor, t_tensor)

    key_columns = [c for c in ("HOUSEHOLD_KEY", "CAMPAIGN", "WINDOW_START_DAY") if c in analysis_df.columns]
    latent_df = analysis_df[key_columns].copy()
    latent_values = mu.detach().cpu().numpy()
    for idx in range(latent_values.shape[1]):
        latent_df[f"latent_{idx}"] = latent_values[:, idx]
    return latent_df


def evaluate_latent_metrics(
    latent_df: pd.DataFrame,
    attributes_df: pd.DataFrame,
    mig_method: str = "sklearn",
    mig_bins: int = 16,
    mig_binning: str = "quantile",
    sap_method: str = "sklearn",
) -> dict[str, float]:
    """Evaluate MIG and SAP for aligned latent and observable attributes."""
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    attribute_cols = [
        c for c in attributes_df.columns
        if c not in {"HOUSEHOLD_KEY", "WINDOW_START_DAY", "CAMPAIGN", "window_type", "attribute_name", "attribute_family"}
        and pd.api.types.is_numeric_dtype(attributes_df[c])
    ]
    if not latent_cols or not attribute_cols:
        return {"mig_score": 0.0, "sap_score": 0.0}

    latent = latent_df[latent_cols].to_numpy(dtype=float)
    attrs = attributes_df[attribute_cols].to_numpy(dtype=float)
    return {
        "mig_score": calculate_mig_with_method(
            latent,
            attrs,
            method=mig_method,
            bins=mig_bins,
            strategy=mig_binning,
        ),
        "sap_score": calculate_sap_with_method(latent, attrs, method=sap_method),
    }


def load_run_latents(
    run_dir: Path,
    analysis_df: pd.DataFrame,
    filename: str = "best_model.pth",
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a trained run and extract latent snapshots for a feature frame."""
    model, config = ModelFactory.load_model_with_config(run_dir, filename=filename)
    latent_df = extract_latent_snapshots(model, analysis_df, device=device)
    return latent_df, config


def _build_factor_mapping_rows(
    run_id: str,
    model_type: str,
    latent_df: pd.DataFrame,
    attributes_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build factor mapping and stability rows for one run."""
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    attribute_cols = [
        c for c in attributes_df.columns
        if c not in {"HOUSEHOLD_KEY", "WINDOW_START_DAY", "CAMPAIGN", "window_type"}
        and pd.api.types.is_numeric_dtype(attributes_df[c])
    ]

    mapping_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    if not latent_cols or not attribute_cols:
        return mapping_rows, stability_rows

    for latent_column in latent_cols:
        best_attribute = None
        best_score = -1.0
        latent_values = latent_df[latent_column]
        for attribute in attribute_cols:
            attr_values = attributes_df[attribute]
            if attr_values.nunique() <= 1:
                score = 0.0
            else:
                score = abs(float(np.corrcoef(latent_values, attr_values)[0, 1]))
            if np.isnan(score):
                score = 0.0
            if score > best_score:
                best_score = score
                best_attribute = attribute

        stability_status = "stable" if best_score >= 0.3 else "unstable"
        mapping_decision = "validated" if best_score >= 0.3 else "rejected"
        latent_index = int(latent_column.split("_")[1])
        mapping_rows.append(
            {
                "run_id": run_id,
                "model_type": model_type,
                "latent_dimension": latent_index,
                "candidate_attribute": best_attribute,
                "association_strength": best_score,
                "mapping_decision": mapping_decision,
            }
        )
        stability_rows.append(
            {
                "run_id": run_id,
                "model_type": model_type,
                "latent_dimension": latent_index,
                "candidate_attribute": best_attribute,
                "stability_status": stability_status,
                "holdout_status": "confirmed" if best_score >= 0.3 else "failed",
            }
        )
    return mapping_rows, stability_rows


def validate_latent_runs(
    analysis_df: pd.DataFrame,
    attributes_df: pd.DataFrame,
    run_dirs: list[Path],
    device: torch.device | None = None,
    mig_method: str = "sklearn",
    mig_bins: int = 16,
    mig_binning: str = "quantile",
    sap_method: str = "sklearn",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate latent runs and return metrics, mappings, and stability outputs."""
    wide_attributes = pivot_validation_attributes(attributes_df)
    aligned_analysis = align_holdout_frames(analysis_df, wide_attributes)
    aligned_attributes = align_holdout_frames(wide_attributes, analysis_df)

    metrics_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        latent_df, config = load_run_latents(run_dir, aligned_analysis, device=device)
        metrics = evaluate_latent_metrics(
            latent_df,
            aligned_attributes,
            mig_method=mig_method,
            mig_bins=mig_bins,
            mig_binning=mig_binning,
            sap_method=sap_method,
        )
        run_id = run_dir.name
        model_type = str(config.get("arch", "baseline"))
        metrics_rows.append(
            {
                "run_id": run_id,
                "model_type": model_type,
                **metrics,
            }
        )
        run_mapping_rows, run_stability_rows = _build_factor_mapping_rows(
            run_id,
            model_type,
            latent_df,
            aligned_attributes,
        )
        mapping_rows.extend(run_mapping_rows)
        stability_rows.extend(run_stability_rows)

    return metrics_rows, mapping_rows, stability_rows


def validate_latent_factors(*, args: Any) -> None:
    """Run latent-factor validation against observed attributes."""
    analysis_df = pd.read_parquet(args.analysis_data)
    attributes_df = pd.read_parquet(args.attributes)
    if getattr(args, "household_splits", None) is not None:
        splits = load_household_splits(args.household_splits)
        split_role = getattr(args, "split_role", "eval")
        allowed_households = household_ids_for_role(splits, split_role)
        analysis_df = filter_households(analysis_df, allowed_households)
        attributes_df = filter_households(attributes_df, allowed_households)
    run_dirs = [Path(run_id) if Path(run_id).exists() else Path("experiments") / run_id for run_id in args.run_ids]

    metrics_rows, mapping_rows, stability_rows = validate_latent_runs(
        analysis_df=analysis_df,
        attributes_df=attributes_df,
        run_dirs=run_dirs,
        mig_method=getattr(args, "mig_method", "sklearn"),
        mig_bins=getattr(args, "mig_bins", 16),
        mig_binning=getattr(args, "mig_binning", "quantile"),
        sap_method=getattr(args, "sap_method", "sklearn"),
    )
    output_dir = Path(args.output_dir)
    serialize_latent_validation_outputs(output_dir, metrics_rows, mapping_rows, stability_rows)

    print("\n" + "=" * 50 + "\nLATENT VALIDATION SUMMARY\n" + "=" * 50)
    print(f"Runs: {[run_dir.name for run_dir in run_dirs]}")
    print(f"Metrics rows: {len(metrics_rows)}")
    print(f"Factor mappings: {len(mapping_rows)}")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 50 + "\n")

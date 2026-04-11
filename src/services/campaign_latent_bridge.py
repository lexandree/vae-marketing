"""Connect campaign attribute shifts to validated latent mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.services.validation_reporting import write_json_artifact


def build_campaign_latent_bridge(*, args: Any) -> None:
    """Join campaign attribute shifts with validated latent mappings."""
    with open(args.campaign_results, "r") as f:
        campaign_results = json.load(f)
    with open(args.factor_mappings, "r") as f:
        factor_mappings = json.load(f)

    attributes_df = pd.read_parquet(args.attributes)
    validated = {
        row["candidate_attribute"]: row
        for row in factor_mappings
        if row.get("mapping_decision") == "validated" and row.get("candidate_attribute")
    }

    wide_attrs = (
        attributes_df.pivot_table(
            index=["HOUSEHOLD_KEY", "CAMPAIGN", "window_type"],
            columns="attribute_name",
            values="attribute_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_attrs.columns.name = None

    rows: list[dict[str, object]] = []
    for result in campaign_results:
        campaign_id = result["CAMPAIGN"]
        campaign_slice = wide_attrs[wide_attrs["CAMPAIGN"] == campaign_id].copy()
        if campaign_slice.empty:
            continue
        pre_slice = (
            campaign_slice[campaign_slice["window_type"] == "pre"]
            .drop(columns=["window_type"])
            .groupby("CAMPAIGN")
            .mean(numeric_only=True)
        )
        in_slice = (
            campaign_slice[campaign_slice["window_type"] == "campaign"]
            .drop(columns=["window_type"])
            .groupby("CAMPAIGN")
            .mean(numeric_only=True)
        )
        if pre_slice.empty or in_slice.empty:
            continue
        delta = (in_slice - pre_slice).iloc[0].to_dict()
        ranked = sorted(delta.items(), key=lambda item: abs(float(item[1])), reverse=True)
        for attribute_name, delta_value in ranked[: args.top_k_attributes]:
            mapping = validated.get(attribute_name, {})
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "outcome_name": result["outcome_name"],
                    "evidence_classification": result["evidence_classification"],
                    "attribute_name": attribute_name,
                    "campaign_delta": float(delta_value),
                    "latent_run_id": mapping.get("run_id"),
                    "latent_dimension": mapping.get("latent_dimension"),
                    "latent_association_strength": mapping.get("association_strength"),
                    "latent_mapping_decision": mapping.get("mapping_decision"),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "campaign_latent_bridge.json", rows)
    pd.DataFrame(rows).to_parquet(output_dir / "campaign_latent_bridge.parquet", index=False)

    print("\n" + "=" * 50 + "\nCAMPAIGN LATENT BRIDGE SUMMARY\n" + "=" * 50)
    print(f"Rows: {len(rows)}")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 50 + "\n")

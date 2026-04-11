"""Evidence-based validation reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json_artifact(path: Path, payload: Any) -> None:
    """Write a JSON artifact to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def serialize_campaign_validation_outputs(
    output_dir: Path,
    effects_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    event_study_df: pd.DataFrame,
    balance_details_df: pd.DataFrame | None = None,
    cohort_summary_df: pd.DataFrame | None = None,
) -> None:
    """Write campaign validation outputs in report-ready form."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "campaign_effects.json", effects_df.to_dict(orient="records"))
    write_json_artifact(
        output_dir / "campaign_diagnostics.json",
        diagnostics_df.to_dict(orient="records"),
    )
    event_study_df.to_parquet(output_dir / "campaign_event_study.parquet", index=False)
    if balance_details_df is not None:
        write_json_artifact(
            output_dir / "campaign_balance_details.json",
            balance_details_df.to_dict(orient="records"),
        )
    if cohort_summary_df is not None:
        write_json_artifact(
            output_dir / "campaign_cohort_summary.json",
            cohort_summary_df.to_dict(orient="records"),
        )


def serialize_latent_validation_outputs(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    stability: list[dict[str, Any]],
) -> None:
    """Write latent validation outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "latent_metrics.json", metrics)
    write_json_artifact(output_dir / "factor_mappings.json", mappings)
    write_json_artifact(output_dir / "latent_stability.json", stability)


def build_claim_recommendations(
    campaign_results: list[dict[str, Any]],
    latent_results: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Build keep/soften recommendations from campaign and latent findings."""
    campaign_keep = []
    campaign_withdraw = []
    for row in campaign_results:
        label = row.get("evidence_classification")
        descriptor = f"Campaign {row.get('CAMPAIGN')} ({row.get('outcome_name')})"
        if label == "supported":
            campaign_keep.append(descriptor)
        else:
            campaign_withdraw.append(descriptor)

    latent_keep = []
    latent_withdraw = []
    for row in latent_results:
        descriptor = f"{row.get('run_id')} latent {row.get('latent_dimension')}"
        if row.get("mapping_decision") == "validated":
            latent_keep.append(descriptor)
        else:
            latent_withdraw.append(descriptor)

    return {
        "campaign_claims": {
            "keep": campaign_keep,
            "withdraw_or_soften": campaign_withdraw,
        },
        "latent_claims": {
            "keep": latent_keep,
            "withdraw_or_soften": latent_withdraw,
        },
    }


def build_validation_report_markdown(
    *,
    campaign_results: list[dict[str, Any]],
    campaign_diagnostics: list[dict[str, Any]],
    latent_results: list[dict[str, Any]],
    latent_metrics: list[dict[str, Any]],
    claim_recommendations: dict[str, dict[str, list[str]]],
) -> str:
    """Build a markdown validation report."""
    lines = [
        "# Validation Report",
        "",
        "## Campaign Findings",
    ]
    for row in campaign_results:
        lines.append(
            f"- Campaign {row.get('CAMPAIGN')} / {row.get('outcome_name')}: "
            f"effect={row.get('effect_size')} label={row.get('evidence_classification')}"
        )

    lines.extend(["", "## Campaign Diagnostics"])
    for row in campaign_diagnostics:
        lines.append(
            f"- Campaign {row.get('CAMPAIGN')} / {row.get('outcome_name')}: "
            f"balance_pass={row.get('balance_pass')} placebo_pass={row.get('placebo_pass')}"
        )

    lines.extend(["", "## Latent Metrics"])
    for row in latent_metrics:
        lines.append(
            f"- Run {row.get('run_id')} ({row.get('model_type')}): "
            f"MIG={row.get('mig_score')} SAP={row.get('sap_score')}"
        )

    lines.extend(["", "## Latent Factor Mappings"])
    for row in latent_results:
        lines.append(
            f"- Run {row.get('run_id')} latent {row.get('latent_dimension')}: "
            f"{row.get('candidate_attribute')} -> {row.get('mapping_decision')}"
        )

    lines.extend(
        [
            "",
            "## Claim Recommendations",
            "",
            "### Campaign Claims To Keep",
        ]
    )
    for item in claim_recommendations["campaign_claims"]["keep"]:
        lines.append(f"- {item}")

    lines.extend(["", "### Campaign Claims To Withdraw Or Soften"])
    for item in claim_recommendations["campaign_claims"]["withdraw_or_soften"]:
        lines.append(f"- {item}")

    lines.extend(["", "### Latent Claims To Keep"])
    for item in claim_recommendations["latent_claims"]["keep"]:
        lines.append(f"- {item}")

    lines.extend(["", "### Latent Claims To Withdraw Or Soften"])
    for item in claim_recommendations["latent_claims"]["withdraw_or_soften"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def generate_validation_report(*, args: Any) -> None:
    """Generate the campaign and latent validation report."""
    with open(args.campaign_results, "r") as f:
        campaign_results = json.load(f)
    with open(args.campaign_diagnostics, "r") as f:
        campaign_diagnostics = json.load(f)
    with open(args.latent_results, "r") as f:
        latent_results = json.load(f)
    with open(args.latent_metrics, "r") as f:
        latent_metrics = json.load(f)

    claim_recommendations = build_claim_recommendations(campaign_results, latent_results)
    report_text = build_validation_report_markdown(
        campaign_results=campaign_results,
        campaign_diagnostics=campaign_diagnostics,
        latent_results=latent_results,
        latent_metrics=latent_metrics,
        claim_recommendations=claim_recommendations,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)
    write_json_artifact(output_path.parent / "claim_recommendations.json", claim_recommendations)

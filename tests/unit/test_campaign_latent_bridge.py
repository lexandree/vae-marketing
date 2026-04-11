import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.services.campaign_latent_bridge import build_campaign_latent_bridge


def test_build_campaign_latent_bridge_writes_expected_rows(tmp_path: Path) -> None:
    campaign_results = tmp_path / "campaign_effects.json"
    factor_mappings = tmp_path / "factor_mappings.json"
    attributes = tmp_path / "validation_attributes.parquet"
    output_dir = tmp_path / "out"

    campaign_results.write_text(
        json.dumps(
            [
                {
                    "CAMPAIGN": 26,
                    "outcome_name": "total_spend",
                    "evidence_classification": "supported",
                }
            ]
        )
    )
    factor_mappings.write_text(
        json.dumps(
            [
                {
                    "run_id": "beta-best",
                    "latent_dimension": 3,
                    "candidate_attribute": "promo_share",
                    "association_strength": 0.42,
                    "mapping_decision": "validated",
                }
            ]
        )
    )
    pd.DataFrame(
        {
            "HOUSEHOLD_KEY": [1, 1, 1, 1],
            "CAMPAIGN": [26, 26, 26, 26],
            "window_type": ["pre", "campaign", "pre", "campaign"],
            "attribute_name": ["promo_share", "promo_share", "trip_count", "trip_count"],
            "attribute_value": [0.1, 0.4, 1.0, 2.0],
        }
    ).to_parquet(attributes, index=False)

    args = SimpleNamespace(
        campaign_results=campaign_results,
        factor_mappings=factor_mappings,
        attributes=attributes,
        output_dir=output_dir,
        top_k_attributes=2,
    )

    build_campaign_latent_bridge(args=args)

    bridge_json = output_dir / "campaign_latent_bridge.json"
    bridge_parquet = output_dir / "campaign_latent_bridge.parquet"
    assert bridge_json.exists()
    assert bridge_parquet.exists()

    payload = json.loads(bridge_json.read_text())
    assert payload
    assert any(row["attribute_name"] == "promo_share" for row in payload)

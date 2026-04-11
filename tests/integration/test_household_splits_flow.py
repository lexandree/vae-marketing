import json
from pathlib import Path

import polars as pl
import pytest

from main import main


def test_build_household_splits_cli_writes_split_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_table = tmp_path / "campaign_table.csv"
    pl.DataFrame(
        {
            "DESCRIPTION": ["A", "B", "C", "D"],
            "HOUSEHOLD_KEY": [1, 2, 3, 4],
            "CAMPAIGN": [26, 30, 8, 13],
        }
    ).write_csv(campaign_table)

    output = tmp_path / "household_splits.json"
    test_args = [
        "main.py",
        "build-household-splits",
        "--campaign-table",
        str(campaign_table),
        "--eval-campaign-ids",
        "26",
        "30",
        "--output",
        str(output),
        "--seed",
        "42",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["eval_campaign_ids"] == [26, 30]
    assert set(payload["eval_households"]) == {1, 2}

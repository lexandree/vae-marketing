import json
from pathlib import Path

import polars as pl

from src.data.splits import (
    build_household_splits,
    filter_households,
    household_ids_for_role,
    load_household_splits,
    write_household_splits,
)


def test_build_household_splits_excludes_eval_campaign_households() -> None:
    campaign_table = pl.DataFrame(
        {
            "HOUSEHOLD_KEY": ["H1", "H2", "H3", "H4", "H5"],
            "CAMPAIGN": [26, 30, 8, 13, 18],
        }
    )

    splits = build_household_splits(campaign_table, eval_campaign_ids=[26, 30], seed=42)

    assert set(splits["eval_households"]) == {"H1", "H2"}
    assert "H1" not in splits["train_households"]
    assert "H2" not in splits["validation_households"]
    assert set(household_ids_for_role(splits, "all")) == {
        *splits["train_households"],
        *splits["validation_households"],
        *splits["eval_households"],
    }


def test_write_and_load_household_splits_round_trip(tmp_path: Path) -> None:
    splits = {
        "eval_campaign_ids": [26, 30],
        "eval_households": ["H1"],
        "train_households": ["H2"],
        "validation_households": ["H3"],
        "metadata": {"split_seed": 42},
    }
    output = tmp_path / "splits.json"

    write_household_splits(splits, output)
    loaded = load_household_splits(output)

    assert loaded == json.loads(output.read_text())
    assert loaded["eval_campaign_ids"] == [26, 30]


def test_filter_households_preserves_frame_type() -> None:
    frame = pl.DataFrame({"HOUSEHOLD_KEY": ["H1", "H2"], "x": [1, 2]})
    filtered = filter_households(frame, ["H2"])

    assert isinstance(filtered, pl.DataFrame)
    assert filtered["HOUSEHOLD_KEY"].to_list() == ["H2"]

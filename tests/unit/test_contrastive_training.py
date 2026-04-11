from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.services.contrastive_training import train_contrastive_vae


def test_train_contrastive_vae_writes_run_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "target.parquet"
    background = tmp_path / "background.parquet"

    base = {
        "HOUSEHOLD_KEY": ["1", "2", "3", "4"],
        "WINDOW_START_DAY": [0, 7, 14, 21],
        "COMMODITY_A_SPEND": [1.0, 0.8, 0.6, 0.4],
        "COMMODITY_A_QTY": [1.0, 1.0, 1.0, 1.0],
        "TEMPORAL_WEEK_SIN": [0.0, 0.5, -0.5, 0.0],
        "TEMPORAL_WEEK_COS": [1.0, 0.5, 0.5, 1.0],
        "CAMPAIGN": [8, 8, 8, 8],
        "contrastive_role": ["target", "target", "target", "target"],
    }
    pd.DataFrame(base).to_parquet(target, index=False)
    bg = dict(base)
    bg["COMMODITY_A_SPEND"] = [0.1, 0.2, 0.1, 0.2]
    bg["contrastive_role"] = ["background"] * 4
    pd.DataFrame(bg).to_parquet(background, index=False)

    args = SimpleNamespace(
        target_data=target,
        background_data=background,
        run_id="contrastive-test-run",
        shared_dim=2,
        salient_dim=2,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        beta_shared=1.0,
        beta_salient=1.0,
        salient_background_weight=1.0,
        verbosity=0,
    )

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        train_contrastive_vae(args=args)
        run_dir = Path("experiments") / args.run_id
        assert (run_dir / "config.json").exists()
        assert (run_dir / "best_model.pth").exists()
        assert (run_dir / "metrics.json").exists()
    finally:
        os.chdir(original_cwd)

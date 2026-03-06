import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb

logger = logging.getLogger(__name__)


def init_wandb(
    project_name: str,
    run_id: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    verbosity: int = 1
) -> None:
    """Initialize Weights & Biases logging with configurable verbosity.

    Args:
        project_name: Name of the WandB project.
        run_id: Unique identifier for the run.
        config: Experiment hyperparameters.
        entity: WandB entity name.
        verbosity: 0 for silent, 1 for normal, 2 for verbose.
    """
    if verbosity == 0:
        os.environ["WANDB_QUIET"] = "true"
        settings = wandb.Settings(silent=True, console="off")
    else:
        settings = wandb.Settings(silent=False, console="wrap")

    wandb.init(
        project=project_name,
        name=run_id,
        config=config,
        entity=entity,
        reinit=True,
        settings=settings
    )


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to WandB."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def save_artifact(file_path: Path, artifact_name: str, artifact_type: str) -> None:
    """Save a file as a WandB artifact."""
    if wandb.run is not None:
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(file_path))
        wandb.log_artifact(artifact)


def finish_logging(verbosity: int = 1) -> None:
    """Finish WandB logging."""
    if wandb.run is not None:
        quiet = (verbosity == 0)
        wandb.finish(quiet=quiet)

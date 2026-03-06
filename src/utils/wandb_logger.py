import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb

logger = logging.getLogger(__name__)


def init_wandb(
    project_name: str,
    run_id: str,
    config: Dict[str, Any],
    entity: Optional[str] = None
) -> None:
    """Initialize Weights & Biases logging.

    Args:
        project_name: Name of the WandB project.
        run_id: Unique identifier for the run.
        config: Dictionary containing experiment hyperparameters.
        entity: WandB entity (user or team) name.
    """
    wandb.init(
        project=project_name,
        name=run_id,
        config=config,
        entity=entity,
        reinit=True
    )


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to WandB.

    Args:
        metrics: Dictionary of metrics to log.
        step: Optional training step/epoch number.
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def save_artifact(file_path: Path, artifact_name: str, artifact_type: str) -> None:
    """Save a file as a WandB artifact.

    Args:
        file_path: Path to the file to save.
        artifact_name: Name for the artifact.
        artifact_type: Type of artifact (e.g., 'model', 'dataset').
    """
    if wandb.run is not None:
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(file_path))
        wandb.log_artifact(artifact)


def save_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    epoch: int,
    is_best: bool = False
) -> None:
    """Save model checkpoint locally and to WandB."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)

    if is_best:
        best_path = run_dir / "best_model.pth"
        torch.save(model.state_dict(), best_path)
        save_artifact(best_path, "best_model", "model")


def finish_logging() -> None:
    """Finish WandB logging."""
    if wandb.run is not None:
        wandb.finish()

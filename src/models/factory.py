import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from src.models.baseline_vae import build_vae_model
from src.models.beta_vae import build_beta_vae_model


class ModelFactory:
    """Factory for creating and loading VAE models."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        """Creates a model based on the configuration."""
        arch = config.get("arch", "baseline")
        latent_dim = config.get("latent_dim", 16)
        num_categories = config.get("num_categories", 10)
        num_temporal = config.get("num_temporal_features", 6)

        if arch == "beta_vae":
            return build_beta_vae_model(latent_dim, num_categories, num_temporal)
        return build_vae_model(latent_dim, num_categories, num_temporal)

    @staticmethod
    def load_model(run_dir: Path, filename: str = "best_model.pth") -> nn.Module:
        """Loads a model and its weights from the run directory."""
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found in {run_dir}")

        with open(config_path, "r") as f:
            config = json.load(f)

        model = ModelFactory.create_model(config)
        weights_path = run_dir / filename
        if not weights_path.exists():
            # Fallback to checkpoint.pth if best_model.pth doesn't exist
            weights_path = run_dir / "checkpoint.pth"

        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found in {run_dir}")

        model.load_state_dict(torch.load(weights_path))
        return model

    @staticmethod
    def save_config(config: Dict[str, Any], run_dir: Path) -> None:
        """Save the experiment configuration to config.json.

        Args:
            config: Dictionary containing the configuration.
            run_dir: Path to the directory where the config will be saved.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def save_metrics(metrics: Dict[str, float], run_dir: Path) -> None:
        """Save evaluation metrics to metrics.json.

        Args:
            metrics: Dictionary containing the metrics.
            run_dir: Path to the directory where the metrics will be saved.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

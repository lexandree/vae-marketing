"""Models package containing VAE architectures and factory."""

from .baseline_vae import build_vae_model as build_baseline_vae
from .factory import ModelFactory

__all__ = [
    "build_baseline_vae",
    "ModelFactory",
]

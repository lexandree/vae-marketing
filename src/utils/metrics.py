import logging
import sys

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression


def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a standard logger for the application.

    Args:
        name: The name of the logger, typically __name__.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def calculate_mig(latent_factors: np.ndarray, attributes: np.ndarray) -> float:
    """Calculate the Mutual Information Gap (MIG) for disentanglement.

    Args:
        latent_factors: Array of shape (N, D) containing latent representations.
        attributes: Array of shape (N, K) containing ground truth attributes.

    Returns:
        The mean MIG score across all attributes.
    """
    if latent_factors.shape[0] != attributes.shape[0]:
        raise ValueError("Number of samples in latent_factors and attributes must match.")

    k_attr = attributes.shape[1]
    d_latent = latent_factors.shape[1]

    if d_latent < 2:
        return 0.0

    mi_matrix = np.zeros((k_attr, d_latent))
    for k in range(k_attr):
        # We assume attributes are continuous, so we use mutual_info_regression
        mi_matrix[k, :] = mutual_info_regression(latent_factors, attributes[:, k])

    mig_scores = []
    for k in range(k_attr):
        sorted_mi = np.sort(mi_matrix[k, :])
        gap = sorted_mi[-1] - sorted_mi[-2]
        # Normalize by maximum MI (proxy for entropy of continuous attribute)
        norm = sorted_mi[-1] if sorted_mi[-1] > 0 else 1.0
        mig_scores.append(gap / norm)

    return float(np.mean(mig_scores))


def calculate_sap(latent_factors: np.ndarray, attributes: np.ndarray) -> float:
    """Calculate the Separated Attribute Predictability (SAP) score.

    Args:
        latent_factors: Array of shape (N, D) containing latent representations.
        attributes: Array of shape (N, K) containing ground truth attributes.

    Returns:
        The mean SAP score across all attributes.
    """
    if latent_factors.shape[0] != attributes.shape[0]:
        raise ValueError("Number of samples in latent_factors and attributes must match.")

    k_attr = attributes.shape[1]
    d_latent = latent_factors.shape[1]

    if d_latent < 2:
        return 0.0

    score_matrix = np.zeros((k_attr, d_latent))
    for k in range(k_attr):
        for d in range(d_latent):
            # Predict attribute k using only latent factor d
            reg = LinearRegression().fit(latent_factors[:, d:d+1], attributes[:, k])
            # Use R^2 score
            score = reg.score(latent_factors[:, d:d+1], attributes[:, k])
            score_matrix[k, d] = max(0.0, score)

    sap_scores = []
    for k in range(k_attr):
        sorted_scores = np.sort(score_matrix[k, :])
        gap = sorted_scores[-1] - sorted_scores[-2]
        sap_scores.append(gap)

    return float(np.mean(sap_scores))


def validate_model_architecture(baseline_metrics: dict, beta_metrics: dict) -> bool:
    """Validate Beta-VAE against Baseline VAE according to Stage 003 Success Criteria.

    SC-002: Beta-VAE achieves a MIG score > 0.15 higher than the Baseline VAE.
    SC-003: Reconstruction error (MSE) for Beta-VAE is within 15% of the Baseline VAE's error.

    Args:
        baseline_metrics: Dictionary containing metrics for the baseline model.
        beta_metrics: Dictionary containing metrics for the Beta-VAE model.

    Returns:
        True if all criteria are met, False otherwise.
    """
    logger = logging.getLogger(__name__)

    base_mig = baseline_metrics.get("mig_score", 0.0)
    beta_mig = beta_metrics.get("mig_score", 0.0)
    mig_diff = beta_mig - base_mig

    base_mse = baseline_metrics.get("mse_loss", float('inf'))
    beta_mse = baseline_metrics.get("mse_loss", float('inf'))

    # Check SC-002
    mig_passed = mig_diff > 0.15
    if not mig_passed:
        logger.warning(f"SC-002 Failed: MIG improvement {mig_diff:.4f} <= 0.15")

    # Check SC-003 (Beta MSE <= 1.15 * Base MSE)
    mse_passed = beta_mse <= base_mse * 1.15
    if not mse_passed:
        logger.warning(
            f"SC-003 Failed: Beta MSE {beta_mse:.4f} > 15% worse than Base MSE {base_mse:.4f}"
        )

    passed = mig_passed and mse_passed
    if passed:
        logger.info("All Success Criteria met successfully.")

    return passed

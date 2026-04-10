import logging
import sys

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression


_BINNED_COLUMN_CACHE: dict[tuple[int, tuple[int, ...], str, int, str], np.ndarray] = {}


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


def _digitize_equal_width(values: np.ndarray, bins: int) -> np.ndarray:
    """Digitize a 1D array into equal-width bins."""
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or min_value == max_value:
        return np.zeros(values.shape[0], dtype=np.int32)

    edges = np.linspace(min_value, max_value, num=bins + 1)
    interior = edges[1:-1]
    return np.digitize(values, interior, right=False).astype(np.int32)


def _digitize_quantiles(values: np.ndarray, bins: int) -> np.ndarray:
    """Digitize a 1D array into quantile bins."""
    quantiles = np.linspace(0.0, 1.0, num=bins + 1)
    edges = np.quantile(values, quantiles)
    interior = np.unique(edges[1:-1])
    if interior.size == 0:
        return np.zeros(values.shape[0], dtype=np.int32)
    return np.digitize(values, interior, right=False).astype(np.int32)


def _digitize_columns(values: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """Digitize each column in a 2D array using the selected strategy."""
    if bins < 2:
        raise ValueError("bins must be at least 2")

    digitized = np.zeros(values.shape, dtype=np.int32)
    for column_idx in range(values.shape[1]):
        column = np.asarray(values[:, column_idx], dtype=float)
        if strategy == "quantile":
            digitized[:, column_idx] = _digitize_quantiles(column, bins)
        elif strategy == "uniform":
            digitized[:, column_idx] = _digitize_equal_width(column, bins)
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")
    return digitized


def _binned_cache_key(values: np.ndarray, bins: int, strategy: str) -> tuple[int, tuple[int, ...], str, int, str]:
    """Build a process-local cache key for a binned 2D array."""
    array = np.asarray(values)
    pointer = int(array.__array_interface__["data"][0])
    return (pointer, array.shape, str(array.dtype), bins, strategy)


def clear_binned_cache() -> None:
    """Clear the process-local cache for discretized matrices."""
    _BINNED_COLUMN_CACHE.clear()


def get_binned_columns_cached(values: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """Return discretized columns from the process-local cache when available."""
    array = np.asarray(values, dtype=float)
    key = _binned_cache_key(array, bins, strategy)
    cached = _BINNED_COLUMN_CACHE.get(key)
    if cached is None:
        cached = _digitize_columns(array, bins=bins, strategy=strategy)
        _BINNED_COLUMN_CACHE[key] = cached
    return cached


def calculate_mig_binned(
    latent_factors: np.ndarray,
    attributes: np.ndarray,
    bins: int = 16,
    strategy: str = "quantile",
) -> float:
    """Calculate an approximate MIG using discretized columns and mutual_info_score."""
    if latent_factors.shape[0] != attributes.shape[0]:
        raise ValueError("Number of samples in latent_factors and attributes must match.")

    k_attr = attributes.shape[1]
    d_latent = latent_factors.shape[1]

    if d_latent < 2:
        return 0.0

    latent_binned = get_binned_columns_cached(latent_factors, bins=bins, strategy=strategy)
    attr_binned = get_binned_columns_cached(attributes, bins=bins, strategy=strategy)

    mi_matrix = np.zeros((k_attr, d_latent), dtype=float)
    for k in range(k_attr):
        for d in range(d_latent):
            mi_matrix[k, d] = mutual_info_score(attr_binned[:, k], latent_binned[:, d])

    mig_scores = []
    for k in range(k_attr):
        sorted_mi = np.sort(mi_matrix[k, :])
        gap = sorted_mi[-1] - sorted_mi[-2]
        norm = sorted_mi[-1] if sorted_mi[-1] > 0 else 1.0
        mig_scores.append(gap / norm)

    return float(np.mean(mig_scores))


def calculate_mig_with_method(
    latent_factors: np.ndarray,
    attributes: np.ndarray,
    method: str = "sklearn",
    bins: int = 16,
    strategy: str = "quantile",
) -> float:
    """Calculate MIG with a selectable implementation backend."""
    if method == "sklearn":
        return calculate_mig(latent_factors, attributes)
    if method == "binned":
        return calculate_mig_binned(latent_factors, attributes, bins=bins, strategy=strategy)
    raise ValueError(f"Unsupported MIG method: {method}")


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


def calculate_sap_vectorized(latent_factors: np.ndarray, attributes: np.ndarray) -> float:
    """Calculate SAP using vectorized correlation-based R^2.

    For univariate linear regression with intercept, the in-sample R^2 equals the
    squared Pearson correlation between the predictor and target. This provides a
    much cheaper alternative to fitting a separate sklearn regression per
    (attribute, latent-dimension) pair while preserving the same scoring target.
    """
    if latent_factors.shape[0] != attributes.shape[0]:
        raise ValueError("Number of samples in latent_factors and attributes must match.")

    k_attr = attributes.shape[1]
    d_latent = latent_factors.shape[1]

    if d_latent < 2:
        return 0.0

    latent = np.asarray(latent_factors, dtype=float)
    attrs = np.asarray(attributes, dtype=float)

    latent_centered = latent - latent.mean(axis=0, keepdims=True)
    attrs_centered = attrs - attrs.mean(axis=0, keepdims=True)

    latent_norm = np.linalg.norm(latent_centered, axis=0)
    attr_norm = np.linalg.norm(attrs_centered, axis=0)
    denom = np.outer(attr_norm, latent_norm)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (attrs_centered.T @ latent_centered) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    score_matrix = np.clip(corr ** 2, 0.0, None)

    sap_scores = []
    for k in range(k_attr):
        sorted_scores = np.sort(score_matrix[k, :])
        gap = sorted_scores[-1] - sorted_scores[-2]
        sap_scores.append(gap)

    return float(np.mean(sap_scores))


def calculate_sap_with_method(
    latent_factors: np.ndarray,
    attributes: np.ndarray,
    method: str = "sklearn",
) -> float:
    """Calculate SAP with a selectable implementation backend."""
    if method == "sklearn":
        return calculate_sap(latent_factors, attributes)
    if method == "vectorized":
        return calculate_sap_vectorized(latent_factors, attributes)
    raise ValueError(f"Unsupported SAP method: {method}")


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

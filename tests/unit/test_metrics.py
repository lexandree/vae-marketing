import numpy as np

from src.utils.metrics import (
    calculate_mig,
    calculate_mig_with_method,
    calculate_sap,
    calculate_sap_with_method,
    clear_binned_cache,
    get_binned_columns_cached,
)


def test_calculate_mig() -> None:
    # 1-to-1 mapping
    latent = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    attr = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    mig_score = calculate_mig(latent, attr)
    assert isinstance(mig_score, float)
    assert mig_score >= 0.0


def test_calculate_mig_binned() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(128, 3))
    attr = np.column_stack(
        [
            latent[:, 0] + 0.05 * rng.normal(size=128),
            latent[:, 1] - 0.10 * rng.normal(size=128),
        ]
    )

    mig_score = calculate_mig_with_method(
        latent,
        attr,
        method="binned",
        bins=16,
        strategy="quantile",
    )
    assert isinstance(mig_score, float)
    assert mig_score >= 0.0

def test_calculate_sap() -> None:
    # Create a clear 1-to-1 mapping that regression can solve
    # latent has shape (4, 2)
    latent = np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0]
    ])
    # attr has shape (4, 2)
    attr = np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0]
    ])

    sap_score = calculate_sap(latent, attr)
    assert isinstance(sap_score, float)
    # The gap should be 1.0 - something small, so > 0.5
    assert sap_score > 0.5


def test_calculate_sap_vectorized_tracks_sklearn() -> None:
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(64, 4))
    attr = np.column_stack(
        [
            0.9 * latent[:, 0] + 0.1 * rng.normal(size=64),
            -0.7 * latent[:, 1] + 0.2 * rng.normal(size=64),
            0.5 * latent[:, 2] - 0.3 * latent[:, 3] + 0.2 * rng.normal(size=64),
        ]
    )

    sklearn_score = calculate_sap_with_method(latent, attr, method="sklearn")
    vectorized_score = calculate_sap_with_method(latent, attr, method="vectorized")

    assert isinstance(vectorized_score, float)
    assert abs(sklearn_score - vectorized_score) < 1e-9


def test_binned_cache_reuses_same_array() -> None:
    clear_binned_cache()
    values = np.arange(24, dtype=float).reshape(8, 3)

    first = get_binned_columns_cached(values, bins=4, strategy="quantile")
    second = get_binned_columns_cached(values, bins=4, strategy="quantile")

    assert first is second

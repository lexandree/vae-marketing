import numpy as np

from src.utils.metrics import calculate_mig, calculate_sap


def test_calculate_mig() -> None:
    # 1-to-1 mapping
    latent = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    attr = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    mig_score = calculate_mig(latent, attr)
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

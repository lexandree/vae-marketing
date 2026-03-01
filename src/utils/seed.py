import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Sets the global random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The integer seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


def train_val_test_split_indices(
    n_samples: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Returns indices for train, validation, and test split.
    """
    idx_all = np.arange(n_samples)
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=(test_size + val_size), random_state=random_state, shuffle=True
    )

    relative_val_size = val_size / (test_size + val_size)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1 - relative_val_size),
        random_state=random_state,
        shuffle=True,
    )

    return {"train": idx_train, "val": idx_val, "test": idx_test}


def split_arrays_by_indices(
    X_rna: np.ndarray,
    X_cnv: np.ndarray,
    X_meth: np.ndarray,
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split omics arrays and labels according to indices.
    """
    out = {}
    for split_name, idx in splits.items():
        out[split_name] = {
            "rna": X_rna[idx],
            "cnv": X_cnv[idx],
            "meth": X_meth[idx],
            "y": y[idx] if y is not None else None,
        }
    return out

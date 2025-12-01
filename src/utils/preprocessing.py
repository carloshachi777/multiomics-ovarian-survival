import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional


def load_omics_from_csv(
    path_rna: str,
    path_cnv: str,
    path_meth: str,
    index_col: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load omics matrices (features x samples) from CSV.

    Assumes:
    - rows: features (genes, probes, etc.)
    - columns: samples (TCGA barcodes)
    """
    rna = pd.read_csv(path_rna, index_col=index_col)
    cnv = pd.read_csv(path_cnv, index_col=index_col)
    meth = pd.read_csv(path_meth, index_col=index_col)
    return rna, cnv, meth


def align_omics_by_samples(
    rna: pd.DataFrame,
    cnv: pd.DataFrame,
    meth: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match samples by common TCGA barcodes across all omics layers.
    """
    common_samples = list(set(rna.columns) & set(cnv.columns) & set(meth.columns))
    common_samples.sort()
    rna_aligned = rna[common_samples]
    cnv_aligned = cnv[common_samples]
    meth_aligned = meth[common_samples]
    return rna_aligned, cnv_aligned, meth_aligned


def zscore_per_sample(
    df: pd.DataFrame,
    winsorize: bool = False,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Z-score normalization by feature across samples.
    Optionally winsorize extreme values.

    This is just one reasonable default – adjust based on your manuscript.
    """
    values = df.values

    if winsorize:
        lower = np.quantile(values, lower_q)
        upper = np.quantile(values, upper_q)
        values = np.clip(values, lower, upper)

    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True) + 1e-8

    z = (values - mean) / std
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def scale_omics(
    rna: pd.DataFrame,
    cnv: pd.DataFrame,
    meth: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, StandardScaler]]:
    """
    Apply StandardScaler to each omics matrix (samples x features).
    Returns:
    - X_rna, X_cnv, X_meth (numpy arrays, shape (n_samples, n_features))
    - scalers dict
    """
    # transpose so that rows are samples (n_samples, n_features)
    rna_T = rna.T
    cnv_T = cnv.T
    meth_T = meth.T

    scaler_rna = StandardScaler()
    scaler_cnv = StandardScaler()
    scaler_meth = StandardScaler()

    X_rna = scaler_rna.fit_transform(rna_T.values)
    X_cnv = scaler_cnv.fit_transform(cnv_T.values)
    X_meth = scaler_meth.fit_transform(meth_T.values)

    scalers = {
        "rna": scaler_rna,
        "cnv": scaler_cnv,
        "meth": scaler_meth,
    }

    return X_rna, X_cnv, X_meth, scalers


def load_labels(
    path_labels: str,
    sample_column: str = "sample",
    label_column: str = "label",
    index_col: Optional[int] = None,
) -> pd.Series:
    """
    Loads labels from a CSV.

    Expected columns:
      - 'sample': TCGA sample ID
      - 'label': class label / outcome

    Returns Series indexed by sample ID.
    """
    df = pd.read_csv(path_labels, index_col=index_col)
    df = df[[sample_column, label_column]].dropna()
    df = df.set_index(sample_column)
    return df[label_column]


def align_labels_with_omics(
    labels: pd.Series,
    omics_df: pd.DataFrame,
) -> np.ndarray:
    """
    Align label vector with omics samples (columns of omics_df).
    Returns a numpy array of labels in the same order as omics_df.columns.
    """
    samples = omics_df.columns
    y = labels.reindex(samples)

    # drop samples without labels if any
    mask = ~y.isna()
    if not mask.all():
        # we assume you will filter omics matrices externally if needed
        y = y[mask]

    return y.values.astype(int)

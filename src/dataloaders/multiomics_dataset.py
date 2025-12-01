import torch
from torch.utils.data import Dataset


class MultiOmicsDataset(Dataset):
    """
    PyTorch Dataset for TCGA multi-omics data.

    Expects:
    - X_rna:  np.ndarray shape (n_samples, n_features_rna)
    - X_cnv:  np.ndarray shape (n_samples, n_features_cnv)
    - X_meth: np.ndarray shape (n_samples, n_features_meth)
    - y:      np.ndarray shape (n_samples,) or None for unsupervised use
    """

    def __init__(self, X_rna, X_cnv, X_meth, y=None, device: str = "cpu"):
        super().__init__()
        assert len(X_rna) == len(X_cnv) == len(X_meth), \
            "All omics arrays must have the same number of samples."

        self.X_rna = torch.tensor(X_rna, dtype=torch.float32, device=device)
        self.X_cnv = torch.tensor(X_cnv, dtype=torch.float32, device=device)
        self.X_meth = torch.tensor(X_meth, dtype=torch.float32, device=device)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long, device=device)
        else:
            self.y = None

    def __len__(self):
        return self.X_rna.shape[0]

    def __getitem__(self, idx):
        x_rna = self.X_rna[idx]
        x_cnv = self.X_cnv[idx]
        x_meth = self.X_meth[idx]

        if self.y is None:
            return {"rna": x_rna, "cnv": x_cnv, "meth": x_meth}

        return {
            "rna": x_rna,
            "cnv": x_cnv,
            "meth": x_meth,
            "label": self.y[idx],
        }

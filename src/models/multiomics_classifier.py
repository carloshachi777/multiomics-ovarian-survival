import torch.nn as nn
import torch.nn.functional as F


class MultiOmicsClassifier(nn.Module):
    """
    Simple MLP classifier operating on latent space z.

    - input_dim: dimensionality of latent vector
    - num_classes: number of output classes (e.g., subtypes, risk groups)
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dims=None, dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [128, 64]

        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        h = self.backbone(z)
        logits = self.head(h)
        return logits

    @staticmethod
    def loss_function(logits, labels):
        return F.cross_entropy(logits, labels)

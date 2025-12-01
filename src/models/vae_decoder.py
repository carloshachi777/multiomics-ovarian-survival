import torch.nn as nn


class Decoder(nn.Module):
    """
    Generic MLP decoder block for a VAE.

    It maps latent vector z -> reconstructed feature vector
    (for one omics modality).
    """

    def __init__(self, latent_dim: int, hidden_dims, output_dim: int):
        super().__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [128, 256]

        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

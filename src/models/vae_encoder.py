import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_decoder import Decoder


class Encoder(nn.Module):
    """
    Generic MLP encoder block for a VAE.

    It maps a feature vector x -> (mu, logvar) in latent space.
    """

    def __init__(self, input_dim: int, hidden_dims, latent_dim: int):
        super().__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [256, 128]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(in_dim, latent_dim)
        self.logvar_layer = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class MultiOmicsVAE(nn.Module):
    """
    Multi-omics VAE with three encoders and three decoders
    (one branch per modality: RNA, CNV, methylation).

    During training:
      - Encode each omics separately -> mu, logvar
      - Sample z_rna, z_cnv, z_meth
      - Optionally average (or concatenate) z's

    Here we use a simple *shared latent* by averaging the three z's.
    """

    def __init__(
        self,
        dim_rna: int,
        dim_cnv: int,
        dim_meth: int,
        latent_dim: int = 64,
        enc_hidden=None,
        dec_hidden=None,
    ):
        super().__init__()

        if enc_hidden is None:
            enc_hidden = [512, 256]

        if dec_hidden is None:
            dec_hidden = [256, 512]

        # Encoders, one per omics
        self.encoder_rna = Encoder(dim_rna, enc_hidden, latent_dim)
        self.encoder_cnv = Encoder(dim_cnv, enc_hidden, latent_dim)
        self.encoder_meth = Encoder(dim_meth, enc_hidden, latent_dim)

        # Decoders, one per omics
        self.decoder_rna = Decoder(latent_dim, dec_hidden, dim_rna)
        self.decoder_cnv = Decoder(latent_dim, dec_hidden, dim_cnv)
        self.decoder_meth = Decoder(latent_dim, dec_hidden, dim_meth)

        self.latent_dim = latent_dim

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x_rna, x_cnv, x_meth):
        mu_rna, logvar_rna = self.encoder_rna(x_rna)
        mu_cnv, logvar_cnv = self.encoder_cnv(x_cnv)
        mu_meth, logvar_meth = self.encoder_meth(x_meth)

        # Shared latent mean/logvar as average (you can change to concatenation if desired)
        mu = (mu_rna + mu_cnv + mu_meth) / 3.0
        logvar = (logvar_rna + logvar_cnv + logvar_meth) / 3.0

        return mu, logvar, {
            "rna": (mu_rna, logvar_rna),
            "cnv": (mu_cnv, logvar_cnv),
            "meth": (mu_meth, logvar_meth),
        }

    def decode(self, z):
        x_rna_hat = self.decoder_rna(z)
        x_cnv_hat = self.decoder_cnv(z)
        x_meth_hat = self.decoder_meth(z)
        return x_rna_hat, x_cnv_hat, x_meth_hat

    def forward(self, x_rna, x_cnv, x_meth):
        mu, logvar, per_branch = self.encode(x_rna, x_cnv, x_meth)
        z = self._reparameterize(mu, logvar)
        x_rna_hat, x_cnv_hat, x_meth_hat = self.decode(z)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "x_rna_hat": x_rna_hat,
            "x_cnv_hat": x_cnv_hat,
            "x_meth_hat": x_meth_hat,
            "per_branch": per_branch,
        }

    @staticmethod
    def loss_function(
        x_rna,
        x_cnv,
        x_meth,
        x_rna_hat,
        x_cnv_hat,
        x_meth_hat,
        mu,
        logvar,
        beta: float = 1.0,
    ):
        """
        Standard VAE loss:
         - Reconstruction (MSE) over all omics
         - KL divergence term
        """
        recon_rna = F.mse_loss(x_rna_hat, x_rna, reduction="mean")
        recon_cnv = F.mse_loss(x_cnv_hat, x_cnv, reduction="mean")
        recon_meth = F.mse_loss(x_meth_hat, x_meth, reduction="mean")

        recon_loss = recon_rna + recon_cnv + recon_meth

        # KL divergence (mean over batch)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl, {
            "recon_total": recon_loss.detach().cpu().item(),
            "kl": kl.detach().cpu().item(),
        }

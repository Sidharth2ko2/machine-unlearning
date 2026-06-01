"""
Week 10: Conditional β-VAE (Cβ-VAE) for multi-class DKF.

The original β-VAE encodes forget and retain images without knowing which
class they belong to. With 10 forget classes the encoders see a mixed
signal and learn a blurry average of what "shared" vs "unique" means.

Fix: inject a learned class embedding into both encoders and the decoder
so each component receives an explicit class signal:

  encoder_s(x_f, embed(y_f)) → S_f   knows which forget class to encode shared features for
  encoder_u(x_r, embed(y_r)) → U_r   knows which retain class to encode unique features for
  decoder(S_f, U_r, embed(y_f)) → x_cf  knows which forget identity to strip out

This is a Conditional VAE (CVAE) extension of the β-VAE.
The β hyperparameter and ELBO loss are unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BETA, EMBED_DIM, LATENT_DIM, NUM_CLASSES


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),                                   # → [B, 2048]
        )
        # class embedding concatenated before projection to latent space
        self.fc_mu     = nn.Linear(128 * 4 * 4 + embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 + embed_dim, latent_dim)

    def forward(self, x, class_emb):
        h = torch.cat([self.net(x), class_emb], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim_s, latent_dim_u, embed_dim):
        super().__init__()
        # forget-class embedding tells the decoder which identity to suppress
        self.fc = nn.Linear(latent_dim_s + latent_dim_u + embed_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,   3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, s, u, class_emb):
        h = self.fc(torch.cat([s, u, class_emb], dim=1)).view(-1, 128, 4, 4)
        return self.net(h)


class CondBetaVAE(nn.Module):
    """Conditional β-VAE: class-aware disentanglement for multi-class forgetting."""

    def __init__(
        self,
        latent_dim_s = LATENT_DIM,
        latent_dim_u = LATENT_DIM,
        num_classes  = NUM_CLASSES,
        beta         = BETA,
        embed_dim    = EMBED_DIM,
    ):
        super().__init__()
        self.beta = beta
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.encoder_s   = ConvEncoder(latent_dim_s, embed_dim)
        self.encoder_u   = ConvEncoder(latent_dim_u, embed_dim)
        self.decoder     = ConvDecoder(latent_dim_s, latent_dim_u, embed_dim)
        self.classifier_o = nn.Linear(latent_dim_s, num_classes)
        self.classifier_y = nn.Linear(latent_dim_u, num_classes)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x_f, x_r, y_f, y_r):
        """
        x_f, y_f : forget images and their class labels
        x_r, y_r : retain images and their class labels

        Returns same tuple shape as the unconditional BetaVAE for drop-in
        compatibility with the training loop.
        """
        c_f = self.class_embed(y_f)   # [B, embed_dim]
        c_r = self.class_embed(y_r)   # [B, embed_dim]

        # Shared encoder sees forget image + forget class label
        mu_s,   logvar_s   = self.encoder_s(x_f, c_f)
        s_f = self.reparameterize(mu_s, logvar_s)

        # Unique encoder for retain samples (what makes y_r special)
        mu_u_r, logvar_u_r = self.encoder_u(x_r, c_r)
        u_r = self.reparameterize(mu_u_r, logvar_u_r)

        # Unique encoder for forget samples (needed for reconstruction only)
        mu_u_f, logvar_u_f = self.encoder_u(x_f, c_f)
        u_f = self.reparameterize(mu_u_f, logvar_u_f)

        x_recon = self.decoder(s_f, u_f, c_f)   # reconstruct forget image
        x_cf    = self.decoder(s_f, u_r, c_f)   # counterfactual: forget shared + retain unique

        pred_o = self.classifier_o(s_f)
        pred_y = self.classifier_y(u_r)

        kl_s = self.kl_divergence(mu_s,   logvar_s)
        kl_u = self.kl_divergence(mu_u_r, logvar_u_r)

        return x_recon, x_cf, pred_o, pred_y, kl_s, kl_u, s_f, u_r

    def compute_loss(self, x_f, x_r, y_r, y_f):
        x_recon, x_cf, pred_o, pred_y, kl_s, kl_u, _, _ = self(x_f, x_r, y_f, y_r)
        recon_loss = F.mse_loss(x_recon, x_f)
        cls_o      = F.cross_entropy(pred_o, y_f)
        cls_y      = F.cross_entropy(pred_y, y_r)
        kl_loss    = self.beta * (kl_s + kl_u)
        return recon_loss + cls_o + cls_y + kl_loss, x_cf.detach()

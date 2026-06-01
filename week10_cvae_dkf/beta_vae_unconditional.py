"""Unconditional β-VAE — same as Week 9, kept here for baseline comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BETA, LATENT_DIM, NUM_CLASSES


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim_s, latent_dim_u):
        super().__init__()
        self.fc = nn.Linear(latent_dim_s + latent_dim_u, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,   3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, s, u):
        h = self.fc(torch.cat([s, u], dim=1)).view(-1, 128, 4, 4)
        return self.net(h)


class BetaVAE(nn.Module):
    def __init__(self, latent_dim_s=LATENT_DIM, latent_dim_u=LATENT_DIM, num_classes=NUM_CLASSES, beta=BETA):
        super().__init__()
        self.beta        = beta
        self.encoder_s   = ConvEncoder(latent_dim_s)
        self.encoder_u   = ConvEncoder(latent_dim_u)
        self.decoder     = ConvDecoder(latent_dim_s, latent_dim_u)
        self.classifier_o = nn.Linear(latent_dim_s, num_classes)
        self.classifier_y = nn.Linear(latent_dim_u, num_classes)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x_f, x_r):
        mu_s,   logvar_s   = self.encoder_s(x_f)
        s_f = self.reparameterize(mu_s, logvar_s)
        mu_u_r, logvar_u_r = self.encoder_u(x_r)
        u_r = self.reparameterize(mu_u_r, logvar_u_r)
        mu_u_f, logvar_u_f = self.encoder_u(x_f)
        u_f = self.reparameterize(mu_u_f, logvar_u_f)
        x_recon = self.decoder(s_f, u_f)
        x_cf    = self.decoder(s_f, u_r)
        pred_o  = self.classifier_o(s_f)
        pred_y  = self.classifier_y(u_r)
        kl_s    = self.kl_divergence(mu_s,   logvar_s)
        kl_u    = self.kl_divergence(mu_u_r, logvar_u_r)
        return x_recon, x_cf, pred_o, pred_y, kl_s, kl_u, s_f, u_r

    def compute_loss(self, x_f, x_r, y_r, y_f):
        x_recon, x_cf, pred_o, pred_y, kl_s, kl_u, _, _ = self(x_f, x_r)
        recon_loss = F.mse_loss(x_recon, x_f)
        cls_o      = F.cross_entropy(pred_o, y_f)
        cls_y      = F.cross_entropy(pred_y, y_r)
        kl_loss    = self.beta * (kl_s + kl_u)
        return recon_loss + cls_o + cls_y + kl_loss, x_cf.detach()

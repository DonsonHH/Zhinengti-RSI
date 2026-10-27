"""Shared node conditional VAE; targets are accepted only by training forward."""
import torch
from torch import nn


class TemperatureCVAE(nn.Module):
    def __init__(self, inputs=56, width=64, latent=8):
        super().__init__()
        def encoder(n):
            return nn.Sequential(nn.Linear(n, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 2 * latent))
        self.prior = encoder(inputs)
        self.posterior = encoder(inputs + 1)
        self.decoder = nn.Sequential(nn.Linear(inputs + latent, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    @staticmethod
    def distribution(raw):
        mu, logvar = raw.chunk(2, dim=-1)
        return mu, logvar.clamp(-8, 4)

    def forward(self, context, target):
        p = self.distribution(self.prior(context))
        q = self.distribution(self.posterior(torch.cat((context, target), -1)))
        z = q[0] + torch.randn_like(q[0]) * torch.exp(q[1] / 2)
        prediction = self.decoder(torch.cat((context, z), -1))
        kl = .5 * (p[1] - q[1] + (q[1].exp() + (q[0] - p[0]).square()) / p[1].exp() - 1).mean()
        return ((prediction - target).square().mean() + .01 * kl), kl

    def predict(self, context):
        mean, _ = self.distribution(self.prior(context))
        return self.decoder(torch.cat((context, mean), -1))

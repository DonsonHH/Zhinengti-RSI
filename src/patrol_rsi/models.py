"""Small paper-derived networks for engineering smoke tests.

Widths and normalization differ from the paper. The prior/posterior and U-Net
structure, five policy channels, shared dueling DDQN and PER are retained.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def block(inside: int, outside: int) -> nn.Sequential:
    # Small sparse-input batches otherwise drove unnormalized deep features to
    # sigmoid saturation. GroupNorm is a declared scaled-reference adaptation.
    return nn.Sequential(nn.Conv2d(inside, outside, 3, padding=1), nn.GroupNorm(4, outside), nn.LeakyReLU(0.1),
                         nn.Conv2d(outside, outside, 3, padding=1), nn.GroupNorm(4, outside), nn.LeakyReLU(0.1))


class GaussianEncoder(nn.Module):
    def __init__(self, channels: int, latent: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, 16, 3, stride=2, padding=1),
                                      nn.LeakyReLU(0.1), nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                      nn.LeakyReLU(0.1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mu = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.features(value)
        return self.mu(hidden), self.logvar(hidden).clamp(-8, 6)


class ConditionalVAEUNet(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.encoder = nn.ModuleList([block(2, width), block(width, width * 2),
                                      block(width * 2, width * 4), block(width * 4, width * 8),
                                      block(width * 8, width * 16)])
        self.decoder = nn.ModuleList([block(width * 24, width * 8), block(width * 12, width * 4),
                                      block(width * 6, width * 2), block(width * 3, width)])
        self.prior = GaussianEncoder(2, width)
        self.posterior = GaussianEncoder(3, width)
        self.fusion = nn.Sequential(nn.Conv2d(width * 2, width, 1), nn.LeakyReLU(0.1),
                                    nn.Conv2d(width, width, 1), nn.LeakyReLU(0.1))
        self.output = nn.Conv2d(width, 1, 1)

    def _features(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.encoder[0](observation)
        skips = [x]
        for layer in self.encoder[1:]:
            x = layer(F.max_pool2d(x, 2))
            skips.append(x)
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        return x

    def _decode(self, observation: torch.Tensor, distribution: tuple[torch.Tensor, torch.Tensor],
                stochastic: bool) -> torch.Tensor:
        hidden = self._features(observation)
        mu, logvar = distribution
        latent = mu
        if stochastic:
            # One vector per conditional Gaussian, broadcast spatially. Drawing
            # independent noise at every pixel would not match the KL above.
            latent = latent + torch.randn_like(mu) * torch.exp(logvar / 2)
        latent = latent[:, :, None, None].expand_as(hidden)
        return torch.sigmoid(self.output(self.fusion(torch.cat([hidden, latent], dim=1))))

    def forward(self, observation: torch.Tensor, target: torch.Tensor):
        prior = self.prior(observation)
        posterior = self.posterior(torch.cat([observation, target], dim=1))
        return self._decode(observation, posterior, True), prior, posterior

    def predict(self, observation: torch.Tensor, samples: int = 1) -> torch.Tensor:
        """Inference uses partial observations only, never a target/posterior."""
        if samples < 1:
            raise ValueError("At least one prior sample is required")
        prior = self.prior(observation)
        return torch.stack([self._decode(observation, prior, True) for _ in range(samples)]).mean(0)


def vae_loss(output: torch.Tensor, target: torch.Tensor, prior, posterior,
             kl_weight: float = 0.01, perceptual_network: nn.Module | None = None,
             perceptual_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    if perceptual_weight > 0 and perceptual_network is None:
        raise ValueError("Perceptual loss requires explicitly staged pretrained features")
    mu_q, lv_q = posterior
    mu_p, lv_p = prior
    reconstruction = F.mse_loss(output, target)
    # KL(posterior || prior), as in the candidate implementation. Paper Eq.(10)
    # prints the opposite ordering; this discrepancy is recorded in provenance.
    kl = 0.5 * (lv_p - lv_q + ((lv_q.exp() + (mu_q - mu_p).square()) / lv_p.exp()) - 1).mean()
    features = output.new_tensor(0.0)
    if perceptual_weight:
        features = F.mse_loss(perceptual_network(output.repeat(1, 3, 1, 1)),
                              perceptual_network(target.repeat(1, 3, 1, 1)))
    total = reconstruction + kl_weight * kl + perceptual_weight * features
    return total, {"reconstruction": float(reconstruction.detach()), "kl": float(kl.detach()),
                   "perceptual": float(features.detach()), "total": float(total.detach())}


class SharedDuelingQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(),
                                     nn.Linear(1152, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, 8)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(observations)
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplay:
    def __init__(self, capacity: int = 1024, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.items: list[tuple] = []
        self.priorities = np.zeros(capacity, np.float64)
        self.cursor = 0

    def add(self, observation, action, reward, following, done, legal):
        item = (observation.copy(), int(action), float(reward), following.copy(), bool(done), legal.copy())
        priority = max(1.0, float(self.priorities[:len(self.items)].max(initial=1.0)))
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.cursor] = item
        self.priorities[self.cursor] = priority
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count: int):
        if not self.items:
            raise ValueError("Empty replay")
        probabilities = self.priorities[:len(self.items)] ** 0.6
        probabilities /= probabilities.sum()
        ids = self.rng.choice(len(self.items), count, replace=True, p=probabilities)
        weights = (len(self.items) * probabilities[ids]) ** -0.4
        weights /= weights.max()
        return ids, list(zip(*(self.items[int(i)] for i in ids))), weights.astype(np.float32)


def ddqn_update(online: SharedDuelingQ, target: SharedDuelingQ, optimizer,
                replay: PrioritizedReplay, device: torch.device, batch_size: int = 16) -> float:
    ids, batch, weights = replay.sample(batch_size)
    obs, actions, rewards, next_obs, dones, legal = batch
    obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device)
    following = torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=device)
    action_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    with torch.no_grad():
        allowed = torch.as_tensor(np.stack(legal), dtype=torch.bool, device=device)
        next_action = online(following).masked_fill(~allowed, -1e9).argmax(dim=1)
        value = target(following).gather(1, next_action[:, None]).squeeze(1)
        value *= allowed.any(1) & ~torch.as_tensor(dones, device=device)
        expected = torch.as_tensor(rewards, device=device) + 0.99 * value
    actual = online(obs_t).gather(1, action_t[:, None]).squeeze(1)
    delta = expected - actual
    loss = (torch.as_tensor(weights, device=device) * F.smooth_l1_loss(actual, expected, reduction="none")).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite DDQN loss")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10)
    optimizer.step()
    for index, error in zip(ids, delta.detach().abs().cpu().numpy()):
        replay.priorities[index] = float(error) + 1e-5
    return float(loss.detach())

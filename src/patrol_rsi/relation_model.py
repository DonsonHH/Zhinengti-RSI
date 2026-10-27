"""Matched M8 networks: same full critic, actor relation context on/off."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .dispatch_model import generalized_advantage


class RelationActorCritic(nn.Module):
    def __init__(self, relational):
        super().__init__()
        self.relational = bool(relational)
        self.encoder = nn.Sequential(nn.Linear(14, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.edge_encoder = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.team_encoder = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(160, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1))

    def edge_context(self, relations):
        hidden = self.edge_encoder(relations)
        mask = (~torch.eye(8, dtype=torch.bool, device=hidden.device)).to(hidden.dtype)
        return (hidden * mask[None, :, :, None]).sum(-2) / 7

    def target_logits(self, features, target_relations, team_relations):
        hidden = self.encoder(features)
        local_context = hidden.mean(-2, keepdim=True).expand_as(hidden)
        edge = self.edge_context(target_relations)
        team = self.team_encoder(team_relations).mean(-2)
        if not self.relational:
            edge = torch.zeros_like(edge)
            team = torch.zeros_like(team)
        return self.actor(torch.cat([hidden, local_context, edge, team], -1)).squeeze(-1)

    def value(self, features, target_relations, team_relations):
        local = self.encoder(features).mean((-3, -2))
        edge = self.edge_context(target_relations).mean(-2)
        team = self.team_encoder(team_relations).mean((-4, -3, -2))
        return self.critic(torch.cat([local, edge, team], -1)).squeeze(-1)

    def joint(self, features, free, allowed, reserved, order, target_relations, team_relations,
              choices=None, coordinated=True, deterministic=False):
        if not coordinated:
            raise ValueError('All M8 groups use the same joint deduplication protocol')
        batch = features.shape[0]
        indices = torch.arange(batch, device=features.device)
        reserved = reserved.clone()
        picked = torch.full((batch, 4), -1, dtype=torch.long, device=features.device)
        conditional_team = team_relations.clone()
        logp = features.new_zeros(batch)
        entropy = features.new_zeros(batch)
        value = self.value(features, target_relations, team_relations)
        for slot in range(4):
            actor = order[:, slot]
            active = free[indices, actor]
            local = features[indices, actor].clone()
            local[:, :, 6] = reserved.to(features.dtype)
            edges = target_relations.clone()
            edges[:, :, :, 5] = reserved[:, None, :].to(features.dtype)
            team = conditional_team[indices, actor].clone()
            legal = allowed[indices, actor].clone() & ~reserved
            legal[~active] = True  # distribution only; busy rows contribute zero
            if not legal.any(-1).all():
                raise ValueError('No eligible target in M8 joint selector')
            logits = self.target_logits(local, edges, team).masked_fill(~legal, -1e9)
            distribution = Categorical(logits=logits)
            if choices is None:
                action = logits.argmax(-1) if deterministic else distribution.sample()
            else:
                action = choices[indices, actor].clamp_min(0)
                if ((~legal[indices, action]) & active).any():
                    raise ValueError('M8 replay choice violates collection mask')
            picked[indices, actor] = torch.where(active, action, -1)
            logp += distribution.log_prob(action) * active
            entropy += distribution.entropy() * active
            reserved[indices[active], action[active]] = True
            # Apply the same conditional public commitment in collection and
            # replay; each focal executor sees earlier new choices.
            rows, executors, goals = indices[active], actor[active], action[active]
            travel = features[rows, executors, goals, 4] * 22 / 48
            onward = target_relations[rows, :, :, 0].gather(2, goals[:, None, None].expand(-1, 8, 1)).squeeze(-1)
            conditional_team[rows, :, :, executors, 1] = travel[:, None, None].expand(-1, 4, 8)
            conditional_team[rows, :, :, executors, 2] = onward[:, None, :].expand(-1, 4, 8)
            conditional_team[rows, :, :, executors, 3] = 1
            here = (torch.arange(8, device=features.device)[None, :] == goals[:, None]).to(features.dtype)
            conditional_team[rows, :, :, executors, 4] = here[:, None, :].expand(-1, 4, 8)
        return picked, logp, entropy, value


def tensors(states, device):
    floats = {'features', 'target_relations', 'team_relations'}
    return {key: torch.as_tensor(np.stack([state[key] for state in states]), device=device,
                                dtype=torch.float32 if key in floats else torch.long if key == 'order' else torch.bool)
            for key in ('features', 'free', 'allowed', 'reserved', 'order', 'target_relations', 'team_relations')}


def ppo_update(model, optimizer, rollout, device, epochs=4, batch_size=128):
    state = tensors([row['state'] for row in rollout], device)
    choices = torch.as_tensor(np.stack([row['choices'] for row in rollout]), dtype=torch.long, device=device)
    old_logp = torch.as_tensor([row['logp'] for row in rollout], dtype=torch.float32, device=device)
    advantage, returns = generalized_advantage([row['reward'] for row in rollout], [row['value'] for row in rollout], [row['done'] for row in rollout])
    advantage = torch.as_tensor(advantage, device=device)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    returns = torch.as_tensor(returns, device=device)
    with torch.no_grad():
        _, replayed, _, _ = model.joint(**state, choices=choices)
        replay_error = float((replayed - old_logp).abs().max())
    if replay_error > 2e-5:
        raise RuntimeError(f'M8 behavior likelihood mismatch: {replay_error}')
    updates = 0
    last = {}
    for _ in range(epochs):
        for ids in torch.randperm(len(rollout), device=device).split(batch_size):
            batch = {key: value[ids] for key, value in state.items()}
            _, logp, entropy, value = model.joint(**batch, choices=choices[ids])
            ratio = (logp - old_logp[ids]).exp()
            actor_loss = -torch.minimum(ratio * advantage[ids], ratio.clamp(.8, 1.2) * advantage[ids]).mean()
            value_loss = .5 * (value - returns[ids]).square().mean()
            loss = actor_loss + .5 * value_loss - .01 * entropy.mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite M8 PPO update')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            updates += 1
            last = {'loss': float(loss.detach()), 'actor_loss': float(actor_loss.detach()),
                    'value_loss': float(value_loss.detach()), 'entropy': float(entropy.detach().mean()),
                    'approx_kl': float((old_logp[ids] - logp).detach().mean())}
    return last | {'gradient_updates': updates, 'behavior_logp_max_error': replay_error}

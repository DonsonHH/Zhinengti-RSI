"""Shared task-score PPO with public-state critic and hard legal masks."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .active_inspection import CANDIDATE_FEATURES


class TaskPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(len(CANDIDATE_FEATURES), 32), nn.Tanh(),
                                   nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.critic = nn.Sequential(nn.Linear(8, 32), nn.Tanh(),
                                    nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def distribution(self, features, mask):
        if not mask.any(dim=-1).all():
            raise ValueError('At least one legal task is required')
        logits = self.actor(features).squeeze(-1)
        return Categorical(logits=logits.masked_fill(~mask, -1e9))

    def values(self, public):
        return self.critic(public).squeeze(-1)


def policy_decisions(env, model, guard=True, stochastic=False):
    decisions, reserved, samples = {}, env.reserved(), []
    logprob, entropy = 0., 0.
    forced = 0
    order = np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents)))
    for i in order:
        if env.agents[i]['task'] is not None:
            continue
        ids, features, allowed = env.options(int(i), reserved, guard=guard)
        if not len(ids):
            decisions[int(i)] = None
            continue
        with torch.inference_mode():
            distribution = model.distribution(torch.as_tensor(features), torch.as_tensor(allowed))
            action = distribution.sample() if stochastic else distribution.logits.argmax()
            lp = float(distribution.log_prob(action))
            ent = float(distribution.entropy())
        number = int(action)
        task = int(ids[number])
        decisions[int(i)] = task
        reserved.add(task)
        samples.append({'features': features.copy(), 'mask': allowed.copy(), 'action': number})
        logprob += lp
        entropy += ent
        forced += int(allowed.sum() == 1 and len(allowed) > 1)
    return decisions, samples, logprob, entropy, forced


def ppo_update(model, optimizer, rollout):
    rewards = np.asarray([r['reward'] for r in rollout], dtype=np.float32)
    values = np.asarray([r['value'] for r in rollout], dtype=np.float32)
    advantage = np.zeros(len(rewards), dtype=np.float32)
    carry, following = 0., 0.
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + .99 * following - values[step]
        carry = delta + .99 * .95 * carry
        advantage[step], following = carry, values[step]
    returns = torch.as_tensor(advantage + values)
    advantages = torch.as_tensor((advantage - advantage.mean()) / (advantage.std() + 1e-8))
    old_logprob = torch.as_tensor([r['logprob'] for r in rollout])
    critic = torch.as_tensor(np.stack([r['public_value'] for r in rollout]))
    samples = [(step, sample) for step, row in enumerate(rollout) for sample in row['samples']]
    if not samples:
        raise RuntimeError('Training episode has no task choices')
    width = max(len(row['features']) for _, row in samples)
    features = np.zeros((len(samples), width, len(CANDIDATE_FEATURES)), dtype=np.float32)
    masks = np.zeros((len(samples), width), dtype=bool)
    actions, groups = [], []
    for i, (step, row) in enumerate(samples):
        n = len(row['features'])
        features[i, :n], masks[i, :n] = row['features'], row['mask']
        actions.append(row['action'])
        groups.append(step)
    features, masks = torch.as_tensor(features), torch.as_tensor(masks)
    actions, groups = torch.as_tensor(actions), torch.as_tensor(groups)
    losses = []
    for _ in range(4):
        distribution = model.distribution(features, masks)
        logs = torch.zeros(len(rollout)).index_add(0, groups, distribution.log_prob(actions))
        entropy = torch.zeros(len(rollout)).index_add(0, groups, distribution.entropy())
        counts = torch.zeros(len(rollout)).index_add(0, groups, torch.ones(len(groups)))
        ratio = torch.exp((logs - old_logprob).clamp(-15, 15))
        policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
        value_loss = (model.values(critic) - returns).square().mean()
        loss = policy_loss + .5 * value_loss - .01 * (entropy / counts.clamp_min(1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite PPO loss')
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {'losses': losses, 'team_steps': len(rollout), 'task_choices': len(samples),
            'full_rollout_epochs': 4, 'mean_public_reward': float(rewards.mean())}

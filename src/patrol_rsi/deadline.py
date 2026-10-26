"""Public deadline-aware observations and common training reward components."""
from __future__ import annotations

import numpy as np

from .environment import PatrolEnv
from .revisit import geometry


def target_deadlines(layout_seed, profile):
    values={'balanced':(16,24,32),'tight':(12,16,24)}
    if profile not in values:raise ValueError('Unknown public deadline profile')
    result=np.resize(np.asarray(values[profile],dtype=int),8).copy()
    np.random.default_rng(layout_seed+447).shuffle(result)
    return result


def measured_mean_estimate(env):
    """No learned extrapolation; measured targets still use real observations."""
    estimate=np.full_like(env.values,.5)
    estimate[env.measured]=env.values[env.measured]
    return estimate


def observations(env,deadlines,deadline_visible=True):
    targets=env.scenario.targets
    dmap=np.zeros_like(env.values);dmap[targets]=deadlines
    # Zeroed channels, rather than fewer parameters, implement the information
    # ablation. All policies have exactly the same eight-channel architecture.
    age=np.zeros_like(env.values)
    age[targets]=np.minimum(env.raw_age[targets]/(deadlines if deadline_visible else 24),4)
    deadline_channel=dmap/32 if deadline_visible else np.zeros_like(dmap)
    base=env.policy_observations()
    return np.stack([np.stack([base[i,0],base[i,1],age,deadline_channel,
                              base[i,3],base[i,4],targets.astype(np.float32),
                              (targets&~env.ever_measured).astype(np.float32)])
                     for i in range(len(env.positions))]).astype(np.float32)


def public_potential(env,deadlines,layout_seed):
    """Explicit map-derived shaping; independently ablated and not an action override."""
    goals,costs=geometry(layout_seed)
    ratio=env.raw_age[env.scenario.targets]/deadlines
    urgency=1+np.minimum(ratio,2)+(~env.ever_measured[env.scenario.targets])
    nearest=np.array([min(max(0,cost[tuple(p)]-1) for p in env.positions) for cost in costs])
    return -float((urgency*nearest).mean()/22)


def reward_components(env,deadlines,actions,layout_seed,shaping=True,gamma=.99,terminal=False):
    """Step once; reward reads only public ages, observations and known geometry.

    The VAE does not supply reward weights. This keeps training targets identical
    between full and no-VAE groups given the same observable transition.
    """
    before_age=env.raw_age.copy();before_seen=env.ever_measured.copy()
    weights=.1+measured_mean_estimate(env)
    deadline_map=np.ones_like(env.values);deadline_map[env.scenario.targets]=deadlines
    potential_before=public_potential(env,deadlines,layout_seed) if shaping else 0.0
    event=env.step(actions)
    masks=np.stack([env.neighborhood(p) for p in env.positions]);overlap=np.maximum(masks.sum(0),1)
    urgency=np.minimum(before_age/deadline_map,4)
    gains=urgency*weights+1.5*(~before_seen&env.scenario.targets)
    local=np.array([(gains*mask/overlap).sum() for mask in masks],dtype=np.float32)
    ages=env.raw_age[env.scenario.targets]/deadlines
    penalty=float((ages>1).mean()+.1*np.minimum(ages,4).mean())
    potential_after=public_potential(env,deadlines,layout_seed) if shaping and not terminal else 0.0
    shaping_reward=gamma*potential_after-potential_before
    rewards=local-penalty+shaping_reward
    return rewards.astype(np.float32),event,{'local_gain':float(local.sum()),'public_penalty':penalty,
        'potential_before':potential_before,'potential_after':potential_after,'shaping_reward':shaping_reward,
        'mean_deadline_violation':float((ages>1).mean())}

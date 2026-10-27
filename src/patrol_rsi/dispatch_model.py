"""Shared entity actor, centralized public-state critic, exact joint likelihood."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from .dispatch import FEATURE_NAMES


class DispatchActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(len(FEATURE_NAMES),64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))

    def target_logits(self,features):
        h=self.encoder(features);context=h.mean(-2,keepdim=True).expand_as(h)
        return self.actor(torch.cat([h,context],dim=-1)).squeeze(-1)

    def value(self,features):
        return self.critic(self.encoder(features).mean((-3,-2))).squeeze(-1)

    def joint(self,features,free,allowed,reserved,order,choices=None,coordinated=True,deterministic=False):
        """Use the same conditional masks for collection and PPO replay.

        Batched arguments: [B,4,8,F], [B,4], [B,4,8], [B,8], [B,4].
        Busy executors carry action -1 and contribute zero joint log-likelihood.
        """
        batch=features.shape[0];indices=torch.arange(batch,device=features.device)
        reserved=reserved.clone();picked=torch.full((batch,4),-1,dtype=torch.long,device=features.device)
        logp=features.new_zeros(batch);entropy=features.new_zeros(batch)
        for slot in range(4):
            agent=order[:,slot];active=free[indices,agent]
            f=features[indices,agent].clone()
            if coordinated:f[:,:,6]=reserved.to(features.dtype)
            mask=allowed[indices,agent].clone()
            if coordinated:mask &= ~reserved
            # Busy rows have a dummy legal target solely to form a distribution.
            mask[~active]=True
            if not mask.any(-1).all():raise ValueError('No eligible target in joint likelihood')
            logits=self.target_logits(f).masked_fill(~mask,-1e9);distribution=Categorical(logits=logits)
            action=(logits.argmax(-1) if deterministic else distribution.sample()) if choices is None else choices[indices,agent].clamp_min(0)
            if choices is not None and ((~mask[indices,action])&active).any():raise ValueError('Replay choice violated collection mask')
            picked[indices,agent]=torch.where(active,action,-1)
            logp += distribution.log_prob(action)*active
            entropy += distribution.entropy()*active
            if coordinated:reserved[indices[active],action[active]]=True
        return picked,logp,entropy,self.value(features)


def tensors(states,device):
    return {key:torch.as_tensor(np.stack([s[key] for s in states]),device=device,
                               dtype=torch.float32 if key=='features' else torch.long if key=='order' else torch.bool)
            for key in ('features','free','allowed','reserved','order')}


def generalized_advantage(rewards,values,dones,gamma=.99,lam=.95):
    advantages=np.zeros(len(rewards),np.float32);running=0.0
    for t in reversed(range(len(rewards))):
        next_value=0.0 if dones[t] or t==len(rewards)-1 else values[t+1]
        delta=rewards[t]+gamma*next_value-values[t]
        running=delta+gamma*lam*(not dones[t])*running
        advantages[t]=running
    return advantages,advantages+np.asarray(values,np.float32)


def ppo_update(model,optimizer,rollout,device,coordinated=True,epochs=4,batch_size=128):
    state=tensors([r['state'] for r in rollout],device)
    choices=torch.as_tensor(np.stack([r['choices'] for r in rollout]),dtype=torch.long,device=device)
    old_logp=torch.as_tensor([r['logp'] for r in rollout],dtype=torch.float32,device=device)
    advantages,returns=generalized_advantage([r['reward'] for r in rollout],[r['value'] for r in rollout],[r['done'] for r in rollout])
    adv=torch.as_tensor(advantages,device=device);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    ret=torch.as_tensor(returns,device=device)
    # Validate behavior likelihood before any parameter update, not just shapes.
    with torch.no_grad():
        _,replayed,_,_=model.joint(**state,choices=choices,coordinated=coordinated)
        replay_error=float((replayed-old_logp).abs().max())
    if replay_error>2e-5:raise RuntimeError(f'Behavior likelihood replay mismatch: {replay_error}')
    last={};updates=0
    for _ in range(epochs):
        for ids in torch.randperm(len(rollout),device=device).split(batch_size):
            batch={k:v[ids] for k,v in state.items()}
            _,logp,entropy,value=model.joint(**batch,choices=choices[ids],coordinated=coordinated)
            ratio=(logp-old_logp[ids]).exp()
            clipped=ratio.clamp(.8,1.2)*adv[ids]
            actor=-torch.minimum(ratio*adv[ids],clipped).mean()
            critic=.5*(value-ret[ids]).square().mean()
            loss=actor+.5*critic-.01*entropy.mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite PPO update')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();updates+=1
            last={'loss':float(loss.detach()),'actor_loss':float(actor.detach()),'value_loss':float(critic.detach()),
                  'entropy':float(entropy.detach().mean()),'approx_kl':float((old_logp[ids]-logp).detach().mean())}
    return last|{'gradient_updates':updates,'behavior_logp_max_error':replay_error}

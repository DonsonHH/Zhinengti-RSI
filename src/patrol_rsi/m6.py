"""M6: target-level shared policy and likelihood-consistent team training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .deadline import measured_mean_estimate
from .dispatch import DispatchEnv,FEATURE_NAMES,heuristic_choices

VARIANTS=('joint_vae','no_vae','independent_targets')
METRICS=('target_coverage','deadline_violation','weighted_idleness','public_cost',
         'peak_age','movement_distance','estimator_calls','target_prediction_mse',
         'target_assignments','duplicate_assignment_fraction')


def contract():
    return {'preflight':{'split':'m6_preflight','layout_seed':16000,'field_seed':23000000,'profile':'balanced','noise':0.0},
        'train_layout_seeds':list(range(17000,17064)), 'training_field_rule':'24000000 + seed*100000 + episode',
        'validation':[{'split':'m6_validation','layout_seed':18000+i,'field_seed':25000000+i,
                       'profile':'balanced' if i<4 else 'tight','noise':0.0} for i in range(8)],
        'test':[{'split':'m6_test','layout_seed':19000+i,'field_seed':26000000+i,'profile':profile,'noise':noise}
                for profile in ('balanced','tight') for noise in (0.0,.03) for i in range(12)],
        'variants':list(VARIANTS),'training_seeds':[0,1,2],'episodes':1200,'preflight_episodes':256,
        'horizon':96,'validation_interval':100,'episodes_per_update':4,'ppo_epochs':4,'minibatch':128,
        'discount':.99,'gae_lambda':.95,'clip':.2,'entropy_coefficient':.01,'value_coefficient':.5,'learning_rate':.0003,
        'feature_names':list(FEATURE_NAMES),'high_level_action':'Next inspection point for free executors; retain until observed by any executor',
        'common_low_level':'Known-map unit-step shortest-distance scores; same cyclic obstacle/collision resolver for every method',
        'joint_protocol':'Autoregressive free-executor choices in cyclic order; mask existing and newly reserved targets; replay the exact same masks in PPO',
        'independent_control':'Mask existing commitments but not newly chosen targets; allow simultaneous duplicate destinations; same actor/critic and PPO budget',
        'reward':'New coverage gain - public deadline violation - 0.1 capped mean age/deadline - 0.25 publicly weighted idleness; no hidden truth or distance shaping',
        'ranking':'coverage loss, deadline violation, weighted idleness (6 decimals)',
        'test_rule':'Validation-only checkpoint selection; all held-out layouts remain unread until final frozen evaluation',
        'scope':'Centralized team dispatch with a shared entity actor and public-state centralized critic; not decentralized deployment, not original DDQN numerical reproduction'}


def rank(score):return tuple(round(float(x),6) for x in (1-score['target_coverage'],score['deadline_violation'],score['weighted_idleness']))
def mean(rows):return {k:float(np.mean([r[k] for r in rows])) for k in METRICS}


def refresh(env,estimator,device,variant,case_seed,step):
    import torch
    if variant=='no_vae':
        env.set_estimate(measured_mean_estimate(env));return 0
    # Prior sampling must not reset or consume the policy's exploration RNG.
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
        torch.manual_seed(case_seed*1000+step)
        with torch.no_grad():
            x=torch.as_tensor(env.estimator_input()[None],dtype=torch.float32,device=device)
            env.set_estimate(estimator.predict(x)[0,0].cpu().numpy())
    return 1


def episode(estimator,policy,case,device,variant='joint_vae',method='learned',collect=False,trace=False):
    import torch
    from .dispatch_model import tensors
    wrapper=DispatchEnv(case);env=wrapper.env;rng=np.random.default_rng(case['field_seed']+471)
    ages=[env.raw_age[env.scenario.targets].copy()];positions=[env.positions.copy()]
    idle=[];violations=[];rewards=[];mse=[];calls=0;rollout=[];events=[]
    coordinated=variant!='independent_targets'
    for t in range(96):
        calls+=refresh(env,estimator,device,variant,case['field_seed'],t)
        mse.append(env.evaluation()['reconstruction_mse']);state=wrapper.prepare()
        if method in ('learned','stochastic_policy'):
            with torch.no_grad():
                choice,logp,_,value=policy.joint(**tensors([state],device),coordinated=coordinated,
                                               deterministic=not collect and method!='stochastic_policy')
            choices=choice[0].cpu().numpy();old_logp=float(logp[0]);old_value=float(value[0])
        else:
            choices=heuristic_choices(state,method,rng);old_logp=0.0;old_value=0.0
        reward,event=wrapper.step(choices,state,joint=coordinated if method in ('learned','stochastic_policy') else True)
        ages.append(env.raw_age[env.scenario.targets].copy());positions.append(env.positions.copy())
        idle.append(env.evaluation()['normalized_weighted_idleness']);violations.append(event['deadline_violation']);rewards.append(reward)
        if collect:rollout.append({'state':state,'choices':choices.copy(),'logp':old_logp,'value':old_value,'reward':reward,'done':t==95})
        if trace:events.append(event|{'choices':choices.tolist(),'free_executors':state['free'].tolist(),
                                    'order':state['order'].tolist(),'behavior_logp':old_logp})
    row=case|{'variant':variant,'method':method,'target_coverage':env.evaluation()['target_coverage'],
              'deadline_violation':float(np.mean(violations)),'weighted_idleness':float(np.mean(idle)),
              'public_cost':-float(np.mean(rewards)),'peak_age':int(np.max(ages)),
              'movement_distance':env.evaluation()['total_movement_distance'],'estimator_calls':calls,
              'target_prediction_mse':float(np.mean(mse)),'target_assignments':wrapper.assignments,
              'duplicate_assignment_fraction':wrapper.duplicate_assignments/max(1,wrapper.assignments),
              'target_deadlines':wrapper.deadlines.tolist()}
    if trace:row.update({'target_ages':[a.tolist() for a in ages],'positions':[p.tolist() for p in positions],
                        'events':events,'prediction_mse_series':mse,'weighted_idleness_series':idle})
    return row,rollout


def evaluate(estimator,policy,cases,device,variant,method='learned',trace=False):
    rows=[episode(estimator,policy,case,device,variant,method,trace=trace)[0] for case in cases]
    return mean(rows),rows


def parameter_hash(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def train(output,estimator_path,seed,variant,preflight=False,smoke=False):
    import torch
    from .models import ConditionalVAEUNet
    from .dispatch_model import DispatchActorCritic,ppo_update
    from .train import setup
    c=contract();device=setup(seed);output.mkdir(parents=True,exist_ok=False)
    (output/'contract.json').write_text(json.dumps(c,indent=2))
    estimator=ConditionalVAEUNet().to(device).eval()
    estimator.load_state_dict(torch.load(estimator_path,map_location=device,weights_only=True)['model'])
    policy=DispatchActorCritic().to(device);initial_state={k:v.detach().clone() for k,v in policy.state_dict().items()}
    initial_hash=parameter_hash(policy)
    optimizer=torch.optim.Adam(policy.parameters(),lr=c['learning_rate']);rng=np.random.default_rng(seed)
    if smoke:episodes=4
    elif preflight:episodes=c['preflight_episodes']
    else:episodes=c['episodes']
    validation=[c['preflight']] if preflight or smoke else c['validation']
    interval=32 if preflight else 4 if smoke else c['validation_interval']
    initial_score,initial_rows=evaluate(estimator,policy,validation,device,variant)
    (output/'initial-validation.json').write_text(json.dumps({'metrics':initial_score,'cases':initial_rows}))
    best=None;selected_episode=None;buffer=[];total_updates=0;logp_max=0.0;start=time.monotonic();last_update={}
    with (output/'learning.jsonl').open('x') as log:
        for e in range(episodes):
            if preflight or smoke:case=c['preflight'].copy()
            else:
                case={'split':'m6_train','layout_seed':int(rng.choice(c['train_layout_seeds'])),
                      'field_seed':24000000+seed*100000+e,'profile':'balanced' if rng.random()<.5 else 'tight','noise':0.0}
            _,rollout=episode(estimator,policy,case,device,variant,collect=True)
            buffer+=rollout
            if (e+1)%c['episodes_per_update']==0:
                last_update=ppo_update(policy,optimizer,buffer,device,coordinated=variant!='independent_targets')
                total_updates+=last_update['gradient_updates'];logp_max=max(logp_max,last_update['behavior_logp_max_error']);buffer=[]
            if (e+1)%interval==0 or e+1==episodes:
                cpu_state=torch.random.get_rng_state();cuda_state=torch.cuda.get_rng_state()
                score,_=evaluate(estimator,policy,validation,device,variant)
                torch.random.set_rng_state(cpu_state);torch.cuda.set_rng_state(cuda_state)
                row={'episode':e+1,'team_steps':(e+1)*96,'gradient_updates':total_updates,'validation':score,
                     'last_ppo':last_update,'seconds':time.monotonic()-start}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if best is None or rank(score)<rank(best):
                    best=score;selected_episode=e+1
                    torch.save({'model':policy.state_dict(),'seed':seed,'variant':variant,'selected_episode':selected_episode,
                                'validation':best,'contract_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()},output/'best-policy.pt')
    selected=torch.load(output/'best-policy.pt',map_location=device,weights_only=True);policy.load_state_dict(selected['model'])
    selected_hash=parameter_hash(policy)
    if selected_hash==initial_hash:raise RuntimeError('Policy parameters did not update')
    tests={}
    if not preflight and not smoke:
        for method in ('learned','slack','risk_age_distance','random_targets'):
            score,rows=evaluate(estimator,policy,c['test'],device,variant,method,trace=True);tests[method]={'metrics':score,'cases':rows}
        untrained=DispatchActorCritic().to(device);untrained.load_state_dict(initial_state)
        score,rows=evaluate(estimator,untrained,c['test'],device,variant,trace=True)
        tests['untrained_network']={'metrics':score,'cases':rows}
    replay_case=validation[0] if preflight or smoke else c['test'][0]
    expected=episode(estimator,policy,replay_case,device,variant,trace=True)[0] if preflight or smoke else tests['learned']['cases'][0]
    fresh=DispatchActorCritic().to(device)
    fresh.load_state_dict(torch.load(output/'best-policy.pt',map_location=device,weights_only=True)['model'])
    repeated=episode(estimator,fresh,replay_case,device,variant,trace=True)[0]
    if repeated!=expected:raise RuntimeError('Fresh checkpoint trajectory replay mismatch')
    if preflight or smoke:(output/'preflight-trace.json').write_text(json.dumps(repeated))
    summary={'seed':seed,'variant':variant,'episodes':episodes,'preflight':preflight,'smoke':smoke,
        'initial_validation':initial_score,'best_validation':best,'selected_episode':selected_episode,
        'preflight_improved_vs_initial':rank(best)<rank(initial_score),'test':tests,'gradient_updates':total_updates,
        'behavior_logp_max_error':logp_max,'fresh_checkpoint_trace_replay_matches':True,
        'initial_parameter_sha256':initial_hash,'selected_parameter_sha256':selected_hash,
        'checkpoint_sha256':hashlib.sha256((output/'best-policy.pt').read_bytes()).hexdigest(),
        'estimator_sha256':hashlib.sha256(estimator_path.read_bytes()).hexdigest(),'parameters':sum(p.numel() for p in policy.parameters()),
        'seconds':time.monotonic()-start,'job_id':os.environ.get('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
        'test_used_for_selection':False,'rsi_enabled':False,'paper_reproduction':False}
    (output/'summary.json').write_text(json.dumps(summary))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--estimator',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();train(a.output,a.estimator,a.seed,a.variant,a.preflight,a.smoke)


if __name__=='__main__':main()

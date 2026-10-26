"""M5 matched retraining and attribution of deadline-aware shared DDQN."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .deadline import measured_mean_estimate,observations,public_potential,reward_components,target_deadlines
from .environment import PatrolEnv
from .experiment import room
from .workflow import Workflow,WorkflowExecutor,dispatch,human_workflow
from .revisit import RevisitSkill

VARIANTS=('deadline_vae','no_vae','no_deadline','no_shaping')
METRICS=('target_coverage','deadline_violation','weighted_idleness','peak_age','movement_distance',
         'estimator_calls','target_prediction_mse','intervention_fraction')


def contract():
    return {'train_layout_seeds':list(range(13000,13064)),
        'training_field_rule':'20000000 + policy_seed*100000 + episode',
        'validation':[{'split':'m5_validation','layout_seed':14000+i,'field_seed':21000000+i,
                       'profile':'balanced' if i<4 else 'tight','noise':0.0} for i in range(8)],
        'test':[{'split':'m5_test','layout_seed':15000+i,'field_seed':22000000+i,
                  'profile':profile,'noise':noise} for profile in ('balanced','tight') for noise in (0.0,.03) for i in range(12)],
        'episodes':2400,'horizon':96,'policy_seeds':[0,1,2],'variants':list(VARIANTS),
        'validation_interval':200,'channels':8,'discount':.99,'replay_capacity':20000,'batch_size':64,
        'learning_rate':.0003,'update_every_steps':4,'target_update_steps':500,
        'ranking':'coverage loss, deadline violation, weighted idleness; six decimal rounding',
        'observation_channels':['navigation','estimated_importance_on_targets','age_over_deadline','deadline_over_32','self','others','target_mask','unseen_targets'],
        'reward':'overlap-normalized public age/measurement/new-visit gain minus public deadline penalty; optional gamma*Phi(next)-Phi(current)',
        'potential':'negative mean publicly weighted nearest-team target distance; map-derived human guidance, separately ablated',
        'test_rule':'Each training group selected only on validation; test run after training selection, never drives hyperparameters',
        'scope':'New trained-policy attribution on static synthetic fields, not full paper reproduction or RSI test'}


def rank(score):
    return tuple(round(float(x),6) for x in (1-score['target_coverage'],score['deadline_violation'],score['weighted_idleness']))


def mean(rows):return {k:float(np.mean([r[k] for r in rows])) for k in METRICS}


def make_policy():
    import torch
    from .models import SharedDuelingQ
    model=SharedDuelingQ()
    model.encoder[0]=torch.nn.Conv2d(8,16,3,padding=1)
    return model


def refresh(env,estimator,device,variant,case_seed,step):
    import torch
    if variant=='no_vae':
        env.set_estimate(measured_mean_estimate(env));return 0
    torch.manual_seed(case_seed*1000+step)
    with torch.no_grad():
        x=torch.as_tensor(env.estimator_input()[None],dtype=torch.float32,device=device)
        env.set_estimate(estimator.predict(x)[0,0].cpu().numpy())
    return 1


def evaluate(estimator,policy,cases,device,variant='deadline_vae',method='learned',trace=False):
    import torch
    estimator.eval();policy.eval();rows=[]
    for case_index,case in enumerate(cases):
        env=PatrolEnv(room(case['layout_seed'],case['noise']),case['field_seed'])
        d=target_deadlines(case['layout_seed'],case['profile'])
        if method=='deadline_planner':
            plan=Workflow.from_dict({'steps':[{'op':'release_stalled'},dispatch('always','least_slack','global_min'),{'op':'fallback','policy':'ddqn'}]})
            workflow=WorkflowExecutor(env,case['layout_seed'],d,plan)
        else:workflow=WorkflowExecutor(env,case['layout_seed'],d,human_workflow()) if method=='human_workflow' else None
        recovery=RevisitSkill(env,case['layout_seed']) if method=='guard12' else None
        rng=np.random.default_rng(case['field_seed']+77);ages=[env.raw_age[env.scenario.targets].copy()]
        points=[env.positions.copy()];idle=[];violation=[];mse=[];calls=0;score_rows=[];actions_log=[]
        for t in range(96):
            calls+=refresh(env,estimator,device,variant,case['field_seed'],t)
            mse.append(env.evaluation()['reconstruction_mse'])
            if method=='random':scores=rng.random((4,8))
            elif method=='deadline_planner':scores=np.zeros((4,8))
            else:
                state=env.policy_observations() if method=='old_m1' else observations(env,d,variant!='no_deadline')
                with torch.no_grad():scores=policy(torch.as_tensor(state,device=device)).cpu().numpy()
            if workflow:scores=workflow.scores(env,scores)
            if recovery:scores=recovery.scores(env,scores,12)
            actions=env.safe_actions(scores);env.step(actions)
            ages.append(env.raw_age[env.scenario.targets].copy());points.append(env.positions.copy())
            idle.append(env.evaluation()['normalized_weighted_idleness']);violation.append(float((ages[-1]>d).mean()))
            if trace:
                if case_index==0:score_rows.append(scores.tolist())
                actions_log.append(actions.tolist())
        intervention=workflow.interventions/384 if workflow else recovery.executor_steps/384 if recovery else 0.0
        row=case|{'target_coverage':env.evaluation()['target_coverage'],'deadline_violation':float(np.mean(violation)),
                 'weighted_idleness':float(np.mean(idle)),'peak_age':int(np.max(ages)),
                 'movement_distance':env.evaluation()['total_movement_distance'],'estimator_calls':calls,
                 'target_prediction_mse':float(np.mean(mse)),'intervention_fraction':intervention,
                 'variant':variant,'method':method,'target_deadlines':d.tolist()}
        if trace:row.update({'target_ages':[a.tolist() for a in ages],'positions':[p.tolist() for p in points],
                            'action_scores_first_case_only':score_rows,'actions':actions_log,
                            'prediction_mse_series':mse,'deadline_violation_series':violation,'weighted_idleness_series':idle})
        rows.append(row)
    return mean(rows),rows


def train(output,estimator_path,seed,variant,episodes=None,smoke=False):
    import torch
    from .models import ConditionalVAEUNet,PrioritizedReplay,ddqn_update
    from .train import setup
    if variant not in VARIANTS:raise ValueError('Unknown attribution variant')
    c=contract();episodes=episodes or c['episodes'];device=setup(seed)
    output.mkdir(parents=True,exist_ok=False);(output/'contract.json').write_text(json.dumps(c,indent=2))
    estimator=ConditionalVAEUNet().to(device).eval()
    estimator.load_state_dict(torch.load(estimator_path,map_location=device,weights_only=True)['model'])
    online,target=make_policy().to(device),make_policy().to(device)
    target.load_state_dict(online.state_dict());target.eval()
    optimizer=torch.optim.Adam(online.parameters(),lr=c['learning_rate'])
    replay=PrioritizedReplay(capacity=c['replay_capacity'],seed=seed)
    rng=np.random.default_rng(seed);cached={};steps=0;updates=0;best=None;best_episode=None;start=time.monotonic()
    initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in online.parameters())).hexdigest()
    initial_state={k:v.detach().clone() for k,v in online.state_dict().items()}
    with (output/'learning.jsonl').open('x') as log:
        for episode in range(episodes):
            layout=int(rng.choice(c['train_layout_seeds']));profile='balanced' if rng.random()<.5 else 'tight'
            field_seed=20000000+seed*100000+episode
            env=PatrolEnv(room(layout),field_seed);env._neighborhoods=cached.setdefault(layout,{})
            d=target_deadlines(layout,profile);epsilon=max(.05,1-episode/(episodes*.7));last_loss=None
            refresh(env,estimator,device,variant,field_seed,0)
            for t in range(96):
                state=observations(env,d,variant!='no_deadline')
                with torch.no_grad():scores=online(torch.as_tensor(state,device=device)).cpu().numpy()
                for agent in range(4):
                    if rng.random()<epsilon:scores[agent]=rng.random(8)
                actions=env.safe_actions(scores)
                rewards,_,parts=reward_components(env,d,actions,layout,shaping=variant!='no_shaping',terminal=t==95)
                if t<95:refresh(env,estimator,device,variant,field_seed,t+1)
                next_state=observations(env,d,variant!='no_deadline');legal=env.legal_masks()
                for a,action in enumerate(actions):
                    if action>=0:replay.add(state[a],action,rewards[a],next_state[a],t==95,legal[a])
                steps+=1
                if steps%4==0 and len(replay.items)>=256:
                    last_loss=ddqn_update(online,target,optimizer,replay,device,64);updates+=1
                if steps%500==0:target.load_state_dict(online.state_dict())
            if (episode+1)%c['validation_interval']==0 or episode+1==episodes:
                cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
                validation,_=evaluate(estimator,online,c['validation'][:2] if smoke else c['validation'],device,variant)
                torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);online.train()
                row={'episode':episode+1,'steps':steps,'updates':updates,'epsilon':epsilon,'last_loss':last_loss,
                     'last_reward_components':parts,'validation':validation,'seconds':time.monotonic()-start}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if best is None or rank(validation)<rank(best):
                    best=validation;best_episode=episode+1
                    torch.save({'model':online.state_dict(),'variant':variant,'seed':seed,'selected_episode':best_episode,
                                'validation':best,'channels':8,'contract_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()},output/'best-policy.pt')
    selected=torch.load(output/'best-policy.pt',map_location=device,weights_only=True)
    online.load_state_dict(selected['model']);online.eval()
    final_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in online.parameters())).hexdigest()
    if final_hash==initial_hash:raise RuntimeError('Neural policy parameters did not update')
    tests={}
    if not smoke:
        methods=['learned','guard12']
        if variant=='deadline_vae':methods+=['human_workflow','deadline_planner','random']
        for method in methods:
            average,rows=evaluate(estimator,online,c['test'],device,variant,method,trace=True)
            tests[method]={'metrics':average,'cases':rows}
        if variant=='deadline_vae':
            average,rows=evaluate(estimator,online,c['test'],device,'no_vae','learned',trace=True)
            tests['testtime_no_vae']={'metrics':average,'cases':rows}
            untrained=make_policy().to(device);untrained.load_state_dict(initial_state);untrained.eval()
            average,rows=evaluate(estimator,untrained,c['test'],device,variant,'learned',trace=True)
            tests['untrained_network']={'metrics':average,'cases':rows}
            from .models import SharedDuelingQ
            old_policy_path=estimator_path.parent.parent/f'seed-{seed}/best-policy.pt'
            old_policy=SharedDuelingQ().to(device)
            old_policy.load_state_dict(torch.load(old_policy_path,map_location=device,weights_only=True)['model']);old_policy.eval()
            average,rows=evaluate(estimator,old_policy,c['test'],device,variant,'old_m1',trace=True)
            tests['old_m1']={'metrics':average,'cases':rows,'checkpoint_sha256':hashlib.sha256(old_policy_path.read_bytes()).hexdigest()}
    # Bind the reported action trace to a fresh read of the selected checkpoint.
    # Smoke uses validation only; formal runs repeat the already frozen first
    # test case without using it to select anything.
    replay_case=c['validation'][0] if smoke else c['test'][0]
    if smoke:
        _,reference=evaluate(estimator,online,[replay_case],device,variant,'learned',trace=True)
        expected=reference[0]
    else:expected=tests['learned']['cases'][0]
    fresh=make_policy().to(device)
    fresh.load_state_dict(torch.load(output/'best-policy.pt',map_location=device,weights_only=True)['model']);fresh.eval()
    _,repeated=evaluate(estimator,fresh,[replay_case],device,variant,'learned',trace=True)
    if repeated[0]!=expected:raise RuntimeError('Fresh selected-checkpoint replay did not match saved first-case trace')
    result={'seed':seed,'variant':variant,'episodes':episodes,'smoke':smoke,'selected_episode':best_episode,
            'best_validation':best,'test':tests,'steps':steps,'ddqn_updates':updates,'initial_parameter_sha256':initial_hash,
            'selected_parameter_sha256':final_hash,'estimator_sha256':hashlib.sha256(estimator_path.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256((output/'best-policy.pt').read_bytes()).hexdigest(),
            'fresh_checkpoint_trace_replay_matches':True,'replay_case_split':replay_case['split'],
            'seconds':time.monotonic()-start,'job_id':os.environ.get('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
            'test_used_for_selection':False,'paper_reproduction':False,'rsi_enabled':False}
    (output/'summary.json').write_text(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--estimator',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();train(a.output,a.estimator,a.seed,a.variant,episodes=4 if a.smoke else None,smoke=a.smoke)


if __name__=='__main__':main()

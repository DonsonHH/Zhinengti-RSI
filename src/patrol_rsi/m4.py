"""M4 executable workflow synthesis with evaluated improver updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .environment import PatrolEnv
from .experiment import room
from .m3 import load_models
from .revisit import RevisitSkill
from .workflow import (ASSIGNMENTS, COMMITS, CONDITIONS, PRIORITIES, Workflow,
                       WorkflowExecutor, human_workflow, random_workflow)

DIAGNOSTICS = ('coverage', 'deadlines', 'oscillation', 'intervention')
MEMORIES = ('none', 'recent', 'best_and_failures')
OPTIMIZERS = ('random_workflow', 'lm_no_memory', 'lm_memory', 'validated_random_meta', 'validated_model_meta')
METRICS = ('target_coverage', 'mean_deadline_violation', 'weighted_idleness', 'peak_age',
           'intervention_fraction', 'movement_distance', 'ddqn_command_fraction')


def snapshot(value):
    return json.loads(json.dumps(value))


def model_message(user):
    task=user.get('task')
    if task=='synthesize_patrol_workflow':
        instruction=('Generate a NEW executable patrol workflow using the evidence. Your entire response must contain exactly '
                     'diagnosis (one short sentence), workflow (an object with steps only), and expected_effect (one short sentence). '
                     'Do not output task, generation, current_workflow, metrics, memory, tool_results or grammar. '
                     'Propose a complete executable workflow that follows the supplied grammar, not a copy of this evidence.')
    elif task=='improve_the_workflow_improver':
        instruction=('Generate a NEW improver protocol. Your entire response must contain exactly diagnostics (list), memory (string), '
                     'and procedure (string <=600 characters). Do not output task, round, current_protocol or previous_meta_feedback.')
    else:
        raise ValueError('Unknown model request')
    return (instruction+'\nBEGIN READ-ONLY INPUT EVIDENCE\n'+json.dumps(user)+
            '\nEND READ-ONLY INPUT EVIDENCE\nNow output ONLY the new JSON proposal. Do not reproduce the input evidence.')


def contract():
    return {
        'meta_development': [[{'split': 'm4_meta_development', 'layout_seed': 10000+2*r+i,
                              'field_seed': 11000000+2*r+i, 'profile': 'balanced' if i==0 else 'tight', 'noise': 0.0}
                             for i in range(2)] for r in range(2)],
        'development': [{'split': 'm4_development', 'layout_seed': 11000+i, 'field_seed': 12000000+i,
                         'profile': 'balanced' if i<3 else 'tight', 'noise': 0.0} for i in range(6)],
        'test': [{'split': 'm4_test', 'layout_seed': 12000+i, 'field_seed': 13000000+i,
                  'profile': 'balanced' if i<6 else 'tight', 'noise': noise} for noise in (0.0,.03) for i in range(12)],
        'policy_seeds': [0,1,2], 'proposal_seeds': [641,743,853],
        'task_candidate_slots': 3, 'meta_rounds': 2, 'inner_candidate_slots': 2,
        'model_input_cap': 6144, 'model_output_cap': 768,
        'rank': 'coverage loss, per-target deadline violation fraction, weighted idleness, workflow intervention fraction',
        'model_methods': list(OPTIMIZERS), 'deadline_profiles': {'balanced':[16,24,32], 'tight':[12,16,24]},
        'weights_and_estimator_schedule': 'Frozen M1 VAE and DDQN; one prior sample each patrol step for every method',
        'meta_rule': 'Paired incumbent/candidate inner synthesis on two independent meta-development cases per round; reset task workflow and history; retain only strictly better inner outcome',
        'budget_note': 'Same task candidate slots and evaluation cases; meta-trained methods use extra offline inner evaluations, reported separately; random-meta controls this preparation budget'}


def deadlines(case):
    values=np.resize(np.asarray(contract()['deadline_profiles'][case['profile']],dtype=int),8).copy()
    np.random.default_rng(case['layout_seed']+447).shuffle(values)
    return values


def rank(score):
    return tuple(round(float(x),6) for x in (1-score['target_coverage'],score['mean_deadline_violation'],
                                             score['weighted_idleness'],score['intervention_fraction']))


def mean(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in METRICS}


@dataclass(frozen=True)
class ImproverProtocol:
    diagnostics: tuple[str,...] = ('deadlines','coverage')
    memory: str = 'recent'
    procedure: str = 'Use public failure evidence to propose one reusable branch-and-dispatch workflow. Respect the DSL grammar.'

    def __post_init__(self):
        if not 1<=len(self.diagnostics)<=3 or len(set(self.diagnostics))!=len(self.diagnostics) or any(x not in DIAGNOSTICS for x in self.diagnostics):
            raise ValueError('Protocol diagnostic tools must be 1..3 distinct allowed tools')
        if self.memory not in MEMORIES or not isinstance(self.procedure,str) or not 1<=len(self.procedure)<=600:
            raise ValueError('Invalid protocol memory or procedure')

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)!={'diagnostics','memory','procedure'} or not isinstance(value['diagnostics'],list):
            raise ValueError('Protocol requires diagnostics list, memory and procedure only')
        return cls(tuple(value['diagnostics']),value['memory'],value['procedure'])

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]


def select_memory(history,mode):
    if mode=='none':return []
    if mode=='recent':return history[-2:]
    if not history:return []
    valid=[r for r in history if 'metrics' in r]
    best=min(valid,key=lambda r:rank(r['metrics'])) if valid else history[-1]
    failures=[r for r in history if not r['accepted']]
    return [best]+([failures[-1]] if failures and failures[-1] is not best else [])


def evaluate(estimator,policy,program,cases,device,control=None,trace=False):
    import torch
    rows=[];diagnostics=[]
    for case in cases:
        env=PatrolEnv(room(case['layout_seed'],case['noise']),case['field_seed'])
        due=deadlines(case);executor=WorkflowExecutor(env,case['layout_seed'],due,program)
        old=RevisitSkill(env,case['layout_seed']) if control=='trigger12' else None
        ages=[env.raw_age[env.scenario.targets].copy()];positions=[env.positions.copy()]
        violations=[];idleness=[]
        rng=np.random.default_rng(case['field_seed']+11)
        for t in range(96):
            torch.manual_seed(case['field_seed']*1000+t)
            with torch.no_grad():
                x=torch.as_tensor(env.estimator_input()[None],dtype=torch.float32,device=device)
                env.set_estimate(estimator.predict(x)[0,0].cpu().numpy())
                learned=policy(torch.as_tensor(env.policy_observations(),device=device)).cpu().numpy()
            if control=='plain_ddqn':scores=learned
            elif old:scores=old.scores(env,learned,12)
            else:scores=executor.scores(env,rng.random((4,8)) if control=='random_fallback' else learned)
            env.step(env.safe_actions(scores))
            ages.append(env.raw_age[env.scenario.targets].copy());positions.append(env.positions.copy())
            violations.append(float((ages[-1]>due).mean()));idleness.append(env.evaluation()['normalized_weighted_idleness'])
        fraction=0.0 if control=='plain_ddqn' else old.executor_steps/384 if old else executor.interventions/384
        row=case|{'target_coverage':env.evaluation()['target_coverage'],'mean_deadline_violation':float(np.mean(violations)),
                  'weighted_idleness':float(np.mean(idleness)),'peak_age':int(np.max(ages)),
                  'intervention_fraction':fraction,'movement_distance':env.evaluation()['total_movement_distance'],
                  'ddqn_command_fraction':1-fraction if control!='random_fallback' else 0.0,
                  'program_id':program.version_id,'executed_rules':executor.executed_rules,
                  'assignments':executor.assignments,'stalled_releases':executor.releases,
                  'target_deadlines':due.tolist()}
        rows.append(row)
        goals=np.argwhere(env.scenario.targets).tolist();a=np.asarray(ages);p=np.asarray(positions)
        worst=np.argsort(-(a/due).max(0))[:3]
        views={
            'coverage':{'unseen_positions':[g for j,g in enumerate(goals) if not env.ever_measured[tuple(g)]],
                        'coverage':row['target_coverage']},
            'deadlines':{'worst_targets':[{'position':goals[j],'deadline':int(due[j]),'peak_age':int(a[:,j].max()),
                                          'overdue_steps':int((a[1:,j]>due[j]).sum())} for j in worst]},
            'oscillation':{'two_step_returns':(p[2:]==p[:-2]).all(-1).sum(0).tolist()},
            'intervention':{'command_fraction':fraction,'executed_rules':executor.executed_rules,
                            'assignments':executor.assignments,'stalled_releases':executor.releases}}
        diagnostics.append({'layout_seed':case['layout_seed'],'profile':case['profile'],**views})
        if trace:row.update({'target_ages':a.tolist(),'positions':p.tolist(),'targets':goals})
    return mean(rows),rows,diagnostics


class ModelCaller:
    def __init__(self,model_directory,device):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(model_directory,torch_dtype=torch.float16,attn_implementation='eager',
                         local_files_only=True,trust_remote_code=False).to(device).eval()
        self.calls=0

    def generate(self,user,seed):
        import torch
        user=snapshot(user)
        system='Generate a bounded JSON workflow or improver protocol as requested. Use provided development evidence only. Do not change models, evaluation, data, budgets or the interpreter. Output one JSON object.'
        rendered=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':model_message(user)}],
                          tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(rendered,return_tensors='pt').to(self.device)
        if inputs['input_ids'].numel()>contract()['model_input_cap']:raise RuntimeError('Prompt token budget exceeded')
        torch.manual_seed(seed);start=time.monotonic()
        with torch.no_grad():ids=self.model.generate(**inputs,max_new_tokens=768,do_sample=True,temperature=.7,top_p=.8,top_k=20)
        answer=self.tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True);self.calls+=1
        return {'prompt':user,'rendered_prompt':rendered,'raw_response':answer,'input_tokens':int(inputs['input_ids'].numel()),
                'output_tokens':int(ids.shape[1]-inputs['input_ids'].shape[1]),'generation_seconds':time.monotonic()-start}


def json_object(answer):
    left,right=answer.find('{'),answer.rfind('}')
    if left<0 or right<left:raise ValueError('No JSON object')
    value=json.loads(answer[left:right+1])
    if not isinstance(value,dict):raise ValueError('Expected object')
    return value


def workflow_prompt(parent,score,diagnostics,protocol,history,generation):
    # New snapshot; DSL semantics, no filled candidate workflow example.
    return json.loads(json.dumps({'task':'synthesize_patrol_workflow','generation':generation,
        'current_workflow':parent.to_dict(),'metrics':score,'improver_protocol':asdict(protocol),
        'tool_results':[{key:d[key] for key in protocol.diagnostics}|{'layout_seed':d['layout_seed'],'profile':d['profile']}
                        for d in diagnostics[:2]],
        'memory':select_memory(history,protocol.memory),
        'grammar':{'top_level':['diagnosis','workflow','expected_effect'],'workflow_keys':['steps'],
                   'steps':'1..3 ordered dispatch operations, optional release_stalled at start, final fallback',
                   'dispatch_keys':['op','when','priority','assignment','commit'],
                   'dispatch_op':'dispatch','when':list(CONDITIONS),'priority':list(PRIORITIES),
                   'assignment':list(ASSIGNMENTS),'commit':list(COMMITS),
                   'release':'object with op=release_stalled only; optional first step',
                   'fallback':'final object with op=fallback and policy=ddqn only'},
        'semantics':{'due_soon':'age + nearest known travel distance >= .75 * public target deadline',
                     'overdue':'age >= public target deadline','unseen':'never observed',
                     'stalled':'some free executor repeats the same positions for two two-step periods',
                     'least_slack':'prioritize deadline minus age minus distance',
                     'age_ratio':'prioritize age/deadline','risk_age':'prioritize observed/estimated importance times age/deadline',
                     'nearest':'rank target first, then nearest free executor',
                     'global_min':'greedily choose best distance-minus-priority pair over all free executors and eligible targets',
                     'until_observed':'keep assignment until observed by any executor',
                     'one_step':'release and recompute next step','release_stalled':'release stuck committed executor before dispatch'},
        'acceptance':'Prioritize coverage, then heterogeneous-deadline violation, then weighted idleness, then intervention cost. No metric/parameter edits.',
        'rules':['Return JSON only. workflow must be an object containing steps, not prose or code.',
                 'The listed domains are exact strings, no placeholder names. Preserve final DDQN fallback.',
                 'All model weights, VAE refreshes (one sample per step), deadlines and interpreter are fixed.']}))


def optimize(estimator,policy,caller,cases,device,protocol,seed,slots,directory,random_search=False):
    directory.mkdir(parents=True,exist_ok=False)
    parent=human_workflow();score,rows,diagnostics=evaluate(estimator,policy,parent,cases,device)
    initial=score;history=[];rng=np.random.default_rng(seed);calls=[]
    for g in range(slots):
        record={'generation':g,'parent_id':parent.version_id,'parent_workflow':parent.to_dict(),
                'protocol':asdict(protocol),'accepted':False}
        if random_search:
            candidate=random_workflow(rng);record['origin']='random_workflow_search'
        else:
            reply=caller.generate(workflow_prompt(parent,score,diagnostics,protocol,history,g),seed*100+g)
            record.update(reply);record['origin']='language_model';calls.append(reply)
            try:
                obj=json_object(reply['raw_response'])
                if set(obj)!={'diagnosis','workflow','expected_effect'}:raise ValueError('Expected diagnosis, workflow, expected_effect only')
                candidate=Workflow.from_dict(obj['workflow']);record['diagnosis']=obj['diagnosis'];record['expected_effect']=obj['expected_effect']
            except (ValueError,TypeError) as error:
                record['error']=str(error)
                history.append({'generation':g,'accepted':False,'error':str(error)})
                (directory/f'candidate-{g}.json').write_text(json.dumps(record))
                print(json.dumps({'phase':str(directory.name),'generation':g,'error':str(error)}),flush=True)
                continue
        candidate_score,candidate_rows,candidate_diagnostics=evaluate(estimator,policy,candidate,cases,device)
        accepted=rank(candidate_score)<rank(score)
        record.update({'candidate_id':candidate.version_id,'candidate_workflow':candidate.to_dict(),
                       'metrics':candidate_score,'cases':candidate_rows,'accepted':accepted})
        if accepted:parent,score,rows,diagnostics=candidate,candidate_score,candidate_rows,candidate_diagnostics
        history.append({'generation':g,'workflow':candidate.to_dict(),'metrics':candidate_score,'accepted':accepted})
        (directory/f'candidate-{g}.json').write_text(json.dumps(record))
        print(json.dumps({'phase':str(directory.name),'generation':g,'accepted':accepted,'coverage':candidate_score['target_coverage'],
                          'violation':candidate_score['mean_deadline_violation']}),flush=True)
    summary={'initial_metrics':initial,'workflow':parent.to_dict(),'workflow_id':parent.version_id,'metrics':score,
             'protocol':asdict(protocol),'slots':slots,'selection_cases':cases,
             'cost':{k:sum(x[k] for x in calls) for k in ('input_tokens','output_tokens','generation_seconds')},
             'model_calls':len(calls),'test_used_for_selection':False}
    (directory/'selection-frozen.json').write_text(json.dumps(summary))
    return summary


def meta_accept(incumbent,candidate):
    return rank(candidate['metrics'])<rank(incumbent['metrics'])


def random_protocol(rng):
    tools=tuple(str(x) for x in rng.choice(DIAGNOSTICS,size=int(rng.integers(1,4)),replace=False))
    return ImproverProtocol(tools,str(rng.choice(MEMORIES)),ImproverProtocol().procedure)


def train_protocol(estimator,policy,caller,device,seed,directory,model_proposals,rounds=2):
    directory.mkdir(parents=True,exist_ok=False)
    incumbent=ImproverProtocol();records=[];rng=np.random.default_rng(seed)
    for r in range(rounds):
        entry={'round':r,'parent':asdict(incumbent),'accepted':False,'proposal_origin':'language_model' if model_proposals else 'random_protocol_search'}
        if model_proposals:
            request={'task':'improve_the_workflow_improver','round':r,'current_protocol':asdict(incumbent),
                     'previous_meta_feedback':records,
                     'allowed_diagnostics':list(DIAGNOSTICS),'allowed_memory':list(MEMORIES),
                     'output_grammar':'Return exactly diagnostics (1..3 distinct strings in allowed_diagnostics), memory (one allowed string), procedure (reusable instruction string <=600 chars). No other fields.',
                     'objective':'Your protocol controls which public diagnostics and memories the workflow generator sees. A candidate is adopted only if it produces a better workflow on independent meta-development tasks after two inner proposal slots. Do not encode task coordinates or test data.',
                     'fixed_boundary':'No model, interpreter, evaluation, data or candidate budget changes. No example answer is supplied.'}
            reply=caller.generate(request,seed*1000+r);entry['model_proposal']=reply
            try:candidate=ImproverProtocol.from_dict(json_object(reply['raw_response']))
            except (ValueError,TypeError) as error:
                entry['error']=str(error);records.append(entry)
                (directory/f'round-{r}.json').write_text(json.dumps(entry));continue
        else:candidate=random_protocol(rng)
        cases=contract()['meta_development'][r];pair_seed=seed+3000+r
        before=optimize(estimator,policy,caller,cases,device,incumbent,pair_seed,2,directory/f'round-{r}-incumbent')
        after=optimize(estimator,policy,caller,cases,device,candidate,pair_seed,2,directory/f'round-{r}-candidate')
        accepted=meta_accept(before,after)
        entry.update({'candidate':asdict(candidate),'incumbent_inner':before,'candidate_inner':after,
                      'accepted':accepted,'evaluation_cases':cases})
        if accepted:incumbent=candidate
        # Feedback for protocol generation is compact, immutable and only meta-dev.
        records.append({'round':r,'parent':entry['parent'],'candidate':asdict(candidate),'accepted':accepted,
                        'incumbent_metrics':before['metrics'],'candidate_metrics':after['metrics']})
        (directory/f'round-{r}.json').write_text(json.dumps(entry))
    result={'protocol':asdict(incumbent),'records':records,'test_used_for_selection':False}
    (directory/'protocol-frozen.json').write_text(json.dumps(result));return incumbent


def run(output,model_directory,checkpoints,trial,smoke=False):
    output.mkdir(parents=True,exist_ok=False)
    c=contract();policy_seed=trial//3;proposal_seed=c['proposal_seeds'][trial%3]
    estimator,policy,device,hashes=load_models(checkpoints,policy_seed)
    (output/'contract.json').write_text(json.dumps(c,indent=2))
    caller=ModelCaller(model_directory,device);meta_rounds=1 if smoke else 2
    random_meta=train_protocol(estimator,policy,caller,device,proposal_seed,output/'meta-random',False,meta_rounds)
    model_meta=train_protocol(estimator,policy,caller,device,proposal_seed,output/'meta-model',True,meta_rounds)
    # No task program or adaptation memory crosses the meta-development boundary.
    protocols={'random_workflow':ImproverProtocol(), 'lm_no_memory':ImproverProtocol(memory='none'),
               'lm_memory':ImproverProtocol(), 'validated_random_meta':random_meta, 'validated_model_meta':model_meta}
    retained={};slots=1 if smoke else c['task_candidate_slots']
    for method in OPTIMIZERS:
        retained[method]=optimize(estimator,policy,caller,c['development'],device,protocols[method],proposal_seed,slots,
                                  output/method,random_search=method=='random_workflow')
    (output/'all-selections-frozen.json').write_text(json.dumps(retained))
    tests={}
    if not smoke:
        controls=[('plain_ddqn',human_workflow(),'plain_ddqn'),('fixed_trigger12',human_workflow(),'trigger12'),
                  ('human_workflow',human_workflow(),None),('human_workflow_random_fallback',human_workflow(),'random_fallback')]
        controls += [(name,Workflow.from_dict(value['workflow']),None) for name,value in retained.items()]
        for name,program,control in controls:
            score,rows,_=evaluate(estimator,policy,program,c['test'],device,control=control,trace=True)
            tests[name]={'metrics':score,'cases':rows,'workflow':program.to_dict(),'control':control}
    result={'trial':trial,'policy_seed':policy_seed,'proposal_seed':proposal_seed,'smoke':smoke,
            'retained':retained,'protocols':{name:asdict(p) for name,p in protocols.items()},'test':tests,
            'checkpoint_hashes':hashes,'model_revision':json.loads((model_directory/'verified_manifest.json').read_text())['revision'],
            'actual_total_model_calls':caller.calls,'test_used_for_selection':False,'job_id':os.environ.get('SLURM_JOB_ID')}
    (output/'summary.json').write_text(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.output,a.model,a.checkpoints,a.trial,a.smoke)


if __name__=='__main__':main()

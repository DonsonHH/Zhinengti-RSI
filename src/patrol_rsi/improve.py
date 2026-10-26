"""Real open-LM proposal/evaluation pilot on frozen patrolling checkpoints.

This bounded pilot is not evidence of open-ended RSI or meta-improvement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import json
import hashlib
from pathlib import Path
import time

from .versions import AgentVersion


def prompt_snapshot(user: dict) -> dict:
    """Freeze nested history before the current proposal can append feedback."""
    return copy.deepcopy(user)


def parse_proposal(content: str, parent: AgentVersion, update_improver: bool):
    begin,end=content.find('{'),content.rfind('}')
    if begin<0 or end<begin:raise ValueError('No JSON object in model output')
    obj=json.loads(content[begin:end+1])
    if not isinstance(obj,dict) or not isinstance(obj.get('patch'),dict):raise ValueError('Expected patch object')
    patch=obj['patch'].copy()
    allowed={'estimator_samples','estimator_refresh_interval','coordination_order'}
    if set(patch)-allowed:raise ValueError('Model attempted to change protected fields')
    if update_improver:
        instruction=obj.get('next_diagnostic_instruction')
        if not isinstance(instruction,str):raise ValueError('Updated diagnostic instruction missing')
        patch['diagnostic_instruction']=instruction
    candidate=parent.proposed(patch)
    return candidate,obj


def run(output: Path, model_directory: Path, estimator_path: Path, policy_path: Path, rounds: int=3):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .experiment import evaluate_case,metrics_mean,split_manifest
    from .models import ConditionalVAEUNet,SharedDuelingQ
    output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('Allocated CUDA required')
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    estimator=ConditionalVAEUNet().to(device)
    estimator.load_state_dict(torch.load(estimator_path,map_location=device,weights_only=True)['model']);estimator.eval()
    policy=SharedDuelingQ().to(device)
    policy.load_state_dict(torch.load(policy_path,map_location=device,weights_only=True)['model']);policy.eval()
    tokenizer=AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
    language_model=AutoModelForCausalLM.from_pretrained(model_directory,torch_dtype=torch.float16,attn_implementation='eager',local_files_only=True,trust_remote_code=False).to(device).eval()
    manifest=split_manifest()
    rsi_test=[{'split':'rsi_test','layout_seed':5000+i,'field_seed':600000+i} for i in range(12)]
    def evaluate(version,split):
        cases=rsi_test if split=='rsi_test' else manifest[split]
        rows=[evaluate_case(estimator,policy,case,device,version=version)[0] for case in cases]
        return metrics_mean(rows),rows
    initial=AgentVersion()
    initial_score,initial_rows=evaluate(initial,'development')
    (output/'initial-development.json').write_text(json.dumps(initial_rows,indent=2))
    all_records=[];retained={}
    for group,update_improver in [('fixed_improver',False),('updated_improver',True)]:
        parent=initial;score=initial_score;parent_rows=initial_rows
        history=[]
        for generation in range(rounds):
            system='You improve a fixed VAE and shared-DDQN patrolling agent. Use only provided development evidence. Return one JSON object. Do not edit weights, tests, evaluator, observations, or budgets.'
            user={'current_version':asdict(parent),'development_metrics':score,'history':history,
                  'development_trajectory_logs':[{'layout_seed':r['layout_seed'],**r['public_diagnostics']} for r in parent_rows[:2]],
                  'diagnostic_instruction':parent.diagnostic_instruction,
                  'fields':{'estimator_samples':'integer 1..8','estimator_refresh_interval':'integer 1..8','coordination_order':['q_priority','cyclic']},
                  'budget':'Prior samples <= twice initial baseline. Accept lower mean idleness, or equivalent idleness within .001 with at least 25% fewer prior samples. Report costs separately.',
                  'required_output':{'diagnosis':'specific failure hypothesis','patch':'changed fields only','expected_effect':'testable prediction','next_diagnostic_instruction':'reusable method for analyzing the next generation'}}
            user=prompt_snapshot(user)
            text=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':json.dumps(user)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(text,return_tensors='pt').to(device)
            torch.manual_seed(600+generation)
            start=time.monotonic()
            with torch.no_grad():ids=language_model.generate(**inputs,max_new_tokens=512,do_sample=True,temperature=.7,top_p=.8,top_k=20)
            answer=tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            record={'group':group,'generation':generation,'parent':parent.version_id,'raw_model_response':answer,
                    'system_prompt':system,'user_prompt':user,'rendered_prompt':text,
                    'input_tokens':int(inputs['input_ids'].numel()),'output_tokens':int(ids.shape[1]-inputs['input_ids'].shape[1]),
                    'generation_seconds':time.monotonic()-start,'origin':'language_model','accepted':False}
            try:
                candidate,parsed=parse_proposal(answer,parent,update_improver)
                candidate_score,candidate_rows=evaluate(candidate,'development')
                budget_ok=(candidate.estimator_samples*((96+candidate.estimator_refresh_interval-1)//candidate.estimator_refresh_interval)<=192)
                gain=score['mean_normalized_weighted_idleness']-candidate_score['mean_normalized_weighted_idleness']
                parent_cost=parent.estimator_samples*((96+parent.estimator_refresh_interval-1)//parent.estimator_refresh_interval)
                candidate_cost=candidate.estimator_samples*((96+candidate.estimator_refresh_interval-1)//candidate.estimator_refresh_interval)
                accepted=budget_ok and (gain>0.001 or (gain>=-0.001 and candidate_cost<=.75*parent_cost))
                record.update({'parsed':parsed,'candidate':asdict(candidate),'development':candidate_score,'development_cases':candidate_rows,'budget_ok':budget_ok,'gain':gain,'accepted':accepted})
                if accepted:parent,score,parent_rows=candidate,candidate_score,candidate_rows
            except (ValueError,TypeError,json.JSONDecodeError) as error:
                record['error']=str(error)
            history.append({k:record[k] for k in ['generation','accepted','gain','error','parsed','development'] if k in record})
            all_records.append(record)
            (output/f'{group}-{generation}.json').write_text(json.dumps(record,indent=2))
            print(json.dumps({'group':group,'generation':generation,'accepted':record['accepted'],'error':record.get('error'),'gain':record.get('gain')}),flush=True)
        retained[group]={'version':asdict(parent),'development':score}
    # New test data are inspected only after all proposal selection has ended.
    test={}
    for name,version in [('initial',initial)]+[(k,AgentVersion(**v['version'])) for k,v in retained.items()]:
        average,rows=evaluate(version,'rsi_test');test[name]={'metrics':average,'cases':rows}
    summary={'model_manifest':json.loads((model_directory/'verified_manifest.json').read_text()),'rounds_per_group':rounds,
             'selection_split':'development','reporting_split':'rsi_test_distinct_from_M1_test','test_used_for_selection':False,'initial_development':initial_score,'retained':retained,'test':test,
             'accepted_candidates':sum(r['accepted'] for r in all_records),'real_language_model_calls':len(all_records),
             'estimator_sha256':hashlib.sha256(estimator_path.read_bytes()).hexdigest(),
             'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),
             'interpretation':'bounded_pipeline_pilot_not_recursive_improvement_effectiveness_proof'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--estimator',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    a=p.parse_args();run(a.output,a.model,a.estimator,a.policy)


if __name__=='__main__':main()

"""Exactly one preregistered real Qwen call, no task evaluation or selection."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.limited_prediction import need,read,sha,write
from patrol_rsi.m12 import verify_open_model,ModelCaller
from patrol_rsi.paired_campaign import protocols
from patrol_rsi.rsi_controlled import ControlledImprover,public_prompt


def main():
    p=argparse.ArgumentParser()
    for key in ('paired','rsi','model','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume-before-generation',action='store_true')
    a=p.parse_args();os.umask(0o077);torch.set_num_threads(4);_,protocol=protocols(a.paired,a.rsi)
    if a.resume_before_generation:
        need((a.output/'attempt.json').is_file() and not(a.output/'generation-started.json').exists() and not(a.output/'preflight.json').exists(),'Recover startup only if no actual generation ever started')
        write(a.output/'startup-recovery.json',{'prior_attempt_SHA256':sha(a.output/'attempt.json'),'generation_calls_before_recovery':0,
            'reason':'Original module lacked transformers;use existing historical model-env,one original generation slot unchanged.',
            'job_id':os.environ.get('SLURM_JOB_ID'),'all_failed_allocation_costs_retained':True})
    else:
        a.output.mkdir(exist_ok=False)
        write(a.output/'attempt.json',{'attempt':1,'call_cap':1,'max_output_tokens':128,'input_cap':6144,
            'seed':9199,'purpose':'Format/runtime preflight only;no workflow retained/evaluated','job_id':os.environ.get('SLURM_JOB_ID'),'start_unix':time.time()})
    start=time.perf_counter();info=verify_open_model(a.model);need(torch.cuda.is_available(),'Allocated GPU required')
    caller=ModelCaller(a.model,'cuda');load_s=time.perf_counter()-start
    versions=tuple(v._version for v in caller.model.parameters())
    clean=public_prompt(PatrolProgram(),{'coverage':{},'professional_evidence':{},'efficiency':{}},[],ControlledImprover())
    user={'task':'Format preflight only. Return ONE workflow object, no explanation. Use only the exact keys and legal domains.',
        'workflow_keys':clean['workflow_keys'],'legal_example':PatrolProgram().to_dict(),
        'allowed_refresh_s':[30,60,120],'allowed_cooldown_s':[0,30,60,120],'allowed_fallback':['tour','ppo'],
        'rules':'Use an empty list for this non-task format preflight.'}
    rendered=caller.tokenizer.apply_chat_template([{'role':'system','content':'Return JSON only. Do not think aloud. No markdown.'},
        {'role':'user','content':json.dumps(user)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=caller.tokenizer(rendered,return_tensors='pt').to('cuda');n=int(inputs['input_ids'].numel());need(n<=6144,'Preflight input budget')
    write(a.output/'generation-started.json',{'actual_generation_attempt':1,'prompt':user,'rendered_prompt':rendered,'input_tokens':n,
        'model':info,'thinking':False,'parameters':protocol['model']['generation'],'max_new_tokens':128,'source_rsi_SHA256':sha(a.rsi)})
    torch.manual_seed(9199);tick=time.perf_counter()
    with torch.inference_mode():ids=caller.model.generate(**inputs,max_new_tokens=128,do_sample=True,temperature=.7,top_p=.8,top_k=20)
    answer=caller.tokenizer.decode(ids[0,n:],skip_special_tokens=True);seconds=time.perf_counter()-tick;count=int(ids.shape[1]-n)
    valid=False;error=None;workflow=None
    try:
        parsed=json.loads(answer[answer.index('{'):answer.rindex('}')+1]);workflow=PatrolProgram.from_dict(parsed).to_dict();valid=True
    except (ValueError,TypeError,KeyError) as exc:error=type(exc).__name__+': '+str(exc)
    need(tuple(v._version for v in caller.model.parameters())==versions,'No open model weight updates')
    write(a.output/'preflight.json',{'real_model_calls':1,'valid_workflow_output':valid,'error':error,'workflow_not_used':workflow,
        'raw_response':answer,'input_tokens':n,'output_tokens':count,'generation_s':seconds,'model_load_and_verify_s':load_s,
        'model':info,'gpu':torch.cuda.get_device_name(),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'thinking':False,'formal_task_proposals':0,'meta_updates':0,'environment_evaluations':0,'total_wall_s':time.perf_counter()-start,
        'within_token_caps':n<=6144 and count<=128,'preflight_only_not_RSI_gain':True,'job_id':os.environ.get('SLURM_JOB_ID')})
    print('M12D_QWEN_SINGLE_PREFLIGHT',valid,'tokens',n,count,flush=True)


if __name__=='__main__':main()

"""New24-slot real format experiment; no task program search or old rescoring."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import torch

from patrol_rsi.e_stage import verify_stage_identity
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.m12 import ModelCaller,verify_open_model
from patrol_rsi.meta_schema import parse_proposal,strict_json,changed_fields,reset_receipt
from patrol_rsi.rsi_controlled import ControlledImprover,INSTRUCTION_TEMPLATES

MODEL=Path('/share/home/renly/renly-patrol-rsi/models/Qwen3-8B-b968826d')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    os.umask(0o077);torch.set_num_threads(4);stage=a.source_root/'research/2026-10-31-m12e-timing-schema';protocol=stage/'E0-Schema-v1.json'
    prompts_path=stage/'E0-prompts.json'
    identity_path=a.run_root/'E0-execution-identity.json';identity=verify_stage_identity(identity_path,{'E0':protocol,'E0_prompts':prompts_path,'Qwen_manifest':MODEL/'verified_manifest.json'},scope='E0')
    need(a.source_root.resolve()==Path(identity['source_root']).resolve(),'CLI source root equals actual imported E code')
    identity_sha=sha(identity_path);config=read(protocol);root=a.run_root/'E0';root.mkdir(exist_ok=False);(root/'calls').mkdir()
    need(sha(prompts_path)==config['prompts_SHA256'],'Exact prompts and parent/challenge mapping preregistered before calls')
    planned={(v['slot'],v['method']):v for v in read(prompts_path)['calls']}
    write(root/'started.json',{'protocol_SHA256':sha(protocol),'identity_SHA256':identity_sha,'job_id':os.environ['SLURM_JOB_ID'],'real_call_cap':24})
    begin=time.monotonic();model=verify_open_model(MODEL);need(torch.cuda.is_available(),'E0 allocated GPU required')
    caller=ModelCaller(MODEL,'cuda');versions=tuple(v._version for v in caller.model.parameters());receipts=[];seen={m:set() for m in config['methods']}
    def timeout(*_):raise TimeoutError('E0 fixed60second generation timeout')
    signal.signal(signal.SIGALRM,timeout)
    for slot,seed in enumerate(config['paired_generation_seeds']):
        for method in config['methods']:
            need(time.monotonic()-begin<config['budgets']['soft_stop_s'],'E0 allocation soft stop; no recovery or extra calls')
            fixed=planned[slot,method];need(fixed['generation_seed']==seed,'Exact frozen paired generation seed')
            parent=ControlledImprover.from_dict(fixed['parent_protocol']);user=fixed['user_prompt']
            rendered=caller.tokenizer.apply_chat_template([{'role':'system','content':'Return exactly one JSON object matching the complete supplied schema. No markdown or prose outside JSON.'},
                {'role':'user','content':json.dumps(user)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=caller.tokenizer(rendered,return_tensors='pt').to('cuda');n=int(inputs['input_ids'].numel());need(n<=4096,'Frozen E0 input token limit')
            name=f'{slot:02d}--{method}';start={'slot':slot,'method':method,'generation_seed':seed,'input_tokens':n,'prompt':user,'rendered_prompt':rendered,
                'parent':parent.to_dict(),'thinking':False,'charged_before_generation':True,'start_unix':time.time()}
            write(root/'calls'/(name+'.start.json'),start);torch.manual_seed(seed);tick=time.perf_counter();raw=None;tokens=None;error=None
            try:
                signal.alarm(60)
                with torch.inference_mode():ids=caller.model.generate(**inputs,max_new_tokens=768,do_sample=True,temperature=.7,top_p=.8,top_k=20)
                tokens=int(ids.shape[1]-n);raw=caller.tokenizer.decode(ids[0,n:],skip_special_tokens=True)
            except Exception as exc:error=type(exc).__name__+': '+str(exc)
            finally:signal.alarm(0)
            receipt={'slot':slot,'method':method,'input_tokens':n,'output_tokens':tokens,'output_tokens_charged_upper_bound':tokens if tokens is not None else 768,
                'generation_s':time.perf_counter()-tick,'raw_response':raw,'generation_error':error,'raw_JSON_valid':False,'raw_full_ControlledImprover_valid':False,
                'interface_schema_valid':False,'compiled_protocol_valid':False,'different_parent':False,'duplicate':False,'changed_fields':[],
                'fallback_protocol':parent.to_dict(),'candidate_protocol':None,'transfer_reset':None,'parse_error':None}
            if raw is not None:
                try:
                    value=strict_json(raw);receipt['raw_JSON_valid']=True
                    try:ControlledImprover.from_dict(value['protocol']);receipt['raw_full_ControlledImprover_valid']=True
                    except (ValueError,TypeError,KeyError):pass
                    candidate,_=parse_proposal(raw,method);receipt.update(interface_schema_valid=True,compiled_protocol_valid=True,
                        candidate_protocol=candidate.to_dict(),different_parent=candidate!=parent,changed_fields=changed_fields(parent,candidate),transfer_reset=reset_receipt(candidate))
                    receipt['duplicate']=candidate.version_id in seen[method];seen[method].add(candidate.version_id)
                except (ValueError,TypeError,KeyError) as exc:receipt['parse_error']=type(exc).__name__+': '+str(exc)
            write(root/'calls'/(name+'.result.json'),receipt);receipts.append(receipt)
            if error is not None:
                write(root/'STOP.json',{'reason':'Generation runtime/timeout failure; no retry or later calls','failed_slot':slot,'method':method,
                    'attempted_calls':len(receipts),'last_failure':error,'unknown_output_cost_charged_upper_bound':768})
                raise RuntimeError(error)
            need(tuple(v._version for v in caller.model.parameters())==versions,'Frozen Qwen model parameters unchanged')
            print('M12E_SCHEMA_SLOT',slot,method,receipt['compiled_protocol_valid'],receipt['different_parent'],flush=True)
    need(sha(identity_path)==identity_sha,'No mid-execution identity refresh')
    summaries={}
    for method in config['methods']:
        rows=[v for v in receipts if v['method']==method];legal=sum(v['compiled_protocol_valid'] for v in rows);different=sum(v['compiled_protocol_valid'] and v['different_parent'] for v in rows)
        distinct=len({json.dumps(v['candidate_protocol'],sort_keys=True) for v in rows if v['compiled_protocol_valid'] and v['different_parent']})
        gate=legal>=11 and different>=8 and distinct>=3 and all(v['transfer_reset']['only_protocol_transferred'] for v in rows if v['compiled_protocol_valid'])
        summaries[method]={'calls':len(rows),'raw_JSON_valid':sum(v['raw_JSON_valid'] for v in rows),'raw_full_ControlledImprover_valid':sum(v['raw_full_ControlledImprover_valid'] for v in rows),
            'interface_schema_valid':sum(v['interface_schema_valid'] for v in rows),'compiled_protocol_valid':legal,'valid_and_different_parent':different,
            'distinct_nonparent_legal_protocols':distinct,'duplicates':sum(v['duplicate'] for v in rows),'timeouts':sum(v['generation_error'] is not None for v in rows),'passed':gate,
            'machine_structure_guarantee':method=='compact_leaves'}
    preferred=next((m for m in config['methods'] if summaries[m]['passed']),None)
    write(root/'complete.json',{'complete':True,'protocol_SHA256':sha(protocol),'execution_identity_SHA256':identity_sha,'model':model,'summaries':summaries,
        'E0_passed':preferred is not None,'future_interface':preferred,'real_model_calls':len(receipts),'input_tokens':sum(v['input_tokens'] for v in receipts),
        'known_output_tokens':sum(v['output_tokens'] or 0 for v in receipts),'unknown_output_calls':sum(v['output_tokens'] is None for v in receipts),
        'output_token_cost_upper_bound':sum(v['output_tokens_charged_upper_bound'] for v in receipts),'wall_s':time.monotonic()-begin,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'weights_updated':False,'real_environment_episodes':0,'task_programs_selected':0,
        'RSI_gain_claim':False,'old_M12D_reply_or_score_changed':False})
    print('M12E_SCHEMA_COMPLETE',preferred,len(receipts),flush=True)


if __name__=='__main__':main()

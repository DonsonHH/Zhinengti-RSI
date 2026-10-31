"""At mostsix already-frozen synthetic patch requests, only after actual G0 gain."""
import argparse
import json
from pathlib import Path
import os
import signal
import time
import torch

from patrol_rsi.g_timing_patch import parse_patch_reply,digest
from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.m4 import ModelCaller
from patrol_rsi.m12 import verify_open_model
from run_m12g_g0 import verify,stop

MODEL=Path('/share/home/renly/renly-patrol-rsi/models/Qwen3-8B-b968826d')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    root,source=a.run_root.resolve(),a.source_root.resolve();torch.set_num_threads(1);os.umask(0o077)
    try:
        ident=root/'G-execution-identity.json';binding=verify(ident,source,root,'G1a');identity_sha=sha(ident)
        gate=read(root/'G0-complete.json')
        need(gate['complete'] and gate['local_mechanism_supported'] and gate['identity_SHA256']==identity_sha,'Actual G0 local-skill gain and same-source evidence before model calls')
        for artifact in gate['sealed_batches']:need(sha(Path(artifact['path']))==artifact['sha256'],'G0 results unchanged')
        need(not(root/'G-STOP.json').exists(),'No source/runtime/safety stop')
        area=root/'G1a-precheck';area.mkdir(exist_ok=False);(area/'calls').mkdir()
        write(area/'launch.json',{'job_id':os.environ['SLURM_JOB_ID'],'identity_SHA256':identity_sha,'G0_receipt_SHA256':sha(root/'G0-complete.json')})
        path=Path(binding['bound_inputs']['prompts']);prompts=read(path)
        need(len(prompts['rows'])==prompts['calls_cap']==6 and len({v['id'] for v in prompts['rows']})==6,'Exactly6fixedsyntheticrequests')
        need(sha(MODEL/'verified_manifest.json')=='a1a327b7aee47f77c9348796bcb2d459d8792a8bc6644339b713668528af4361','Fixed Qwen revision manifest')
        model_verified=verify_open_model(MODEL)
        need(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'V100S' in torch.cuda.get_device_name(0),'One allocated V100S, no CPU fallback')
        caller=ModelCaller(MODEL,'cuda');versions=tuple(v._version for v in caller.model.parameters());rows=[]
        write(area/'model.json',{'verification':model_verified,'device':torch.cuda.get_device_name(0),'parameter_tensors':len(versions),'actual_training':False})
        for i,case in enumerate(prompts['rows']):
            need(not(root/'G-STOP.json').exists(),'No next call after G stop')
            system='Return exactly one JSON object: diagnosis, patch, expected_effect. Follow the typed patch schema and original IDs/hashes. No text outside JSON.'
            rendered=caller.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':json.dumps(case['prompt'])}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=caller.tokenizer(rendered,return_tensors='pt');n=int(inputs['input_ids'].numel());need(n<=6144,'Synthetic input token cap')
            base=area/'calls'/f'call-{i:02d}'
            write(base.with_suffix('.start.json'),{'case':case,'input_tokens':n,'rendered_prompt':rendered,'attempt_charged':True,'output_token_cap':1024,'unknown_output_charge_upper_bound':1024})
            def timeout(signum,frame):raise TimeoutError('Frozen90s synthetic call timeout')
            previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,90);start=time.perf_counter();output=None
            try:
                torch.manual_seed(case['generation_seed'])
                with torch.inference_mode():output=caller.model.generate(**inputs.to('cuda'),max_new_tokens=1024,do_sample=True,temperature=.7,top_p=.8,top_k=20)
                text=caller.tokenizer.decode(output[0,n:],skip_special_tokens=True)
            except BaseException as error:
                write(base.with_suffix('.failed.json'),{'type':type(error).__name__,'message':str(error),'input_tokens':n,'output_tokens':int(output.shape[1]-n) if output is not None else None,'output_charge_upper_bound':int(output.shape[1]-n) if output is not None else 1024,'no_retry':True});raise
            finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
            value={'id':case['id'],'raw_response':text,'input_tokens':n,'output_tokens':int(output.shape[1]-n),'generation_wall_s':time.perf_counter()-start}
            try:
                child,patch=parse_patch_reply(text,TimingSkill.initial_e1());value.update(raw_patch_valid=True,compiled_valid=True,patch=patch,compiled_skill=child.to_dict(),different_parent=digest(child.to_dict())!=digest(TimingSkill.initial_e1().to_dict()))
            except (ValueError,TypeError,KeyError,OverflowError) as error:value.update(raw_patch_valid=False,compiled_valid=False,error_type=type(error).__name__,error=str(error))
            need(tuple(v._version for v in caller.model.parameters())==versions,'Frozen Qwen parameter versions')
            write(base.with_suffix('.result.json'),value);rows.append(value);print('G_SYNTHETIC_CALL',i,value['compiled_valid'],flush=True)
        need(sha(ident)==identity_sha and sha(root/'G0-complete.json')==read(area/'launch.json')['G0_receipt_SHA256'],'Identity and G0 gate held through precheck')
        verify(ident,source,root,'G1a')
        write(area/'complete.json',{'complete':True,'identity_SHA256':identity_sha,'rows':rows,'real_model_calls':len(rows),'raw_compiled_valid':sum(v['compiled_valid'] for v in rows),
            'interface_passed':sum(v['compiled_valid'] for v in rows)>=5,'new_environment_episodes':0,'synthetic_only':True,'no_repair_or_resampling':True,'no_skill_gain_claim':True})
    except BaseException as error:stop(root,error);raise


if __name__=='__main__':main()

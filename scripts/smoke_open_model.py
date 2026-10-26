"""Allocated-GPU offline generation check, not an improvement experiment."""
import argparse
import json
from pathlib import Path
import time
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from patrol_rsi.improve import parse_proposal
from patrol_rsi.versions import AgentVersion

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
assert torch.cuda.is_available()
torch.set_num_threads(4);torch.manual_seed(600)
start=time.monotonic()
tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,attn_implementation='eager',local_files_only=True,trust_remote_code=False).to('cuda').eval()
prompt=tokenizer.apply_chat_template([{'role':'user','content':'Return exactly this JSON structure for an interface smoke test: {"patch":{"estimator_refresh_interval":2},"next_diagnostic_instruction":"Analyze redundant estimator calls."}. The instruction is outside patch. No other keys inside patch.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
x=tokenizer(prompt,return_tensors='pt').to('cuda')
with torch.no_grad():y=model.generate(**x,max_new_tokens=192,do_sample=False,pad_token_id=tokenizer.eos_token_id)
answer=tokenizer.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True)
(a.output/'raw-response.txt').write_text(answer)
print(answer,flush=True)
candidate,_=parse_proposal(answer,AgentVersion(),True)
assert candidate.estimator_refresh_interval==2
report={'purpose':'offline_generation_interface_smoke_only','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__,'dtype':str(model.dtype),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,'raw_response':answer,'input_tokens':int(x['input_ids'].numel()),'output_tokens':int(y.shape[1]-x['input_ids'].shape[1]),'parsed_version_id':candidate.version_id}
(a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

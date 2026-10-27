"""Read selected M6 weights, metadata and parameter hashes directly."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from patrol_rsi.dispatch_model import DispatchActorCritic
from patrol_rsi.m6 import VARIANTS,contract,parameter_hash
from m6_evidence_io import read_json

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2);rows=[];counts=set();initial={}
contract_sha=hashlib.sha256(json.dumps(contract(),sort_keys=True).encode()).hexdigest()
for variant in VARIANTS:
    for seed in range(3):
        root=a.results/f'{variant}-seed-{seed}';s=read_json(root/'summary.json')
        path=root/'best-policy.pt';state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['seed']==seed and state['variant']==variant and state['selected_episode']==s['selected_episode']
        assert state['validation']==s['best_validation'] and state['contract_sha256']==contract_sha
        model=DispatchActorCritic();model.load_state_dict(state['model'])
        count=sum(p.numel() for p in model.parameters());counts.add(count)
        assert parameter_hash(model)==s['selected_parameter_sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==s['checkpoint_sha256']
        if seed in initial:assert initial[seed]==s['initial_parameter_sha256']
        else:initial[seed]=s['initial_parameter_sha256']
        rows.append({'variant':variant,'seed':seed,'selected_episode':s['selected_episode'],'parameters':count,
            'checkpoint_sha256':s['checkpoint_sha256'],'selected_parameter_sha256':s['selected_parameter_sha256'],
            'actual_metadata_and_parameters_match_summary':True})
assert len(counts)==1 and len(set(initial.values()))==3
result={'actual_weight_files_read':9,'same_parameter_count':counts.pop(),'paired_initializations_across_variants':True,'rows':rows}
a.output.write_text(json.dumps(result,indent=2));print('M6 WEIGHTS VERIFIED',len(rows),'files;',result['same_parameter_count'],'parameters each')

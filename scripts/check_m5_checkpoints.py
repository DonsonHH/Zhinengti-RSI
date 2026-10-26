"""Read actual M5 selected weights and verify their declared metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from patrol_rsi.m5 import VARIANTS,contract,make_policy

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2);rows=[];initial={};counts=set()
contract_sha=hashlib.sha256(json.dumps(contract(),sort_keys=True).encode()).hexdigest()
for variant in VARIANTS:
    for seed in range(3):
        root=a.results/f'{variant}-seed-{seed}';s=json.loads((root/'summary.json').read_text())
        checkpoint=root/'best-policy.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['channels']==8 and state['variant']==variant and state['seed']==seed
        assert state['selected_episode']==s['selected_episode'] and state['validation']==s['best_validation']
        assert state['contract_sha256']==contract_sha
        model=make_policy();model.load_state_dict(state['model'])
        count=sum(p.numel() for p in model.parameters());counts.add(count)
        digest=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        assert digest==s['selected_parameter_sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==s['checkpoint_sha256']
        if seed in initial:assert initial[seed]==s['initial_parameter_sha256']
        else:initial[seed]=s['initial_parameter_sha256']
        rows.append({'variant':variant,'seed':seed,'selected_episode':state['selected_episode'],'parameters':count,
                     'checkpoint_sha256':s['checkpoint_sha256'],'parameter_sha256':digest,
                     'actual_weight_metadata_matches_summary':True})
assert len(counts)==1 and len(set(initial.values()))==3
report={'actual_selected_checkpoints_read':12,'same_architecture_parameter_count':counts.pop(),
        'three_distinct_initializations_paired_across_variants':True,'rows':rows}
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

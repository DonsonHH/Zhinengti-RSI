"""Post-validation behavior attribution, no new strategies or model calls."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import hashlib


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();r=a.run_root;results=[]
    for phase,seeds in (('development',(31000,31001)),('validation',(31020,31021))):
        allrows=[]
        for seed in seeds:allrows.extend(read(r/phase/f'seed-{seed}'/'complete.json')['rows'])
        for arm in ('Tour','frozen_PPO','human_professional','fixed_improver_retained','revised_PPO_evidence'):
            own=[v for v in allrows if v['arm']==arm];times=[];tools=Counter();before=0;noextra=0;rule_indices=Counter()
            for row in own:
                path=Path(row['record_path']);assert sha(path)==row['record_SHA256']
                record=json.loads(gzip.decompress(path.read_bytes()));optional=[v for v in record['workflow_decisions'] if v['task']!=v['baseline_next']]
                assert len(optional)<=1
                if not optional:noextra+=1;continue
                action=optional[0];times.append(action['time_s']);tools[action['tool']]+=1;rule_indices[str(action['rule_index'])]+=1
                before+=action['time_s']<row['specification']['fault_start_s']
            results.append({'phase':phase,'arm':arm,'conditions':len(own),'conditions_without_extra':noextra,
                'extra_dispatch_time_histogram_s':{str(k):v for k,v in sorted(Counter(times).items())},
                'extra_before_injection_conditions':before,'extra_selection_tools':dict(tools),'extra_rule_index_counts':dict(rule_indices),
                'interpretation':'Actualposthoc behavior counts. Injection time is evaluator-only comparison,never a policyfeature. Timingassociation doesnot alone identify a causal fix.'})
    out=r/'behavior-mechanism.json'
    with out.open('x',encoding='utf-8') as f:json.dump({'rows':results,'new_episodes':0,'new_model_calls':0,'new_candidates':0,'final_data_access':False},f,indent=2)
    print('M12D_BEHAVIOR_MECHANISM_DONE',len(results))


if __name__=='__main__':main()

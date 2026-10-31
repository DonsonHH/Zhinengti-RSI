"""Publish nonfinal receipts and final seed aggregates; keep final rows private."""
import argparse
import json
import hashlib
import tarfile
from pathlib import Path

import numpy as np


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run_root;o=a.output;o.mkdir(exist_ok=False)
    files=[]
    if any((r/f'rsi-trial-{i}'/'final-complete.json').exists() for i in range(3)):
        audit=read(r/'rsi-independent-final-audit.json');assert audit['passed'] and audit['global_final_seal_SHA256']==sha(r/'rsi-final-seal.json')
        assert len(audit['private_final_receipts'])==3
        for receipt in audit['private_final_receipts']:assert sha(Path(receipt['path']))==receipt['sha256']
    for name in ('development-complete.json','validation-complete.json','resource-check-before-final.json','execution-identity.json',
        'rsi-execution-identity.json','formal-launch-conditions.json','historical-identities.json','rsi-cost-profile.json','rsi-final-seal.json',
        'rsi-resource-before-final.json','rsi-first-repeat-resource.json','rsi-stop.json','resource-ledger.json','storage-final.json','resource-sacct.psv',
        'behavior-mechanism.json','rsi-independent-final-audit.json','resource-sacct-before-final.psv','resource-memory.psv','meta-schema-diagnosis.json','final-launch-accounting.psv',
        'm12d_guarded_final.sbatch','submit_m12d_final.py','ledger_m12d.py','audit_m12d_rsi_final.py','package_m12d_results.py'):
        if(r/name).exists():files.append(r/name)
    for name in ('qwen-preflight','jobs','gate','reviewed/gate','rsi-regression/gate','rsi-regression-v3/gate'):
        if(r/name).exists():files.extend(p for p in (r/name).rglob('*') if p.is_file())
    for phase,seeds in (('smoke',(30990,)),('development',(31000,31001)),('validation',(31020,31021))):
        for seed in seeds:
            folder=r/phase/f'seed-{seed}'
            for name in ('complete.json','started.json','negative-executor-tests.json'):
                if(folder/name).exists():files.append(folder/name)
            files.extend((folder/'sources').glob('*/complete.json'))
    trial_summaries=[]
    for i in range(3):
        root=r/f'rsi-trial-{i}'
        if not root.exists():continue
        for name in ('started.json','adapt-validation-complete.json','validation-mapping-seal.json','final-started.json','independent-search-audit.json'):
            if(root/name).exists():files.append(root/name)
        if(root/'final-one-shot-claim/claim.json').exists():files.append(root/'final-one-shot-claim/claim.json')
        if(root/'final-submission-claim').exists():files.extend(p for p in (root/'final-submission-claim').iterdir() if p.is_file())
        files.extend((root/'calls').glob('*.json'))
        files.extend(path for path in (root/'evaluations').glob('*.json') if read(path).get('phase')!='fresh_task_final')
        final=root/'final-complete.json'
        if not final.exists():
            trial_summaries.append({'trial':i,'adaptation_complete':(root/'adapt-validation-complete.json').exists(),'final_complete':False,
                'attempted_model_calls':len(list((root/'calls').glob('*.start.json'))),'returned_model_calls':len(list((root/'calls').glob('*.result.json'))),
                'environment_attempts':len(list((root/'cases').glob('*.attempt.json'))),'completed_environment_records':len(list((root/'cases').glob('*.json.gz')))})
            continue
        d=read(final);ad=read(root/'adapt-validation-complete.json');index={v['record_path']:v for v in d['new_final_actual_cases']}
        released={k:v for k,v in d.items() if k not in ('new_final_actual_cases','groups')};costs={};roles={}
        for name,group in d['groups'].items():
            rows=[index[path] for path in group['case_ids']]
            for row in rows:
                assert sha(Path(row['record_path']))==row['record_SHA256'] and row['audit']['passed']
            costs[name]={'conditions':len(rows),'mean_travel_m':float(np.mean([v['metrics']['travel_m'] for v in rows])),
                'mean_service_s':float(np.mean([v['metrics']['service_s'] for v in rows])),
                'mean_idle_s':float(np.mean([v['metrics']['idle_s'] for v in rows])),
                'mean_extra_checks':float(np.mean([v['metrics']['selected_detours'] for v in rows])),
                'actual_PPO_dispatches':sum(v['usage']['actual_PPO_dispatches'] for v in rows),
                'actual_PPO_multichoice_dispatches':sum(v['usage']['PPO_multichoice_dispatches'] for v in rows),
                'total_dispatches':sum(v['usage']['dispatches'] for v in rows),
                'PPO_single_legal_candidate_dispatches':sum(v['usage']['PPO_single_legal_candidate_dispatches'] for v in rows),
                'exclusive_shadow_actor_scoring_calls':sum(v['usage']['exclusive_shadow_actor_scoring_calls'] for v in rows),
                'rule_selected_dispatches':sum(v['usage']['rule_selected_dispatches'] for v in rows),
                'forced_return_Tour_dispatches':sum(v['usage']['forced_return_Tour_dispatches'] for v in rows),
                'maximum_exact_gap_ratio':max(v['metrics']['maximum_exact_gap_ratio'] for v in rows),
                'maximum_exact_gap_s':max(v['metrics']['maximum_exact_gap_s'] for v in rows),
                'revisit_violating_task_count':sum(v['metrics']['tasks_violating_exact_gap'] for v in rows),
                'CVAE_calls':sum(v['usage']['estimator_calls'] for v in rows),'all_independent_audits_passed':True}
            costs[name]['actual_PPO_multichoice_takeover_rate']=costs[name]['actual_PPO_multichoice_dispatches']/costs[name]['total_dispatches']
            roles[name]={role:{'segments_noise_repeated':sum(v['by_role'][role]['segments'] for v in rows),
                'confirmed_noise_repeated':sum(v['by_role'][role]['confirmed'] for v in rows)} for role in ('thermal','electrical','cooling','control')}
        released.update(costs=costs,professional_confirmation=roles,private_final_original_SHA256=sha(final),
            final_raw_rows_exported=False,final_private_record_inventory=[{'path':v['record_path'],'sha256':v['record_SHA256']} for v in index.values()])
        dest=o/f'rsi-trial-{i}-final-release.json';write(dest,released)
        trial_summaries.append({'trial':i,'adaptation_complete':True,'final_complete':True,'actual_model_calls':len(ad['model_call_receipts']),
            'input_tokens':ad['allcalls_input_tokens'],'output_tokens':ad['allcalls_output_tokens'],
            'meta_updates_accepted':sum(v['accepted'] for v in ad['meta_rounds']),'environment_attempts':sum(v['attempted_episodes'] for v in ad['costs'].values())+d['final_cost']['attempted_episodes'],
            'final_statistics':d['statistics'],'validation_statistics':ad['validation'],'final_costs':costs,'actual_programs':d['actual_program_mappings']})
    write(o/'trial-summary.json',{'trials':trial_summaries,'five_arm_final_never_generated':not(r/'final_test').exists(),
        'preflight_generations':1,'formal_calls_charged':sum(v.get('actual_model_calls',v.get('attempted_model_calls',0)) for v in trial_summaries),
        'final_samples_exported':False,'old_historical_results_unchanged':True})
    for path in files:
        target=o/path.relative_to(r);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    inventory=[{'path':str(p.relative_to(o)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in o.rglob('*') if p.is_file()]
    write(o/'export-inventory.json',{'files':inventory,'raw_final_rows_exported':False})
    archive=o.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in o.rglob('*'):
            if path.is_file():tar.add(path,arcname=str(path.relative_to(o)),recursive=False)
    print('M12D_RELEASE',str(archive),archive.stat().st_size,sha(archive),flush=True)


if __name__=='__main__':main()

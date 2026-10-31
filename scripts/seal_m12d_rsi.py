"""Seal all three real adaptations before generating any final processes."""
import argparse
from pathlib import Path
import time

from patrol_rsi.limited_prediction import read,sha,write,need
from patrol_rsi.paired_campaign import verify_identity
from patrol_rsi.rsi_campaign import validate_mapping


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();r=a.run_root
    identity_path=r/'rsi-execution-identity.json';identity=read(identity_path);verify_identity(identity)
    artifacts=[{'path':str(identity_path),'sha256':sha(identity_path)}]+identity['artifacts']
    calls=episodes=0;fps_by_phase={}
    for trial in range(3):
        root=r/f'rsi-trial-{trial}';path=root/'adapt-validation-complete.json';d=read(path)
        need(d['complete'] and not d['final_source_access'] and d['execution_identity_SHA256']==sha(identity_path),'All adaptations complete before final')
        validate_mapping(d,read(root/'validation-mapping-seal.json'),sha(identity_path),identity['RSI_SHA256'])
        need(len(d['model_call_receipts'])<=26 and d['allcalls_input_tokens']<=159744 and d['allcalls_output_tokens']<=19968,'Perrepeat exact modelbudget')
        calls+=len(d['model_call_receipts']);episodes+=sum(v['attempted_episodes'] for v in d['costs'].values())
        for row in d['case_index']:
            phase=row['specification']['phase'];fp=row['process_fingerprint']
            need(fp not in fps_by_phase or fps_by_phase[fp]==phase,'No physical fingerprint transfer across task partitions')
            fps_by_phase[fp]=phase
            need(row['audit']['passed'] and sha(Path(row['record_path']))==row['record_SHA256'],'Actual environment outcome independently audited')
        for name in ('adapt-validation-complete.json','validation-mapping-seal.json'):artifacts.append({'path':str(root/name),'sha256':sha(root/name)})
    need(calls<=78 and episodes<=5184,'All nonfinal model/environmentcaps before final')
    costs=read(r/'rsi-resource-before-final.json');need(costs['within_caps'],'Actual GPU/CPU budget before final')
    artifacts.append({'path':str(r/'rsi-resource-before-final.json'),'sha256':sha(r/'rsi-resource-before-final.json')})
    write(r/'rsi-final-seal.json',{'approved':True,'all_three_adaptations_frozen':True,'artifacts':artifacts,
        'formal_model_calls_consumed':calls,'nonfinal_environment_attempts':episodes,'final_environment_cap':1296,
        'sealed_unix':time.time(),'no_final_source_or_metric_used':True})
    print('M12D_RSI_ALL_MAPPING_SEAL',calls,episodes,flush=True)


if __name__=='__main__':main()

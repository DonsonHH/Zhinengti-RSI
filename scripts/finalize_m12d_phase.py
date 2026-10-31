"""Aggregate exact phase scope, verify pairing, stop or seal without tuning."""
import argparse
import time
from pathlib import Path

from patrol_rsi.aligned_patrol import arm_programs
from patrol_rsi.limited_prediction import need,read,sha,write
from patrol_rsi.paired_campaign import aggregate,validation_gate,protocols,PAIRED_SHA,exclude_history,verify_identity


def main():
    p=argparse.ArgumentParser()
    for name in ('paired','rsi','manifest','run-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('development','validation','final_test'),required=True);a=p.parse_args()
    protocol,_=protocols(a.paired,a.rsi);rows=[];artifacts=[];process_seen={};all_fps={}
    identity_path=a.run_root/'execution-identity.json';identity=read(identity_path)
    need(identity['paired_SHA256']==PAIRED_SHA and identity['RSI_SHA256']==sha(a.rsi),'Executed protocol identities')
    verify_identity(identity,{'paired':a.paired,'rsi':a.rsi,'manifest':a.manifest,'history':a.run_root/'historical-identities.json'})
    historical=read(a.run_root/'historical-identities.json')
    for phase in ('development','validation','final_test'):
        if phase=='final_test' and a.phase!='final_test':continue
        for seed in protocol['partition'][phase]:
            path=a.run_root/phase/f'seed-{seed}'/'complete.json'
            if phase==a.phase:need(path.is_file(),'All intended phase seeds completed')
            if not path.exists():continue
            d=read(path);need(d['complete'] and d['phase']==phase and d['seed']==seed and d['conditions']==18 and d['formal_episodes']==90,'Exact phase scope')
            need(d['execution_identity_SHA256']==sha(identity_path),'Completed phase produced under same immutable executed identity')
            need(read(path.parent/'started.json')['execution_identity_SHA256']==d['execution_identity_SHA256'],'Started and completed same executed identity')
            need(len(set(d['processes'].values()))==9,'Exactly9 numeric processes each seed')
            for key,fp in d['processes'].items():
                exclude_history(seed,fp,historical)
                need(fp not in all_fps,'No cross-seed/phase physical fingerprint overlap');all_fps[fp]=(phase,seed,key)
            if phase!=a.phase:continue
            rows.extend(d['rows']);artifacts.append({'path':str(path),'sha256':sha(path)})
    expected=len(protocol['partition'][a.phase])*90;need(len(rows)==expected,'All five arms across all condition pairs')
    for row in rows:
        key=row['physical_key'],row['noise_c'];process_seen.setdefault(key,[]).append(row)
        need(sha(Path(row['record_path']))==row['record_SHA256'],'Closed audited record unchanged')
    for own in process_seen.values():
        need(len(own)==5 and {v['arm'] for v in own}==set(arm_programs()),'Exactly5 methods same condition')
        need(len({v['process_fingerprint'] for v in own})==len({v['source_trace_file_SHA256'] for v in own})==1,'Same fixed physical arrays across arms')
        need(len({v['public_telemetry_sequence_SHA256'] for v in own})==1,'Same360 noisy observations across all five methods')
    report={'complete':True,'phase':a.phase,'formal_episodes':len(rows),'replayed_episodes':len(rows),
        'protocol_SHA256':PAIRED_SHA,'manifest_SHA256':sha(a.manifest),'artifacts':artifacts,'aggregate':aggregate(rows),
        'execution_identity_SHA256':sha(identity_path),
        'unique_numeric_processes':len({v['process_fingerprint'] for v in rows}),'cross_phase_fingerprint_overlap':0,
        'final_sources_opened':a.phase=='final_test','all_five_arms_reported':True}
    if a.phase=='validation':
        # The independently reviewed resource wrapper constrains each phase.
        # Measured allocated CPU costs are recorded before final opening.
        resource=read(a.run_root/'resource-check-before-final.json')
        report['gate']=validation_gate(report['aggregate'],sources_ok=True,budget_ok=resource['within_caps'])
    write(a.run_root/(a.phase+'-complete.json'),report)
    if a.phase=='validation' and report['gate']['passed']:
        write(a.run_root/'validation-seal.json',{'protocol_SHA256':PAIRED_SHA,'manifest_SHA256':sha(a.manifest),
            'gate':report['gate'],'arms':{k:v.to_dict() for k,v in arm_programs().items()},
            'RSI_SHA256':sha(a.rsi),'artifacts':artifacts+identity['artifacts']+[{'path':str(identity_path),'sha256':sha(identity_path)},
                {'path':str(a.run_root/'validation-complete.json'),'sha256':sha(a.run_root/'validation-complete.json')},
                {'path':str(a.run_root/'resource-check-before-final.json'),'sha256':sha(a.run_root/'resource-check-before-final.json')}],
            'sealed_unix':time.time(),'final_selection_or_substitution':False})
    print('M12D_PHASE_COMPLETE',a.phase,report.get('gate',{}).get('status','all_audited'),flush=True)


if __name__=='__main__':main()

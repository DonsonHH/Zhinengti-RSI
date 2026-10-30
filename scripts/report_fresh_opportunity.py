"""Audit complete fresh coverage and dependence; no model scores or fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np

from patrol_rsi.fresh_opportunity import (KIND_IDS, ROOTS, SPLITS, coverage_gate,
    fingerprint_conflicts, frozen_inputs, need, read, sha, write)
from patrol_rsi.opportunity_data import FEATURE_NAMES, HORIZONS
from report_opportunity_data import Coverage


def partial_inventory(root, manifest):
    """Known finished conditions only; conflicts prevent further phase launch."""
    rows=[]
    for item in manifest:
        p=root/item['split']/f"seed-{item['group_seed']}"/item['id']/'summary.json'
        if p.exists():
            s=read(p)
            rows.append({k:s[k] for k in ('id','split','group_seed','numeric_process_fingerprint')})
    return {'conditions_generated':len(rows),'expected':len(manifest),'complete':len(rows)==len(manifest),
        'cross_split_fingerprint_conflicts':fingerprint_conflicts(rows)}


def diagnostic_entropy(groups):
    total=sum(sum(v) for v in groups.values())
    entropy=0.
    for counts in groups.values():
        n=sum(counts)
        if n:
            entropy-=sum(c/total*math.log2(c/n) for c in counts if c)
    return entropy


def development_clock_diagnostic(root, manifest):
    """In-sample fixed-bin description, not a fitted classifier or ablation gain."""
    cols={name:i for i,name in enumerate(FEATURE_NAMES)}
    groups={k:defaultdict(lambda:[0,0]) for k in ('role_horizon','role_horizon_clock','role_horizon_clock_schedule','role_horizon_clock_schedule_reading')}
    by_clock=defaultdict(Counter)
    # Frozen threshold bins use original professional thresholds and availability.
    for item in manifest:
        if item['split']!='development':continue
        p=root/'development'/f"seed-{item['group_seed']}"/item['id']
        with np.load(p/'public-features.npz') as values:
            base,completion=values['base'],values['completion']
        with np.load(p/'targets-evaluator-only.npz') as values:
            b=values['B_action']
        for t,now in enumerate(ROOTS):
            for h,horizon in enumerate(HORIZONS):
                for j in np.flatnonzero(b[t,h]>=0):
                    kind=int(KIND_IDS[j]); y=int(b[t,h,j]); key=(kind,horizon)
                    clock=key+(now,)
                    # Planned dispatch relative to root, optional flag and in-flight flag.
                    schedule=clock+(float(completion[t,h,j,3]),float(completion[t,h,j,4]),float(completion[t,h,j,11]))
                    channel,threshold,side=(('temperature_measurement',65.,1),('contact',60.,1),('flow',33.6,-1),('sensor_error',1.,2))[kind]
                    available=bool(base[t,j,cols[channel+'.available']])
                    v=base[t,j,cols[channel+'.value']]
                    alarm=bool(abs(v)>threshold) if side==2 else bool((v-threshold)*side>0)
                    reading=schedule+(available,alarm if available else False)
                    for name,k in zip(groups,(key,clock,schedule,reading)):
                        groups[name][k][y]+=1
                    by_clock[f'{kind}/{horizon}/{now}'].update(valid=1,positive=y)
    entropy={name:diagnostic_entropy(v) for name,v in groups.items()}
    return {'scope':'Development only; fixed-bin descriptive conditional entropy. This is not out-of-sample skill, a predictor fit, unique causal contribution or a formal selection result.',
        'conditional_entropy_bits':entropy,
        'clock_association_bits':entropy['role_horizon']-entropy['role_horizon_clock'],
        'schedule_additional_association_bits':entropy['role_horizon_clock']-entropy['role_horizon_clock_schedule'],
        'reading_additional_association_bits':entropy['role_horizon_clock_schedule']-entropy['role_horizon_clock_schedule_reading'],
        'counts_by_role_horizon_root':{k:dict(v) for k,v in by_clock.items()},
        'interpretation':'Fixed335/705 onset can create repeatable clock priors. More conditioning bins can mechanically lower empirical entropy; independent prediction gains require future prespecified comparisons. No model was searched here.'}


def report(root, protocol, manifest, output, audit_root):
    inv=partial_inventory(root,manifest)
    need(inv['complete'],'All1280 frozen conditions must be generated')
    output.mkdir(exist_ok=False)
    write(output/'cross-split-conflicts.json',inv)
    dimensions=defaultdict(dict); negatives=defaultdict(set)
    extras=defaultdict(Counter); sums=[]; fingerprints={}; noises=defaultdict(list)
    audits=[]; audit_counts=Counter(); storage=0; private_process_storage=0; private_electrical=[]; public_electrical=[]
    file_inventory=[]
    groups=[(split,seed) for split,seeds in protocol['new_seed_groups'].items() for seed in seeds]
    for gi,(split,seed) in enumerate(groups):
        path=audit_root/f'group-{gi}.json'; v=read(path)
        need(v['status']=='passed' and v['counts']['conditions']==20 and (v['split'],v['seed'])==(split,seed),'Independent group audit')
        audit_counts.update(v['counts']); audits.append({'group_index':gi,'split':split,'seed':seed,'sha256':sha(path),'counts':v['counts']})
    for item in manifest:
        group=root/item['split']/f"seed-{item['group_seed']}"; p=group/item['id']
        s=read(p/'summary.json')
        need(all(s[k]==v for k,v in item.items()),'Frozen item exact identity')
        fp=s['numeric_process_fingerprint']; fingerprints[item['physical_key']]=fp
        noise_key=(item['split'],item['physical_key'])
        noises[noise_key].append({'noise_c':item['specification']['noise_c'],'fingerprint':fp,
            'labels_SHA':sha(p/'labels-only.npz'),'targets_SHA':sha(p/'targets-evaluator-only.npz')})
        events={v['id']:v for v in s['event_definitions']}
        condition={'condition':item['specification'],'numeric_process_fingerprint':fp}
        for row in s['strata']:
            keys=[('split',item['split']+'/'+row['kind']+'/'+str(row['horizon_s'])),
                ('scenario',item['split']+'/'+item['specification']['scenario']+'/'+row['kind']+'/'+str(row['horizon_s'])),
                ('phase',item['split']+'/'+str(item['specification']['fault_start_s'])+'/'+row['kind']+'/'+str(row['horizon_s'])),
                ('process',item['split']+'/'+fp+'/'+row['kind']+'/'+str(row['horizon_s']))]
            for dim,key in keys:
                cell=dimensions[dim].setdefault(key,Coverage());cell.add(row,condition,events)
                if row['B_action_negative']:negatives[dim,key].add(item['group_seed'])
                for field in ('A_positive_without_legal_B','A_positive_B_reference_negative','B_positive_A_negative'):
                    extras[dim,key][field]+=row[field]
        # Never export final row labels/event timing/continuous diagnostic values.
        sums.append(dict(item,numeric_process_fingerprint=fp,dataset_samples=s['dataset_samples'],
            feature_audit=s['feature_audit'],counts=[{k:v for k,v in row.items() if not k.endswith('_ids')} for row in s['strata']]))
        if 'electrical_diagnostic' in s and item['specification']['noise_c']==.2:
            v=dict(item,numeric_process_fingerprint=fp,diagnostic=s['electrical_diagnostic'])
            private_electrical.append(v)
            if item['split']!='final_test':public_electrical.append(v)
        for file in p.iterdir():
            if file.is_file():
                storage+=file.stat().st_size
                file_inventory.append({'path':str(file),'bytes':file.stat().st_size,'sha256':sha(file),
                    'final_test_sealed':item['split']=='final_test'})
    for split,seed in groups:
        group=root/split/f'seed-{seed}'
        for file in group.rglob('*'):
            if file.is_file() and file.parent in (group, group/'processes'):
                storage+=file.stat().st_size
                if file.parent==group/'processes':private_process_storage+=file.stat().st_size
                file_inventory.append({'path':str(file),'bytes':file.stat().st_size,'sha256':sha(file),'final_test_sealed':split=='final_test'})
    for pair in noises.values():
        need(len(pair)==2 and {v['noise_c'] for v in pair}=={.2,1.},'Two exact noise versions')
        need(all(len({v[k] for v in pair})==1 for k in ('fingerprint','labels_SHA','targets_SHA')),'Same physical/noise A/B/legality identity')
    public_derived_storage=storage-private_process_storage
    need(len(noises)==640 and public_derived_storage<=6*1000**3,'Manifest and6GB public-derived storage cap')
    values={}
    for dim,cells in dimensions.items():
        values[dim]={}
        for key,cell in cells.items():
            v=cell.value();v['B_negative_seed_groups']=sorted(negatives[dim,key]);v.update(extras[dim,key])
            v['B_action_valid_fraction']=v['B_action_valid']/v['total_samples']
            values[dim][key]=v
    table=[]
    for split in SPLITS:
        for kind in ('thermal','electrical','cooling','control'):
            for horizon in HORIZONS:
                v=values['split'][split+'/'+kind+'/'+str(horizon)]
                table.append(dict(split=split,**v,gate=coverage_gate(v,protocol['data_gate']['per_kind_per_horizon_per_split_minimum'][split])))
    passed=[v for v in table if v['gate']['passed']]
    conflict=inv['cross_split_fingerprint_conflicts']
    verdict='NO-GO' if conflict or not passed else 'GO' if len(passed)==48 else 'LIMITED'
    result={'stage':'B4-B1a','generation_complete':True,'generated_conditions':1280,'named_physical_processes':640,
        'numeric_processes':len(set(fingerprints.values())),'seed_groups':64,
        'sample_rows':sum(v['dataset_samples'] for v in sums),'coverage_cells':table,
        'passed_cells':len(passed),'failed_cells':48-len(passed),'coverage_verdict':verdict,
        'B4_B1b_full_domain_fitting_allowed':verdict=='GO',
        'limited_protocol_automatically_authorized':False,
        'cross_split_fingerprint_conflicts':conflict,'all640_noise_pairs_labels_identical':True,
        'derived_and_source_data_bytes':storage,'private_physical_source_bytes':private_process_storage,
        'public_derived_data_bytes':public_derived_storage,'public_derived_cap_bytes':6*1000**3,
        'independent_counts':dict(audit_counts),
        'predictor_fits':0,'CVAE_updates':0,'PPO_updates':0,'LLM_calls':0,
        'denominators':'Sample rows correlated; deduplicated retained event keys=(numeric fingerprint,task,start,end). Separate counts for A endpoints and legal B; events across horizons never summed as independent.',
        'final_test_seal':'Final row labels, process arrays, public histories and continuous per-process diagnostics remain700/600 at source. Released statistics only; no predictor score or model selection.'}
    write(output/'coverage.json',result)
    write(output/'coverage-dimensions.json',values)
    write(output/'independent-audits.json',{'groups':audits,'totals':dict(audit_counts)})
    write(output/'raw-storage.json',{'bytes':storage,'public_derived_bytes':public_derived_storage,
        'private_physical_source_bytes':private_process_storage,'public_derived_cap_bytes':6*1000**3,'files':file_inventory})
    (output/'condition-counts.json.gz').write_bytes(gzip.compress(json.dumps(sums,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    write(output/'electrical-nonfinal-diagnostics.json',public_electrical)
    # This path is separate and sealed. It is hashed, not downloaded for modelling.
    private_path=root.parent/'sealed-electrical-all-processes.json'
    write(private_path,private_electrical);private_path.chmod(0o600)
    write(output/'sealed-diagnostic-receipt.json',{'path':str(private_path),'bytes':private_path.stat().st_size,'sha256':sha(private_path),'processes':len(private_electrical),'exported':False})
    write(output/'development-clock-diagnostic.json',development_clock_diagnostic(root,manifest))
    lines=['# 四专业×三视野×四划分覆盖验收','',f'结论：{verdict}。通过{len(passed)}/48单元；全域拟合允许={verdict=="GO"}。','',
        '|划分|专业|秒|有效B|B正例|正例种子|正例过程|保留事件|负例种子|缺口（种子/过程/事件/负种子）|通过|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for v in table:
        a=v['gate']['actual'];d=v['gate']['deficits']
        lines.append('|'+ '|'.join(map(str,[v['split'],v['kind'],v['horizon_s'],v['B_action_valid'],v['B_action_positive'],
            a['positive_seed_groups'],a['positive_numeric_processes'],a['distinct_retained_positive_events'],a['valid_negative_seed_groups'],
            '/'.join(str(x) for x in d.values()),v['gate']['passed']]))+'|')
    (output/'覆盖表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FRESH_COVERAGE_DONE',verdict,len(passed),storage,flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('root','protocol','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-root',type=Path);p.add_argument('--partial',action='store_true')
    args=p.parse_args();protocol,manifest=frozen_inputs(args.protocol,args.manifest)
    if args.partial:
        value=partial_inventory(args.root,manifest);write(args.output,value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
        if value['cross_split_fingerprint_conflicts']:raise SystemExit(2)
    else:report(args.root,protocol,manifest,args.output,args.audit_root)


if __name__=='__main__':main()

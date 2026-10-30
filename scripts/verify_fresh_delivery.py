"""Portable released B1a evidence checks without opening sealed final data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from patrol_rsi.fresh_opportunity import MANIFEST_SHA, PROTOCOL_SHA, WEIGHTS_SHA, need, read, sha, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.workspace;b=root/'research/2026-10-30-m12c1b4b1a-fresh';e=b/'evidence'
    delivery=read(e/'delivery-manifest.json');archive=b/'source/delivery.tar.gz'
    need(sha(archive)==delivery['sha256'] and archive.stat().st_size==delivery['bytes'],'Actual delivered archive')
    with tarfile.open(archive,'r:gz') as tar:
        members={v.name:v for v in tar.getmembers() if v.isfile()}
        need(set(members)==set(delivery['files']),'Exact allow-listed package members')
        for name,row in delivery['files'].items():
            raw=tar.extractfile(members[name]).read()
            need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Every archive member bytes')
            need((e/name).stat().st_size==row['bytes'] and sha(e/name)==row['sha256'],'Every extracted artifact bytes')
            need('final_test' not in name and ('sealed' not in name or name=='report/sealed-diagnostic-receipt.json'),
                'No final original or sealed evaluator file delivered; hash-only receipt allowed')
            need(not name.startswith('jobs/final-'),'Final per-seed event counts stay in sealed original logs')
    archives=[]
    latest={}
    manifests=[p for p in b.glob('*-manifest.json') if 'created_utc' in read(p)]
    for manifest_path in sorted(manifests,key=lambda p:read(p)['created_utc']):
        m=read(manifest_path)
        if 'created_utc' not in m:continue
        path=b/'source'/m['archive'];need(sha(path)==m['sha256'],'All immutable source archive SHA')
        with tarfile.open(path,'r:gz') as tar:
            source={v.name:v for v in tar.getmembers() if v.isfile()}
            need(set(source)==set(m['files']),'Exact frozen source members')
            for name,row in m['files'].items():
                raw=tar.extractfile(source[name]).read()
                need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'All frozen source member SHA')
                latest[name]=row
        archives.append(m['archive'])
    for name,row in latest.items():
        need(sha(root/name)==row['sha256'],'Current tested source bound to latest archive: '+name)
    protected=read(e/'protected-baseline.json');allowed={'.github/workflows/cpu-checks.yml','CURRENT_RESEARCH.md','README.md'}
    unchanged=0
    for name,row in protected['files'].items():
        if name in allowed:continue
        need(sha(root/name)==row['sha256'] and (root/name).stat().st_size==row['bytes'],'Old evidence/code must remain exact: '+name)
        unchanged+=1
    old=root/'research/2026-10-30-m12c1b4b0-data'
    need(sha(old/'B4-B1冻结协议.json')==PROTOCOL_SHA and sha(old/'B4-B1新数据清单.json')==MANIFEST_SHA,'Original frozen protocol and manifest')
    acceptance=read(e/'final-acceptance.json')
    need(len(delivery['sealed_final_log_receipts'])==48 and
        all(not v['content_exported'] for v in delivery['sealed_final_log_receipts']), 'All24 final stdout/stderr pairs sealed; hashes only')
    for name,row in acceptance['bindings'].items():
        need(sha(e/name)==row['sha256'] and (e/name).stat().st_size==row['bytes'],'Acceptance evidence SHA')
    coverage=read(e/'report/coverage.json');table=coverage['coverage_cells'];need(len(table)==48,'Complete48 cells')
    protocol=read(old/'B4-B1冻结协议.json')
    for v in table:
        gate=v['gate'];minimum=protocol['data_gate']['per_kind_per_horizon_per_split_minimum'][v['split']]
        need(gate['minimum']==minimum and gate['deficits']=={k:max(0,n-gate['actual'][k]) for k,n in minimum.items()},'Exact frozen cell thresholds/deficits')
        need(gate['passed']==(not any(gate['deficits'].values())),'Cell readiness exact')
        need(v['B_action_valid']==v['B_action_positive']+v['B_action_negative'] and
             v['total_samples']==v['B_action_valid']+v['B_action_invalid'],'No denominator or mask mixing')
        if v['split']=='final_test':
            need(not any(k in v for k in ('seed_groups','B_positive_seed_groups','B_negative_seed_groups')),'Final seed outcome sets sealed')
    dimensions=read(e/'report/coverage-dimensions.json')
    need(not any(k.startswith('final_test/') for d,vals in dimensions.items() if d!='split' for k in vals),'No final process/scenario/phase label table')
    summaries=json.loads(gzip.decompress((e/'report/condition-counts-nonfinal.json.gz').read_bytes()))
    need(len(summaries)==800 and all(v['split']!='final_test' for v in summaries),'Nonfinal counts only')
    need(acceptance['engineering_execution_and_audit']=='complete' and not acceptance['new_predictor_fits'] and
         not acceptance['PPO_retraining'] and not acceptance['CVAE_updates'] and not acceptance['Qwen_calls'],'No premature training')
    write(args.output,{'status':'passed','protected_old_files_unchanged':unchanged,'source_archives':archives,
        'delivered_files':len(delivery['files']),'full_generated_conditions_verified_by_remote_audit':1280,
        'physical_processes_replayed':640,'coverage_cells':48,'released_nonfinal_counts':800,
        'all_final_originals_not_exported':True,'coverage_verdict':acceptance['coverage_verdict'],
        'fitting_allowed':acceptance['B4_B1b_full_domain_fitting_allowed'],
        'scope':'Local byte/archive/protocol/denominator/decision/seal audit; full1280 independent labels/features and640 replay are remote verified originals, bound by delivered receipts. No local opening of final row labels.'})
    print('FRESH_PORTABLE_VERIFICATION_PASSED',acceptance['coverage_verdict'],flush=True)


if __name__=='__main__':main()

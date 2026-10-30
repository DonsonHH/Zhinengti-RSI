"""Stream original B1a features into limited-cohort float32 memory maps."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from patrol_rsi.fresh_opportunity import frozen_inputs
from patrol_rsi.limited_prediction import CELLS,load_reviewed,need,read,sha,write


B1A_STORAGE_SHA='ef404d3ecadb8b6e6b503e6bfac11551a32e9a10d66c117c74648f25acae8065'
B1A_COVERAGE_SHA='4d08cbd6e9350f73ee4a750dd7958470a08cf72b50a133d7fff9ce11f5a25854'


def prepare(args):
    protocol,review=load_reviewed(args.protocol,args.review)
    old,manifest=frozen_inputs(args.old_protocol,args.old_manifest)
    need(sha(args.source_inventory)==B1A_STORAGE_SHA,'Immutable independently replayed B1a source inventory')
    need(sha(args.coverage)==B1A_COVERAGE_SHA,'Immutable B1a48cell coverage authority')
    source_index={v['path']:v for v in read(args.source_inventory)['files']}
    if args.phase=='final_test':
        need(args.selection_receipt is not None,'Final source cache requires frozen selection first')
        receipt=read(args.selection_receipt)
        need(receipt['status']=='frozen_before_final_open' and receipt['protocol_SHA256']==sha(args.protocol),'Matching pre-final selection receipt')
        for v in receipt['artifacts']:
            need(sha(Path(v['path']))==v['sha256'],'All selected model/calibrator/threshold bytes frozen')
        write(args.output/'final-open.json',{'protocol_SHA256':sha(args.protocol),'selection_SHA256':sha(args.selection_receipt),
            'opened_unix':time.time(),'purpose':'Prepare and evaluate fixed models once, no final-based selection','job_id':os.environ.get('SLURM_JOB_ID')})
    else:need(args.phase in ('development','validation_calibration','validation_selection'),'Only allowed nonfinal splits')
    def checked(path):
        row=source_index[str(path)]
        need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Every used original B1a byte')
    selected=[];seen=set();seed_fingerprints=defaultdict(set)
    for item in manifest:
        if item['split']!=args.phase:continue
        folder=args.source_root/item['split']/f"seed-{item['group_seed']}"/item['id']
        p=folder/'source-identity-not-features.json';checked(p);identity=read(p)
        need(all(identity[k]==v for k,v in item.items()),'Frozen original condition identity')
        fp=identity['numeric_process_fingerprint'];key=(item['group_seed'],fp,item['specification']['noise_c'])
        seed_fingerprints[item['group_seed']].add(fp)
        if key in seen:continue
        seen.add(key);selected.append((item,folder,fp))
    nseeds=len(old['new_seed_groups'][args.phase]);need(len(seed_fingerprints)==nseeds,'All declared seed groups')
    need(len(selected)==nseeds*18 and all(len(v)==9 for v in seed_fingerprints.values()),'Original healthy-phase dedup,9numericprocesses x2noise perseed')
    destination=args.output/args.phase;destination.mkdir(exist_ok=False)
    upper={};coverage=read(args.coverage)
    for ci,(kind,horizon) in enumerate(CELLS):
        row=next(v for v in coverage['coverage_cells'] if (v['split'],v['kind'],v['horizon_s'])==(args.phase,kind,horizon))
        need(row['gate']['passed'],'Only the seven cells with prior coverage authority')
        upper[ci]=row['B_action_valid']
    existing_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    planned_bytes=sum(upper.values())*(98*4+8+2+8*4)+7*4*256+1000000
    need(existing_bytes+planned_bytes<=protocol['resources']['new_derived_storage_cap_bytes'],
        'Preallocate only within4GB:existingcache plus every declaredarray/header/metadata reserve')
    arrays={};counts=Counter();proc_ids={};condition_ids={item['id']:i for i,item in enumerate(manifest)}
    for ci in range(7):
        folder=destination/f'cell-{ci}';folder.mkdir()
        arrays[ci]={key:np.lib.format.open_memmap(folder/name,mode='w+',dtype=dtype,shape=shape) for key,name,dtype,shape in
            [('x','features.npy',np.float32,(upper[ci],98)),('w','weights.npy',np.float64,(upper[ci],)),
             ('y','labels.npy',np.int8,(upper[ci],2)),('m','evaluator-meta.npy',np.int32,(upper[ci],8))]}
    start=time.perf_counter();file_checks=0
    for item,folder,fp in selected:
        for name in ('public-features.npz','targets-evaluator-only.npz','labels-only.npz'):
            checked(folder/name);file_checks+=1
        with np.load(folder/'public-features.npz') as f:base,actions=f['base'],f['completion']
        with np.load(folder/'targets-evaluator-only.npz') as f:ta,tb=f['A'],f['B_action']
        with np.load(folder/'labels-only.npz') as f:raw_ids,retained_ids=f['event_id'],f['retained_event_id']
        process=proc_ids.setdefault(fp,len(proc_ids));seed=item['group_seed']
        for ci,(kind,horizon) in enumerate(CELLS):
            h=(30,60,120).index(horizon);lo,hi={'thermal':(0,128),'electrical':(128,144),'control':(160,176)}[kind]
            ri,local=np.nonzero(tb[:,h,lo:hi]>=0);tasks=local+lo;n=len(ri)
            need(n>0,'A zero-opportunity condition cannot silently disappear')
            need(np.all((ta[ri,h,tasks]>=0)&(tb[ri,h,tasks]>=0)) and not np.any((tb[ri,h,tasks]==1)&(ta[ri,h,tasks]!=1)),
                'Only legal cohort,distinct A/B,Bpositive impliesA')
            at=counts[ci];end=at+n;need(end<=upper[ci],'Original source count upper bound')
            output=arrays[ci]
            output['x'][at:end]=np.concatenate((base[ri,tasks],actions[ri,h,tasks]),axis=1)
            output['w'][at:end]=1/(nseeds*len(seed_fingerprints[seed])*2*n)
            output['y'][at:end]=np.column_stack((ta[ri,h,tasks],tb[ri,h,tasks]))
            ticks=((ri+1)*30+horizon)//5-1
            output['m'][at:end]=np.column_stack((ri,tasks,np.full(n,condition_ids[item['id']]),np.full(n,seed),
                np.full(n,process),retained_ids[ticks,tasks],raw_ids[ticks,tasks],(ri+1)*30+horizon))
            counts[ci]=end
    summaries=[]
    for ci,(kind,horizon) in enumerate(CELLS):
        folder=destination/f'cell-{ci}';output=arrays[ci]
        for v in output.values():v.flush()
        n=counts[ci];need(abs(float(output['w'][:n].sum())-1)<1e-9,'Exact equal seed/process/noise row hierarchy')
        summary={'cell_index':ci,'kind':kind,'horizon_s':horizon,'phase':args.phase,'rows':n,'capacity':upper[ci],
            'A_positive':int((output['y'][:n,0]==1).sum()),'B_positive':int((output['y'][:n,1]==1).sum()),
            'cohort':'Both A/Bon samelegalB>=0samples','meta_names':['root_index','task','condition_id','seed','process','retained_event_id','raw_event_id','completion_time_s'],
            'protocol_SHA256':sha(args.protocol),'source_inventory_SHA256':B1A_STORAGE_SHA,
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.iterdir() if p.is_file()}}
        write(folder/'complete.json',summary);summaries.append(summary)
    write(destination/'complete.json',{'phase':args.phase,'conditions_after_dedup':len(selected),'unique_processes':len(proc_ids),
        'seed_groups':nseeds,'protocol_SHA256':sha(args.protocol),'source_files_checked':file_checks,
        'source_physics_regenerated':0,'cells':summaries,'wall_s':time.perf_counter()-start,
        'new_cache_bytes':sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())})
    need(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<=protocol['resources']['new_derived_storage_cap_bytes'],'4GB bounded derived cache')
    print('LIMITED_CACHE_DONE',args.phase,[counts[i] for i in range(7)],flush=True)


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','old-protocol','old-manifest','source-root','source-inventory','coverage','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--phase',required=True);p.add_argument('--selection-receipt',type=Path)
    args=p.parse_args();os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True);prepare(args)


if __name__=='__main__':main()

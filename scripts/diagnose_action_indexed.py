"""Development-only exact legal-action confirmation coverage;no predictor fit."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gzip
import os
from pathlib import Path
import time

import numpy as np

from patrol_rsi.fresh_opportunity import frozen_inputs
from patrol_rsi.limited_prediction import load_reviewed,need,read,sha,write


KINDS=('thermal','electrical','cooling','control')


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','old-protocol','old-manifest','source-root','source-inventory','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--seed-index',type=int,required=True);args=p.parse_args();os.umask(0o077)
    protocol,review=load_reviewed(args.protocol,args.review);old,manifest=frozen_inputs(args.old_protocol,args.old_manifest)
    need(args.seed_index in range(24),'Development-only preregistered Action-Indexed diagnostic')
    seed=old['new_seed_groups']['development'][args.seed_index]
    inventory={v['path']:v for v in read(args.source_inventory)['files']}
    need(sha(args.source_inventory)=='ef404d3ecadb8b6e6b503e6bfac11551a32e9a10d66c117c74648f25acae8065','Immutable B1a source audit')
    def check(path):
        row=inventory[str(path)];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Original diagnostic source bytes')
    args.output.mkdir(exist_ok=False);group=args.source_root/'development'/f'seed-{seed}'
    check(group/'geometry.npz')
    with np.load(group/'geometry.npz') as f:geometry={k:f[k] for k in f.files}
    statistics=defaultdict(Counter);events=defaultdict(set);processes=defaultdict(set)
    raw_events=defaultdict(set);physical=defaultdict(set);fingerprints=set();sourcefiles=[];started=time.perf_counter();conds=0
    for item in (v for v in manifest if v['split']=='development' and v['group_seed']==seed):
        folder=group/item['id']
        for name in ('tour-record.json.gz','labels-only.npz','events-evaluator-only.json','source-identity-not-features.json'):
            check(folder/name);sourcefiles.append({'path':str(folder/name),'sha256':sha(folder/name)})
        identity=read(folder/'source-identity-not-features.json');fp=identity['numeric_process_fingerprint'];fingerprints.add(fp)
        need(identity['source_geometry_sha256']==sha(group/'geometry.npz'),'Per-condition original full-return geometry binding')
        tour=__import__('json').loads(gzip.decompress((folder/'tour-record.json.gz').read_bytes()))
        need(tour['optional_checks']==0 and tour['exact_gap_violations']==0,'ProtectedTour leaves original one-extra budget unspent')
        with np.load(folder/'labels-only.npz') as f:
            a,b,retained,raw=f['A_physical'],f['B_reference'],f['retained_event_id'],f['event_id']
        own_events=read(folder/'events-evaluator-only.json')
        for event in own_events:
            if event['retained_at_least_15s']:physical[event['kind']].add((fp,event['id']))
        # B1a geometry enumerates every actual5s free decision state's hard-legal
        # extraCHECK. Its full-return arithmetic and public schedule were
        # independently audited across all1280 original conditions.
        dispatch=geometry['dispatch_time_s'].astype(int);finish=geometry['finish_s'].astype(int)
        tasks=geometry['task'].astype(int);agents=geometry['agent'].astype(int)
        optional=np.ones(len(tasks),bool)
        planned=[]
        for assignment in tour['assignments']:
            start=assignment['time_s'];end=start+int(np.ceil((assignment['distance_m']+assignment['service_s'])/5-1e-10))*5
            if end<=1800:planned.append((start,end,assignment['task'],assignment['agent']))
        plan=np.asarray(planned,int)
        dispatch=np.r_[dispatch,plan[:,0]];finish=np.r_[finish,plan[:,1]];tasks=np.r_[tasks,plan[:,2]];agents=np.r_[agents,plan[:,3]]
        optional=np.r_[optional,np.zeros(len(plan),bool)]
        need(np.all((finish>dispatch)&(finish<=1800)&(finish%5==0)&(dispatch>=5)),'Exact legal finish,not arbitrary window')
        task_kind=np.where(tasks<128,0,np.where(tasks<144,1,np.where(tasks<160,2,3)))
        agent_kind=np.where(agents<12,0,np.where(agents<14,1,np.where(agents<16,2,3)))
        need(np.array_equal(task_kind,agent_kind),'Original profession qualification')
        ticks=finish//5-1;positive=b[ticks,tasks];true=a[ticks,tasks];rid=retained[ticks,tasks];eid=raw[ticks,tasks];lead=finish-dispatch
        need(not np.any(positive&~true),'Confirmation impliesphysical anomaly')
        bins=np.digitize(lead,[30,60,120],right=True)
        for grid,rootmask in (('all_5s_roots',np.ones(len(tasks),bool)),('same_30s_roots',dispatch%30==0)):
            for k,kind in enumerate(KINDS):
                for bin_id,name in ((-1,'all'),(0,'0-30'),(1,'30-60'),(2,'60-120'),(3,'120-1800')):
                    mask=rootmask&(task_kind==k)
                    if bin_id>=0:mask&=bins==bin_id
                    key=grid+'/'+kind+'/'+name;c=statistics[key]
                    c.update(legal_candidates=int(mask.sum()),optional_candidates=int((mask&optional).sum()),
                        protected_Tour_candidates=int((mask&~optional).sum()),B_reference_positive=int((mask&positive).sum()),
                        A_physical_positive=int((mask&true).sum()))
                    good=mask&positive&(rid>0)
                    events[key].update((fp,int(v)) for v in np.unique(rid[good]))
                    raw_events[key].update((fp,int(v)) for v in np.unique(eid[mask&positive]) if v>0)
                    if good.any():processes[key].add(fp)
                    if mask.any():
                        c['sum_horizon_s']+=int(lead[mask].sum());c['minimum_horizon_s']=min(c.get('minimum_horizon_s',1801),int(lead[mask].min()))
                        c['maximum_horizon_s']=max(c.get('maximum_horizon_s',0),int(lead[mask].max()))
        conds+=1
    result={'stage':'Action-Indexed exploratory feasibility','seed':seed,'split':'development','conditions':conds,
        'unique_numeric_processes':len(fingerprints),'sourcefiles':sourcefiles,'protocol_SHA256':sha(args.protocol),
        'statistics':{key:dict(c,distinct_retained_positive_events=len(events[key]),
            distinct_raw_positive_events=len(raw_events[key]),positive_numeric_processes=len(processes[key]),
            positive_seed_groups=int(bool(events[key])),event_keys=[list(v) for v in sorted(events[key])]) for key,c in statistics.items()},
        'unique_physical_retained_events_by_profession':{k:len(physical[k]) for k in KINDS},
        'elapsed_s':time.perf_counter()-started,'predictor_fits':0,'new_physics':0,'final_data_access':False,
        'scope':'Eachcurrent candidate evaluated independently using originalhard-legal geometry/quotas/Tour andexactB_reference endpoint. Root-grid andETA effects separated;not allcandidates jointlyexecutable.'}
    write(args.output/'complete.json',result)
    print('ACTION_INDEXED_DEVELOPMENT_DONE',seed,conds,flush=True)


if __name__=='__main__':main()

"""Recompute every published cell from numeric targets, not summary claims."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from patrol_rsi.fresh_opportunity import KIND_IDS, ROOTS, SPLITS, frozen_inputs, need, read, sha, write
from patrol_rsi.opportunity_data import HORIZONS


def main():
    p=argparse.ArgumentParser()
    for n in ('root','protocol','manifest','report','output'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();protocol,manifest=frozen_inputs(args.protocol,args.manifest)
    cells=defaultdict(Counter); sets=defaultdict(lambda:defaultdict(set)); conditions=rows=0
    for item in manifest:
        folder=args.root/item['split']/f"seed-{item['group_seed']}"/item['id']
        with np.load(folder/'targets-evaluator-only.npz') as f:
            a,ref,b=f['A'],f['B_reference'],f['B_action']
        with np.load(folder/'labels-only.npz') as f:
            raw_ids,retained_ids=f['event_id'],f['retained_event_id']
        identity=read(folder/'source-identity-not-features.json');fp=identity['numeric_process_fingerprint']
        events={v['id']:v for v in read(folder/'events-evaluator-only.json')}
        summary=read(folder/'summary.json')
        for ki,kind in enumerate(('thermal','electrical','cooling','control')):
            mask=KIND_IDS==ki
            for hi,horizon in enumerate(HORIZONS):
                av,rv,bv=a[:,hi,mask],ref[:,hi,mask],b[:,hi,mask]
                key=(item['split'],kind,horizon); v=cells[key]
                counts={'total_samples':int(av.size),'A_valid':int((av>=0).sum()),'A_positive':int((av==1).sum()),
                    'A_negative':int((av==0).sum()),'B_reference_valid':int((rv>=0).sum()),'B_reference_positive':int((rv==1).sum()),
                    'B_action_valid':int((bv>=0).sum()),'B_action_positive':int((bv==1).sum()),'B_action_negative':int((bv==0).sum()),
                    'B_action_invalid':int((bv<0).sum()),'A_positive_without_legal_B':int(((av==1)&(bv<0)).sum()),
                    'A_positive_B_reference_negative':int(((av==1)&(rv==0)).sum()),'B_positive_A_negative':int(((bv==1)&(av==0)).sum())}
                row=next(v for v in summary['strata'] if (v['kind'],v['horizon_s'])==(kind,horizon))
                need(all(row[k]==value for k,value in counts.items()),'Every per-condition count')
                v.update(counts)
                sets[key]['numeric'].add(fp)
                if (bv==1).any():sets[key]['positive_numeric'].add(fp);sets[key]['positive_seeds'].add(item['group_seed'])
                if (bv==0).any():sets[key]['negative_seeds'].add(item['group_seed'])
                ticks=np.minimum((np.asarray(ROOTS)+horizon)//5-1,359)
                for arr,name,field in ((raw_ids,'events','B_positive_event_ids'),(retained_ids,'retained','B_positive_retained_event_ids')):
                    positive_ids=[int(v) for v in np.unique(arr[ticks][:,mask][bv==1]) if v>0]
                    need(positive_ids==row[field],'Per-condition predictable event identity')
                    for i in positive_ids:
                        event=events[i];sets[key][name].add((fp,event['task'],event['first_tick'],event['last_exclusive_tick']))
                need(not np.any((bv>=0)&(av<0)) and not np.any((bv==1)&(av!=1)), 'No legal B outside physical endpoints; B positive implies A')
        conditions+=1;rows+=a.size
    report=read(args.report)
    for row in report['coverage_cells']:
        key=(row['split'],row['kind'],row['horizon_s']);counts=cells[key];g=sets[key]
        need(all(row[k]==v for k,v in counts.items()),'Every48-cell pooled denominator/count')
        need(row['B_positive_seed_groups']==sorted(g['positive_seeds']) and row['B_negative_seed_groups']==sorted(g['negative_seeds']),'Independent positive/negative seed coverage')
        need(row['B_positive_numeric_processes']==len(g['positive_numeric']) and row['distinct_B_positive_events']==len(g['events']) and
             row['distinct_B_positive_retained_events']==len(g['retained']),'Independent fingerprint/event de-duplication')
        expected={'positive_seed_groups':len(g['positive_seeds']),'positive_numeric_processes':len(g['positive_numeric']),
            'distinct_retained_positive_events':len(g['retained']),'valid_negative_seed_groups':len(g['negative_seeds'])}
        minimum=protocol['data_gate']['per_kind_per_horizon_per_split_minimum'][row['split']]
        deficits={k:max(0,v-expected[k]) for k,v in minimum.items()}
        need(row['gate']=={'actual':expected,'minimum':minimum,'deficits':deficits,'passed':not any(deficits.values())},'Independent frozen gate each cell')
    need(conditions==1280 and len(cells)==48 and rows==1280*59*3*176,'Every source condition and48 cells')
    write(args.output,{'status':'passed','conditions':conditions,'sample_rows':int(rows),'cells':48,
        'scope':'All A/B target counts, invalid masks, independent positive/negative seed/process/event sets and frozen deficits recomputed from NPZ; no producer summary arithmetic trusted.',
        'coverage_SHA256':sha(args.report),'no_predictor_fitting':True})
    print('FRESH_COUNTS_INDEPENDENT_DONE',conditions,rows,flush=True)


if __name__=='__main__':main()

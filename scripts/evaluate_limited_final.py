"""One final score pass of previously frozen candidates;no fitting allowed."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from patrol_rsi.limited_prediction import CELLS,HeadData,load_reviewed,need,read,sha,write
from patrol_rsi.limited_metrics import evaluate
from select_limited_predictors import predict


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','cache','selection','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cell',type=int,required=True);args=p.parse_args();os.umask(0o077)
    protocol,review=load_reviewed(args.protocol,args.review);selection=read(args.selection)
    need(selection['status']=='frozen_before_final_open' and selection['protocol_SHA256']==sha(args.protocol),'Frozen selected modelreceipt')
    opened=read(args.cache/'final-open.json');need(opened['selection_SHA256']==sha(args.selection) and opened['opened_unix']>=selection['frozen_unix'],'No final sourceopen before artifactfreeze')
    need(args.cell in range(7),'One ofseven finalhead cells')
    kind,horizon=CELLS[args.cell];data=HeadData(args.cache/'final_test'/f'cell-{args.cell}',expected_phase='final_test',expected_cell=args.cell,expected_protocol=sha(args.protocol))
    need(data.info['phase']=='final_test','Finalevaluation cohortonly')
    dest=args.output/f'cell-{args.cell}';dest.mkdir(exist_ok=False)
    write(dest/'started.json',{'selection_SHA256':sha(args.selection),'protocol_SHA256':sha(args.protocol),
        'job_id':os.environ.get('SLURM_JOB_ID'),'start_unix':time.time(),'fitting_or_tuning_allowed':False})
    out=[];start=time.perf_counter()
    for group in (v for v in selection['selected_heads'] if v['cell']==args.cell):
        li=0 if group['label']=='A_physical' else 1
        for chosen in group['views']:
            path=Path(chosen['model_path']);need(sha(path)==chosen['model_sha256'],'No changed modelafter testopen')
            candidate=read(path);prob=predict(data,candidate,kind)
            # Original per-row predictions remain in private output for export
            # correctness repairs. Never exported or re-used for selection.
            predictions=dest/(chosen['id']+'--private-probabilities.npy');np.save(predictions,prob)
            result=evaluate(data,li,prob,chosen['threshold'])
            private=dest/(chosen['id']+'--private-metrics.json');write(private,result)
            released={'cell':args.cell,'kind':kind,'horizon_s':horizon,'label':group['label'],'id':chosen['id'],
                'family':chosen['family'],'L2':chosen['L2'],'threshold':chosen['threshold'],'summary':result['summary'],
                'seeds':result['seeds'],'selection_roles':{key:group[key] for key in ('primary_full','time_comparator','fixed_comparator')},
                'raw_predictions_SHA256':sha(predictions),'raw_metrics_SHA256':sha(private),'raw_rows_exported':False,
                'freeze_receipt_SHA256':sha(args.selection)}
            out.append(released)
    write(dest/'released-metrics.json',{'cell':args.cell,'kind':kind,'horizon_s':horizon,'models':out,'wall_s':time.perf_counter()-start,
        'selection_SHA256':sha(args.selection),'model_refits':0,'calibration_refits':0,'threshold_changes':0})
    print('LIMITED_FINAL_CELL_SCORED',args.cell,'frozen_models',len(out),flush=True)


if __name__=='__main__':main()

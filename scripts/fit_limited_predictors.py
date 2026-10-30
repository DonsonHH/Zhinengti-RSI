"""One fixed cell: stream development LR,calibrate only on calibration split."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from patrol_rsi.limited_prediction import (CELLS,HEADS,SCALARS,HeadData,load_reviewed,need,read,sha,write,scalar_score,weighted_scaler)
from patrol_rsi.limited_model import fit_lr,lr_scores,fit_calibrator


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','gate','cache','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cell',type=int,required=True);args=p.parse_args();os.umask(0o077);torch.set_num_threads(1)
    protocol,review=load_reviewed(args.protocol,args.review);gate=read(args.gate)
    need(gate['success'] and gate['tests']>=262 and not(gate['skipped'] or gate['failures'] or gate['errors']),'Actual fullTorch>=262 before any predictor fitting')
    need(args.cell in range(7),'One ofseven fixed cells')
    kind,horizon=CELLS[args.cell];dev=HeadData(args.cache/'development'/f'cell-{args.cell}',expected_phase='development',expected_cell=args.cell,expected_protocol=sha(args.protocol))
    cal=HeadData(args.cache/'validation_calibration'/f'cell-{args.cell}',expected_phase='validation_calibration',expected_cell=args.cell,expected_protocol=sha(args.protocol))
    need(dev.info['phase']=='development' and cal.info['phase']=='validation_calibration','Strict fit/calibration splits')
    need(dev.info['protocol_SHA256']==cal.info['protocol_SHA256']==sha(args.protocol),'Reviewed cache provenance')
    output=args.output/f'cell-{args.cell}';output.mkdir(exist_ok=False)
    write(output/'started.json',{'cell':args.cell,'kind':kind,'horizon_s':horizon,'protocol_SHA256':sha(args.protocol),
        'review_SHA256':sha(args.review),'actualTorch_gate_SHA256':sha(args.gate),'job_id':os.environ.get('SLURM_JOB_ID'),
        'development_cache_SHA256':sha(dev.folder/'complete.json'),'calibration_cache_SHA256':sha(cal.folder/'complete.json'),
        'final_split_access':False,'started_unix':time.time()})
    views=['time_lr','public_lr']+(['public_raw_lr','public_assimilated_lr','public_all_lr'] if kind=='thermal' else [])
    index=[];lr_count=cal_count=0;begin=time.perf_counter()
    for view in views:
        scaler=weighted_scaler(dev,view,kind)
        for label_index,label in enumerate(HEADS):
            for l2 in protocol['candidates']['L2']:
                identity=f'{label}--{view}--L2-{l2}'
                model=fit_lr(dev,view,kind,label_index,l2,scaler);lr_count+=1
                calibration=None
                if model['status']=='completed_within_fixed_optimizer_budget':
                    scores=lr_scores(cal,model)
                    calibration=fit_calibrator(scores,cal.y[:cal.n,label_index],cal.w[:cal.n]);cal_count+=1
                path=output/(identity+'.json')
                write(path,{'id':identity,'label':label,'label_index':label_index,'family':view,'L2':l2,'model':model,
                    'calibrator':calibration,'protocol_SHA256':sha(args.protocol),'fits_on':'development','calibrates_on':'validation_calibration'})
                index.append({'id':identity,'label':label,'label_index':label_index,'family':view,'L2':l2,'path':str(path),
                    'sha256':sha(path),'valid':model['status']=='completed_within_fixed_optimizer_budget' and calibration is not None and calibration['status']=='completed_within_fixed_optimizer_budget'})
                print('LIMITED_FIXED_FIT',args.cell,identity,index[-1]['valid'],model['closure_evaluations'],flush=True)
    for label_index,label in enumerate(HEADS):
        y=np.asarray(dev.y[:dev.n,label_index]);w=np.asarray(dev.w[:dev.n]);n=dev.n
        frequency=(float(np.dot(w,y)/w.sum())*n+1)/(n+2)
        for family in SCALARS:
            if kind!='thermal' and family.startswith('frozen_CVAE'):continue
            identity=f'{label}--{family}';calibration=None
            if family!='constant':
                scores=np.empty(cal.n)
                for sl,raw,_,_ in cal.batches():scores[sl]=scalar_score(raw,family,kind)
                calibration=fit_calibrator(scores,cal.y[:cal.n,label_index],cal.w[:cal.n]);cal_count+=1
            path=output/(identity+'.json')
            write(path,{'id':identity,'label':label,'label_index':label_index,'family':family,'L2':None,
                'frequency':frequency if family=='constant' else None,'calibrator':calibration,'kind':kind,
                'protocol_SHA256':sha(args.protocol),'fits_on':'development','calibrates_on':'validation_calibration' if family!='constant' else None})
            index.append({'id':identity,'label':label,'label_index':label_index,'family':family,'L2':None,'path':str(path),
                'sha256':sha(path),'valid':family=='constant' or calibration['status']=='completed_within_fixed_optimizer_budget'})
    write(output/'complete.json',{'cell':args.cell,'kind':kind,'horizon_s':horizon,'candidates':index,
        'LR_slots_consumed':lr_count,'calibration_slots_consumed':cal_count,'wall_s':time.perf_counter()-begin,
        'protocol_SHA256':sha(args.protocol),'final_access':False,'neural_retraining':0,'LLM_calls':0})
    print('LIMITED_CELL_FIT_DONE',args.cell,lr_count,cal_count,flush=True)


if __name__=='__main__':main()

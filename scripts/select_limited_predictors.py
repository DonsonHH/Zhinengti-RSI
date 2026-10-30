"""Score selection only;freeze all chosen models/calibrators/thresholds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from patrol_rsi.limited_prediction import CELLS,HEADS,HeadData,load_reviewed,need,read,sha,write,scalar_score
from patrol_rsi.limited_model import lr_scores,calibrated
from patrol_rsi.limited_metrics import evaluate,fpr_threshold,rank


def predict(data,candidate,kind):
    if candidate['family']=='constant':return np.full(data.n,candidate['frequency'])
    if 'model' in candidate:score=lr_scores(data,candidate['model'])
    else:
        score=np.empty(data.n)
        for sl,raw,_,_ in data.batches():score[sl]=scalar_score(raw,candidate['family'],kind)
    return calibrated(score,candidate['calibrator'])


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','cache','fits','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();os.umask(0o077);protocol,review=load_reviewed(args.protocol,args.review)
    args.output.mkdir(exist_ok=False);chosen=[];artifacts=[];total_fit=total_cal=0;started=time.perf_counter()
    for ci,(kind,horizon) in enumerate(CELLS):
        info=read(args.fits/f'cell-{ci}'/'complete.json')
        need(info['protocol_SHA256']==sha(args.protocol) and not info['final_access'],'Original complete fixed-budget fit phase')
        total_fit+=info['LR_slots_consumed'];total_cal+=info['calibration_slots_consumed']
        data=HeadData(args.cache/'validation_selection'/f'cell-{ci}',expected_phase='validation_selection',expected_cell=ci,expected_protocol=sha(args.protocol))
        results=[]
        for row in info['candidates']:
            if not row['valid']:continue
            path=Path(row['path']);need(sha(path)==row['sha256'],'Frozen trained/cailbrated candidate bytes')
            candidate=read(path);li=row['label_index'];prob=predict(data,candidate,kind)
            threshold=fpr_threshold(data.y[:data.n,li],prob)
            result=evaluate(data,li,prob,threshold)
            identity=f"cell-{ci}--{row['id']}"
            out=args.output/(identity+'.json');write(out,result)
            results.append({'id':row['id'],'family':row['family'],'label':row['label'],'label_index':li,'L2':row['L2'],
                'model_path':str(path),'model_sha256':sha(path),'threshold':threshold,
                'selection_result':str(out),'selection_result_sha256':sha(out),'summary':result['summary'],
                'ranking':list(rank(result,f"{row['L2']}--{row['id']}"))})
        frozen=[]
        for label in HEADS:
            own=[v for v in results if v['label']==label];families=sorted({v['family'] for v in own})
            for family in families:frozen.append(min((v for v in own if v['family']==family),key=lambda v:tuple(v['ranking'])))
            selected={v['family']:v for v in frozen if v['label']==label}
            primary='public_all_lr' if kind=='thermal' else 'public_lr'
            need(primary in selected and 'time_lr' in selected and 'constant' in selected,'Required fixed primary/time/frequency models;no rescue')
            comparator=min((selected[v] for v in ('constant','professional_hold','bounded_trend','nominal_physical') if v in selected),key=lambda v:tuple(v['ranking']))
            chosen.append({'cell':ci,'kind':kind,'horizon_s':horizon,'label':label,
                'primary_full':selected[primary]['id'],'time_comparator':selected['time_lr']['id'],'fixed_comparator':comparator['id'],
                'views':[v for v in frozen if v['label']==label]})
        print('LIMITED_SELECTION_CELL_DONE',ci,flush=True)
    need(total_fit==138 and total_cal<=192,'Fixed global fit/calibration slot accounting')
    for group in chosen:
        for v in group['views']:
            artifacts.extend([{'path':v['model_path'],'sha256':v['model_sha256']},
                {'path':v['selection_result'],'sha256':v['selection_result_sha256']}])
    receipt={'status':'frozen_before_final_open','protocol_SHA256':sha(args.protocol),'review_SHA256':sha(args.review),
        'frozen_unix':time.time(),'job_id':os.environ.get('SLURM_JOB_ID'),'LR_slots_consumed':total_fit,
        'calibration_slots_consumed':total_cal,'final_labels_or_scores_used':False,'selected_heads':chosen,'artifacts':artifacts,
        'wall_s':time.perf_counter()-started,'selection_split':'validation_selection',
        'post_scope_claim':'Exploratory coverage-selected domain;requiresnew scope-blind confirmation before reliabilityclaim.'}
    write(args.output/'selection-receipt.json',receipt)
    print('LIMITED_SELECTION_FROZEN',sha(args.output/'selection-receipt.json'),flush=True)


if __name__=='__main__':main()

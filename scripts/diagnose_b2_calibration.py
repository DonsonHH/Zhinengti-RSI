"""Development/calibration only; never opens selection or final caches."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from patrol_rsi.b2_calibration import fit,sigmoid,weighted_auc,distribution,derivative
from patrol_rsi.limited_model import fit_calibrator,lr_scores,calibrated
from patrol_rsi.limited_metrics import probabilities,weighted_ap
from patrol_rsi.limited_prediction import CELLS,HeadData,need,read,sha,write


def main():
    parser=argparse.ArgumentParser()
    for key in ('protocol','review','old-protocol','cache','fits','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);protocol=read(args.protocol);review=read(args.review)
    need(protocol['protocol']=='B4-B2-development-v1' and review['protocol_SHA256']==sha(args.protocol) and review['approved'], 'Reviewed independent developmentdiagnostic')
    args.output.mkdir(exist_ok=False);start=time.perf_counter();all_rows=[];replays=[];models=[];slots=0
    for ci,(kind,horizon) in enumerate(CELLS):
        sources={phase:HeadData(args.cache/phase/f'cell-{ci}',expected_phase=phase,expected_cell=ci,expected_protocol=sha(args.old_protocol))
            for phase in ('development','validation_calibration')}
        index=read(args.fits/f'cell-{ci}'/'complete.json')
        for view in ('time_lr','public_all_lr' if kind=='thermal' else 'public_lr'):
            identity=f'B_action--{view}--L2-0.1';path=args.fits/f'cell-{ci}'/(identity+'.json')
            original=next(v for v in index['candidates'] if v['id']==identity)
            need(original['valid'] and sha(path)==original['sha256'],'Unchanged fixed developmentLR candidate')
            candidate=read(path);scores={phase:lr_scores(data,candidate['model']) for phase,data in sources.items()}
            cal=sources['validation_calibration'];sc=scores['validation_calibration'];target=cal.y[:cal.n,1];w=cal.w[:cal.n]
            tag=f'cell-{ci}--{view}'
            write(args.output/(tag+'--replay-start.json'),{'kind':'original_ridge1_replay','candidate':identity,'slot_consumed_on_attempt':True,'source_SHA256':sha(path)})
            replay=fit_calibrator(sc,target,w);old=candidate['calibrator']
            write(args.output/(tag+'--replay-result.json'),{'original':old,'replayed':replay})
            need(replay['status']==old['status'] and np.allclose([replay['a'],replay['b']],[old['a'],old['b']],rtol=1e-9,atol=1e-10),'Originala2/2 calibration exactnumericalreplay')
            replays.append({'cell':ci,'id':identity,'source_SHA256':sha(path),'original':old,'replayed':replay})
            variants=[('frozen_ridge1',old|{'rho':1.,'balanced':False,'prior_correction_logit':0.})]
            for ablation in protocol['probability']['new_ablation_calibrators']:
                write(args.output/(tag+'--'+ablation['id']+'--start.json'),{'kind':'new_calibration_diagnostic','slot':slots+1,'candidate':identity,'ablation':ablation,'slot_consumed_on_attempt':True})
                result=fit(sc,target,w,ablation['rho'],ablation['id']=='balanced_ridge0');slots+=1
                write(args.output/(tag+'--'+ablation['id']+'--result.json'),result)
                variants.append((ablation['id'],result));models.append({'cell':ci,'id':identity,'ablation':ablation['id'],'calibrator':result})
            gradients=[]
            for name,result in variants:
                if result['status']=='completed_within_fixed_optimizer_budget':
                    gradients.append({'ablation':name,**derivative(sc,target,w,result['a'],result['b'],result['rho'],result['balanced'])})
            for phase,data in sources.items():
                s=scores[phase];y=data.y[:data.n,1];weights=data.w[:data.n];rows=[]
                outputs=[('raw_sigmoid',sigmoid(s)),('calibration_weighted_constant',np.full(data.n,float(w@target/w.sum())))]
                for name,result in variants:
                    if result['status']!='completed_within_fixed_optimizer_budget':continue
                    native=sigmoid(result['a']*s+result['b'])
                    if result['balanced']:outputs.append((name+'--native_changed_prior',native))
                    outputs.append((name+('--prior_corrected' if result['balanced'] else ''),sigmoid(result['a']*s+result['b']+result['prior_correction_logit'])))
                for name,p in outputs:
                    rows.append({'probability':name,'metrics':probabilities(y,p,weights),'AUROC':weighted_auc(y,p,weights),
                        'overall':distribution(p,weights),'positive':distribution(p[y==1],weights[y==1]),'negative':distribution(p[y==0],weights[y==0])})
                all_rows.append({'cell':ci,'kind':kind,'horizon_s':horizon,'view':view,'phase':phase,'id':identity,'LR_source_SHA256':sha(path),
                    'raw_score_AUPRC':weighted_ap(y,s,weights),'raw_score_AUROC':weighted_auc(y,s,weights),
                    'raw_scores':{'overall':distribution(s,weights),'positive':distribution(s[y==1],weights[y==1]),'negative':distribution(s[y==0],weights[y==0])},
                    'probability_outputs':rows,'calibration_objective_derivatives':gradients if phase=='validation_calibration' else None,
                    'interpretation':'Calibration data are in-sample for calibration;development data fitted the originalLR. No independent confirmatory score or selected replacement.'})
            print('B2_CALIBRATION_DIAGNOSED',ci,view,flush=True)
    need(slots==42 and len(replays)==14,'Exactly42newplus14replay slots;no newLRfit')
    write(args.output/'calibration-diagnosis.json',{'protocol_SHA256':sha(args.protocol),'rows':all_rows,'new_calibrators':models,'original_replays':replays,
        'new_calibration_fits':slots,'replay_fits':len(replays),'new_LR_fits':0,'phases_opened':['development','validation_calibration'],
        'selection_or_final_access':False,'old_models_changed':False,'wall_s':time.perf_counter()-start,'job_id':os.environ.get('SLURM_JOB_ID')})


if __name__=='__main__':main()

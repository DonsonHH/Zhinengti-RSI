"""Frozen C1-B1 timing controls; old models, no new neural/LLM training."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .m12 import ROOT, actor_scorer, load_actors, sha, write
from .m11 import frozen_predictor, seed_digest
from .m12b import specification, DEVELOPMENT, TEST
from .m12c1b import aggregate, rank
from .value_learning import scorer, identity
from .value_dataset import labels
from .timing_run import run
from .timing_atlas import Atlas

GATES=(0,60,180,300,'off')
B0=ROOT/'research/2026-10-28-m12c1b-value/evidence'


def contract():
    return {'stage':'C1-B1 one-global-check timing and signal diagnosis','development_seeds':[8100,8101],
        'validation_seeds':[8150,8151],'test_seeds':[8200,8201],'smoke_seed':8199,
        'test_physical_processes':16,'test_observation_conditions':32,'frozen_gates':list(GATES),
        'public_lookahead_s':120,'future_schedule_grid_s':5,'one_global_optional_action':True,
        'main_test_groups':8,'nominal_substitution_groups':1,'main_test_rollouts':288,'development_tours':8,
        'branch_sampling':'first, last and minimum-completion-cost valid legal optional witness per event; merge identical actions',
        'branch_scope':'Original tour states only, separate counterfactuals, not a global upper bound or unavoidable-miss proof',
        'surrogate':'weight*professional_excess*completion_lead/300 - .05*extra_duration/300 - .05*lost_slack/300',
        'wait_discount':'exp(-wait_s/300)','new_neural_training':False,'LLM_calls':0,'improver_updated':False}


def specs(phase):
    if phase=='smoke':
        return [specification('smoke',a,b,8199) for a,b in DEVELOPMENT[:2]]
    if phase in ('development','validation'):
        seeds=(8100,8101) if phase=='development' else (8150,8151)
        return [specification(phase,a,b,seed) for a,b in DEVELOPMENT for seed in seeds]
    if phase=='test':
        return [specification(phase,a,b,seed,noise) for a,b in TEST for seed in (8200,8201) for noise in (.2,1.)]
    raise ValueError('Unknown split')


def frozen_models():
    result={}
    for seed in range(3):
        folder=B0/f'fit-{seed}';pack=json.loads((folder/'models/net.json').read_bytes())
        selection=json.loads((folder/'selections-frozen.json').read_bytes())['models']['net']
        if identity(pack)!=selection['model_id']:
            raise RuntimeError('Old net scorer/selection mismatch')
        result[seed]={'pack':pack,'model_id':identity(pack),'threshold':selection['threshold']}
    return result


def configs(selected_gate,models):
    return {'tour':{'method':'tour'},'cue_immediate':{'method':'cue'},
        **{f'old_net_{i}':{'method':'net','threshold':models[i]['threshold'],'seed':i} for i in range(3)},
        'cue_gate':{'method':'cue','not_before_s':selected_gate},
        'evidence_now':{'method':'evidence'},'lookahead':{'method':'lookahead'}}


class Store:
    def __init__(self,output,predictor,actors,fingerprints,estimator,models):
        self.output,self.predictor,self.actors,self.models=output,predictor,actors,models
        self.actor_score=actor_scorer(actors[0]);self.rows=[];self.processes={}
        output.mkdir(parents=True,exist_ok=False)
        (output/'cases').mkdir();(output/'process-traces').mkdir()
        write(output/'contract.json',contract())
        write(output/'immutable-inputs.json',{'actors':fingerprints,'estimator_sha256':sha(estimator),
            'old_net_models':{str(k):{'model_id':v['model_id'],'threshold':v['threshold']} for k,v in models.items()},
            'job_id':os.environ.get('SLURM_JOB_ID'),'torch':torch.__version__,'numpy':np.__version__})

    def execute(self,spec,tag,config,**kwargs):
        seed=config.get('seed');model=self.models[seed] if seed is not None else None
        if model:
            kwargs.update(score=scorer(model['pack']),model_id=model['model_id'])
        record,env=run(spec,self.predictor,self.actor_score,config['method'],config=config,**kwargs)
        key='--'.join(str(spec[k]) for k in ('phase','scenario','duration_s','seed','noise_c','estimator'))+'--'+tag
        path=self.output/'cases'/(key+'.json.gz')
        if path.exists():
            raise RuntimeError('No silent case overwrite')
        path.write_bytes(gzip.compress((json.dumps(record)+'\n').encode(),mtime=0))
        process='--'.join(str(spec[k]) for k in ('scenario','duration_s','seed','fault_start_s'))
        if process not in self.processes:
            dest=self.output/'process-traces'/(process+'.npz');np.savez_compressed(dest,**env.trace.arrays)
            self.processes[process]={'path':str(dest.relative_to(self.output)),'sha256':sha(dest),'bytes':dest.stat().st_size,
                'numeric_digest':env.trace.fingerprint}
        elif self.processes[process]['numeric_digest']!=env.trace.fingerprint:
            raise RuntimeError('Check/wait changed physical process')
        row={'case':key,'tag':tag,'specification':spec,'deployment':record['deployment'],
            'path':str(path.relative_to(self.output)),'sha256':sha(path),'bytes':path.stat().st_size,'process_key':process,
            'metrics':record['metrics'],'anomaly_intervals':record['anomaly_intervals'],
            'event_weight_sum':sum(r['weight'] for r in record['anomaly_intervals']),
            'weighted_delay_sum':sum(r['weight']*r['delay_or_censor_s'] for r in record['anomaly_intervals'])}
        self.rows.append(row)
        if len(self.rows)%16==0:
            print('B1_CASES',len(self.rows),tag,flush=True)
        return row,record

    def finish(self,**extra):
        write(self.output/'case-index.json',self.rows);write(self.output/'process-index.json',self.processes)
        write(self.output/'summary.json',{'actual_cases':len(self.rows),'physical_processes':len(self.processes),
            'new_neural_training':False,'LLM_calls':0,'job_id':os.environ.get('SLURM_JOB_ID'),**extra})


def validate(store):
    options=[]
    for gate in GATES:
        group=[]
        for spec in specs('validation'):
            row,_=store.execute(spec,'gate-'+str(gate),{'method':'cue','not_before_s':gate});group.append(row)
        options.append({'gate':gate,'case_ids':[r['case'] for r in group],'validation':aggregate(group)})
    selected=min(options,key=lambda r:rank(r['validation']))
    write(store.output/'selection-frozen.json',{'selected_gate':selected['gate'],'all_options':options,
        'all_configs':configs(selected['gate'],store.models),'frozen_at_unix':time.time(),'test_not_started':True})
    store.finish(stage='validation')


def formal(store,validation,smoke=False):
    selection=json.loads((validation/'selection-frozen.json').read_bytes()) if not smoke else None
    frozen=configs(180,store.models) if smoke else selection['all_configs']
    write(store.output/'test-start-gate.json',{'selection_sha256':sha(validation/'selection-frozen.json') if not smoke else None,
        'selection_frozen_at_unix':selection['frozen_at_unix'] if selection else None,'started_at_unix':time.time(),
        'frozen_configs':frozen,'all_model_ids':{str(k):v['model_id'] for k,v in store.models.items()}})
    atlas,baselines,groups={}, {}, {}
    if not smoke:
        for spec in specs('development'):
            collector=Atlas();row,record=store.execute(spec,'development-tour',{'method':'tour'},observer=collector)
            atlas[row['case']]=collector.finish(record);baselines[row['case']]=record
    for tag,config in frozen.items():
        group=[]
        for spec in specs('smoke' if smoke else 'test'):
            collector=Atlas() if tag=='tour' else None
            row,record=store.execute(spec,tag,config,**({'observer':collector} if collector is not None else {}))
            group.append(row)
            if collector is not None:
                atlas[row['case']]=collector.finish(record);baselines[row['case']]=record
        groups[tag]={'case_ids':[r['case'] for r in group],'aggregate':aggregate(group)}
    if not smoke:
        group=[]
        for spec in specs('test'):
            row,_=store.execute(dict(spec,estimator='nominal'),'lookahead-nominal',{'method':'lookahead'});group.append(row)
        groups['lookahead-nominal']={'case_ids':[r['case'] for r in group],'aggregate':aggregate(group)}
    write(store.output/'groups.json',groups)
    write(store.output/'atlas-before-branches.json',atlas)
    branches=[]
    for base_key,description in atlas.items():
        for i,sample in enumerate(description['samples']):
            base=baselines[base_key]
            row,branch=store.execute(base['specification'],'branch-'+str(i),{'method':'tour'},intervention=sample['intervention'])
            branches.append({'base_case':base_key,'branch_case':row['case'],**sample,
                'labels':labels(base,branch,sample['intervention']['task'])})
    write(store.output/'branch-index.json',branches)
    store.finish(stage='smoke' if smoke else 'formal',main_cases=sum(len(g['case_ids']) for g in groups.values()),
        development_tours=0 if smoke else 8,independent_branches=len(branches),atlas_baselines=len(atlas))


def main():
    p=argparse.ArgumentParser()
    for key in ('output','actors','estimator-weights','estimator-manifest'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--phase',choices=('smoke','validation','formal'),required=True);p.add_argument('--validation',type=Path)
    args=p.parse_args();torch.set_num_threads(1);started=time.perf_counter()
    actors,fingerprints=load_actors(args.actors);predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    store=Store(args.output,predictor,actors,fingerprints,args.estimator_weights,frozen_models())
    if args.phase=='validation':
        validate(store)
    else:
        formal(store,args.validation,smoke=args.phase=='smoke')
    for seed,model in actors.items():
        if seed_digest(model)!=fingerprints[str(seed)]['parameters_sha256']:
            raise RuntimeError('Frozen old actor changed')
    write(args.output/'completed.json',{'phase':args.phase,'seconds':time.perf_counter()-started,
        'old_actors_unchanged':True,'estimator_sha256':sha(args.estimator_weights)})
    print('B1_COMPLETE',args.phase,len(store.rows),flush=True)


if __name__=='__main__':
    main()

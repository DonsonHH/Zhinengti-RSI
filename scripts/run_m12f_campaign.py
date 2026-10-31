"""Bounded real timing-skill search, independent confirmation and meta transfer."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import torch

from patrol_rsi.f_identity import verify_identity
from patrol_rsi.limited_prediction import need,read,write,sha
from patrol_rsi.rsi_controlled import ControlledImprover
from patrol_rsi.timing_skill import TimingSkill,information_baseline
from patrol_rsi.timing_learning import (parse_skill_reply,parse_meta_reply,skill_context,skill_prompt,meta_prompt,
    sample_skill,strict_gain,nonregression,strictly_better_key,per_seed_no_more_misses,aggregate,exact_policy_result)
from timing_skill_execution import skill_policy,continue_skill
from timing_skill_runtime import (EvaluationStore,ActualCaller,source_cases,init_worker,fingerprint,check_runtime,fail_stop,MODEL,ACTORS,CVAE)


def skill_key(skill,mode):
    value=skill.to_dict()
    if mode=='canonical_rule_order':
        rules=[]
        for row in value['rules']:
            atoms=[]
            for atom in row['all']:
                if atom not in atoms:atoms.append(atom)
            if {'all':atoms} not in rules:rules.append({'all':atoms})
        value['rules']=rules
    return fingerprint(value)


def references():
    return {'Tour':skill_policy(continue_skill()),'human_professional':{'executor':'original-human-v1','ranker':'rules','skill':None},
        'initial_E1_skill':skill_policy(TimingSkill.initial_e1()),'manual_information':skill_policy(information_baseline())}


class Search:
    """Every slot describes the same initial parent; only own feedback can transfer."""
    def __init__(self,store,caller,initial,protocol,area,*,prior=False):
        self.store,self.caller,self.initial,self.protocol=store,caller,initial,protocol
        self.area=Path(area);self.area.mkdir(parents=True,exist_ok=False);self.prior=prior
        self.history=[];self.entries=[];self.best=initial
        self.seen={skill_key(TimingSkill.initial_e1(),protocol.dedup)}

    def context(self):
        histories=[]
        for row in self.history:
            value=dict(row)
            if not self.prior and row.get('evaluation_key') is not None:
                value['cards']=self.store.cards(self.store.results[row['evaluation_key']],self.protocol.protocol.diagnostics)
            histories.append(value)
        baseline=[] if self.prior else self.store.cards(self.initial,self.protocol.protocol.diagnostics)
        return skill_context(TimingSkill.initial_e1(),histories,baseline,protocol=self.protocol,prior_only=self.prior)

    def slot(self,seed,*,simple=False):
        number=len(self.entries);path=self.area/f'slot-{number:02d}'
        write(path.with_suffix('.attempt.json'),{'slot':number,'generation_seed':seed,'initial_parent':TimingSkill.initial_e1().to_dict(),
            'generator':'random_program' if simple else self.protocol.generator,'prior_only':self.prior,'one_opportunity_consumed':True})
        entry={'slot':number,'generation_seed':seed,'valid':False,'accepted':False,'reason':'invalid_schema'}
        try:
            if simple or self.protocol.generator=='random_program':
                skill=sample_skill(seed);entry['source']='simple_same_grammar_sampler'
            else:
                generation=self.caller.call(skill_prompt(self.context()),seed)
                entry['source']='fixed_Qwen3-8B';entry['generation']=generation
                skill=parse_skill_reply(generation['raw_response'])
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            # Infrastructure/source failures never enter this parse-only path.
            if 'generation' not in entry and not(simple or self.protocol.generator=='random_program'):raise
            entry['schema_error']=type(error).__name__+': '+str(error)
            self.history.append({'skill':None,'accepted':False,'schema_error':True})
            self.entries.append(entry);write(path.with_suffix('.complete.json'),entry);return entry
        entry.update(valid=True,skill=skill.to_dict(),candidate_id=skill.version_id)
        key=skill_key(skill,self.protocol.dedup)
        if key in self.seen:
            entry['reason']='duplicate_consumes_slot';entry['behavior']={'behavior_changed':False}
            self.history.append({'skill':skill.to_dict(),'accepted':False,'evaluation_key':None})
        else:
            self.seen.add(key);result=self.store.evaluate(skill_policy(skill))
            allowed=per_seed_no_more_misses(result['statistics'],self.initial['statistics'])
            accepted=allowed and strictly_better_key(result['statistics'],self.best['statistics'])
            entry.update(accepted=accepted,reason='strict_development_improvement' if accepted else 'no_strict_development_improvement',
                evaluation_key=result['key'],statistics=result['statistics'],behavior=self.store.behavior(result,self.initial))
            self.history.append({'skill':skill.to_dict(),'accepted':accepted,'evaluation_key':result['key']})
            if accepted:self.best=result
        self.entries.append(entry);write(path.with_suffix('.complete.json'),entry)
        return entry

    def summary(self):
        return copy.deepcopy({'initial_policy':self.initial['policy'],'selected_policy':self.best['policy'],'selected_evaluation_key':self.best['key'],
            'selected_statistics':self.best['statistics'],'protocol':self.protocol.to_dict(),'prior_only':self.prior,
            'valid_proposals':sum(v['valid'] for v in self.entries),'slots':len(self.entries),'entries':self.entries,'history':self.history})


def frozen_inputs(paths):return [{'path':str(path),'sha256':sha(path)} for path in paths]


def checked_complete(path,identity_sha):
    value=read(path);need(value['complete'] and value['identity_SHA256']==identity_sha,'Same-source completed previous substage')
    for item in value.get('sealed_artifacts',[]):need(sha(Path(item['path']))==item['sha256'],'Frozen selection mapping unchanged')
    return value


def recheck_seal(artifacts):
    for item in artifacts:need(sha(Path(item['path']))==item['sha256'],'Selection seal unchanged throughout confirmation')


def store(root,source,pool,area,cases,cap,context,deadline):
    return EvaluationStore(root,source,pool,area,cases,cap,context,deadline=deadline)


def methods_f1(mapping):
    result=references()
    result.update({name:mapping[search]['selected_policy'] for name,search in (('simple3','simple'),('prior3','prior'),('experienced3','experienced'))})
    result['experienced3_PPOtool']=dict(result['experienced3'],ranker='ppo')
    return result


def f1_search(root,source,pool,cases,study,context,deadline,identity_sha):
    area=root/'F1/search';area.mkdir(parents=True,exist_ok=False)
    cohort=source_cases(root,cases['known_development'],known=True)
    evaluator=store(root,source,pool,area/'evaluations',cohort,680,context,deadline)
    initial=evaluator.evaluate(references()['initial_E1_skill'],capture=True)
    for name,policy in references().items():evaluator.evaluate(policy)
    caller=ActualCaller(root,area/'calls',18,deadline=deadline);mappings=[];sealed=[];mechanisms=[]
    for generation in study['F1']['generation_repeats']:
        searches={name:Search(evaluator,caller,initial,ControlledImprover(),area/f'generation-{generation}/{name}',prior=name=='prior') for name in ('simple','prior','experienced')}
        for slot in range(3):
            searches['simple'].slot(generation*10000+200+slot,simple=True)
            searches['prior'].slot(generation*10000+100+slot)
            searches['experienced'].slot(generation*10000+100+slot)
        mapping={name:search.summary() for name,search in searches.items()}
        mapping['generation_seed']=generation;mapping['methods']=methods_f1(mapping)
        evaluator.evaluate(mapping['methods']['experienced3_PPOtool'])
        path=area/f'mapping-{generation}.json';write(path,mapping);mappings.append(mapping);sealed.append(path)
        mechanisms.append({'generation_seed':generation,'prior_valid':mapping['prior']['valid_proposals'],
            'experienced_valid':mapping['experienced']['valid_proposals'],
            'experienced_behavior_changed':sum(v.get('behavior',{}).get('behavior_changed',False) for v in mapping['experienced']['entries'])})
    cards=evaluator.cards(initial,ControlledImprover().protocol.diagnostics)
    available_roles=sorted({v['measurement']['kind'] for v in cards if v['elapsed_to_actual_measurement_s']>0})
    exposed=[card for receipt in caller.receipts for card in receipt['prompt']['data']['experience_cards']]
    roles=sorted({v['measurement']['kind'] for v in exposed if v['elapsed_to_actual_measurement_s']>0})
    gate=all(v['prior_valid']>=2 and v['experienced_valid']>=2 and v['experienced_behavior_changed']>=1 for v in mechanisms) and roles==['control','cooling','electrical','thermal']
    ledger=evaluator.finish()
    value={'complete':True,'identity_SHA256':identity_sha,'mappings':mappings,'sealed_artifacts':frozen_inputs(sealed),
        'mechanism_gate':{'passed':gate,'per_repeat':mechanisms,'actual_qualified_feedback_professions':roles,'available_feedback_professions':available_roles,
            'not_a_positive_task_or_PPO_requirement':True},'model_calls':len(caller.receipts),'evaluation_ledger':ledger}
    write(area/'complete.json',value);print('M12F_F1_SEARCH_COMPLETE',gate,len(caller.receipts),flush=True)


def f1_confirmation(root,source,pool,cases,study,context,deadline,identity_sha,phase,history):
    search_receipt=root/'F1/search/complete.json'
    selection_seal=frozen_inputs([search_receipt])
    selected=checked_complete(search_receipt,identity_sha)
    selection_seal+=selected['sealed_artifacts']
    if phase=='F1_final':
        validation_receipt=root/'F1/validation/complete.json'
        selection_seal+=frozen_inputs([validation_receipt])
        previous=checked_complete(validation_receipt,identity_sha)
        need(previous['final_open'],'F1 validation did not permit final opening')
    area=root/'F1'/('validation' if phase=='F1_validation' else 'final');area.mkdir(parents=True,exist_ok=False)
    cohort=source_cases(root,[v for v in cases['future'] if v['partition']==phase],history=history)
    evaluator=store(root,source,pool,area/'evaluations',cohort,320 if phase=='F1_validation' else 640,context,deadline)
    comparisons=[]
    for mapping in selected['mappings']:
        stats={name:evaluator.evaluate(policy)['statistics'] for name,policy in mapping['methods'].items()}
        task=all(strict_gain(stats['experienced3'],stats[name]) for name in ('Tour','human_professional','initial_E1_skill','manual_information','simple3'))
        experience=strict_gain(stats['experienced3'],stats['prior3']) and all(nonregression(stats['experienced3'],stats[name]) for name in references())
        comparisons.append({'generation_seed':mapping['generation_seed'],'statistics':stats,'TaskGO':task,'ExperienceGO':experience})
    task_go=all(v['TaskGO'] for v in comparisons);experience_go=all(v['ExperienceGO'] for v in comparisons)
    recheck_seal(selection_seal)
    confirmed_task=previous['TaskGO'] and task_go if phase=='F1_final' else None
    confirmed_experience=previous['ExperienceGO'] and experience_go if phase=='F1_final' else None
    value={'complete':True,'identity_SHA256':identity_sha,'phase':phase,'comparisons':comparisons,
        'TaskGO':task_go,'ExperienceGO':experience_go,'final_open':True if phase=='F1_final' else task_go or experience_go,
        'final_opened':phase=='F1_final','confirmed_TaskGO':confirmed_task,'confirmed_ExperienceGO':confirmed_experience,
        'registered_success':confirmed_task or confirmed_experience if phase=='F1_final' else task_go or experience_go,
        'sealed_artifacts':selection_seal,'evaluation_ledger':evaluator.finish(),'model_calls':0}
    write(area/'complete.json',value);print('M12F_'+phase+'_COMPLETE',task_go,experience_go,flush=True)


def f2_adapt(root,source,pool,cases,study,context,deadline,identity_sha,trial,history):
    first=checked_complete(root/'F1/search/complete.json',identity_sha)
    checked_complete(root/'F1/validation/complete.json',identity_sha)
    need(first['mechanism_gate']['passed'],'No usable F1 mechanism; F2 remains closed')
    need(trial in (0,1,2),'Three predeclared meta repeats')
    generation=study['F2']['generation_repeats'][trial]
    area=root/f'F2/trial-{trial}/adaptation';area.mkdir(parents=True,exist_ok=False)
    caller=ActualCaller(root,area/'calls',19,deadline=deadline)
    meta_cases=source_cases(root,[v for v in cases['future'] if v['partition']=='F2_meta'],history=history)
    meta_store=store(root,source,pool,area/'meta-evaluations',meta_cases,100,context,deadline)
    initial_meta=meta_store.evaluate(references()['initial_E1_skill'],capture=True)
    incumbent=ControlledImprover();old=Search(meta_store,caller,initial_meta,incumbent,area/'old-inner')
    for slot in range(2):old.slot(generation*10000+1000+slot)
    response=caller.call(meta_prompt(incumbent,old.context()),generation*10000+2000,meta=True)
    try:candidate_protocol=parse_meta_reply(response['raw_response']);valid=True
    except (ValueError,TypeError,KeyError,OverflowError):candidate_protocol=incumbent;valid=False
    new=Search(meta_store,caller,initial_meta,candidate_protocol,area/'candidate-inner')
    for slot in range(2):new.slot(generation*10000+1000+slot)
    changed=candidate_protocol.version_id!=incumbent.version_id
    retained=valid and changed and strictly_better_key(new.best['statistics'],old.best['statistics'])
    protocol=candidate_protocol if retained else incumbent
    meta={'proposal_valid':valid,'protocol_changed':changed,'retained':retained,'incumbent':incumbent.to_dict(),'candidate':candidate_protocol.to_dict(),
        'transferred_protocol':protocol.to_dict(),'response':response,'old':old.summary(),'new':new.summary(),
        'ledger':meta_store.finish(),'task_reset':{'initial_skill':TimingSkill.initial_e1().to_dict(),'history':[],'candidate_archive':[],'outcomes':[]}}
    write(area/'meta-complete.json',meta)
    adapt_cases=source_cases(root,[v for v in cases['future'] if v['partition']=='F2_adapt'],history=history)
    evaluator=store(root,source,pool,area/'adapt-evaluations',adapt_cases,540,context,deadline)
    initial=evaluator.evaluate(references()['initial_E1_skill'],capture=True)
    for name,policy in references().items():evaluator.evaluate(policy)
    simple=Search(evaluator,caller,initial,incumbent,area/'simple8')
    prior=Search(evaluator,caller,initial,incumbent,area/'prior3',prior=True)
    fixed=Search(evaluator,caller,initial,incumbent,area/'fixed8')
    meta_search=Search(evaluator,caller,initial,protocol,area/'meta3')
    prefix=None
    for slot in range(8):
        simple.slot(generation*10000+5000+slot,simple=True)
        fixed.slot(generation*10000+3000+slot)
        if slot==2:prefix=fixed.summary();write(area/'fixed-prefix3.json',prefix)
    for slot in range(3):prior.slot(generation*10000+4000+slot)
    for slot in range(3):meta_search.slot(generation*10000+3000+slot)
    methods=references()
    methods.update(simple8=simple.best['policy'],prior3=prior.best['policy'],fixed_experience3=prefix['selected_policy'],
        fixed_experience8=fixed.best['policy'],meta3=meta_search.best['policy'],meta3_PPOtool=dict(meta_search.best['policy'],ranker='ppo'))
    evaluator.evaluate(methods['meta3_PPOtool'])
    value={'complete':True,'identity_SHA256':identity_sha,'trial':trial,'generation_seed':generation,'meta':meta,'methods':methods,
        'simple':simple.summary(),'prior':prior.summary(),'fixed8':fixed.summary(),'fixed3':prefix,'meta3':meta_search.summary(),
        'model_calls':len(caller.receipts),'adapt_ledger':evaluator.finish()}
    write(area/'complete.json',value);print('M12F_F2_ADAPT_COMPLETE',trial,valid,retained,len(caller.receipts),flush=True)


def f2_confirmation(root,source,pool,cases,context,deadline,identity_sha,phase,batch,history):
    mapping_paths=[root/f'F2/trial-{trial}/adaptation/complete.json' for trial in range(3)]
    sealed=frozen_inputs(mapping_paths)
    mappings=[checked_complete(path,identity_sha) for path in mapping_paths]
    if phase=='F2_final':
        validation_receipt=root/'F2/validation/complete.json'
        sealed+=frozen_inputs([validation_receipt])
        previous=checked_complete(validation_receipt,identity_sha)
        need(previous['MetaGO'] and batch in (0,1),'F2 final requires sealed validated meta gain and fixed batch')
    area=root/'F2'/('validation' if phase=='F2_validation' else f'final-batch-{batch}');area.mkdir(parents=True,exist_ok=False)
    selected=[v for v in cases['future'] if v['partition']==phase]
    if phase=='F2_final':selected=[v for v in selected if v['specification']['seed'] in ((36140,36141) if batch==0 else (36142,36143))]
    cohort=source_cases(root,selected,history=history)
    comparisons=[];ledgers=[]
    for mapping in mappings:
        evaluator=store(root,source,pool,area/f'trial-{mapping["trial"]}',cohort,200,context,deadline)
        stats={name:evaluator.evaluate(policy)['statistics'] for name,policy in mapping['methods'].items()}
        meta_go=strict_gain(stats['meta3'],stats['fixed_experience3']) and all(nonregression(stats['meta3'],stats[name]) for name in (*references(),'simple8'))
        comparisons.append({'trial':mapping['trial'],'generation_seed':mapping['generation_seed'],'statistics':stats,'MetaGO':meta_go})
        ledgers.append(evaluator.finish())
    recheck_seal(sealed)
    value={'complete':True,'identity_SHA256':identity_sha,'phase':phase,'batch':batch,'comparisons':comparisons,
        'MetaGO':all(v['MetaGO'] for v in comparisons),'sealed_artifacts':sealed,'ledgers':ledgers,'model_calls':0}
    write(area/'complete.json',value);print('M12F_'+phase+'_COMPLETE',batch,value['MetaGO'],flush=True)
    if phase=='F2_final':merge_f2_final(root,cases,identity_sha)


def merge_f2_final(root,cases,identity_sha):
    """Judge the preregistered four-seed final cohort, not each two-seed batch."""
    import fcntl
    with (root/'F2/final-merge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=[root/f'F2/final-batch-{batch}/complete.json' for batch in (0,1)]
        if not all(p.exists() for p in paths):return
        batches=[checked_complete(path,identity_sha) for path in paths]
        mappings=[checked_complete(root/f'F2/trial-{trial}/adaptation/complete.json',identity_sha) for trial in range(3)]
        expected=[v['specification'] for v in cases['future'] if v['partition']=='F2_final'];comparisons=[]
        for trial,mapping in enumerate(mappings):
            stats={}
            for name,policy in mapping['methods'].items():
                rows=[]
                for batch in batches:
                    result=exact_policy_result(batch['ledgers'][trial]['results'],policy)
                    rows.extend(result['rows'])
                stats[name]=aggregate(rows,expected)
            passed=strict_gain(stats['meta3'],stats['fixed_experience3']) and all(nonregression(stats['meta3'],stats[name]) for name in (*references(),'simple8'))
            comparisons.append({'trial':trial,'generation_seed':mapping['generation_seed'],'statistics':stats,'MetaGO':passed})
        value={'complete':True,'identity_SHA256':identity_sha,'comparisons':comparisons,'MetaGO':all(v['MetaGO'] for v in comparisons),
            'final_opened':True,'original_environment_evaluations_added_by_merge':0,'sealed_artifacts':frozen_inputs(paths)}
        path=root/'F2/final-complete.json'
        if path.exists():need(read(path)==value,'Identical one-time final summary if concurrent merge observed')
        else:write(path,value)
        fcntl.flock(lock,fcntl.LOCK_UN)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--mode',choices=('F1_search','F1_validation','F1_final','F2_adapt','F2_validation','F2_final'),required=True)
    p.add_argument('--trial',type=int,default=-1);p.add_argument('--batch',type=int,default=-1);a=p.parse_args()
    try:return run(a)
    except BaseException as error:fail_stop(a.run_root.resolve(),error);raise


def run(a):
    os.umask(0o077);torch.set_num_threads(1);root,source=a.run_root.resolve(),a.source_root.resolve()
    deadline=time.monotonic()+(5280 if a.mode in ('F1_search','F2_adapt') else 3480)
    need(source==Path(__file__).resolve().parents[1],'CLI source equals actual imported source')
    stage=source/'research/2026-10-31-m12f-timing-skill';identity_path=root/'campaign-execution-identity.json'
    required={'study':stage/'M12F-Study-v1.json','cases':stage/'cases-v1.json','prompts':stage/'prompt-templates-v1.json','history':root/'historical-identities.json',
        'CVAE':CVAE,'CVAE_manifest':source/'research/2026-10-28-m111-coverage/frozen_estimator.json',
        'actor_manifest':source/'research/2026-10-27-m11-scheduling/evidence/policy-weight-retrieval.json',
        **{f'actor{i}':ACTORS/f'history-seed-{i}/selected.pt' for i in range(3)},'Qwen_manifest':MODEL/'verified_manifest.json'}
    verify_identity(identity_path,required,a.mode);identity_sha=sha(identity_path)
    study,cases,history=(read(required[k]) for k in ('study','cases','history'))
    need(study['cases_SHA256']==sha(required['cases']),'Actual study/cases binding')
    need(study['prompt_templates_SHA256']==sha(required['prompts']),'Actual preregistered prompt template bytes')
    templates=read(required['prompts']);empty=skill_context(TimingSkill.initial_e1(),[],[],protocol=ControlledImprover())
    need(templates['task_empty_context_template']==skill_prompt(empty) and templates['meta_empty_context_template']==meta_prompt(ControlledImprover(),empty),'Actual prompt-building source equals registered literal/schema templates')
    need(study['skill_space']['initial_skill']==TimingSkill.initial_e1().to_dict() and study['skill_space']['manual_information_skill']==information_baseline().to_dict(),'Frozen initial/reference skill grammar')
    check_runtime(root,deadline)
    claim=root/'launches'/f'{a.mode}--trial{a.trial}--batch{a.batch}.json';claim.parent.mkdir(exist_ok=True)
    write(claim,{'mode':a.mode,'trial':a.trial,'batch':a.batch,'identity_SHA256':identity_sha,'job_id':os.environ['SLURM_JOB_ID'],'unix':time.time(),'single_launch_only':True})
    context={'source_archive_SHA256':read(Path(read(identity_path)['bound_inputs']['source_manifest']))['sha256'],
        'CVAE_SHA256':sha(CVAE),'actor0_SHA256':sha(ACTORS/'history-seed-0/selected.pt'),'study_SHA256':sha(required['study'])}
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(source),)) as pool:
            if a.mode=='F1_search':f1_search(root,source,pool,cases,study,context,deadline,identity_sha)
            elif a.mode.startswith('F1_'):f1_confirmation(root,source,pool,cases,study,context,deadline,identity_sha,a.mode,history)
            elif a.mode=='F2_adapt':f2_adapt(root,source,pool,cases,study,context,deadline,identity_sha,a.trial,history)
            else:f2_confirmation(root,source,pool,cases,context,deadline,identity_sha,a.mode,a.batch,history)
        need(sha(identity_path)==identity_sha,'Immutable campaign identity throughout execution')
        verify_identity(identity_path,required,a.mode)
        from timing_skill_runtime import check_storage
        check_storage(root)
    except BaseException as error:fail_stop(root,error);raise


if __name__=='__main__':main()

"""Freeze the prescribed24 G0 policies and first-goal resource envelope, no evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from patrol_rsi.g_timing_patch import patch_view,compile_patch,digest
from patrol_rsi.timing_skill import TimingSkill,information_baseline

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-11-01-m12g-local'
F=ROOT/'research/2026-10-31-m12f-timing-skill'
BASE='a5ab5880e33e3df3cd9aeffc3a90a63c40273c41'


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def atom(field,op,value):return {'field':field,'op':op,'value':value}


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    STAGE.mkdir(exist_ok=False);(STAGE/'evidence').mkdir();(STAGE/'source').mkdir()
    parent=TimingSkill.initial_e1();view=patch_view(parent);policies=[]
    def candidate(name,category,edits,description):
        patch={'schema_version':'timing-patch-v1','parent_id':view['parent_id'],'edits':edits}
        child=compile_patch(parent,patch)
        policies.append({'slot':len(policies)+1,'id':name,'category':category,'description':description,'patch':patch,
            'policy':{'executor':'timing-skill-v1','ranker':'urgency','skill':child.to_dict()},'eligible_local_mechanism':True})
    def target(op,node,**values):return {'op':op,'target':node,'expected_hash':(view['rules'] if node=='r0' else view['predicates'])[node]['hash'],**values}
    for i,value in enumerate((.01,.1,.25),1):candidate(f'A{i:02d}','positive_evidence',[target('replace_threshold','r0.p4',value=value)],'One margin coefficient change')
    for i,value in enumerate((.1,.25),4):candidate(f'A{i:02d}','positive_evidence',[target('replace_threshold','r0.p1',value=value)],'One professional threshold change')
    candidate('A06','positive_evidence',[target('delete_predicate','r0.p4')],'Remove margin only')
    for name,predicate in [('A07',atom('shift_s','le',30)),('A08',atom('direct_age_s','le',60))]:
        candidate(name,'positive_evidence',[target('add_predicate','r0',predicate=predicate)],'Add one bounded condition')
    cues=('pod_thermal_signal','pod_flow_deficit','pod_service_gap','thermal_disagreement')
    for offset,condition in ((0,atom('direct_age_s','is_missing',None)),(4,atom('direct_age_s','ge',120))):
        for i,cue in enumerate(cues,1):
            rule={'all':[atom('already_alarm','eq',False),condition,atom('inspection_age_ratio','ge',.5),atom('lead_s','ge',60),atom('shift_s','le',30),atom(cue,'ge',.05)]}
            candidate(f'B{offset+i:02d}','independent_information',[{'op':'add_rule','rule':rule}],'Independent missing/stale information branch; no own positive cue/margin requirement')
    for i,age in enumerate((60,120,300),1):
        rule={'all':[atom('already_alarm','eq',True),atom('direct_available','eq',True),atom('direct_reading_normal','eq',False),atom('direct_age_s','ge',age),atom('lead_s','ge',60),atom('shift_s','le',30)]}
        candidate(f'C{i:02d}','professional_semantics',[{'op':'add_rule','rule':rule}],'Persistent known-alarm recheck; no duplicate discovery credit')
    for i,lead in enumerate((30,60,120),4):
        rule={'all':[atom('already_alarm','eq',True),atom('direct_reading_normal','eq',True),atom('direct_is_qualified','eq',False),atom('direct_age_s','le',60),atom('lead_s','ge',lead),atom('shift_s','le',30)]}
        candidate(f'C{i:02d}','professional_semantics',[{'op':'add_rule','rule':rule}],'Qualified recovery check after actual remote normal reading')
    policies.append({'slot':23,'id':'H01','category':'bridge','description':'Original human trigger/rule order/baseline semantics, actual thermal rule action signal->urgency',
        'patch':None,'policy':{'executor':'g-human-urgency-v1','ranker':'urgency','skill':None},'eligible_local_mechanism':False})
    rule={'all':[atom('kind','eq','thermal'),atom('professional_signal','ge',.05),atom('lead_s','gt',0)]}
    candidate('H02','bridge',[target('delete_rule','r0'),{'op':'add_rule','rule':rule}],'Thermal intention in extra-onlyTimingSkill; not claimed equivalent to H01')
    assert len(policies)==24 and len({digest(p['policy']) for p in policies})==24
    cases=read(F/'cases-v1.json')['known_development'];assert len(cases)==20
    write(STAGE/'G0-candidates.json',{'baseline':BASE,'initial_skill':parent.to_dict(),'parent_view':view,'policies':policies,'candidate_order_fixed_before_scores':True})
    write(STAGE/'known-cases.json',{'baseline':BASE,'cases':cases,'scope':'Only already-used F development; no F validation/final or new physics'})
    resource={'CPU_partition':'cu','CPU_cores':2,'CPU_RAM_GB':8,'G0_jobs':2,'G0_minutes_each':45,'gate_jobs':1,'gate_minutes':15,
        'audit_jobs':1,'audit_minutes':20,'conditional_GPU_jobs':1,'GPU_partition':'4gpu','GPU_GRES':'gpu:V100S:1','GPU_CPU_cores':4,
        'GPU_RAM_GB':48,'GPU_minutes':15,'total_allocated_CPU_core_minutes_cap':360,'total_GPU_seconds_cap':900,
        'parallel_allocations_cap':2,'new_storage_bytes_cap':2000000000,'environment_attempts_cap':480,'model_calls_cap':6,
        'max_reserved_CPU_minutes_initial_schedule':30+180+40+60,'freeze_basis':'Live2026-11-01snapshot; Fmaxepisode9.472s;240cases/twoCPUworkers fits45min. Allsetup/failure/audit costs counted.'}
    write(STAGE/'G0-G1a-Study-v1.json',{'baseline':BASE,'scope':'T0-T5 only; no G1b/G2/newconfirmation','candidates_SHA256':sha(STAGE/'G0-candidates.json'),
        'cases_SHA256':sha(STAGE/'known-cases.json'),'resources':resource,'execution':'All24policies real20-case closedloops, no result-based pruning/replacement; observational screening is diagnostics only.',
        'screening':'From archived initial public probe; report finite-probe scope only. No universal never-trigger claim; no synthetic continuation after divergence.',
        'reference_authority':{'F_search_complete_SHA256':sha(F/'evidence/run/F1/search/complete.json'),'F_manifest_SHA256':sha(F/'evidence/source-manifest-v6.json')},
        'inherit_core':['src/patrol_rsi/timing_skill.py','src/patrol_rsi/timing_skill_features.py','src/patrol_rsi/timing_skill_control.py','scripts/timing_skill_execution.py'],
        'gain_gate':'Original strict_gain against initialE1: eachseed no moremiss; fewermacroMiss OR sameMiss>=5%waitgain OR samemiss/wait>=5%CHECKless andwalknotincrease; abs1e-9. H01alone cannotopen localpatchmodelprecheck.',
        'precheck':'Only if a compilableTimingSkill has G0strictgain and compiler/feedback tests+publicboundary pass. Exactlysix synthetic1-2edit prompts, no realdata/evaluationlabels. >=5/6rawvalidcompiled; no repairs/retries.',
        'report':'All24slots, roles, metrics, shadow/actualtrajectory distinctions, completecosts, G1b executableDRAFT only; finalsets remainunopened.'})
    source_inputs=[]
    for p in [ROOT/'reports/2026-11-01-M12-G-后续研究与Sol执行计划.md',*sorted((ROOT/'research/2026-11-01-m12g-plan/source').glob('*'))]:
        source_inputs.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    tracked=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
    write(STAGE/'evidence/protected-baseline.json',{'baseline':BASE,'files':{name:sha(ROOT/name) for name in filter(None,tracked)},'source_inputs':source_inputs})
    print('G_CANDIDATES_FROZEN',len(policies),len(cases),sha(STAGE/'G0-candidates.json'),sha(STAGE/'G0-G1a-Study-v1.json'),flush=True)


if __name__=='__main__':main()

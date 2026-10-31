"""Validate prepared F manifests against original authorities; never regenerate outcomes.

The initial creator is preserved in code-v1/v2/v3 source archives. The reviewed
Study JSON is the design authority; keeping another embedded study would drift.
"""
import hashlib
import json
from pathlib import Path

from patrol_rsi.timing_skill import TimingSkill,information_baseline,NUMERIC_DOMAINS,BOOL_FIELDS

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-31-m12f-timing-skill'
OLD=ROOT/'research/2026-10-31-m12e-timing-schema'
BASE='39d06ce5edb4dc7f7fff18bedc76909350a946d4'
FIXTURES=(('healthy',900),('coldplate_fouling',180),('dc_contact_hotspot',180),('sensor_bias',900),('aux_feeder_loss+cdu_fouling',180))
SPLITS={'F1_validation':[35020,35021],'F1_final':[35040,35041,35042,35043],
    'F2_meta':[36000,36001],'F2_adapt':[36100,36101],'F2_validation':[36120,36121],'F2_final':[36140,36141,36142,36143]}


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study=read(STAGE/'M12F-Study-v1.json');cases=read(STAGE/'cases-v1.json')
    assert study['baseline']==cases['baseline']==BASE
    assert study['cases_SHA256']==sha(STAGE/'cases-v1.json')
    assert study['skill_space']['initial_skill']==TimingSkill.initial_e1().to_dict()
    assert study['skill_space']['manual_information_skill']==information_baseline().to_dict()
    assert study['skill_space']['numeric_domains']=={k:list(v) for k,v in NUMERIC_DOMAINS.items()}
    assert study['skill_space']['boolean_fields']==sorted(BOOL_FIELDS)
    expected=[]
    for seed in (33000,33001):
        complete=read(OLD/'evidence/run/E1'/f'seed-{seed}/complete.json');seen=set()
        for ordinal,diagnostic in enumerate(complete['diagnostics']):
            spec=diagnostic['specification'];key=spec['scenario'],spec['noise_c']
            if key in seen:continue
            seen.add(key);case=next(v for v in cases['known_development'] if v['case_id']==f'known-{seed}-{ordinal:02d}')
            row=next(v for v in complete['rows'] if v['episode_purpose']=='main' and v['method']=='evidence_gate_urgency' and v['specification']==spec)
            release=read(OLD/'evidence/run/E1'/f'seed-{seed}/public-release'/f'condition-{ordinal:02d}--public-selections-release.json')
            assert case['specification']==spec and case['reference_E1_metrics']==row['metrics']
            assert case['process_fingerprint']==diagnostic['process_fingerprint']
            assert case['reference_E1_action_SHA256']==release['all5_actual_action_log_SHA256']['evidence_gate_urgency']
            expected.append(case['case_id'])
    assert len(expected)==20 and set(expected)=={v['case_id'] for v in cases['known_development']}
    expected=[];pool=[v for v in range(95,900,5) if v not in (150,335,520,705)]
    for phase,seeds in SPLITS.items():
        for seed in seeds:
            for family,duration in FIXTURES:
                number=int(hashlib.sha256(f'M12F-phase-v1|{phase}|{seed}|{family}'.encode()).hexdigest(),16)
                for noise in (.2,1.):
                    spec={'phase':phase,'seed':seed,'scenario':family,'duration_s':duration,'fault_start_s':pool[number%len(pool)],
                        'noise_c':noise,'estimator':'history','control':'feedback','resources':'provisioned','variant':'open'}
                    expected.append({'case_id':f'{phase}-{seed}-{family}-{noise}','partition':phase,'specification':spec,'planned_not_generated':True})
    assert cases['future']==expected and len(expected)==160
    assert study['resources']['all_real_model_calls_cap']==75 and study['resources']['all_environment_attempts_cap']==6000
    assert study['resources']['GPU_total_seconds_cap']==21600
    print(json.dumps({'prepared':True,'known_conditions':20,'future_manifest_conditions':160,
        'future_generation_performed':False,'study_SHA256':sha(STAGE/'M12F-Study-v1.json'),'cases_SHA256':sha(STAGE/'cases-v1.json')}),flush=True)


if __name__=='__main__':main()

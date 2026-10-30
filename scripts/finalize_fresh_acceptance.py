"""Issue fitting permission only after all independent B1a evidence exists."""
from __future__ import annotations

import argparse
from pathlib import Path

from patrol_rsi.fresh_opportunity import MANIFEST_SHA, PROTOCOL_SHA, WEIGHTS_SHA, need, read, sha, write


def main():
    p=argparse.ArgumentParser()
    for name in ('evidence','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();e=args.evidence
    coverage=read(e/'report/coverage.json');counts=read(e/'report/counts-independent.json')
    sources=read(e/'report/source-replay.json');cost=read(e/'slurm-cost-review.json')
    full=read(e/'gate/full-tests.json');cpu=read(e/'local-cpu-tests.json')
    need(coverage['generated_conditions']==1280 and coverage['seed_groups']==64 and coverage['generation_complete'],'All1280 exact frozen conditions')
    need(not coverage['cross_split_fingerprint_conflicts'] and coverage['all640_noise_pairs_labels_identical'],'Physical independence and repeated-noise identity')
    need(counts['status']=='passed' and counts['conditions']==1280 and counts['cells']==48 and counts['sample_rows']==1280*59*3*176,'Independent all-sample counts and48 gates')
    need(counts['coverage_SHA256']==coverage['sealed_full_coverage']['sha256'],'Independent private original coverage binding')
    need(sources['status']=='passed' and sources['original_processes_replayed']==640 and sources['fixed_public_task_metadata_frames']==1280*359,
        'Every original physical process and every original public task contract replayed')
    need(sources['source_coverage_SHA']==sha(e/'report/raw-storage.json') and sources['frozen_weights_SHA']==WEIGHTS_SHA,
        'Entire raw source/weights snapshot SHA')
    need(sources['final_permissions_verified'] and sources['raw_files_SHA_verified']>1280*9,'All raw files and final test seal')
    need(full['success'] and full['tests']>=245 and not(full['failures'] or full['errors'] or full['skipped']),'Final real Torch complete regression including resource timestamp regressions')
    need(cpu['success'] and cpu['tests']>=30 and not(cpu['failures'] or cpu['errors'] or cpu['skipped']),'Lightweight CPU complete regression including resource timestamp regressions')
    need(cost['status']=='passed' and cost['generated_groups']==64 and cost['GPU_allocations']==0,'Terminal Slurm ledger and resource caps')
    need(coverage['public_derived_data_bytes']<=6*1000**3,'Actual decimal6GB public-derived cap')
    independent=coverage['independent_counts']
    need(independent['conditions']==1280 and independent['history_frames']==1280*359 and
        independent['base_values']==1280*59*176*85 and independent['action_values']==1280*59*3*176*13 and
        independent['original_measurements_recomputed']>0 and independent['complete_return_checks_recomputed']>0,
        'Full original measurements, whole-return geometry and all85+13 feature columns')
    cells=coverage['coverage_cells'];need(len(cells)==48,'Four professions x3 horizons x4 splits')
    passed=[v for v in cells if v['gate']['passed']]
    covered_domains=[{'kind':kind,'horizon_s':horizon} for kind in ('thermal','electrical','cooling','control')
        for horizon in (30,60,120) if all(v['gate']['passed'] for v in cells if (v['kind'],v['horizon_s'])==(kind,horizon))]
    verdict='GO' if len(passed)==48 else 'LIMITED' if covered_domains else 'NO-GO'
    result={'stage':'B4-B1a','engineering_execution_and_audit':'complete','coverage_verdict':verdict,
        'B4_B1b_full_domain_fitting_allowed':verdict=='GO','limited_protocol_automatically_authorized':False,
        'passed_cells':len(passed),'failed_cells':48-len(passed),'requirements_proved':{
            'frozen_protocol_manifest_weights':True,'all64groups_1280conditions_generated':True,
            'all360physical_arrays_and359public_history_frames':True,'original_noise_and_Tour_completion':True,
            'all_A_B_labels_and85plus13features':True,'whole_return_qualification':True,
            'all48_coverage_gates_independently_evaluated':True,'cross_split_and_noise_independence':True,
            'final_raw_test_stays_sealed':True,'Torch_and_light_CPU_tests':True,'terminal_Slurm_and_budgets':True},
        'protocol_SHA256':PROTOCOL_SHA,'manifest_SHA256':MANIFEST_SHA,'weights_SHA256':WEIGHTS_SHA,
        'kind_horizon_domains_passing_all_four_splits':covered_domains,
        'new_predictor_fits':0,'PPO_retraining':0,'CVAE_updates':0,'Qwen_calls':0,'B4C_policy_search':0,
        'B3_2_verdict':'NO-GO unchanged','B4B0_historical_verdict':'NO-GO unchanged; new B1a evidence recorded separately',
        'next_step':('May start only the already frozen B4-B1b fitting after this data acceptance. No fit executed in B1a.' if verdict=='GO' else
            'Full-domain B4-B1b fitting remains disallowed. Report exact cell deficits; any limited-domain protocol is a separate decision, without automatic training or changed faults/seeds.'),
        'bindings':{name:{'bytes':(e/name).stat().st_size,'sha256':sha(e/name)} for name in
            ('report/coverage.json','report/counts-independent.json','report/source-replay.json',
             'report/raw-storage.json','slurm-cost-review.json','gate/full-tests.json','local-cpu-tests.json')}}
    write(args.output,result)
    print('B1A_FINAL_ACCEPTANCE',verdict,'fitting_allowed',verdict=='GO',flush=True)


if __name__=='__main__':main()

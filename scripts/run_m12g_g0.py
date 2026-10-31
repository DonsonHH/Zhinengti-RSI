"""Exactly24 preregistered G0 policies on20 known development conditions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import torch

from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.g_timing_patch import digest,compile_patch
from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.timing_learning import aggregate,strict_gain
from run_m12d_paired import StoredTrace,write_gz
import timing_skill_runtime as F
from g_timing_execution import execute_candidate


def stop(root,error):
    try:write(root/'G-STOP.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'job_id':os.environ.get('SLURM_JOB_ID')})
    except FileExistsError:pass


def work(item):
    case,candidate,path,root,parent_row,deadline=item;root=Path(root);path=Path(path)
    try:
        need(not(root/'G-STOP.json').exists(),'No new episode after observed G stop')
        need(time.monotonic()<deadline,'G0 CPU allocation soft runtime cap')
        write(path.with_suffix('.attempt.json'),{'case_id':case['case_id'],'candidate_id':candidate['id'],'charged_actual_episode_attempt':True,'job_id':os.environ['SLURM_JOB_ID']})
        folder=Path(case['trace_folder']);need(sha(folder/'complete.json')==case['source_meta_SHA256'],'Known E1 development source authority')
        trace=StoredTrace(folder);need(trace.fingerprint==case['process_fingerprint'],'Known physical identity')
        record,audit=execute_candidate(case,candidate,trace,F.PREDICTOR,F.SCORER)
        parent_path=Path(parent_row['record_path']);need(sha(parent_path)==parent_row['record_SHA256'],'Historical parent actual trajectory authority')
        parent_record=json.loads(gzip.decompress(parent_path.read_bytes()))
        need(digest(parent_record['specification'])==digest(case['specification']) and parent_record['process_fingerprint']==trace.fingerprint,'Parent and child paired physical/observation condition')
        need(len(parent_record['action_log'])==len(record['action_log'])==359,'Full paired action clocks')
        first=None
        for a,b in zip(parent_record['action_log'],record['action_log']):
            need(a['time_s']==b['time_s'],'Paired action time grid')
            if first is None and digest(a['choices'])!=digest(b['choices']):first={'time_s':a['time_s'],'actual_parent_choices':a['choices'],'actual_child_choices':b['choices'],'scope':'Separate real paired runs; not unobserved shadow outcome'}
        extras=[v for v in record['selected_certificates'] if v['task']!=v['baseline_next']]
        record.update(g_first_actual_trajectory_difference=first,g_first_actual_CHECK=extras[0] if extras else None,
            g_parent_record_path=str(parent_path),g_parent_record_SHA256=parent_row['record_SHA256'])
        write_gz(path,record)
        eligibility=[{k:v[k] for k in ('task','onset_s','end_s','weight','samples')} for v in record['anomaly_intervals']]
        row={'case_id':case['case_id'],'specification':case['specification'],'process_fingerprint':trace.fingerprint,'event_eligibility_SHA256':digest(eligibility),
            'metrics':record['metrics'],'usage':record['usage'],'audit':audit,'record_path':str(path),'record_SHA256':sha(path),
            'action_SHA256':record['action_log_SHA256'],'shadow_counts':record['g_shadow_counts'],'difference_cards':len(record['g_difference_cards']),
            'first_actual_difference':first,'first_CHECK':record['g_first_actual_CHECK']}
        write(path.with_suffix('.complete.json'),row);return row
    except BaseException as error:stop(root,error);raise


def verify(identity_path,source,root,scope='G0'):
    identity=read(identity_path);need(Path(identity['source_root']).resolve()==source.resolve()==Path(__file__).resolve().parents[1],'Actual reviewed G source root')
    need(Path(identity['run_root']).resolve()==root.resolve() and identity_path.resolve()==root.resolve()/'G-execution-identity.json','Frozen actual G run-root identity')
    need(identity['baseline']=='a5ab5880e33e3df3cd9aeffc3a90a63c40273c41','G baseline')
    mandatory={'study','candidates','cases','prompts','source_manifest','source_archive','review','gate','F_manifest','F_search_complete','CVAE','CVAE_manifest','actor_manifest','actor0','actor1','actor2'}
    need(mandatory<=set(identity['bound_inputs']),'Complete source/input/model binding')
    indexed={v['path'] for v in identity['artifacts']}
    need(len(indexed)==len(identity['artifacts']) and set(identity['bound_inputs'].values())<=indexed,'Unique indexed bound inputs')
    for item in identity['artifacts']:need(sha(Path(item['path']))==item['sha256'],'G immutable source/input/model bytes: '+item['path'])
    manifest=read(Path(identity['bound_inputs']['source_manifest']));old=read(Path(identity['bound_inputs']['F_manifest']))
    need(manifest['sha256']==sha(Path(identity['bound_inputs']['source_archive'])),'Actual reviewed G archive bytes')
    study=read(Path(identity['bound_inputs']['study']))
    for name in study['inherit_core']:need(manifest['files'][name]['sha256']==old['files'][name]['sha256'],'Inherited F0-validated kernel unchanged')
    review=read(Path(identity['bound_inputs']['review']));gate=read(Path(identity['bound_inputs']['gate']))
    need(review['approved'] and scope in review['approved_scopes'] and review['source_archive_SHA256']==manifest['sha256'],'Independent G source review authorizes this mode')
    need({'study','candidates','cases','source_manifest','prompts'}<=set(review['reviewed_input_SHA256']),'Review binds complete G declared inputs')
    for key,digest_value in review['reviewed_input_SHA256'].items():need(sha(Path(identity['bound_inputs'][key]))==digest_value,'Reviewed G inputs unchanged')
    need(gate['success'] and gate['tests']>=29 and not(gate['errors'] or gate['failures'] or gate['skipped']) and gate['source_root']==str(source),'Targeted inherited/new seam integration gate')
    tests={'tests/test_'+v+'.py' for v in ('timing_skill','timing_semantics','timing_skill_features','timing_learning','g_timing_patch','g_bridge','g_feedback')}
    need(set(gate['test_source_SHA256'])==tests and gate['runtime_imports_passed'],'Exact new/inherited interfaces exercised')
    for name,value in gate['test_source_SHA256'].items():need(sha(source/name)==value,'Actual targeted test source hash')
    for name,item in manifest['files'].items():need(sha(source/name)==item['sha256'],'Actual G snapshot equals archive manifest')
    actual={p.relative_to(source).as_posix() for folder in ('src','scripts','tests') for p in (source/folder).rglob('*.py')}
    need(actual=={name for name in manifest['files'] if name.endswith('.py') and name.split('/')[0] in ('src','scripts','tests')},'No unreviewed executable source added')
    return identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--batch',type=int,required=True);a=p.parse_args()
    root,source=a.run_root.resolve(),a.source_root.resolve();torch.set_num_threads(1);os.umask(0o077);deadline=time.monotonic()+2640
    try:
        need(a.batch in (0,1),'Two fixed12-policy CPU batches')
        ident=root/'G-execution-identity.json';binding=verify(ident,source,root);identity_sha=sha(ident)
        study=read(Path(binding['bound_inputs']['study']));catalog=read(Path(binding['bound_inputs']['candidates']));cases=read(Path(binding['bound_inputs']['cases']))['cases']
        need(study['candidates_SHA256']==sha(Path(binding['bound_inputs']['candidates'])) and study['cases_SHA256']==sha(Path(binding['bound_inputs']['cases'])),'Frozen G plan/candidate/case joins')
        need(len(catalog['policies'])==24 and len(cases)==20 and all(v['specification']['seed'] in (33000,33001) and v['specification']['phase']=='development' for v in cases),'Only24policies20knownFdevcases')
        old_cases=read(source/'research/2026-10-31-m12f-timing-skill/cases-v1.json')['known_development']
        need(digest(cases)==digest(old_cases) and len({v['case_id'] for v in cases})==20 and [v['slot'] for v in catalog['policies']]==list(range(1,25)),'Exact known development cohort and slot order')
        need(sha(Path(binding['bound_inputs']['F_search_complete']))==study['reference_authority']['F_search_complete_SHA256'] and sha(Path(binding['bound_inputs']['F_manifest']))==study['reference_authority']['F_manifest_SHA256'],'Frozen reference authority joins')
        area=root/f'G0/batch-{a.batch}';area.mkdir(parents=True,exist_ok=False)
        write(area/'launch.json',{'job_id':os.environ['SLURM_JOB_ID'],'identity_SHA256':identity_sha,'batch':a.batch,'episode_cap':240})
        refs=read(Path(binding['bound_inputs']['F_search_complete']))['evaluation_ledger']['results']
        initial_policy={'executor':'timing-skill-v1','ranker':'urgency','skill':TimingSkill.initial_e1().to_dict()}
        initial=next(v for v in refs if digest(v['policy'])==digest(initial_policy))
        need(len(initial['rows'])==20,'Complete historical initial reference')
        initial_by={digest(v['specification']):v for v in initial['rows']}
        probes=[]
        for row in initial['rows']:
            path=Path(row['probe_path']);need(sha(path)==row['probe_SHA256'],'Known parent public probe authority')
            with gzip.open(path,'rt',encoding='utf-8') as f:probes.extend(json.loads(line) for line in f)
        chosen=catalog['policies'][a.batch*12:(a.batch+1)*12];results=[]
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=F.init_worker,initargs=(str(source),)) as pool:
            for candidate in chosen:
                need(not(root/'G-STOP.json').exists(),'No next policy after G stop')
                need(time.monotonic()<deadline,'G0 soft runtime cap before policy')
                need(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<2000000000,'2GB new storage cap')
                parent=TimingSkill.initial_e1()
                if candidate['patch'] is not None:need(digest(compile_patch(parent,candidate['patch']).to_dict())==digest(candidate['policy']['skill']),'Actual frozen patch compilation')
                if candidate['policy']['executor']=='timing-skill-v1':
                    child=TimingSkill.from_dict(candidate['policy']['skill']);admitted=sum(child.action(v)=='CHECK' for v in probes)
                    screen={'rows':len(probes),'CHECK_rows':admitted,'parent_input_differences':sum(child.action(v)!=parent.action(v) for v in probes),
                        'scope':'Finite archived parent inputs only; no universal/trajectory equivalence asserted; all20actualclosedloops stillrun.'}
                else:screen={'scope':'H01 original human rule executor, no TimingSkill shadow claim'}
                dest=area/candidate['id'];dest.mkdir();start=time.perf_counter()
                write(dest/'start.json',{'candidate':candidate,'screening':screen,'scheduled_actual_episodes':20})
                rows=list(pool.map(work,[(case,candidate,str(dest/f'case-{i:02d}.json.gz'),str(root),initial_by[digest(case['specification'])],deadline) for i,case in enumerate(cases)],chunksize=1))
                stats=aggregate(rows,[v['specification'] for v in cases]);gain=strict_gain(stats,initial['statistics'])
                changes=sum(row['action_SHA256']!=initial_by[digest(row['specification'])]['action_SHA256'] for row in rows)
                value={'candidate':candidate,'screening':screen,'statistics':stats,'gain_vs_initial':gain,'eligible_local_mechanism':candidate['eligible_local_mechanism'],
                    'actual_trajectory_differences_vs_initial':changes,'rows':rows,'wall_s':time.perf_counter()-start,'identity_SHA256':identity_sha}
                write(dest/'complete.json',value);results.append(value)
                print('G0_POLICY_COMPLETE',candidate['id'],stats['macro_missed_count'],stats['macro_wait_s'],gain,flush=True)
        need(sha(ident)==identity_sha,'G identity unchanged during batch');verify(ident,source,root)
        write(area/'complete.json',{'complete':True,'batch':a.batch,'identity_SHA256':identity_sha,'results':results,'actual_environment_evaluations':len(results)*20,
            'new_physical_processes':0,'new_model_calls':0})
        print('G0_BATCH_COMPLETE',a.batch,len(results)*20,flush=True)
    except BaseException as error:stop(root,error);raise


if __name__=='__main__':main()

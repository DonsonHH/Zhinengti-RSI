"""Pre-register E0/E1 before new calls or physical processes; read old public evidence."""
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from patrol_rsi.meta_schema import strict_json,full_schema,compact_schema,ENUMS,format_parents,FORMAT_CHALLENGES,prompt
from patrol_rsi.rsi_controlled import ControlledImprover,INSTRUCTION_TEMPLATES
from patrol_rsi.improver_protocol import TOOLS,MEMORY

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-31-m12e-timing-schema'
OLD=ROOT/'research/2026-10-30-m12d-formal'
BASE='6174f2ab3ae227b8bec6f83b05fdca512bb84192'


def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    rows=[]
    for trial in range(3):
        for number in (13,18):
            start=OLD/f'evidence/run/rsi-trial-{trial}/calls/call-{number:02d}.start.json'
            end=start.with_name(start.name.replace('.start.','.result.'));raw=read(end)['raw_response'];v=strict_json(raw);p=v['protocol']
            try:ControlledImprover.from_dict(p);error=None
            except (ValueError,TypeError,KeyError) as e:error=type(e).__name__+': '+str(e)
            assert error
            checks={'outer_envelope_exact':set(v)=={'diagnosis','protocol','expected_effect'},
                'controlled_outer_keys_exact':set(p)=={'protocol',*ENUMS},'missing_internal_Improver':'protocol' not in p,
                'diagnostic_count':len(p.get('diagnostics',[])),'diagnostic_count_valid':1<=len(p.get('diagnostics',[]))<=3,
                'diagnostic_names_valid':all(x in TOOLS for x in p.get('diagnostics',[])),
                'diagnostics_distinct':len(set(p.get('diagnostics',[])))==len(p.get('diagnostics',[])),
                'memory_allowed':p.get('memory') in MEMORY,'instruction_allowed':p.get('instruction') in INSTRUCTION_TEMPLATES,
                **{k+'_allowed':p.get(k) in values for k,values in ENUMS.items()}}
            rows.append({'trial':trial,'call':number,'raw_response':raw,'parser_error':error,'checks':checks,
                'original_start_path':str(start.relative_to(ROOT)),'original_start_SHA256':sha(start),
                'original_result_path':str(end.relative_to(ROOT)),'original_result_SHA256':sha(end),'original_response_edited':False})
    write(STAGE/'evidence/old-six-meta-diagnosis.json',{'source_stage':'M12-D','rows':rows,'valid':0,'old_scores_recomputed':False,
        'old_final_raw_data_accessed':False,'meaning':'JSON syntax succeeded; nested schema and diagnostic cardinality failed. No legal new meta protocol effect was tested.'})
    time=dt.datetime.now(dt.timezone.utc).isoformat()
    prompts={'parents':[v.to_dict() for v in format_parents()],'synthetic_challenges':FORMAT_CHALLENGES,'calls':[
        {'slot':i,'method':method,'generation_seed':940000+i,'challenge_index':i%3,'parent_index':i%3,
            'parent_protocol':format_parents()[i%3].to_dict(),'user_prompt':prompt(method,format_parents()[i%3],i%3)}
        for i in range(12) for method in ('full_nested','compact_leaves')]}
    write(STAGE/'E0-prompts.json',prompts)
    e0={'protocol':'M12E-E0-Schema-v1','baseline':BASE,'status':'frozen_for_independent_review_before_execution','created_utc':time,
        'question':'Can the fixed open model propose legal, different and transferable ControlledImprover protocols? Format/representation only, no performance or RSI gain claim.',
        'methods':['full_nested','compact_leaves'],'calls_each_method':12,'actual_model_call_cap':24,'paired_generation_seeds':[940000+i for i in range(12)],'prompts_SHA256':sha(STAGE/'E0-prompts.json'),
        'schedule':'Interleave full_nested then compact_leaves for each slot; same model seed per pair. Three predeclared synthetic format challenges/parents, four repetitions each. No response-dependent edits or retries.',
        'model':{'id':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218','precision':'float16','thinking':False,
            'temperature':.7,'top_p':.8,'top_k':20,'input_token_cap':4096,'output_token_cap':768,'weights_updated':False},
        'full_schema':full_schema(),'compact_schema':compact_schema(),
        'compact_semantics':'NEW deterministic structural assembly. Model supplies all diagnostic/memory/instruction/enum leaves; compiler supplies nesting only. It cannot repair invalid values or substitute a successful proposal. Compiled format success is not model self-improvement.',
        'public_input':'Only fixed synthetic coverage/violation/check/duplicate counts and current legal parent, schema, allowlists and complete example. No real task trace, seed, injection, outcome label or final-test information in prompt.',
        'raw_parse':'Strict entire JSON object, duplicate keys/nonfinite constants/trailing prose rejected. Full interface validates original nested object. Compact validates leaf schema then compiles and validates unchanged DSL.',
        'metrics':['raw_JSON_syntax_rate','raw_interface_schema_rate','raw_full_ControlledImprover_rate_distinct_from_compiled_rate','compiled_full_schema_rate','valid_different_parent_rate','distinct_valid_protocols','changed_allowed_fields','invalid_duplicate_timeout_counts','all_attempts_tokens_wall_GPU','complete_task_history_reset'],
        'method_gate':{'compiled_or_raw_full_valid_min':11,'valid_and_different_parent_min':8,'distinct_nonparent_legal_protocols_min':3,
            'zero_hidden_or_disallowed_change':True,'all_valid_protocol_transfer_resets':True,'all_attempts_and_costs_accounted':True},
        'decision':'E0 passes if either frozen interface meets ALL per-method criteria; full_nested preferred if both pass, otherwise compact interface explicitly labelled. Failure stops future costly meta search. n=12 diagnostic reliability screen, not population guarantee.',
        'failure_accounting':'Each attempted generation charges one of24slots before model.generate; invalid/duplicate/timeout consumes slot, no resampling; invalid restores identical parent. Unknown failed output tokens charged at cap upper bound and labelled unknown. Model/runtime/source failure stops remaining activity.',
        'budgets':{'input_tokens_total_cap':98304,'output_tokens_total_cap':18432,'GPU_allocations_max':1,'GPU_seconds_max':1800,'per_call_timeout_s':60,'soft_stop_s':1680,'CPU_per_GPU_allocation':4,'RAM_GB':48,'source_storage_cap_bytes':500000000},
        'not_allowed':['oldreplyrepair','oldM12Drescoring','taskprogramsearch','metasearch','newweights','newrealpreflightoutside24slots']}
    write(STAGE/'E0-Schema-v1.json',e0)
    fixtures=read(ROOT/'research/2026-10-30-m12c1b4b2-alignment/MultiAgent-Patrol-Paired-v1.json')['physical_fixtures']
    partitions={'development':[33000,33001],'validation':[33020,33021,33022,33023],'final_test':[33040,33041,33042,33043]};manifest=[]
    old_phases={150,335,520,705}
    for phase,seeds in partitions.items():
        for seed in seeds:
            for scenario,items in fixtures.items():
                used=[]
                for ordinal,(duration,_) in enumerate(items):
                    value=int(hashlib.sha256(f'M12E-independent-phase-v1|{phase}|{seed}|{scenario}|{ordinal}'.encode()).hexdigest(),16)
                    starts=[v for v in range(95,900,5) if v not in old_phases and all(abs(v-p)>=100 for p in used)]
                    onset=starts[value%len(starts)];used.append(onset)
                    for noise in (.2,1.):manifest.append({'phase':phase,'seed':seed,'scenario':scenario,'duration_s':duration,'fault_start_s':onset,
                        'noise_c':noise,'estimator':'history','control':'feedback','resources':'provisioned','variant':'open'})
    write(STAGE/'process-manifest.json',{'phase_rule':'SHA256 fixed namespace/partition/seed/scenario/ordinal;5sgrid95..895 excluding old4onsets;withincondition two phases separated>=100s. Never an online input.',
        'partitions':partitions,'rows':manifest,'validation_final_planned_not_generated':True})
    e1={'protocol':'M12E-E1-Timing-v1','baseline':BASE,'status':'frozen_for_independent_review_before_execution','created_utc':time,
        'question':'Separate using the sole optional check NOW from ranking legal targets. WAIT means continue protected Tour while retaining quota, never physical idle.',
        'arms':['Tour','human_professional','unconditional_PPO','evidence_gate_PPO','evidence_gate_urgency'],
        'fixed_core':'OriginalM9physics,18heterogeneousexecutors,ownedtask/qualification/cooldown60s/fullreturn/measurement/revisit unchanged; shared aligned_patrol.run_episode and global1CHECK gate.',
        'development_manifest_SHA256':sha(STAGE/'process-manifest.json'),'partitions':partitions,'only_current_partition':'development',
        'gate':{'signals':{'thermal_excess':.05,'contact_heat':.05,'flow_deficit':.05,'verified_sensor_bias':.5},
            'candidate':'Hard legal optional target, corresponding public professional evidence available and above fixed threshold, not already measured alarm; positive finish lead over protectedTour; signal*public_weight*lead >0.05*other_task_shift.',
            'signal_sources':'Same typed_signal current history/qualified contact/observed flow/verified sensor bias; observation availability and age explicit. No event oracle, scenario, seed or fault clock.',
            'ranking':'Same qualifying set for both gated arms; PPO uses frozen actor scores, urgency uses original f[:,0]+f[:,1]. Shared rotating dispatch order/common1CHECK determines selected agent.',
            'after_spending':'MandatoryTour; no further optional candidate dispatch or quota reset across owned rounds.'},
        'diagnostic':'Record all Tour legal candidates at every5sroot through1795, currentpublic measurement/age, feature/score at5s, projected downstream shift and exact routes. Evaluate offline oracle ONLY after public decisions saved.',
        'oracle':'Existing analytic one-CHECK Tour continuation diagnostic; no labels to selector/model. Offline best and earliest public-eligible net-beneficial candidate each condition are independently executed once (dedup) then Tour and compared to analytic all-event miss/gain/loss/wait. Other oracle opportunities labelled approximate, never claimed closed-loop measured.',
        'oracle_net_definition':'No lost confirmation, no increased miss AND (fewer misses OR censored wait gain minus0.05*shift >0). This is offline diagnostic, not online gate truth.',
        'metrics':['macro_seed_process_noise_miss_and_weighted_censored_wait','professional_effective_confirmation','WAIT_continueTour_and_CHECK_counts','CHECK_time','actualPPO_multiple_and_single_candidate_rank','specific_Tour_deviation','missed_public_recognizable_opportunities','route_shift_and_work_cost','qualification_fullreturn_exact_initial_tail_revisit','physical_and_noise_pairing'],
        'opportunity_gate':'Each of2developmentseedgroups >=2distinctnumericprocesses with a public-eligible net-beneficial witness verified by actual closed-loop branch, not two noise versions counted twice.',
        'timing_gate':'GatedPPO actually retains quota then uses CHECK in both seeds; all safety/source/cost checks pass; perseedmiss no increase vs Tour AND human_professional AND gated_urgency; each comparator macro fewer misses OR same miss with>=5%lower macro wait.',
        'E2_rule':'E2 remains unexecuted in this development stage. Future task-level launch requires independent reviewed protocol, E1 opportunity AND timing gates; future meta launch additionally requires E0 pass. No gate can replace the other.',
        'budgets':{'new_physical_processes_development':18,'noisy_conditions':36,'main_episodes':180,'closed_oracle_witness_cap':72,'total_new_episode_cap':252,
            'allocated_CPU_minutes_cap':960,'max_CPU_jobs_parallel':2,'CPU_per_job':2,'RAM_GB':8,'per_job_minutes':60,'real_LLM_calls':0,'GPU_allocations':0,'source_storage_cap_bytes':3000000000},
        'stop':'Any source/split/safety/resource failure stops. No addedcandidatearms,thresholdrelaxation,newweights,changeddeadlinesorresultdrivenphasechanges. Negative performance retained; futurevalidation/finalsealed.',
        'statistical_unit':'2developmentseedgroups,9numericprocessesperseed,2noisepaired,5arms; macro sameM12D hierarchicalmetric. Development descriptive screen, no confirmatory population claim.'}
    write(STAGE/'E1-Timing-v1.json',e1)
    files=subprocess.check_output(['git','ls-files','-z']).decode().split('\0');protected={}
    for name in filter(None,files):
        p=ROOT/name;protected[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    write(STAGE/'evidence/protected-baseline.json',{'baseline':BASE,'files':protected})
    print('M12E_PREREGISTERED',len(rows),len(manifest),len(protected),sha(STAGE/'E0-Schema-v1.json'),sha(STAGE/'E1-Timing-v1.json'),flush=True)


if __name__=='__main__':main()

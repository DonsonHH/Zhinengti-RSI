"""Reconstruct real search retention and transfer from immutable receipts only."""
import argparse
import json
from pathlib import Path

from patrol_rsi.improver_protocol import validation_rank
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.rsi_controlled import ControlledImprover,program_key


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--trial',type=int,required=True);a=p.parse_args()
    r=a.run_root;root=r/f'rsi-trial-{a.trial}';d=read(root/'adapt-validation-complete.json')
    lookup={}
    for path in (root/'evaluations').glob('*--complete.json'):
        v=read(path);lookup[v['phase'],PatrolProgram.from_dict(v['program']).version_id]=v['result']
    all_case_index={v['record_path']:v for v in d['case_index']}
    need(d['complete'] and not d['final_source_access'],'Actual nonfinal completion before independent audit')
    for path,row in all_case_index.items():
        need(sha(Path(path))==row['record_SHA256'] and row['audit']['passed'],'Unchanged independently audited actual environment record')
    counts={'task_candidate_attempts':0,'accepted_task_candidates':0,'duplicate_task_candidates':0,'failed_task_candidates':0}
    def rank(result):
        rows=[all_case_index[x] for x in result['case_ids']]
        weight=sum(v['event_weight_sum'] for v in rows)
        rebuilt={'violating_tasks':sum(v['metrics']['tasks_violating_exact_gap'] for v in rows),
            'coverage':sum(v['metrics']['final_professional_coverage'] for v in rows)/len(rows),
            'missed_segments':sum(v['metrics']['missed_segments'] for v in rows),
            'weighted_censored_delay_s':sum(v['weighted_delay_sum'] for v in rows)/weight if weight else None,
            'mean_travel_m':sum(v['metrics']['travel_m'] for v in rows)/len(rows)}
        expected=validation_rank(result['validator']);need(validation_rank(rebuilt)==expected,'Independent actual-case pooled development selector')
        return expected
    def search(value,phase,slots):
        pg=PatrolProgram();protocol=ControlledImprover.from_dict(value['protocol']);seen={program_key(pg,protocol.dedup)}
        need(value['status']=='completed_bounded_slots' and len(value['history'])==len(value['receipts'])==slots,'Every fixed candidate slot accounted')
        for history,receipt in zip(value['history'],value['receipts']):
            counts['task_candidate_attempts']+=1
            need(receipt['parent_id']==pg.version_id and receipt['accepted']==history['accepted'],'Actual parent/child receipt sequence')
            if history['workflow'] is None:
                need(not history['accepted'] and receipt['reason']=='failure_consumes_slot','Invalid/failed consumes slot')
                counts['failed_task_candidates']+=1
                if protocol.failure_fallback=='safe_tour':pg=PatrolProgram()
                continue
            child=PatrolProgram.from_dict(history['workflow']);key=program_key(child,protocol.dedup)
            if key in seen:
                need(not history['accepted'] and history['reason']=='duplicate_consumes_slot','Duplicate consumes without free resample')
                counts['duplicate_task_candidates']+=1;continue
            seen.add(key)
            need((phase,child.version_id) in lookup,'Every distinct valid child actually evaluated')
            accepted=protocol.acceptance=='strict_improvement' and rank(lookup[phase,child.version_id])<rank(lookup[phase,pg.version_id])
            need(accepted==history['accepted'],'Independent strict task retention decision')
            if accepted:pg=child;counts['accepted_task_candidates']+=1
        need(pg.to_dict()==value['program'] and pg.version_id==value['version_id'],'Final selected program reconstructed')
        return pg
    fixed=search(d['direct'],'fresh_task_adapt',13);random=search(d['random'],'fresh_task_adapt',13)
    retained=ControlledImprover();accepted_meta=0
    for number,entry in enumerate(d['meta_rounds']):
        need(entry['initial_program']==PatrolProgram().to_dict() and entry['history_start_lengths']==[0,0],'Both meta arms reset actual task and history')
        old=search(entry['incumbent'],f'meta_round{number}',2);new=search(entry['candidate'],f'meta_round{number}',2)
        proposed=ControlledImprover.from_dict(entry['candidate']['protocol'])
        need(entry['incumbent']['protocol']==retained.to_dict(),'Old protocol sequence')
        accepted=proposed.version_id!=retained.version_id and rank(lookup[f'meta_round{number}',new.version_id])<rank(lookup[f'meta_round{number}',old.version_id])
        need(accepted==entry['accepted'],'Independent strict meta retention')
        if accepted:retained=proposed;accepted_meta+=1
        need(entry['retained_protocol']==retained.to_dict(),'No discarded protocol crosses transfer')
    need(d['transfer']['protocol']==retained.to_dict(),'Only retained protocol starts fresh task search')
    transferred=search(d['transfer'],'fresh_task_adapt',3)
    starts=list((root/'calls').glob('*.start.json'));results=list((root/'calls').glob('*.result.json'))
    need(len(starts)==len(results)==len(d['model_call_receipts'])<=26,'Every actual model attempt has charged complete receipt')
    input_tokens=output_tokens=0
    for receipt in d['model_call_receipts']:
        start=read(root/'calls'/f"call-{receipt['number']:02d}.start.json");end=read(root/'calls'/f"call-{receipt['number']:02d}.result.json")
        need(start['seed']==receipt['seed'] and start['input_tokens']==receipt['input_tokens'] and start['thinking'] is False,'Actual nonthinking call identity and token input')
        need(end['raw_response']==receipt['raw_response'] and end['output_tokens']==receipt['output_tokens'],'No edited model response or token output')
        input_tokens+=receipt['input_tokens'];output_tokens+=receipt['output_tokens']
    need(input_tokens==d['allcalls_input_tokens']<=159744 and output_tokens==d['allcalls_output_tokens']<=19968,'Independent total real token costs')
    need(counts['task_candidate_attempts']==37,'Exactly13random+13fixed+8metainner+3freshmetaslots')
    phase_by_fp={}
    for row in d['case_index']:
        phase=row['specification']['phase'];fp=row['process_fingerprint']
        need(fp not in phase_by_fp or phase_by_fp[fp]==phase,'No same physical process across meta/adapt/validation')
        phase_by_fp[fp]=phase
    report={'passed':True,'trial':a.trial,'actual_model_calls':len(starts),'input_tokens':input_tokens,'output_tokens':output_tokens,
        'counts':counts,'meta_updates_accepted':accepted_meta,'all_actual_task_retention_reconstructed':True,
        'all_actual_meta_retention_reconstructed':True,'only_protocol_transferred_with_task_history_reset':True,
        'numeric_processes':len(phase_by_fp),'case_records_rehashed':len(all_case_index),'model_calls_in_this_audit':0,
        'environment_episodes_in_this_audit':0,'final_source_access':False,'adaptation_receipt_SHA256':sha(root/'adapt-validation-complete.json')}
    write(root/'independent-search-audit.json',report)
    print('M12D_RSI_SEARCH_AUDIT_PASSED',a.trial,len(starts),len(all_case_index),flush=True)


if __name__=='__main__':main()

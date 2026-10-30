"""Pure injected protocol-update/transfer smoke, no model or environment calls."""
from __future__ import annotations

import argparse
from pathlib import Path

from patrol_rsi.improver_protocol import Improver
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.limited_prediction import write
from patrol_rsi.rsi_controlled import (ControlledImprover,INSTRUCTION_TEMPLATES,BudgetedEvaluationStore,
    paired_protocol_update,partitions_disjoint,transfer_reset,run_slots)


def convert(value):
    if hasattr(value,'to_dict'):return value.to_dict()
    if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [convert(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    identities={'meta':[{'seed_group':'stub-meta','process_fingerprint':'stub-physical-meta'}],
        'transfer':[{'seed_group':'stub-transfer','process_fingerprint':'stub-physical-transfer'}]}
    assert partitions_disjoint(identities)
    def evaluate(cases,program):
        return {'case_ids':cases,'public':{'coverage':{},'professional_evidence':{},'efficiency':{}},
            'validator':{'violating_tasks':0,'coverage':1.,'missed_segments':0,
                'weighted_censored_delay_s':10. if program.refresh_s==120 else 20.,'mean_travel_m':100.}}
    calls=[]
    def generator(prompt,seed):
        calls.append({'seed':seed,'start_program':prompt['current_workflow'],'history_count':len(prompt['retrieved_candidates'])})
        changed=prompt['improver_protocol']['instruction']==INSTRUCTION_TEMPLATES[1]
        return {'workflow':PatrolProgram(refresh_s=120 if changed else 30).to_dict()}
    old=ControlledImprover();new=ControlledImprover(Improver(instruction=INSTRUCTION_TEMPLATES[1]))
    store=BudgetedEvaluationStore(evaluate,10)
    retained,pair=paired_protocol_update(old,new,store,{'qwen_once':generator},['stub-meta'],7)
    reset=transfer_reset(retained);fresh_store=BudgetedEvaluationStore(evaluate,10)
    result=run_slots(reset['protocol'],fresh_store,{'qwen_once':generator},['stub-transfer'],70,2)
    assert pair['accepted'] and pair['history_start_lengths']==[0,0] and reset['program']==PatrolProgram() and not reset['history']
    assert calls[0]==calls[2] and calls[4]['start_program']==PatrolProgram().to_dict() and calls[4]['history_count']==0
    write(args.output/'meta-interface-smoke.json',{'scope':'Pure synthetic interface fixture;fixed scorecallback,zero physical simulation,zero actualmodelcalls. Values are not scientific benefits.',
        'partition_identity_fixture':identities,'pair':convert(pair),'fresh_result':convert(result),
        'stub_generator_calls':len(calls),'real_LLM_calls':0,'real_environment_episode_evaluations':0,
        'task_program_or_history_transferred':False,'only_protocol_transferred':True,
        'meta_store':{'requests':store.requests,'cache_hits':store.cache_hits,'synthetic_callback_cases':store.actual_episode_evaluations},
        'fresh_store':{'requests':fresh_store.requests,'cache_hits':fresh_store.cache_hits,'synthetic_callback_cases':fresh_store.actual_episode_evaluations}})


if __name__=='__main__':main()

"""Frozen RSI partition identities, prefix selection and macro validation."""
from .improver_task import PatrolProgram
from .limited_prediction import need
import numpy as np

GROUPS=('Tour','human_fixed','random_search','fixed3','fixed13','controlled_meta')


class FatalCampaignBoundary(BaseException):
    """Never swallowed as an ordinary failed proposal by legacy interfaces."""


def cases(phase,paired,rsi):
    seeds=rsi['partitions'][phase];onsets=(335,705) if phase.startswith('meta') else (150,520)
    rows=[]
    for seed in seeds:
        for scenario,fixtures in paired['physical_fixtures'].items():
            for number,(duration,_) in enumerate(fixtures):
                onset=onsets[number]
                for noise in (.2,1.):
                    rows.append({'phase':phase,'seed':seed,'scenario':scenario,'duration_s':duration,'fault_start_s':onset,
                        'noise_c':noise,'estimator':'history','control':'feedback','resources':'provisioned','variant':'open'})
    need(len(rows)==len(seeds)*18,'9physicalprocesses/two noises in frozen RSI partition')
    return rows


def prefix_program(result,n=3):
    need(len(result['history'])>=n,'Prefix exists in original candidate sequence')
    program=PatrolProgram()
    for row in result['history'][:n]:
        if row['accepted']:program=PatrolProgram.from_dict(row['workflow'])
    return program


def macro(rows):
    seeds=[]
    for seed in sorted({v['specification']['seed'] for v in rows}):
        own=[v for v in rows if v['specification']['seed']==seed];process={}
        for row in own:process.setdefault(row['process_fingerprint'],[]).append(row)
        need(len(process)==9 and all(len(v)==2 for v in process.values()),'Exact9numericprocesses eachseed two noise versions')
        missed=[];wait=[]
        for pair in process.values():
            need({v['specification']['noise_c'] for v in pair}=={.2,1.},'Paired noise cohort')
            missed.append(np.mean([v['metrics']['missed_segments'] for v in pair]))
            values=[v['metrics']['consequence_weighted_censored_delay_s'] for v in pair]
            need((values[0] is None)==(values[1] is None),'Same event eligibility under noise')
            if values[0] is not None:wait.append(float(np.mean(values)))
        seeds.append({'seed':seed,'macro_missed_count':float(np.mean(missed)),'macro_wait_s':float(np.mean(wait)) if wait else None})
    return {'seeds':seeds,'macro_missed_count':float(np.mean([v['macro_missed_count'] for v in seeds])),
        'macro_wait_s':float(np.mean([v['macro_wait_s'] for v in seeds if v['macro_wait_s'] is not None])),
        'conditions':len(rows),'missed_noise_repeated_sum':sum(v['metrics']['missed_segments'] for v in rows)}


def retain_on_validation(group,tour):
    old={v['seed']:v for v in tour['seeds']}
    perseed=all(v['macro_missed_count']<=old[v['seed']]['macro_missed_count'] for v in group['seeds'])
    delta=group['macro_missed_count']-tour['macro_missed_count']
    wait=group['macro_wait_s']-tour['macro_wait_s']
    return {'retain':bool(perseed and (delta<0 or delta==0 and wait<0)),
        'no_seed_miss_increase':perseed,'macro_missed_delta':delta,'macro_wait_delta_s':wait}


def validate_mapping(receipt,mapping,identity_SHA,protocol_SHA):
    need(receipt['complete'] and not receipt['final_source_access'],'Only completed nonfinal adaptation receipt')
    need(receipt['execution_identity_SHA256']==mapping['execution_identity_SHA256']==identity_SHA and mapping['RSI_protocol_SHA256']==protocol_SHA,'Exact original source/protocol mapping identity')
    need(receipt['programs']==mapping['programs'] and set(mapping['programs'])==set(receipt['validation'])==set(GROUPS),'Exactly six nominal/actual logical groups')
    for name,row in mapping['programs'].items():
        nominal=PatrolProgram.from_dict(row['nominal_program']);actual=PatrolProgram.from_dict(row['actual_program'])
        expected=retain_on_validation(receipt['validation'][name],receipt['validation']['Tour']) if name!='Tour' else {'retain':True}
        need(row['validation_gate']==expected,'Recomputed exact validation fallback')
        need(actual==(nominal if expected['retain'] else PatrolProgram()),'Final actual program iff nominal passed else original Tour')
    need(not any(v['specification']['phase']=='fresh_task_final' for v in receipt['case_index']),'No final rows before mapping seal')
    return True

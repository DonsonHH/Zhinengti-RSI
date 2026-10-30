"""Aggregate development legal-action feasibility,without changing frozen heads."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from pathlib import Path

from patrol_rsi.limited_prediction import load_reviewed,need,read,sha,write


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','root','B1a-coverage','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();protocol,review=load_reviewed(args.protocol,args.review)
    need(sha(args.B1a_coverage)=='4d08cbd6e9350f73ee4a750dd7958470a08cf72b50a133d7fff9ce11f5a25854','Original frozen coverage comparison')
    totals=defaultdict(Counter);events=defaultdict(set);positive_seeds=defaultdict(set);positive_processes=defaultdict(set);sources=[];elapsed=0.
    for i in range(24):
        p=args.root/f'seed-index-{i}'/'complete.json';v=read(p)
        need(v['split']=='development' and v['conditions']==20 and not v['final_data_access'] and v['protocol_SHA256']==sha(args.protocol),'Exactly development-only diagnostic')
        sources.append({'seed':v['seed'],'sha256':sha(p)});elapsed+=v['elapsed_s']
        for key,row in v['statistics'].items():
            totals[key].update({k:row[k] for k in ('legal_candidates','optional_candidates','protected_Tour_candidates','B_reference_positive','A_physical_positive','sum_horizon_s') if k in row})
            if 'minimum_horizon_s' in row:totals[key]['minimum_horizon_s']=min(totals[key].get('minimum_horizon_s',1801),row['minimum_horizon_s'])
            if 'maximum_horizon_s' in row:totals[key]['maximum_horizon_s']=max(totals[key].get('maximum_horizon_s',0),row['maximum_horizon_s'])
            for fp,event in row['event_keys']:
                events[key].add((fp,event));positive_processes[key].add(fp);positive_seeds[key].add(v['seed'])
    statistics={}
    for key,counts in totals.items():
        statistics[key]=dict(counts,distinct_retained_positive_events=len(events[key]),positive_seed_groups=len(positive_seeds[key]),
            positive_numeric_processes=len(positive_processes[key]),mean_horizon_s=counts['sum_horizon_s']/counts['legal_candidates'] if counts['legal_candidates'] else None)
    old=read(args.B1a_coverage);contrast=[]
    for kind in ('thermal','electrical','cooling','control'):
        fixed=[v for v in old['coverage_cells'] if v['split']=='development' and v['kind']==kind]
        contrast.append({'kind':kind,'action_all_5s':statistics['all_5s_roots/'+kind+'/all'],
            'action_same_30s':statistics['same_30s_roots/'+kind+'/all'],
            'current_action_ETA_le30_all_5s':statistics['all_5s_roots/'+kind+'/0-30'],
            'current_action_ETA_le30_same_30s':statistics['same_30s_roots/'+kind+'/0-30'],
            'historical_fixed_exact_horizons':[{'horizon_s':v['horizon_s'],'B_valid':v['B_action_valid'],
                'B_positive':v['B_action_positive'],'retained_events':v['distinct_B_positive_retained_events']} for v in fixed]})
    candidates=sum(statistics['all_5s_roots/'+k+'/all']['legal_candidates'] for k in ('thermal','electrical','cooling','control'))
    storage={'flat98float32_plus8int32_meta_and2int8labels_bytes':candidates*(98*4+8*4+2),
        'flat_features_only_bytes':candidates*98*4,'label_only_2int8_bytes':candidates*2,
        'root_base85float32_all359frames_176tasks_480conditions_bytes':480*359*176*85*4,
        'factoring_note':'Current-action rows can reference existing original public histories and shared geometry;storing every root/task base matrix can be larger than storing only legal flat actions. No new action feature dataset actually written.'}
    result={'scope':'Development-only Action-Indexed feasibility;separate from Limited model selection and final scores',
        'protocol_SHA256':sha(args.protocol),'seeds':24,'conditions':480,'sources':sources,'statistics':statistics,'contrasts':contrast,
        'storage_estimates':storage,'sum_diagnostic_wall_s':elapsed,'candidate_rows_total':candidates,
        'interpretation':'Compare same30s roots before attributing change to target indexing;all5s roots additionally changes root coverage. Inflight future protectedTour completion is not a new dispatchable action. Eachlegal candidate is a separate one-CHECK-then-Tour counterfactual,not a jointly executable schedule.',
        'no_window_truth_shortcut':True,'original_qualification_quota_return_constraints':True,'new_physical_data':0,
        'new_action_predictor_training':0,'final_data_access':False,'historical_B1a_changed':False}
    args.output.mkdir(exist_ok=False);write(args.output/'action-indexed-feasibility.json',result)
    lines=['# 开发集Action-Indexed可行性诊断','',
        '|专业|5秒根合法动作|5秒根B正例|5秒根可预测保留事件|同30秒根合法动作|同30秒根事件|ETA≤30秒事件（5秒根）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in contrast:
        a=v['action_all_5s'];b=v['action_same_30s'];c=v['current_action_ETA_le30_all_5s']
        lines.append('|'+ '|'.join(map(str,[v['kind'],a['legal_candidates'],a['B_reference_positive'],a['distinct_retained_positive_events'],
            b['legal_candidates'],b['distinct_retained_positive_events'],c['distinct_retained_positive_events']]))+'|')
    lines+=['','每个动作只用其实际预计完成时刻的原B_reference，不以窗口内任意异常替代确认。不同预测根和噪声共享事件，上表事件按物理指纹／事件去重，不将大量动作当独立故障。','',
        '候选覆盖增加不等于巡检策略收益，也没有用诊断结果更改本轮正式限定模型。']
    (args.output/'行动时刻诊断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('ACTION_INDEXED_DIAGNOSTIC_REPORTED',candidates,flush=True)


if __name__=='__main__':main()

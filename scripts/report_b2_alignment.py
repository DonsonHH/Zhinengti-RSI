"""Aggregate completed development diagnosis; no model or final-data reads."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);args=p.parse_args();stage=args.stage;run=stage/'evidence/run'
    data=read(run/'calibration/calibration-diagnosis.json');cal=[];ridges=[]
    for row in data['rows']:
        assert row['phase'] in ('development','validation_calibration')
        if row['phase']!='validation_calibration':continue
        outputs={v['probability']:v for v in row['probability_outputs']}
        frozen=outputs['frozen_ridge1'];rho0=outputs['hierarchy_ridge0'];small=outputs['hierarchy_ridge0p0001']
        cal.append({'cell':row['cell'],'kind':row['kind'],'horizon_s':row['horizon_s'],'view':row['view'],
            'raw_AP':row['raw_score_AUPRC'],'raw_AUROC':row['raw_score_AUROC'],'weighted_prevalence':frozen['metrics']['weighted_prevalence'],
            'raw_score_range':row['raw_scores']['overall']['range'],'frozen_probability_range':frozen['overall']['range'],
            'ridge0_probability_range':rho0['overall']['range'],'ridge0p0001_probability_range':small['overall']['range'],
            'frozen_Brier':frozen['metrics']['Brier'],'ridge0_Brier':rho0['metrics']['Brier'],'ridge0p0001_Brier':small['metrics']['Brier'],
            'balanced_corrected_Brier':outputs['balanced_ridge0--prior_corrected']['metrics']['Brier'],
            'balanced_native_Brier':outputs['balanced_ridge0--native_changed_prior']['metrics']['Brier'],
            'in_sample_calibration_diagnostic':True,'replacement_selected':False})
        for derivative in row['calibration_objective_derivatives']:
            detail={'cell':row['cell'],'view':row['view'],**derivative}
            if derivative['ablation']=='frozen_ridge1':
                old=next(v['original'] for v in data['original_replays'] if v['cell']==row['cell'] and '--'+row['view']+'--' in v['id'])
                detail.update(calibration_a=old['a'],original_termination=old['termination'],
                    gradient_wrt_softplus_z=derivative['sum_gradient_wrt_a']*(-math.expm1(-old['a'])),
                    parameterization_note='Optimizergradienttol acts onz,not directlya;smallsoftplusaderivative can meetabsolutegradienttol whiledL/da islarger. Descriptiveoriginalreceipt,not extraoptimizedcandidate.')
            ridges.append(detail)
    actions=[]
    for i in (0,1):actions.extend(read(run/'actions'/f'seed-index-{i}'/'action-diagnosis.json')['rows'])
    assert len(actions)==36 and len({r['numeric_process_fingerprint'] for r in actions})==18
    by_rank={};per_seed=[]
    for name in ('professional_cue','public_net','public_net_no_history','time_cost_only'):
        allrows=[]
        for seed in (20000,20001):
            own=[v for v in actions if v['seed']==seed];out=[v['rankers'][name] for v in own]
            process=defaultdict(list)
            for r in own:process[r['numeric_process_fingerprint']].append(r)
            macro_wait=[]
            for rows in process.values():
                values=[v['rankers'][name]['weighted_wait_delta_s'] for v in rows]
                if values[0] is not None:macro_wait.append(float(np.mean(values)))
            value={'ranker':name,'seed':seed,'conditions':len(out),'unique_processes':len(process),
                'selected_conditions':sum(v['selected'] for v in out),'net_beneficial_conditions':sum(v['net_beneficial'] for v in out),
                'net_harmful_conditions':sum(v['net_harmful'] for v in out),'paired_condition_missed_delta_sum':sum(v['missed_delta'] for v in out),
                'macro_process_missed_delta':float(np.mean([np.mean([v['rankers'][name]['missed_delta'] for v in rows]) for rows in process.values()])),
                'macro_eventbearing_process_wait_delta_s':float(np.mean(macro_wait)) if macro_wait else None,
                'macro_net_delay_gain_s':float(np.mean([v['net_delay_gain_s'] for v in out])),
                'lost_confirmations_sum':sum(v['lost_confirmations'] for v in out),'gained_confirmations_sum':sum(v['gained_confirmations'] for v in out)}
            per_seed.append(value);allrows.extend(out)
        by_rank[name]={'conditions':len(allrows),'selected_conditions':sum(v['selected'] for v in allrows),
            'net_beneficial_conditions':sum(v['net_beneficial'] for v in allrows),'net_harmful_conditions':sum(v['net_harmful'] for v in allrows),
            'paired_condition_missed_delta_sum':sum(v['missed_delta'] for v in allrows),'lost_confirmations_sum':sum(v['lost_confirmations'] for v in allrows),
            'gained_confirmations_sum':sum(v['gained_confirmations'] for v in allrows),
            'seed_macro_wait_delta_s':float(np.mean([v['macro_eventbearing_process_wait_delta_s'] for v in per_seed if v['ranker']==name])),
            'seed_macro_net_delay_gain_s':float(np.mean([v['macro_net_delay_gain_s'] for v in per_seed if v['ranker']==name]))}
    # The immutable diagnostic reports best-CHECK. Also provide the proper
    # optional-action ceiling including the no-CHECK Tour outcome.
    corrected_ceiling=[]
    for row in actions:
        oracle=row['oracle_ceiling_not_deployable'];take=oracle['missed_delta']<0 or oracle['missed_delta']==0 and oracle['net_delay_gain_s']>0
        base_missed=oracle['missed_delta'] if take else 0;gain=oracle['net_delay_gain_s'] if take else 0.
        corrected_ceiling.append({'condition':row['condition'],'seed':row['seed'],'ceiling_action':'CHECK' if take else 'Tour',
            'missed_delta':base_missed,'net_delay_gain_s':gain,'private_labels_used_only_for_offline_ceiling':True,
            'regrets':{name:{'miss_count_regret':v['missed_delta']-base_missed,
                'net_delay_regret_s_when_equal_misses':gain-v['net_delay_gain_s'] if v['missed_delta']==base_missed else None} for name,v in row['rankers'].items()}})
    actions_summary={'scope':'Two alreadyknown developmentseeds;18numericprocesses/36noiseconditions. No formaltest or newdistributionconfirmation.',
        'rankers':by_rank,'per_seed':per_seed,'total_legal_current_candidates':sum(v['candidates'] for v in actions),
        'net_useful_condition_count':sum(v['oracle_ceiling_not_deployable']['exists_net_beneficial'] for v in actions),
        'original_current_geometry_and_selected_schedule_verified':True,'independent_selected_schedule_checks':sum(read(run/'actions'/f'seed-index-{i}'/'action-diagnosis.json')['independent_selected_schedule_checks'] for i in (0,1)),
        'oracle_ceiling_including_no_check':corrected_ceiling,'oracle_interpretation':'Original receipts preserved best-CHECK comparator;derivedceiling hereincludesTour toavoidnegative regretwhenallchecks harmful. Nochange to publicselections or actualoutcomes.',
        'duplicates':'Two noiseversionsnot independentfaults;reportseed/processmacro aswell asrawconditioncounts.',
        'CVAE_comparison':'public_net andpublic_net_no_history selections/outcomes identical here;no uniqueCVAEincrement established.'}
    report={'status':'completed_development_diagnostics_and_protocol_freeze','stage_decision':'LIMITED GO for independent mainline experiment preparation;not launchor scientificpositiveclaim',
        'old_B3_2_NO_GO_unchanged':True,'old_B1b_NO_GO_unchanged':True,'probability_summary':cal,'calibration_derivatives':ridges,
        'probability_reason':'Samefrozenlogits withrho1 produceextremelysmalla andnarrowprobabilityrange;rho0/0.0001 expandrange. Somein-sample Briersworsen,balancedcorrectionnot universallybetter. This identifiesregularizationcontribution,not a confirmatoryreplacement or allLRcapacityproof.',
        'new_fit_counts':{'calibrators':data['new_calibration_fits'],'original_replays':data['replay_fits'],'LR':0},'action_summary':actions_summary,
        'protocols':{p.name:{'sha256':sha(p),'review':'freeze_not_launch'} for p in (stage/'MultiAgent-Patrol-Paired-v1.json',stage/'RSI-Controlled-Meta-v1.json')},
        'no_future_formal_tests':True,'real_LLM_calls':0,'CVAE_PPO_updates':0,'new_physical_processes':0}
    write(stage/'evidence/b2-results.json',report)
    lines=['# B4-B2开发诊断结果','', '本轮只做开发诊断、接口验证和下一轮协议冻结。旧B3.2与B1b的NO-GO保留。','',
        '## 概率压缩：同一分数，不同斜率正则','',
        '|专业／秒|原始分数AP|冻结概率范围|去斜率正则后的范围|冻结Brier|去正则Brier|',
        '|---|---:|---:|---:|---:|---:|']
    for r in cal:
        if r['view']=='time_lr':continue
        lines.append(f"|{r['kind']}／{r['horizon_s']}|{r['raw_AP']:.5f}|{r['frozen_probability_range']:.8g}|{r['ridge0_probability_range']:.5f}|{r['frozen_Brier']:.8g}|{r['ridge0_Brier']:.8g}|")
    lines+=['','以上均为校准集上的拟合内诊断；没有用旧最终集比较、选择或替换校准器。42次新校准和14次原校准重放已按预设预算完成。','',
        '## 当前动作Top-1与整回合净效果','',
        '|公开排序方法|执行条件数／36|净有益|净有害|漏检变化合计（噪声重复口径）|种子宏等待变化秒|种子宏净等待收益秒|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in by_rank.items():
        lines.append(f"|{name}|{r['selected_conditions']}|{r['net_beneficial_conditions']}|{r['net_harmful_conditions']}|{r['paired_condition_missed_delta_sum']}|{r['seed_macro_wait_delta_s']:.5f}|{r['seed_macro_net_delay_gain_s']:.5f}|")
    lines+=['','等待下降为好；净等待收益扣除了0.05×顺延秒的固定代价。漏检合计包含同一物理事件的两档噪声，不作独立故障次数。各方法每个条件最多执行一次补查，之后继续Tour；没有联合执行所有有益候选。','',
        f"共核查{actions_summary['total_legal_current_candidates']:,}个当前合法候选、{actions_summary['independent_selected_schedule_checks']}条实际选中反事实分支。32／36个条件存在至少一个离线净有益候选，公开净价值规则只选择11个条件；这说明有未利用的机会，不能用离线最优当可部署策略。",'',
        'public_net与移除history_assimilated的版本在本诊断中行动和结果相同，没有建立CVAE的独立贡献。开发结果不足以支持新种子、布局或RSI迁移结论。']
    (stage/'开发诊断结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('B2_REPORT_READY',len(cal),len(actions),'diagnosticrows',flush=True)


if __name__=='__main__':main()

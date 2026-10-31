"""Released formal aggregates, actual tool use and evidence matrix."""
import argparse
import json
from pathlib import Path

import numpy as np

from patrol_rsi.limited_prediction import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();s=a.stage;r=s/'evidence/run'
    phases={phase:read(r/(phase+'-complete.json')) for phase in ('development','validation')}
    comparisons=[];professional=[];usage=[]
    for phase,seeds in (('development',(31000,31001)),('validation',(31020,31021))):
        rows=[]
        for seed in seeds:rows.extend(read(r/phase/f'seed-{seed}'/'complete.json')['rows'])
        for arm in phases[phase]['aggregate']:
            own=[v for v in rows if v['arm']==arm]
            baselines={(v['physical_key'],v['noise_c']):v for v in rows if v['arm']=='Tour'}
            action_diff=sum(v['action_log_SHA256']!=baselines[v['physical_key'],v['noise_c']]['action_log_SHA256'] for v in own)
            checks_diff=sum(v['inspection_events_SHA256']!=baselines[v['physical_key'],v['noise_c']]['inspection_events_SHA256'] for v in own)
            u={'phase':phase,'arm':arm,'conditions':len(own),'action_logs_different_from_Tour':action_diff,'inspection_event_logs_different_from_Tour':checks_diff,
                'mean_optional_checks':float(np.mean([v['metrics']['selected_detours'] for v in own])),
                'zero_extra_conditions':sum(v['metrics']['selected_detours']==0 for v in own)}
            for key in ('dispatches','actual_PPO_dispatches','PPO_multichoice_dispatches','PPO_single_legal_candidate_dispatches',
                    'exclusive_shadow_actor_scoring_calls','rule_selected_dispatches','forced_return_Tour_dispatches','ordinary_return_Tour_dispatches','estimator_calls'):
                u[key]=sum(v['usage'][key] for v in own)
            u['PPO_multichoice_takeover_rate']=u['PPO_multichoice_dispatches']/u['dispatches']
            usage.append(u)
            for role in ('thermal','electrical','cooling','control'):
                professional.append({'phase':phase,'arm':arm,'role':role,
                    'retained_event_records_noise_repeated':sum(v['by_role'][role]['segments'] for v in own),
                    'effectively_confirmed_event_records_noise_repeated':sum(v['by_role'][role]['confirmed'] for v in own),
                    'scope':'Must be qualified measurement at actual completion;two noiseversions not independentfaults.'})
            stat=phases[phase]['aggregate'][arm]
            comparisons.append({'phase':phase,'arm':arm,'macro_missed_count':stat['macro_missed_count'],'macro_wait_s':stat['macro_wait_s'],
                'mean_travel_m':stat['mean_travel_m'],'mean_service_s':stat['mean_service_s'],'mean_idle_s':stat['mean_idle_s'],
                'mean_optional_checks':stat['mean_optional_checks'],'zero_exact_revisit_violations':stat['safety_ok'],
                'missed_event_records_noise_repeated':stat['missed_noise_repeated_sum'],
                'seed_metrics':stat['seeds'],'audits':stat['all_replays_and_independent_audits_passed']})
    gate=phases['validation']['gate'];q=read(r/'qwen-preflight/preflight.json')
    matrix=[
        {'question':'Taskworkflowincrement vsstrongTour','evidence':'360actualformaldev/validationepisodes+360frozenweightreplays;revisedmatchesTouronvalidationmiss/wait.','verdict':'NO-GO;fivearmfinalremainsunopened.'},
        {'question':'PPO actual decision/usefulness','evidence':'Actor/tool/singlecandidate/shadow/forcedTourcounts storedperrow;differentactualpathsnotautomaticallybetterevents.','verdict':'No incrementalconfirmationgain for frozenPPO inthisvalidationtaskfamily.'},
        {'question':'CVAE independent contribution','evidence':'All5arms share frozenCVAE;callsreported. No newnominal/CVAEformal6tharm. B2same-score/no-historyfindings retainedhistorically.','verdict':'No causalCVAEbenefit identifiedbythisfivearmdesign;not disprovenuniversally.'},
        {'question':'RSI actual model execution','evidence':f"OneQwenpreflightvalid={q['valid_workflow_output']},input={q['input_tokens']},output={q['output_tokens']};formalbudgetseparate.",
         'verdict':'Runtime/formatonly,not selfimprovement evidence.'},
        {'question':'Task skills and meta-improver transfer','evidence':'Separatefrozenfourgroup protocol. Only actualcompletedRSIreceipts canestablish effect;source/interface/preflight approvalcannot.',
         'verdict':'Pendingboundedstageoutcome;must updatewithactualcompletionorstop.'}]
    result={'stage':'M12-D','five_arm_gate':gate,'formal_primary_episodes':360,'formal_replay_episodes':360,
        'five_arm_final_episodes':0,'five_arm_final_source_generated':False,'comparisons':comparisons,
        'actual_tool_usage':usage,'professional_confirmation':professional,'Qwen_preflight':q,
        'B2_vs_this_stage':'B2publicnet counterfactual uses currentlegalCHECK endpoints andthenprescribedTour atsameoldstate;revised_PPO_evidence isdifferenthandwrittenPGwithPPOfallback. Thisstage realmeasurementfeedback/queues runinnewindependentprocesses. NoB2effectcreditedasformalnewgain.',
        'evidence_matrix':matrix,'old_B3_2_B1b_B2_results_unchanged':True}
    write(s/'evidence/m12d-summary.json',result)
    lines=['# M12-D：五臂正式结果与RSI准备','',
        '五臂开发180条、验证180条均实际执行并逐条重放。验证未通过，最终四个种子的物理源没有生成，最终360条没有运行；修订候选没有被Tour替换。','',
        '|阶段|方法|宏漏检数|宏后果加权删失等待秒|平均补查|平均移动米|',
        '|---|---|---:|---:|---:|---:|']
    for v in comparisons:lines.append(f"|{v['phase']}|{v['arm']}|{v['macro_missed_count']:.6f}|{v['macro_wait_s']:.6f}|{v['mean_optional_checks']:.6f}|{v['mean_travel_m']:.3f}|")
    lines+=['','宏统计先平均两档噪声、再过程、再种子；健康过程计入漏检和成本，有事件过程定义等待。不能把重复事件行当独立试验。','',
        '## 实际PPO与工作流行为','',
        '|阶段|方法|PPO多候选派单|PPO单候选派单|仅影子评分|规则选择|强制Tour|与Tour不同的动作日志数|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for v in usage:lines.append('|'+ '|'.join(str(v[k]) for k in ('phase','arm','PPO_multichoice_dispatches','PPO_single_legal_candidate_dispatches','exclusive_shadow_actor_scoring_calls','rule_selected_dispatches','forced_return_Tour_dispatches','action_logs_different_from_Tour'))+'|')
    lines+=['','真实调用或改变路线不能替代异常确认收益。当前固定所有权与全队一次补查限制了持续适应性；本轮没有临时扩大动作空间。CVAE在五组都冻结使用，故单靠本对照不能识别其因果增量。','',
        '## 唯一真实Qwen预检','',
        f"固定Qwen3-8B修订b968826d9c46dd6066d109eabc6255188de91218，13文件SHA核验；非思考模式，输入{q['input_tokens']}、输出{q['output_tokens']}tokens，生成{q['generation_s']:.3f}秒。输出合法工作流，未用于任务选择或元更新。",'',
        '初次GPU启动因模块环境缺transformers失败，未开始generation；恢复使用历史model-env，保留原开始收据，累计GPU101秒。实际generation只有一次，失败与恢复成本都计账。','',
        '## 论文证据矩阵','',
        '|问题|实际证据|当前结论|','|---|---|---|']
    for v in matrix:lines.append(f"|{v['question']}|{v['evidence']}|{v['verdict']}|")
    (s/'五臂正式结果与证据矩阵.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('M12D_REPORT_FORMAL_SCOPE',gate['status'],flush=True)


if __name__=='__main__':main()

"""Recompute historical tool-use counts from already released case aggregates."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);args=p.parse_args();stage=args.stage
    root=stage.parent.parent;rows=[];sources=[]
    for trial in range(3):
        folder=root/'research/2026-10-28-m12b-improver/evidence'/f'trial-{trial}'
        index=read(folder/'case-index.json');groups=read(folder/'test-groups.json');lookup={v['case']:v for v in index}
        for path in (folder/'case-index.json',folder/'test-groups.json'):sources.append({'path':str(path.relative_to(root)).replace('\\','/'),'sha256':sha(path)})
        for group,ids in groups.items():
            own=[lookup[i] for i in ids];public=[v['public_selection'] for v in own]
            def total(k):return sum(v[k] for v in public)
            rows.append({'trial':trial,'group':group,'case_references':len(own),'actual_PPO_dispatches':total('ppo_dispatches'),
                'dispatches':total('dispatches'),'rule_dispatches_including_return_rule':total('rule_dispatches'),
                'forced_single_candidate':total('forced_single_candidate'),'different_from_shadow_actor':total('different_from_actor'),
                'extra_checks':total('extra_checks'),'estimator_calls':total('estimator_calls'),
                'mean_extra_checks':float(np.mean([v['extra_checks'] for v in public])),
                'zero_extra_case_references':sum(v['extra_checks']==0 for v in public),
                'numeric_processes':len({lookup[i]['process_key'] for i in ids}),
                'note':'Released caseaggregates only;actualPPOtoolcounts include forcedsinglecandidate actions. Actorwascomputedforshadowcomparisonevenwhenrule/Tourselected.'})
    aggregate=[]
    for name in dict.fromkeys(v['group'] for v in rows):
        own=[v for v in rows if v['group']==name]
        aggregate.append({'group':name,**{k:sum(v[k] for v in own) for k in ('case_references','actual_PPO_dispatches','dispatches','rule_dispatches_including_return_rule','forced_single_candidate','different_from_shadow_actor','extra_checks','estimator_calls','zero_extra_case_references')},
            'mean_extra_checks':float(np.mean([v['mean_extra_checks'] for v in own])),'generation_repeats':3,'independent_physical_processes':16,
            'scope':'3generationrepeats reference same16physicalprocesses×2noises;not96 independent environments.'})
    old=root/'research/2026-10-27-m11-scheduling/evidence/case-index.json.gz'
    m11=json.loads(gzip.decompress(old.read_bytes()));sources.append({'path':str(old.relative_to(root)).replace('\\','/'),'sha256':sha(old)})
    m11groups=[]
    for method in sorted({v['specification']['method'] for v in m11}):
        own=[v for v in m11 if v['specification']['method']==method]
        m11groups.append({'method':method,'recorded_cases':len(own),'completed_inspections':sum(v['metrics']['completed_inspections'] for v in own),
            'guard_forced_choices':sum(v['metrics']['guard_forced_choices'] for v in own),
            'PPO_decision_count_available_in_released_aggregate':False,
            'interpretation':'Source methoddispatch tellsactualactor/heuristicroute;completedinspections not substitutedforactorcallcount.'})
    smoke=read(stage/'evidence/run/interface-smoke-v3/interface-smoke.json')
    result={'source_baseline':'d55c1613317f9dfb520f5d6eaea1083e32b3ecb2','historical_public_aggregate_sources':sources,
        'M12B_3repeat_groups':aggregate,'M12B_per_repeat':rows,'M11_method_records':m11groups,
        'B2_archived_engineering_rows':smoke['rows'],'B2_CVAE_cross_interventions':smoke['CVAE_deployment_cross_interventions'],
        'B2_receipt_repair':smoke['receipt_correctness_repair'],
        'no_private_final_records_opened':True,'new_scientific_gain_claimed':False,
        'M12B_effect_scope':'Historical fixedimprover descriptive smalltaskgain;metaupdatednotbetter. no_actor/nominalcrossinterventions notproofzero valueeverywhere.',
        'atomic_tool_independence':'No standalonePPO/CVAEcausalbenefit established. B2publicnet-nohistorysamechoices. Returnlegalitytools providefeasibilityinthisfixedmodel,not general safety guarantee.'}
    (stage/'evidence/interface-usage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# M11—M12-B实际工具使用与接口证据','',
        '计数来自已发布的case-index与test-groups聚合，没有重新打开旧最终逐样本记录；脚本为report_b2_usage.py。模型在内存中或计算过影子分数，不代表决定了实际行动。','',
        '## 历史M12-B：三个生成重复的合计','',
        '|组|回合引用|PPO工具派单|全部派单|额外补查|无补查回合|CVAE调用|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in aggregate:
        lines.append('|'+ '|'.join(str(row[k]) for k in ('group','case_references','actual_PPO_dispatches','dispatches','extra_checks','zero_extra_case_references','estimator_calls'))+'|')
    lines+=['','每组96个回合引用重复使用相同16种物理过程、两档噪声和三次生成重复。PPO工具计数仍可能包含只有一个合法候选的派单，不能据此声称每次都存在真正选择。','',
        '## 五个必须回答的问题','',
        '1. **何时由PPO选择？** M11学习调度由actor评分合法候选，守卫可能把候选缩到一个；M12-B只有命中action=ppo规则或fallback=ppo时由PPO排序。choose_program仍会先计算actor，供影子比较。B4-B1a只走Tour并用零评分，B1b完全没有新动作选择。','',
        '2. **哪些流程实质走Tour？** 默认PatrolProgram无规则、fallback=tour逐项执行所属巡回。历史模型元组保留流程fallback=tour，没有PPO规则；大部分回合无补查。固定改进器流程有少量补查，不能笼统称其每回合都等于Tour。','',
        '3. **CVAE改变了什么？** 温度prior输出进入current_history并被后续已完成测量覆盖。历史元流程的nominal交叉干预没有改变漏检与等待；本轮一个归档过程五臂history/nominal动作帧与确认结果均相同。调用次数说明实际运行了工具，尚未建立其独立收益。','',
        '4. **哪些原子工具有独立增量？** 当前支持专业公开测量与返回可行性工具的重要作用，但没有单工具消融证明PPO、CVAE或LLM本身带来收益。本轮公开净价值与去历史版本相同；固定流程的微小历史收益来自组合，不能逐项归因给学习器。','',
        '5. **预算是否过度限制适应性？** 是一个明确限制：固定所有权、只有全队一次补查且随后Tour，使多执行体不能持续重分工，学习策略自由度很小。它适合先做公平窄范围基准；不能据此否定多智能体学习。增加额度或跨所有权分配需另立约束与预算，当前不改环境追逐正结果。','',
        '## 本轮归档过程的五臂工程试跑','',
        '|方法|历史模式实际PPO派单|仅影子actor计算|额外补查|CVAE调用|',
        '|---|---:|---:|---:|---:|']
    for row in smoke['rows']:
        if row['estimator']!='history':continue
        u=row['usage'];lines.append(f"|{row['arm']}|{u['actual_PPO_dispatches']}|{u['exclusive_shadow_actor_scoring_calls']}|{u['global_optional_checks']}|{u['estimator_calls']}|")
    lines+=['','归档试跑仅一个已知开发物理过程，5臂×2估计方式。终末1800秒遥测采用明确的共同延续方式，不能称原件完整精确重放或正式新测试。它证明接口能执行且额度可核查，不用于科研选优。','',
        '原试跑把额度用完后的强制Tour也记为PPO；本轮保留旧收据并修正标签，10条动作与测量哈希全部不变。新统计还分出PPO多候选派单、实际选择与仅影子计算。']
    (stage/'接口调用证据.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('B2_USAGE_READY',len(aggregate),len(m11groups),flush=True)


if __name__=='__main__':main()

"""Novice-readable M12-E results; distinguish representation, task and RSI gains."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

NAMES={'Tour':'固定巡回','human_professional':'原人工专业规则','unconditional_PPO':'无条件PPO','evidence_gate_PPO':'证据门控＋冻结PPO','evidence_gate_urgency':'证据门控＋非学习紧迫度'}


def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();s=a.stage;r=s/'evidence/run';root=s.parents[1]
    e0=read(r/'E0/complete.json');e0audit=read(s/'evidence/E0-independent-audit.json');audit=read(r/'E1-independent-audit.json');ledger=read(r/'resource-ledger.json');storage=read(r/'storage-after-export.json')
    need=e0audit['passed'] and audit['passed'] and ledger['all_allocations_completed_exit0'];assert need
    data=[read(r/'E1'/f'seed-{seed}/complete.json') for seed in (33000,33001)]
    mainrows=[v for d in data for v in d['rows'] if v['episode_purpose']=='main'];diagnostics=[v for d in data for v in d['diagnostics']]
    assert len(mainrows)==180 and sum(d['witness_episodes'] for d in data)==40
    methods={}
    for method in NAMES:
        rows=[v for v in mainrows if v['method']==method];frames=[f for v in rows for f in v['decision_frames']];check=[f for f in frames if f['decision']=='CHECK']
        methods[method]={'statistics':audit['independently_reconstructed_main_statistics'][method],'conditions':len(rows),'frame_counts':dict(Counter(v['decision'] for v in frames)),
            'CHECK_count':len(check),'CHECK_times_s':sorted(v['time_s'] for v in check),'CHECK_global_gate_pool_sizes':[v['gate_eligible_candidates'] for v in check],
            'mean_travel_m':statistics.mean(v['metrics']['travel_m'] for v in rows),'mean_service_s':statistics.mean(v['metrics']['service_s'] for v in rows),
            'mean_idle_s':statistics.mean(v['metrics']['idle_s'] for v in rows),'maximum_exact_gap_ratio':max(v['metrics']['maximum_exact_gap_ratio'] for v in rows),
            'mean_CHECK_shift_s':statistics.mean(v['selected_extra_public_evidence']['shift_s'] for v in check) if check else None,
            'PPO_actual_dispatches':sum(v['usage']['actual_PPO_dispatches'] for v in rows),'PPO_multi_candidate_dispatches':sum(v['usage']['PPO_multichoice_dispatches'] for v in rows),
            'PPO_single_candidate_dispatches':sum(v['usage']['PPO_single_legal_candidate_dispatches'] for v in rows),
            'actor_network_calls':sum(v['usage']['actual_actor_network_calls_in_selector'] for v in rows),
            'CVAE_calls':sum(v['usage']['estimator_calls'] for v in rows),
            'professional_confirmation':{role:{'event_records_noise_repeated':sum(v['by_role'][role]['segments'] for v in rows),'confirmed_records_noise_repeated':sum(v['by_role'][role]['confirmed'] for v in rows)} for role in ('thermal','electrical','cooling','control')}}
    first=[]
    for row in mainrows:
        if row['method']!='unconditional_PPO':continue
        f=next(v for v in row['decision_frames'] if v['decision']=='CHECK');agent=f['selected_extra']['agent'];j=f['selected_extra']['task'];g=next(v for v in row['first_root_actor_scores'] if v['agent']==agent)
        selected=g['ids'].index(j);base=g['ids'].index(g['baseline_next'])
        first.append({'specification':row['specification'],'CHECK_time_s':f['time_s'],'agent':agent,'selected_task':j,'baseline_task':g['baseline_next'],
            'selected_actor_score':g['scores'][selected],'baseline_actor_score':g['scores'][base],'argmax_margin':g['scores'][selected]-g['scores'][base],
            'candidate_count':len(g['ids']),'selected_features':g['features'][selected],'baseline_features':g['features'][base],'feature_names':row['feature_names']})
    candidate=next(v for v in mainrows if v['method']=='evidence_gate_PPO' and v['specification']['seed']==33000 and v['specification']['scenario']=='coldplate_fouling' and v['specification']['fault_start_s']==885 and v['specification']['noise_c']==.2)
    example=next(v for v in candidate['decision_frames'] if v['decision']=='CHECK')
    base=next(v for v in mainrows if v['method']=='Tour' and v['specification']==candidate['specification'])
    example_out={'specification':candidate['specification'],'public_decision':example,'missed_delta':candidate['metrics']['missed_segments']-base['metrics']['missed_segments'],
        'weighted_wait_delta_s':candidate['metrics']['consequence_weighted_censored_delay_s']-base['metrics']['consequence_weighted_censored_delay_s'],'scope':'Post-hoc illustrative development example, not a selected evaluation method.'}
    opportunity={'conditions_with_any_analytic_useful':sum(v['analytic_useful_candidates']>0 for v in diagnostics),
        'conditions_with_actual_public_Top1_useful':sum(v['public_Top1_actual_beneficial'] for v in diagnostics),
        'conditions_with_only_offline_useful':sum(v['analytic_useful_candidates']>0 and not v['public_Top1_actual_beneficial'] for v in diagnostics),
        'actual_witnesses':40,'analytic_actual_mismatch_count':sum(not w['analytic_matches_actual'] for v in diagnostics for w in v['actual_witnesses']),
        'seed_process_witnesses':audit['public_witness_processes'],'noise_versions_not_independent':True}
    passed=e0['E0_passed'] and audit['E1_opportunity_gate'] and audit['E1_timing_gate']
    decision={'E0':'PASS_format_and_legal_protocol_representation_only','E1_opportunity':'PASS' if audit['E1_opportunity_gate'] else 'NO-GO',
        'E1_timing_learning_increment':'PASS' if audit['E1_timing_gate'] else 'NO-GO','E2':'LIMITED GO_prepare_separate_execution_gate' if passed else 'NO-GO',
        'E2_executed':False,'reasons':[] if passed else ['Opportunityrepeatability gate requires>=2publicbeneficialnumericprocessesinEACHseed;observed4and1.',
            'GatedPPO equals nonlearninggatedurgency;12actualCHECKs eachhadonly1eligibleoption,0PPOmultichoiceauthority.'],
        'scope':'Current frozen E2 launch route only; not a universal prerequisite against future experience-driven timing skills. New question requires a new preregistered design, not relabelling this outcome.'}
    interface_costs={}
    for method in ('full_nested','compact_leaves'):
        calls=[read(p) for p in (r/'E0/calls').glob('*--'+method+'.result.json')]
        interface_costs[method]={'calls':len(calls),'input_tokens':sum(v['input_tokens'] for v in calls),'output_tokens':sum(v['output_tokens'] for v in calls),
            'generation_wall_s':sum(v['generation_s'] for v in calls),'allocation_time_split':'Sharedloadandallocationcostnotassignedperinterface;total195GPUallocatedseconds reportedseparately.'}
    results={'stage':'M12-E','decisions':decision,'E0_actual':e0,'E0_interface_costs':interface_costs,'E1_methods':methods,'E1_opportunities':opportunity,'E1_timing_comparisons':audit['E1_timing_comparators'],
        'unconditional_PPO_first_action_evidence':first,'public_wait_then_CHECK_example':example_out,'resources':ledger,'storage_after_export':storage,
        'AI_attribution':{'manual_timing_gate':'Human-designed and frozen before new data; not PPO timing learning or RSI.',
            'PPO_ranking':'No multichoice ranking at12actualgatedCHECKs; allteamgatepools also1. No independentlearningincrement.',
            'CVAE':'SamefrozenCVAEall5arms; no causalestimatorablation/currentclaim.',
            'meta':'E0 tests legal representation only; no task adaptation/meta-improver search or effect tested.'}}
    write(s/'evidence/E-results.json',results);write(s/'evidence/E2-decision.json',decision)
    lines=['# Zhinengti-RSI：M12-E进度、结果与下一步','',
        '核对日期：2026年10月31日。基于提交6174f2ab。本文把本轮结果、原始科研目标、仍未解决的问题和后续协议集中说明；完整机器证据与此前历史另汇入一个JSON。','',
        '## 1. 现在到了哪里','',
        '**两个接口阻塞已得到实际处理：模型能直接生成合法的完整元协议，巡检系统也能先保留额度、出现公开依据后再补查。当前仍没有证明独立的PPO排序增量或递归改进增量。**','',
        'E0完成24次真实Qwen调用。完整嵌套接口12次原始回复全部合法、全部不同于父协议，共9种不同合法协议；组装接口也通过门槛。E1完成18个新物理过程、36个观测条件、180次五组闭环和40次离线见证闭环，另做独立核对。','',
        '**E2决定为NO-GO，当前不启动下一轮正式验证或元搜索。** 两个原因很具体：公开规则可识别的有效机会在第二个种子只有1个物理过程，未到事先要求的2个；门控PPO与非学习紧迫度完全相同。12次实际门内补查都只有1个合格候选，学习排序没有发挥余地。','',
        'NO-GO适用于本轮冻结的启动路线。将来仍可研究“从经验中改进何时保留、何时使用额度”；需要围绕该技能另立问题和新协议，继续保留强基线与独立验证。当前结果不会通过降低阈值、加候选或重选种子来改变。','',
        '## 2. 与导师最初想法怎样对应','',
        '|导师／论文主线|本轮实际推进|尚未证明|','|---|---|---|',
        '|结合设备特征，决定何时去哪里、检查什么|保留18名异质执行者、专业任务、局部观测和一次额度；明确分开保留／使用与目标排序|未知机房、更多协作自由度或现场风险改善|',
        '|持续覆盖并避免长期盲区|全部方法共用受保护巡回、专业资格、完整返回和精确重访要求|覆盖约束由共同执行核保障，不能算AI独立发现的能力|',
        '|原论文的部分观测估计＋多执行者学习|冻结CVAE提供状态估计，冻结PPO评分实际合法候选|本轮共用CVAE，没有其因果消融；PPO门内没有多候选排序机会|',
        '|智能体根据经验改进技能和工具使用|新接口能生成合法且不同的诊断／记忆／提案协议，能在迁移时清空任务状态|E0只检验表达与合法性，没有验证经验带来的任务增量|',
        '|嵌入RSI，改进产生下一次修改的方法|后续合法元改进设计已分出任务收益、元收益和全部准备成本|本轮没有运行元搜索，也没有证明递归能力提升|','',
        '原参考论文研究水域多船巡检，采用变分状态估计与共享Dueling DDQL；当前机房版本是专业任务与PPO的适配。两者的代码和场景不同，不能直接用原论文成绩作为本项目已复现或超越的证据。','',
        '## 3. M12-D的结论完整保留','',
        'M12-D五臂开发和验证各180条、逐条重放完成，修订候选没有超过Tour，五臂最终源仍未生成。RSI曾真实运行78次正式调用与1次单列预检，最终测试已经使用；6份元协议提案全部非法，实际沿用旧协议，正式结论NO-GO。','',
        '本轮只读取原始公开报告、元提案回复与来源标识。没有打开旧私有最终逐行记录、没有向旧实验追加候选或补算新回复，也没有改动旧权重、协议、结果和测试原件。','',
        '## 4. 六份旧元回复究竟错在哪里','',
        '旧回复并非不会写JSON，而是没有写对嵌套层级。外层有三个字段，其中protocol的值还应包含一个内部protocol：','',
        '```text','最外层：diagnosis、protocol、expected_effect',
        '  外层protocol = ControlledImprover','    内部protocol = Improver','      diagnostics：1至3个不同的合法工具','      memory：原枚举','      instruction：三个固定模板之一',
        '    generator、dedup、acceptance、failure_fallback：各自原枚举','```','',
        '|旧重复|调用号|JSON语法|嵌套层级|诊断项数|完整合法|','|---|---|---|---|---:|---|']
    for v in read(s/'evidence/old-six-meta-diagnosis.json')['rows']:lines.append(f"|{v['trial']}|{v['call']}|通过|缺内部protocol|{v['checks']['diagnostic_count']}|未通过|")
    lines+=['','六份回复的诊断工具名称、记忆、指令和枚举本身均在允许范围，但层级不完整，且诊断项有4或5个，超过上限3。完整原文、原文件SHA和逐字段诊断都保留在集中JSON中，没有替它们补层级、删字段或重新计分。','',
        '## 5. E0实际测了什么','',
        '事先固定两个接口，各12次、共24次调用；使用相同Qwen3-8B修订和非思考生成参数。3个父协议、3组明确标注为合成的公开格式挑战、24份提示和生成种子均在调用前封存。没有把机房隐藏事件、物理种子或最终指标放进提示。','',
        '|接口|原始JSON合法|原始完整协议合法|结构转换后合法|合法且不同于父协议|不同合法协议数|重复输出|','|---|---:|---:|---:|---:|---:|---:|']
    for method,name in (('full_nested','完整嵌套提示'),('compact_leaves','字段索引＋结构组装')):
        v=e0['summaries'][method];lines.append(f"|{name}|{v['raw_JSON_valid']}/12|{v['raw_full_ControlledImprover_valid']}/12|{v['compiled_protocol_valid']}/12|{v['valid_and_different_parent']}/12|{v['distinct_nonparent_legal_protocols']}|{v['duplicates']}|")
    lines+=['','通过门事先设为每接口至少11次完整合法、8次合法且不同于父协议、3种不同合法协议，且变更都在白名单内、迁移重置全部通过。两组都通过，按预定优先顺序保留“完整嵌套提示”作为后续接口。重复、非法或超时都占槽，不免费重试。','',
        '结构组装接口的原始输出是另一种字段格式，所以“原始完整协议”列为0；它并非失败后被修好。模型选择所有叶字段，程序只提供嵌套结构，编译后的合法率不能当成模型掌握嵌套或自改进的证据。完整嵌套组的12份原始回复则直接通过原验证器。','',
        '这个结果只支持本组清晰提示下的格式能力。旧6次使用真实任务上下文，新12次使用合成格式挑战，不能把两次合法率之差当作提示改动的配对因果效应。12个样本也不等于普适可靠性保证。','',
        '每个合法对象都验证了允许的变更字段，并调用原迁移重置接口，得到默认Tour、空历史、空候选档案和空任务结果。没有保留任何任务程序，也没有执行任务适配或元搜索。','',
        '## 6. E1怎样让系统表达“保留额度”','',
        '每个回合全队仍只有一次额外检查。保留额度表示继续原巡回，没有原地停着等；这样仍会完成专业检查和重访义务。系统先看“现在是否值得用额度”，门打开后才比较“检查哪个候选”。','',
        '门使用事先固定的公开专业线索与测量可用性，并记录测量年龄；当前门条件本身不按测量年龄调整阈值。它还要求比继续Tour更早完成，且固定的线索收益代理超过顺延成本代理。它不读真实故障起点或未来事件标签，也没有“等到335秒／705秒”的规则。','',
        '五组是Tour、原人工专业规则、无条件PPO、证据门控＋冻结PPO、相同证据门控＋非学习紧迫度。门是人工设计的；两组门控的候选条件完全相同，只替换内部排序。共享轮转顺序仍决定哪位合格执行者先获得补查机会。','',
        '新开发种子为33000、33001；故障相位在生成前独立固定在95至895秒的5秒网格，排除旧四个固定相位。未来验证33020–33023、最终33040–33043只在清单登记，当前未生成。模型、人员、专业资格、测量、时限和物理方程均不改。','',
        '## 7. 五组实际结果','',
        '漏检是每个物理过程的宏平均数，不是百分比。等待包含未发现事件的结束截尾，严重事件权重更高。先平均两档噪声，再平均物理过程，再平均两个种子；健康条件计入漏检、成本和安全，没有事件则等待为空。','',
        '|方法|宏漏检数|宏加权删失等待秒|CHECK条件数／36|真实PPO多候选派单|真实PPO单候选派单|','|---|---:|---:|---:|---:|---:|']
    for method,v in methods.items():
        stat=v['statistics'];lines.append(f"|{NAMES[method]}|{stat['macro_missed_count']:.6f}|{stat['macro_wait_s']:.3f}|{v['CHECK_count']}|{v['PPO_multi_candidate_dispatches']}|{v['PPO_single_candidate_dispatches']}|")
    lines+=['','相对Tour，门控组的宏漏检减少0.222222，等待减少7.142秒；相对原人工规则，宏漏检减少0.055556、等待减少0.465秒。但两种门内排序的结果完全相同，所以这些改善只能归为当前固定证据门与调度组合的描述性收益。','',
        '**12次实际门控补查时，全队通过该门的候选数都只有1个；PPO的多候选排序次数为0。** 所以本数据没有给学习排序提供可区分的选择，不能从相同结果断言PPO普遍无效，也不能声称它带来了增量。','',
        '|种子|Tour漏检／等待|人工规则漏检／等待|两种门控共同漏检／等待|两档噪声都验证有效的公开见证过程数|','|---|---|---|---|---:|']
    for seed in (33000,33001):
        values=[next(x for x in methods[m]['statistics']['seeds'] if x['seed']==seed) for m in ('Tour','human_professional','evidence_gate_PPO')]
        cells=[f"{v['macro_missed_count']:.6f}／{v['macro_wait_s']:.3f}s" for v in values]
        lines.append(f"|{seed}|{'|'.join(cells)}|{audit['public_witness_processes'][str(seed)]['count']}|")
    lines+=['','巡检侧成本也保留在同一预算下：','',
        '|方法|每回合平均移动米|平均检查服务秒（全队）|发生CHECK时平均排班顺延秒|','|---|---:|---:|---:|']
    for method,v in methods.items():
        shift=f"{v['mean_CHECK_shift_s']:.3f}" if v['mean_CHECK_shift_s'] is not None else '无CHECK'
        lines.append(f"|{NAMES[method]}|{v['mean_travel_m']:.3f}|{v['mean_service_s']:.3f}|{shift}|")
    lines+=['','回合固定1800秒，补查可能使末尾某项检查来不及完成，因此服务时长差也可能为负；不能把它直接解释成现实人力节约。全部资格和首尾重访仍须满足共同约束。完整分专业有效确认、精确间隔、空闲和实际工具调用在JSON中保留。']
    lines+=['','## 8. 一个实际的“先保留、后使用”例子','',
        '以下是事后展示的一个开发条件，不用于挑选方法：种子33000、冷板污堵、故障相位885秒、噪声0.2。门控策略此前一直继续Tour，在1005秒使用额度。','',
        '当时可用的融合温度约69.37°C，最近测温年龄为0秒，上次CVAE更新距今15秒。候选检查预计1035秒完成，继续Tour预计1165秒才完成，可提前130秒；额外绕行使后续任务约顺延45秒。候选满足专业资格、返回计划和固定门条件。','',
        '该条件的实际结果相对Tour少漏1段、加权等待少45秒。该例说明接口确实执行了保留后补查；总体是否稳定仍要看全部条件与预注册门，不能由这个例子替代。','',
        '## 9. 为什么原PPO仍在第5秒用掉额度','',
        '36个条件的无条件PPO都在5秒补查，结果与Tour相同。记录保留了首次全部候选的16维特征、actor分数、巡回候选分数和被选目标；独立检查确认它实际选择的是当时分数最大者。','',
        '结构上的原因已经明确：现有actor用于任务排序，输入没有“本回合唯一额度还剩多少”的字段，也没有独立的保留／使用决策头。选择器直接对合法任务取最大分，再由外层一次额度门扣除。Tour是候选之一，但它并未按当前一次额度问题专门学习等待时机。','',
        'actor分数表示候选排序倾向，不是故障概率。本轮没有新训练概率预测器，也没有把CVAE温度估计直接当成可靠的异常概率。','',
        '分数差能说明当时为何选了该目标，不能单凭特征数值断言某个特征造成了错误。当前不重训、不临时调模型；人工证据门解决了表达与触发问题，尚未把时机判断变成经验证的学习能力。','',
        '## 10. 离线机会与真实净收益怎样区分','',
        f"逐一核对{audit['audit_counts']['public_roots']}个Tour公开决策根和{audit['audit_counts']['public_candidates']}个合法额外检查候选。离线Oracle只在五组决策封存后使用未来事件标签，不能进入在线规则、actor或Qwen提示。",'',
        '30个噪声条件存在离线有益候选；固定公开排序能给出实际有益见证的只有10个条件。另20个条件有离线机会，但当前公开门／排序没有识别到；这不等于证明信息在理论上不可观测。','',
        '每条件最多预先选两个见证：离线最优有益候选，以及最早由固定公开排序提出且事后有益的候选；重复则只跑一次。实际执行40条“Tour前缀→一次合法CHECK→Tour”闭环，逐事件核对发现、丢失、等待和首次确认，40条均与解析诊断一致。没有看到结果后另选见证补跑。','',
        '其余候选只是解析反事实，不能称298152条都做了真实闭环。所有“如果先保留、以后会遇到机会”的计数都基于继续Tour的反事实路径，也不等于在已经绕行后的真实排班上仍可同时执行。','',
        '## 11. E2门为什么没有通过','',
        '|门|预先要求|实际结果|决定|','|---|---|---|---|',
        '|E0合法表达|每接口≥11/12合法、≥8/12合法且不同、≥3种不同协议，重置和成本通过|两接口均通过；按优先顺序选完整嵌套|PASS，仅格式能力|',
        '|E1机会可重复性|每个种子至少2个不同物理过程在两档噪声下都有实际有益公开见证|33000为4个，33001为1个|NO-GO|',
        '|E1时机／学习增量|保留后使用在两种子出现；各种子不增漏，且分别胜Tour、人工规则、门控紧迫度|前两项对照改善；与门控紧迫度完全相同|NO-GO|',
        '|E2启动|任务启动需要E1两门；元启动还需要E0和后续独立任务确认|E1未通过|NO-GO；未生成验证／最终源、未运行元搜索|','',
        '这些门是分别回答问题的：合法格式不能代替任务收益，门控优于巡回也不能代替学习排序或递归增量。当前按冻结规则停止，没有降低阈值、扩大任务集合、放宽重访或追加新种子追求通过。','',
        '## 12. 下一轮独立协议已准备到什么程度','',
        '任务级协议保留这五种固定方法，登记4个新验证种子和4个最终种子。主要对照同时包含Tour、原人工规则和非学习门控；验证各组完整重放与审计通过后才可开最终集，最终还需独立满足同一效果门。上限720条主要回合＋720条重放，当前均未启动。','',
        '合法元改进器的未来设计使用新的34000／34100系列划分，固定8候选、保存前三前缀，与元开发后新任务3候选比较。每重复最多16次模型调用，3次重复最多48次；环境上限5292次。元准备成本与新任务机会分别报告，随机搜索和人工／紧迫度参照保留。','',
        '未来缓存把执行器标签、程序、源和模型指纹一起作为身份。原人工与原紧迫度执行器不会因共享默认程序JSON而被错误合并或改用新解释器。任务历史与经验分别清空；相同分区的评价缓存可共享，分区之间绝不共享。','',
        '这些协议已经是可审查的设计，执行器实现、完整回归、独立源和实际启动门仍属于后续阶段。当前E2决定NO-GO，所以没有运行这些48次调用，也没有重新使用M12-D最终集。','',
        '## 13. 当前瓶颈与还不能确定的事','',
        '|问题|现在有直接证据的部分|仍不能下结论的部分|合理下一步|','|---|---|---|---|',
        '|合法协议表达|新完整提示12/12原始回复合法，解决了旧层级错误|复杂真实任务上下文下是否同样可靠；协议是否有用|保持完整Schema，后续另立小规模真实上下文预检与效果对照|',
        '|保留额度|系统能沿Tour保留，12次门控CHECK发生在180至1005秒|智能体是否能从经验学会更好的触发时机|把可改技能明确放在时机选择，不能把人工门称为自改进|',
        '|学习排序缺少区分度|实际门内候选全为1，PPO与紧迫度结果一致|候选较多时PPO是否更好；全队协作排序是否有价值|先核查新开发任务中公开可区分的合法选择，保持硬约束，不以降低阈值制造通过|',
        '|公开线索识别不稳定|有离线机会的30条件中仅10有公开有益见证；第二种子仅1过程重复|是观测不足、门过粗、时机或目标定义错位，各占多少|在既有开发记录中分层诊断，区分缺观测与排序未识别；不动旧最终数据|',
        '|递归增量|格式与迁移重置通过，但尚无真实合法元搜索收益|合法元更新能否提高同预算新任务改进效率|任务目标与机会先明确，再按新独立协议验证；当前不启动昂贵搜索|',
        '|现场与论文贡献|仿真、专业检查和审计接口可复核|真实机房阈值、规程、现场收益及可发表的新机制|与导师核对真实约束；贡献落到可验证机制和独立效果，不以模型名称或测试数量代替|','',
        '## 14. 实际成本与交付完整性','',
        f"本轮6个Slurm分配全部正常结束，累计{ledger['allocated_CPU_minutes_inclusive']:.2f}分配CPU分钟。GPU仅E0一张V100S分配{ledger['GPU_seconds']}秒；24次真实调用共{e0['input_tokens']}输入、{e0['known_output_tokens']}输出tokens，没有超时、未知token或免费重试。E1主回合180、见证40，审计新增模型调用和环境回合均为0。",'',
        f"包含代码、物理源、原始回合和发布包的新目录约{storage['bytes_all_new_run_files_after_export']/1e9:.3f}GB，小于3GB上限。317项真实Torch回归和96项轻量CPU检查通过，工程测试不能替代科学效果。CVAE／PPO／Qwen权重没有更新。",'',
        '独立核对覆盖220条原记录哈希和事件算术、12924个公开根、298152个候选、公开门与排序、真实Tour前缀和40条见证结果。每回合另已执行专业资格、路径、参考测量、完整返回证书和初始／末尾重访的独立检查；本轮没有再执行第二遍神经策略重放。','',
        '完整嵌套组实际使用10712输入、1592输出tokens，纯生成约81.71秒；结构组装组为9524输入、1186输出tokens，约61.07秒。模型加载与分配开销共享，未虚构两组各自的完整GPU占用；总成本以195秒Slurm分配为准。本地检查与文件准备没有折算成Slurm核分钟。','',
        '大体积物理数组、完整原始回合和逐根公开向量留在集群，以路径、来源SHA和已核对的摘要保留。集中JSON包含本报告、所有发布证据、完整E1开发元数据及此前全量历史；E1重复行字段用可逆列式表格压缩，提供解码和逐值还原核对。','',
        '## 15. 给导师的简短汇报','',
        '> 本轮先解决了可表达性：完整元协议提示12次均合法，巡检也能保留唯一额度后再依据公开线索补查。门控方案在新开发条件上减少了漏检，但12次触发都只有一个候选，PPO与非学习排序完全相同；公开机会在第二个种子还不够稳定，因此未通过预注册E2启动门。下一步应围绕经验驱动的时机技能明确学习目标和可区分机会，再验证合法元改进器的独立收益。本轮不会继续追加搜索。','',
        '- [单文件全量JSON](2026-10-31-M12-E-项目全量汇总.json)',
        '- [E0实际回复与独立复核](../research/2026-10-31-m12e-timing-schema/evidence/E0-independent-audit.json)',
        '- [E1实际指标、动作与时机归因](../research/2026-10-31-m12e-timing-schema/evidence/E-results.json)',
        '- [E1独立核对与冻结门](../research/2026-10-31-m12e-timing-schema/evidence/run/E1-independent-audit.json)',
        '- [未来任务级协议](../research/2026-10-31-m12e-timing-schema/E2-Task-Paired-v1.json)；[未来合法元改进设计](../research/2026-10-31-m12e-timing-schema/E2-Legal-Meta-v1.json)',
        '- [M12-D完整历史报告](2026-10-30-M12-D-完整进度与瓶颈.md)','']
    report=root/'reports/2026-10-31-M12-E-完整进度与瓶颈.md';report.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('M12E_REPORT',decision['E2'],len(mainrows),opportunity,flush=True)


if __name__=='__main__':main()

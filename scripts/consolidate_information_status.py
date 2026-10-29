"""Write a self-contained, evidence-bound B4-A report and project JSON."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/2026-10-29-m12c1b4a-information'
EV = BASE / 'evidence'
NAMES = {
    'no_net_helpful_current_action': '当前没有净有益合法补查',
    'helpful_current_lacks_positive_available_unalarmed_signal': '有益动作缺少可用正线索或被已告警排除',
    'helpful_current_rejected_by_cost_threshold_or_veto': '有正线索，但被代价／门槛／守卫拒绝',
    'current_proxy_ranks_nonhelpful_action': '当前代理把非有益动作排在前面',
    'CHECK_executes_net_helpful_action': '实际CHECK执行有益动作',
    'WAIT_defers_net_helpful_current_action': '当前最优动作有益，仍选择WAIT',
}


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + '.gz')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def write(name, value):
    (BASE / name).write_text(value.strip() + '\n', encoding='utf-8', newline='\n')


def main():
    r = read(EV / 'report/aggregate.json')
    independent = read(EV / 'report/independent-verification.json')
    resources = read(EV / 'slurm-cost-review.json')
    stage_gate = read(EV / 'final-gate/full-tests.json')
    if independent['independent_verification'] != 'passed' or r['original_cases_bound'] != 768:
        raise RuntimeError('Full original-data verification has not passed')
    frames = []
    with gzip.open(EV / 'report/all-revised-frames.jsonl.gz', 'rt', encoding='utf-8') as stream:
        frames = [json.loads(line) for line in stream]
    c0 = [v for v in frames if v['alias'] == 'C0']
    layers = Counter(v['layer'] for v in c0)
    relevant = len(c0) - layers['no_net_helpful_current_action']
    weak = layers['helpful_current_lacks_positive_available_unalarmed_signal']
    percent = 100 * weak / relevant if relevant else 0.
    missing = sum(v['helpful_missing_signal_count'] for v in c0)
    zero = sum(v['helpful_zero_signal_count'] for v in c0)
    alarm = sum(v['helpful_known_alarm_excluded_count'] for v in c0)
    by_phase = {}
    for phase in ('development', 'validation'):
        own = [v for v in c0 if v['specification']['phase'] == phase]
        by_phase[phase] = dict(Counter(v['layer'] for v in own))
    diagnostic = {'C0_frames': len(c0), 'C0_layers': dict(layers), 'C0_by_phase': by_phase,
        'C0_frames_with_helpful_current_action': relevant,
        'C0_helpful_frames_no_positive_candidate_fraction': weak / relevant if relevant else None,
        'C0_helpful_candidate_occurrences_missing_signal': missing,
        'C0_helpful_candidate_occurrences_zero_signal': zero,
        'C0_helpful_candidate_occurrences_alarm_excluded': alarm,
        'scope': 'Original C0 development/validation only. Frame and candidate counts are repeated observations, not independent anomalies or causal error fractions.'}
    save(EV / 'interpretation-review.json', diagnostic)
    layer_table = '\n'.join(f'| {NAMES[key]} | {layers[key]:,} |' for key in NAMES)
    fusion = r['thermal_fusion_unique_and_tied_positive_counts']
    cache = r['thermal_raw_and_assimilated_cache_provenance']
    selected = r['error_statistics'].get('selected_future/C0', {})
    write('运行与归因结果.md', f'''
# B4-A：可用动作存在，主要缺口首先出现在风险线索，其次在评分和等待

本轮完成原768条正式记录、64份物理数组的绑定，并从旧动作与冻结模型恢复128个条件、{r['old_public_history_frames_reconstructed']:,}帧巡回公开历史。核对{r['original_public_SHA_frames_verified']:,}个原有公开状态散列；每条件8,282个硬约束合法补查，共{r['hard_legal_actions_across_conditions']:,}个动作，与更早的公开历史组合成{r['factorized_candidate_root_pairs']:,}个候选—时刻对。完整因子保存在集群，所有紧凑统计和两份完整历史在本地交付。

原动作重放只恢复观测，不选择新策略或新实例。包括预审计，共132次旧巡回历史恢复，其中128次为修订版全量；没有新的神经训练、Qwen调用、策略搜索或独立测试。B3.2的NO-GO与最终Tour回退保持原结果。

## 1. 准确信息是否可能产生更好的合法动作？

在{r['conditions_with_net_helpful_legal_check']}/128个旧条件中，至少存在一个“检查一次然后按原队列巡回”的净有益动作；共{r['net_helpful_legal_checks']:,}个有益、{r['net_harmful_legal_checks']:,}个有害动作。它们相互重叠且共享异常，不能全部执行，也不代表在线系统知道该选哪一个。

因此，旧任务里的收益空间并非处处为空。当前证据不支持把失败完全归因于唯一一次预算；是否扩大预算或重排任务，仍需独立设计。这里的“有益”按漏检优先、同漏检再比较加权删失等待，不是已标定业务损失。

## 2. 主要错误在哪一层？

下表只取原C0的64个开发／验证条件，避免将四候选重复算成四份独立证据。共{len(c0):,}个尚未耗尽额度的原决策帧：

| 诊断层次 | 帧数 |
| --- | ---: |
{layer_table}

当前有益动作存在的{relevant:,}帧中，{weak:,}帧（{percent:.2f}%）没有可接受的正风险线索；另有{layers['helpful_current_rejected_by_cost_threshold_or_veto']:,}帧被代价／门槛／守卫拒绝，{layers['current_proxy_ranks_nonhelpful_action']:,}帧当前排序不理想，{layers['WAIT_defers_net_helpful_current_action']:,}帧因未来最大代理选择等待。实际执行{layers['CHECK_executes_net_helpful_action']}帧有益CHECK。

这表明，“等待价值高估”是实在的问题，**当前估计未给有益动作提供正线索，是本批原决策帧更常见的阻断层**。这些是分层诊断频次，不能直接解释成每种机制造成了多少漏检。缺少正线索也不等于信息论上不可预测：可能是缺测、模型漏估、阈值或观测利用方式造成。

有益候选发生次数另计：缺少信号{missing:,}次、信号为零{zero:,}次、已有告警排除{alarm:,}次；同一帧可有多个候选，这些次数不作为上述帧数百分比的分母。

## 3. 未来最大值和融合有什么表现？

全部合法热候选—时刻对中，融合正信号{fusion['positive_fused_pairs']:,}次，其中{fusion['positive_fused_reference_zero']:,}次对应完成时刻参考信号为零。正线索主要由历史有界趋势独立胜出：{fusion['measured_bounded_trend:positive_unique_win']:,}次；名义模型独立胜出{fusion['nominal:positive_unique_win']:,}次。

原C0保存的最优未来点共{selected.get('n', 0):,}个，正预测而参考零{selected.get('false_positive', 0):,}个，信号MAE为{selected.get('MAE', 0):.5f}。本轮同时补上了未胜出候选，避免只对最优分数做结论；完整误差按专业与预测完成视野分层。

![全候选预测视野与误差](evidence/report/b4a-horizon-error.png)

读图：横轴是从当前到预测检查完成的时间；左图越高表示风险信号越高估，右图表示正预测中真实参考为零的比例。fused是最大值融合，history cache是覆盖缓存，nominal是物理名义估计，bounded trend是有界趋势，raw CVAE是未覆盖的CVAE先验输出。远期误差应与异常召回一起看；很少报异常也可能得到较低平均误差。

信号是连续代理，尚不是异常概率；此处不计算伪装成概率的Brier分数。不同来源可用量不同，逐来源误差的分母在完整统计中保留。重复根时刻、候选、噪声和种子存在依赖，图上的海量点不构成海量独立样本。

## 4. CVAE这次提供了什么独立信息？

热任务历史帧{cache['thermal_task_frames']:,}个，原始先验可用{cache['raw_prior_available']:,}个，融合缓存可用{cache['history_available']:,}个。缓存与原始值相同{cache['history_value_equals_raw_prior']:,}次、不同{cache['history_value_differs_from_raw_prior']:,}次。时间戳匹配测量和先验的计数可重叠，不能当互斥来源比例。

history_assimilated在正信号中独立胜出{fusion['history_assimilated:positive_unique_win']:,}次，参与平局{fusion['history_assimilated:positive_tie_member']:,}次（平局容差1e−10）。所有{r['old_public_history_frames_reconstructed']:,}帧都具备可用名义温度；端点界检查发现可由历史缓存独立提高正信号的帧为{r['history_positive_influence_roots']:,}，信号比较容差1e−12。在这批保存的巡回历史上，该融合分量没有产生可检测的额外风险信号，固定评分因此缺少相应的新增信息。

这比仅看显示的胜出来源更具体，范围仍是**冻结模型、旧任务与固定历史上的输入／评分支配分析**。它不替代新训练模型或其他观测条件下的闭环估计器消融，也不说明VAE这一模型类别无价值。cvae_raw本来未直接进入旧task_signal，独立保留它是为了检验原始输出及后续预测目标。

![当前检查与未来等待关系](evidence/report/b4a-check-wait.png)

读图：横轴是当前最优可接受CHECK分数，纵轴是未来WAIT分数；虚线两侧表示二者大小关系。灰点表示最终未补查的WAIT，蓝点表示后来有补查的WAIT，橙叉标记当前最优动作事后有益却仍等待，绿星为实际CHECK。两个面板是原C0／C1，点会重叠。

图中“事后有益”由离线评价器计算，不作为公开输入。C0/C1实际决策只来自开发与验证；最终测试执行的是Tour回退，没有虚构C0/C1测试成绩。

## 5. 核验强度与限制

全部{r['actual_original_branches_verified']}条原实际补查的方法记录，解析计算与原逐异常确认、漏检和加权等待一致。独立程序又直接移时原检查事件、调用原专业测量与告警函数，对全部{independent['counts']['analytic_actions']:,}个合法动作、{independent['counts']['all_event_confirmation_labels']:,}个动作—异常标签核对通过；它不复用主分析器的分组移时／搜索计算。

{stage_gate['tests']}项真实Torch完整测试无跳过、错误或失败。初版评价遗漏热测量噪声、单候选JSON未定义指标导出问题均已修复，并保留版本与测试记录。最终结果仍来自合成机房，无现场标定。

单次检查之后的动作被固定为原巡回，所以后续观测虽然改变，却不会改变处方动作；这使移时评价成立。若后续换成多次自适应动作或检查影响设备状态，必须真实模拟观测和策略反馈，不能沿用这套代数标签冒充闭环。
''')
    write('B4-B下一阶段方案.md', '''
# B4-B建议：先预测有效检查机会，再谈新策略

B4-A已把三个问题分开：一部分旧条件存在有益合法动作；当前模型常未给这些动作正线索；趋势和未来最大代理又会高估其他机会。下一阶段建议优先验证“哪些公开历史能预测未来合格检查取得确认”，并保留完整回合价值作为后续独立问题。

## 先写清标签和信息边界

以合法完成时刻为预测位置，热检查、接点热、流量和控制诊断分别按专业定义。标签要求真实状态达到原条件，并考虑合格测量能否告警；参考时刻、原异常段和完整确认规则分别保留。30／60／120秒预测是起步候选视野，具体取样和总数据量在新协议中冻结。

同一公开历史上比较经验频率、简单逻辑模型或轻量时序模型与现有趋势／物理／CVAE线索；先确认信号的增量信息，再决定是否训练异常敏感CVAE。不能把0至大于1的代理值直接叫异常概率，也不能用传感器误差、后果权重和概率三个不同单位混成一个标签。

## 独立门槛

新开发与验证过程均另选并事先冻结。旧8500至8703属于开发诊断资料；不把噪声、不同方法和同一物理过程的多个时间点当独立训练或测试样本。由过程分组划分并报告分层结果，冻结误报预算后看异常召回，同时看概率校准、视野退化、专业差异与额外测量成本。

只有输入可获得、标签覆盖清楚且验证中有可靠增量信息，才进入B4-C的小规模整回合反事实和新闭环比较。如果现有信号仍无法支持可靠预测，明确记录其证据范围，转向主动感知／传感器信息价值；不宣称已经证明所有可用信息都不足。

B4-C仍需分开目标新增确认、其他任务失去确认、加权等待、资源代价和重访限制；与未来信息有关的离线最佳动作只用于标签。动作空间扩展、PPO重训和RSI恢复均另立协议。

当前这是一份**设计方案**。新数据种子、训练次数、候选上限、评价阈值和算力预算尚未冻结，训练与新闭环尚未提交。下一步先与导师确定主要论文切口、原论文复现范围，以及真实短异常／提前发现的处置价值。
''')
    write('复现与数据说明.md', '''
# B4-A复查入口

全部768条旧记录由原字节SHA绑定；本轮保存完整128份公开历史、共同几何候选和逐动作解析结果。全量公开历史在集群根`/share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b4a/formal`，全部文件位置、字节数及SHA在[evidence/raw-storage.json](evidence/raw-storage.json)。本地保存全部几何、解析结果、逐帧摘要、条件统计及两份完整公开历史。

每个历史359帧，几何表每条件8,282个硬合法动作；在根时刻t只使用派单时刻≥t的候选。状态文件与几何共同构成全部候选的无损分解，而不是抽样。缺测、零信号、已告警、负分和守卫拒绝均能还原。

使用Python3.11和NumPy，先设置PYTHONPATH为src。以下示例还原旧开发正例；输出路径需不存在：

```powershell
$env:PYTHONPATH = "$PWD\\src"
python scripts/inspect_information_candidate.py --folder research/2026-10-29-m12c1b4a-information/evidence/formal/development-0 --condition 14 --root-time 750 --candidate-index 3631 --candidate C0 --output candidate-new.json
```

输出将公开历史与事后标签放在两个并列字段，后者不可传给在线策略。没有异常时等待值为null，不造为零。其他任意条件可从raw-storage索引取得对应完整历史，校验SHA后使用同一入口。

完整独立审计需要原B3.2全量案例与数组，在Slurm内运行`verify_information.py`；它逐帧核对原公开状态SHA、用独立事件矩阵重算全部动作的有效确认及全回合指标。使用保存源码版本、原输入与新输出目录，禁止覆盖旧结果。

当前README有一项明确授权的状态更新，旧README字节已单独保存。其他原4207个跟踪文件不改。若运行旧阶段的严格工作树字节检查，应使用对应历史提交；不能把后来的README更新当成旧实验数值被修改。
''')
    rows = resources['allocations']
    cost_table = '\n'.join(f"| {v['JobID']} | {v['State']} | {v['ElapsedRaw']} | {int(v['CPUTimeRAW'])/60:.3f} |" for v in rows)
    write('Slurm成本与执行记录.md', f'''
# B4-A计算与版本记录

共{resources['allocation_count']}项分配，{resources['allocated_CPU_minutes']:.3f}分配CPU分钟，低于960分钟上限。所有计算使用cu、2CPU、8GB，无GPU；实际峰值内存{resources['maximum_RSS_MiB']:.2f}MiB，同时分配最多{resources['maximum_concurrent_allocations']}项。

| 作业 | 终态 | 墙钟秒 | 分配CPU分钟 |
| --- | --- | ---: | ---: |
{cost_table}

130570／130571为初版完整测试与两条件预审计；修订后130573／130574重新执行完整测试和同两条件。130576的八个数组项覆盖全部128个旧条件，130582执行独立全动作核验、图表和原件打包。最终新增加的CLI空值回归由单独门槛作业复核。

预审计的四次旧历史恢复与全量128次分别记账。它们没有创建新物理实例或新策略成绩；底层物理方程为恢复旧遥测而实际重建，并与所有保存数组精确比较。本轮不把这些重建称为零计算或只读取摘要。

门槛和预审计旧版本保留，不能用初版统计覆盖修订版。历史负结果、未来信息边界与Tour回退未改。实际作业日志、原始账本和独立核算分别保留在evidence下。
''')
    status = {'stage': 'C1-B4-A', 'status': 'completed', 'formal_research_result': 'B3.2 NO-GO unchanged',
              'attribution_result': 'Current positive-cue scarcity is the most frequent observed blockage in C0 helpful-action frames; future-value overestimation and gates also matter.',
              'B4B': 'design proposed, not submitted', 'new_policy_trials': 0, 'new_physical_instances': 0,
              'neural_updates': 0, 'LLM_calls': 0, 'old_history_reconstructions_full': 128,
              'old_history_reconstructions_pilots': 4, 'original_cases_bound': 768}
    save(BASE / '阶段状态.json', status)
    write('阶段验收与导师汇报.md', f'''
# 阶段验收与导师汇报

B4-A完成了全候选信息、预测与整回合机会审计。原768条记录全部绑定，128份旧公开历史恢复与原散列一致，{independent['counts']['analytic_actions']:,}个合法动作的逐异常结果通过独立事件矩阵核验。两张要求的诊断图已生成，具体视觉检查收据单列。

最重要的认识有三项：

1. 有益合法动作在{r['conditions_with_net_helpful_legal_check']}/128个旧条件中存在，预算小不是本轮失败的唯一解释。
2. 原C0有益动作存在的{relevant:,}帧中，{percent:.2f}%缺少可接受的正线索。应先检验估计与信息，不能只调WAIT系数。
3. 当前历史CVAE融合分量没有独立抬高正风险信号；趋势高估和未来最大值另有明确证据。下一步保留CVAE作为待检验工具，转向有效检查机会及异常持续性预测。

B3.2的NO-GO保持。B4-A的完成允许进入**B4-B方案冻结**，尚不授权把新策略称为有效，也不启动扩大搜索、重训PPO或RSI实验。现有旧测试仅作诊断资料。

可向导师这样复述：

> 我们已经把全部旧实验的合法补查机会拆开检查了。理想地知道状态时，一些合法动作确实有收益；实际模型经常不给这些动作正线索，同时又把另一些未来机会估得过高。现在优先要解决的是：公开历史能不能预测未来检查取得确认的机会，而不只是把温度图预测得平均更准。等这条信息链有证据，再恢复巡检策略和RSI的独立验证。
''')
    old_path = ROOT / 'reports/2026-10-29-Zhinengti-RSI-项目全量汇总.json'
    old = read(old_path)
    plots = []
    for name in ('b4a-horizon-error', 'b4a-check-wait'):
        path = EV / 'report' / (name + '.png')
        raw = path.read_bytes()
        plots.append({'name': name, 'path': path.relative_to(ROOT).as_posix(), 'mime': 'image/png',
                      'sha256': hashlib.sha256(raw).hexdigest(), 'encoding': 'base64', 'data': base64.b64encode(raw).decode('ascii')})
    documents = {p.name: p.read_text(encoding='utf-8') for p in sorted(BASE.glob('*.md'))}
    core = {'文档信息': {'标题': 'Zhinengti-RSI 当前全量汇总：B4-A全候选信息与价值审计',
                         '更新时间_北京时间': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                         '最新阶段': 'C1-B4-A', '最近正式策略结论': 'C1-B3.2 NO-GO不变',
                         '旧阶段与报告': '完整23阶段快照原样内嵌，内部当前／下一步属于2026-10-29当时状态。',
                         '当前状态入口': '先读当前结论和本轮通俗说明，再按需要查完整统计与历史。',
                         '原始大文件': '所有紧凑数字及两张新图直接内嵌；全量128历史与模型按路径/SHA索引，不将海量二进制塞入知识文档。'},
            '当前结论': {'一句话': '当前任务有可用检查机会，但风险线索经常没识别出来；未来趋势高估又造成等待，先做信息与价值验证。',
                         '阶段状态': status, '关键数字': diagnostic,
                         '下一步': 'B4-B先冻结有效检查机会预测协议；没有提交新训练或新闭环。'},
            '用户本轮审查建议_完整文本': gzip.decompress((EV / '用户审查建议原文.txt.gz').read_bytes()).decode('utf-8-sig'),
            '目前仍不知道的答案': [
                {'问题': '零风险线索究竟是传感信息不足，还是已有信息没用好？',
                 '已知': '旧轨迹中有益动作经常没有正线索，缺失与零估计已分别计数。',
                 '未知': '没有经过新预测器验证，尚不能证明现有信息在统计意义上不可用。',
                 '下一检验': '同一公开历史、同一预测完成时刻比较简单模型和现有工具的异常确认机会预测。'},
                {'问题': '怎样计算真正的WAIT信息价值？',
                 '已知': '原q_wait为当前信息外推的最大代理，不是收到新观测后的期望价值。',
                 '未知': '未来异常持续性、新信息及机会损失如何统一成可校准目标，尚无实验证实的最佳函数。',
                 '下一检验': '先校准有限视野确认概率；整回合反事实价值另立B4-C协议。'},
                {'问题': '怎样让CVAE提供其他信号没有的增量信息？',
                 '已知': '当前冻结缓存在这些旧巡回历史里没有独立提高正风险信号。',
                 '未知': '改变训练目标或观测条件能否带来动作和任务收益，仍需新实验。',
                 '下一检验': '保留原始输出，在相同决策器下独立比较，不立即重训更大网络。'},
                {'问题': '是否应该扩大一次补查的动作空间？',
                 '已知': '若干旧条件在现有限制内已有有益动作。',
                 '未知': '多次补查、任务交接和局部重排的收益及代价未独立测量。',
                 '下一检验': '先解决可用信息，再以独立任务协议评价更大动作空间。'},
                {'问题': '何时恢复RSI，以及论文主贡献落在哪一层？',
                 '已知': '流程合成已能运行，元更新优势和现场后果权重仍未建立。',
                 '未知': '最佳递归修改对象、必要原论文复现范围和业务价值需要导师与新证据共同确定。',
                 '下一检验': '工具收益可信后比较人工、随机、固定改进器与更新改进器，分别报告成本。'}],
            '本轮通俗说明_markdown': documents['运行与归因结果.md'],
            '完整分层与条件统计': r,
            '全部128条件摘要': read(EV / 'report/all-condition-summaries.json'),
            '独立核验_原件': independent,
            '完整Slurm成本核验': resources,
            '最新真实Torch测试': stage_gate,
            '本轮全部文字材料': documents,
            '两张诊断图_内嵌': plots,
            '全部原始因子文件索引': read(EV / 'raw-storage.json'),
            '传输与无损封装收据': read(EV / 'delivery-receipt.json'),
            '后续研究路线': documents['B4-B下一阶段方案.md'],
            '历史_B3_2及此前全部23阶段_原样': old,
            '历史快照SHA256': hashlib.sha256(old_path.read_bytes()).hexdigest()}
    for name in ('visual-review.json', 'portable-verification.json',
                 'portable-verification-final.json', 'candidate-export-review.json'):
        if (EV / name).exists():
            core[name] = read(EV / name)
    core['单候选还原验证_实际原件'] = {name: read(EV / name) for name in
        ('healthy-candidate-export.json', 'positive-candidate-export.json', 'waiting-candidate-export.json')
        if (EV / name).exists()}
    if (EV / 'github-ci-initial.json').exists():
        core['GitHub_CPU_CI_首次现场验证'] = read(EV / 'github-ci-initial.json')
    target = ROOT / 'reports/2026-10-30-Zhinengti-RSI-项目全量汇总.json'
    save(target, core)
    save(BASE / '本轮汇总.json', {'阶段状态': status, '完整统计': r, '诊断解释': diagnostic,
        '全部文字材料': documents, '全项目单文件入口': target.relative_to(ROOT).as_posix()})
    explanation = ('先用一个例子理解：系统像巡检班组，先按固定路线工作；发现线索后可以临时加查一次。'
        'CVAE等工具负责估计没直接看到的状态，规则负责决定现在查还是保留机会，大模型的外层改进负责修改技能和流程。'
        '本轮没有训练或修改这些模型，而是在旧考试记录里拆开看：系统看到了什么、为何这样决定、当时是否确有更好的合法选择。\n\n')
    human = '# Zhinengti-RSI当前进度与瓶颈：B4-A更新\n\n' + explanation + documents['阶段验收与导师汇报.md'] + '\n' + documents['运行与归因结果.md'] + '\n' + documents['B4-B下一阶段方案.md']
    human = human.replace('(evidence/', '(../research/2026-10-29-m12c1b4a-information/evidence/')
    (ROOT / 'reports/2026-10-30-项目进度与瓶颈说明-新手版.md').write_text(human, encoding='utf-8', newline='\n')
    print(json.dumps({'consolidated_project_JSON': str(target), 'bytes': target.stat().st_size,
                      'C0_helpful_current_frames': relevant, 'no_positive_cue_frames': weak, 'percent': percent}, ensure_ascii=False))


if __name__ == '__main__':
    main()

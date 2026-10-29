"""Evidence-bound B4-B0 report, readiness decision and single-file handoff."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/2026-10-30-m12c1b4b0-data'
EV = BASE / 'evidence'
ROLES = {'thermal': '温度', 'electrical': '电气接点热', 'cooling': '冷却流量', 'control': '控制／传感偏差'}


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + '.gz')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def write(name, content):
    (BASE / name).write_text(content.strip() + '\n', encoding='utf-8', newline='\n')


def main():
    coverage = read(EV / 'report/coverage.json')
    independent = read(EV / 'report/independent-verification.json')
    protocol = read(BASE / 'B4-B1冻结协议.json')
    freeze = read(EV / 'B4-B1-design-freeze-final.json')
    resources = read(EV / 'slurm-cost-review.json')
    gate = read(EV / 'final/gate/full-tests.json')
    if independent['status'] != 'passed' or coverage['conditions'] != 128 or gate['tests'] < 237:
        raise RuntimeError('Full B0 evidence and final real Torch gate required')
    if freeze['sha256'] != hashlib.sha256((BASE / 'B4-B1冻结协议.json').read_bytes()).hexdigest():
        raise RuntimeError('Future protocol bytes do not match frozen receipt')
    for row in coverage['coverage']['profession_horizon'].values():
        if row['horizon_s'] not in (30, 60, 120):
            raise RuntimeError('Superseded horizon-metadata report cannot be final evidence')
    cells = coverage['coverage']['profession_horizon']
    totals = {key: sum(v[key] for v in cells.values()) for key in
        ('A_valid', 'A_positive', 'A_negative', 'B_reference_valid', 'B_reference_positive',
         'B_action_valid', 'B_action_positive', 'B_action_negative', 'B_action_invalid')}
    rows = []
    for kind, title in ROLES.items():
        for horizon in (30, 60, 120):
            v = cells[kind + '/' + str(horizon)]
            rows.append(f"| {title} | {horizon} | {v['B_action_valid']:,} | {v['B_action_positive']:,} | {v['B_action_negative']:,} | {100 * v['valid_B_positive_fraction']:.4f}% | {v['distinct_B_positive_retained_events']} | {len(v['B_positive_seed_groups'])} |")
    positive_table = '\n'.join(rows)
    physical_table = '\n'.join(f"| {title} | {h} | {cells[k+'/'+str(h)]['A_valid']:,} | {cells[k+'/'+str(h)]['A_positive']:,} | {cells[k+'/'+str(h)]['B_reference_positive']:,} |" for k, title in ROLES.items() for h in (30, 60, 120))
    readiness = {'stage': 'C1-B4-B0', 'engineering_preparation': 'complete',
        'B4B1_predictor_training_gate': 'NO-GO', 'B3_2_formal_policy_verdict': 'NO-GO unchanged',
        'reasons': ['All available old8500-8703data have already been used/read and remain diagnostic only.',
            'Electrical task has zero physical and confirmable positive events at all three horizons.',
            'Cooling30s has14 distinct retained positive events despite28 positive noise-repeated sample rows.',
            'Fresh64seed-group development/calibration/selection/final manifest is frozen but not generated, so independent split coverage is unverified.'],
        'fresh_generation': 0, 'new_predictor_training': 0, 'PPO_updates': 0, 'LLM_calls': 0,
        'permitted_next_step': 'Execute only the frozen new-process generation/coverage phase after explicit continuation; fitting starts only after empirical coverage and causality gates pass.',
        'old_diagnostic_totals': totals, 'old_conditions': 128, 'old_seed_groups': coverage['seed_groups'],
        'old_named_processes': coverage['named_physical_processes'], 'old_numeric_processes': coverage['numeric_process_fingerprints'],
        'protocol': protocol['protocol'], 'protocol_sha256': freeze['sha256'],
        'fresh_manifest_sha256': freeze['fresh_manifest_sha256']}
    save(BASE / '阶段状态.json', readiness)
    save(EV / 'training-readiness.json', readiness)
    write('数据可用性与独立性结果.md', f'''
# B4-B0数据准备完成；B4-B1训练启动为NO-GO

这一步要先弄清“未来能不能查到异常”的标签和数据是否可靠，再训练预测器。现在接口、旧数据审计和下一轮设计均已完成；新模型尚未训练，新的独立过程尚未生成。

旧数据共{coverage['conditions']}个观测条件、{coverage['named_physical_processes']}个命名物理配置，实际只有{coverage['numeric_process_fingerprints']}种不同数值过程，来自8个底层种子。每个条件359个预测根、176个专业任务、3个视野，形成{coverage['total_sample_rows']:,}条根—任务—视野记录。这些记录大量共享物理过程和事件，不能当成数千万个独立异常。

## 1. 正负例是否够用

下表统计的是**合法完成检查时刻上的B_action**，不是所有物理端点。无合法动作、超出回合的端点标为-1，均未混入负例。事件按物理数值指纹、任务和真实起止去重，两档噪声不重复增加事件数。

| 专业 | 视野／秒 | 有效B样本 | 正例 | 负例 | 正例比例 | 去重保留事件 | 有正例的种子组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{positive_table}

所有合法B样本合计{totals['B_action_valid']:,}条，其中正例{totals['B_action_positive']:,}条、负例{totals['B_action_negative']:,}条；{totals['B_action_invalid']:,}条无效候选保持缺失。上表同一事件还会在不同视野重复出现，各行事件数不能再相加当独立样本量。

最明确的缺口是电气接点热：旧四种情景里没有形成超过原60°C阈值的电气异常，三个视野均没有正例。冷却30秒也只有14个可覆盖保留事件，28条正例主要来自两档噪声重复。控制任务虽然有更多时间点正例，去重保留事件仍只有16个。事件去重不等于彼此独立，统计抽样单位仍是底层种子组。

因此，靠随机切分时间点、复制事件或对测试正例过采样，都不能修复独立证据不足。热任务的旧事件可用于检查接口和形成假设，也不能把已看过的8700–8703重新称为独立测试。

## 2. A与条件式B也单独保留

| 专业 | 视野／秒 | 有效物理端点 | A物理正例 | B_reference正例 |
| --- | ---: | ---: | ---: | ---: |
{physical_table}

A表示当时真实物理异常；B_reference进一步要求原定合格测量触发真实告警；B_action再加可合法完成的检查安排。至少15秒的保留异常段是另一个事件字段，并未抹掉较短的真实超阈值端点。当前值、事件持续时间、合格确认和整回合收益由不同字段表达。

## 3. 因果与传感器检查

全部公开特征来自根时刻及其更早的历史。独立程序核对原通道的值、来源、时间戳、最近三次专业检查，以及全部85列基础特征和13列时刻／资格／余量特征，亦核对C侧车的动作索引。可用读数的时间检查共{coverage['causal_available_readings_checked']:,}次，未发现未来时间戳。

完整按专业、视野、情景、物理指纹、种子、相位与噪声的统计见[evidence/report/coverage.json.gz](evidence/report/coverage.json.gz)。不同传感器的可用／缺失数同时按全部根任务和有效B样本两个分母报告。缺测补零均配available位；不能由零值判断设备正常。

原始CVAE输出与测量覆盖缓存分列保存，不被称为已经校准的未来异常概率。公开电力、泵指令和温度／流量来自原telemetry；真实故障类型、真实起止、未来告警和C净收益只存在评价侧。

## 4. 独立性与结论

64对物理配置的两档噪声在物理标签、B标签和合法时刻上完全对应。健康情景的不同名义注入相位可能产生相同数值过程，已由指纹去重。新过程清单的64个种子与基准内2,968份JSON／JSON.gz的已记录种子字段没有碰撞；这只证明新编号未复用，不证明新数据已经具备正例。

本轮训练门槛为**NO-GO**：数据工程通过，但四专业独立正例不完整，且新划分尚未实际生成。B3.2原NO-GO不变。下一步最小修复是按冻结清单生成新的过程，补入原有dc_contact_hotspot故障族并检查覆盖；不修改物理阈值、不增强旧事件、不替换种子追逐正结果。
''')
    write('B4-B1最小基线与执行方案.md', f'''
# 下一轮B4-B1：先过数据门槛，再拟合最小预测器

最终设计版本：`{protocol['protocol']}`，SHA256为`{freeze['sha256']}`。完整协议和每个新条件均已落盘；这里只解释主要决定。

## 过程级划分与最小修复

开发使用20000–20023共24个种子；校准使用20200–20207共8个；模型／阈值选择使用20208–20215共8个；最终测试使用20400–20423共24个。同一种子的情景、相位和噪声全部属于同一划分。相同数值指纹在不同划分出现会阻止训练，不能冒充独立。

固定五类情景：健康、180秒冷板污堵、180秒电气接点热、900秒传感偏差、180秒辅助馈电—CDU耦合；各有335／705秒两个起点及0.2／1.0°C观测噪声。共640个命名物理配置、1280个观测条件；实际数值去重数仍须生成后核验。预测根改为30秒间隔，减少时间点重复，视野仍为30／60／120秒。

这是一份尚未执行的受控清单。最先执行数据生成和覆盖审查，不直接拟合。每专业／视野在开发、校准、选择、测试中分别要求至少8／4／4／8个有正例的种子组、12／6／6／12个有正例的数值过程、20／8／8／20个可预测保留事件，另有明确负例覆盖门槛。任何不足都记录缺口并停止，不自动追加事件、加强故障或替换种子。

## 预先限定的比较

| 方法 | 冻结范围 |
| --- | --- |
| 经验频率 | 分专业／视野的开发经验概率，固定Laplace平滑 |
| 逻辑回归 | 2类标签×4专业×3视野，L2仅0.1／1／10，共最多72次拟合；确定性CPU float64 LBFGS及迭代上限固定 |
| 有界趋势 | 使用原温度外推与原专业持有证据，不新搜系数 |
| 物理名义估计 | 根时刻因果名义输出持有到端点；非温度任务明确用专业证据基线，不偷用未来物理过程 |
| 冻结CVAE原始输出 | 温度任务的固定先验输出；不虚构其流量／接点模型能力 |
| 覆盖缓存 | 与原始输出分开作为表示对照，不当新训练估计器 |
| 可选小型时序模型 | 一层16宽GRU的设计备选；当前enabled=false、训练预算0，启用需独立修订，不能在测试后救分 |

拟合标签头明确为A_physical与B_action。B_reference保留全端点数据审计，并在同一合法B子集报告条件可确认性，不增加未预算的第三类模型搜索。C不进入本轮预测器。

## 校准、指标和独立评价

校准集只拟合预设单调sigmoid概率映射，不与开发拟合或最终测试混用。选择集决定正则化、方法、比较基线和1%经验误报阈值，全部在最终测试前冻结。原风险代理不在0至1内，必须经过校准或明确保持连续评分，不能直接称概率。

主指标是固定误报预算下的事件召回和过程／种子宏平均Brier，另报告A／B_reference、实际测试误报、AUPRC、对数损失、可靠性图、最差种子和资源成本。训练与统计先按相同物理指纹去重、噪声版本组内平均，最后按种子组比较。

不确定性使用预定2000次种子组配对bootstrap；LIMITED GO的12个预设单元、两项主收益使用明确的同时下界分位数规则，不临时发现子组。门槛是启动和判断规则，不能保证统计功效或现场泛化。

最终测试在预先声明的独立覆盖清单审计之外保持封存；不得暴露预测分数供选择。先冻结模型、校准器、阈值和比较方法收据，再评分一次。GO要求事件召回至少提高0.05且宏Brier至少相对下降5%，两项配对区间支持收益并满足误报／专业退化条件；限定范围通过才判LIMITED GO，其他情况NO-GO。

概率校准与分组评价参考[官方校准说明](https://scikit-learn.org/stable/modules/calibration.html)、[分组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)和[Brier定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)。具体数值门槛、单调约束和分组权重是本项目预注册设计。概率只对应冻结的合成情景分布，尚不是现场故障发生率。

## 资源和Slurm顺序

下一轮全部使用cu、2CPU／8GB、单任务最多60分钟、最多并发2项，无GPU、无PPO／Qwen调用。新过程生成最多2880分配CPU分钟，拟合／校准最多960，评价／审计最多480，总上限4320；公开衍生数据上限6GB。依次通过源码／清单、数据覆盖、开发拟合、校准选择、冻结、一次最终评分和独立审计。B0没有启动这些B1作业。
''')
    rows = resources['allocations']
    table = '\n'.join(f"| {v['JobID']} | {v['State']} | {v['ElapsedRaw']} | {int(v['CPUTimeRAW'])/60:.3f} |" for v in rows)
    write('测试复核与Slurm成本.md', f'''
# B4-B0工程验证与计算账本

最终真实Torch完整测试{gate['tests']}项，零跳过、错误或失败。新机会数据契约测试7项，同时保留原formal-study和information-audit的CPU检查。全量数据由修订code-v3生成，独立复核和最终汇总使用analysis-v3，来源包与原字节SHA分别保留。

独立核验覆盖128个条件、{independent['counts']['samples']:,}条样本、{independent['counts']['public_base_values_verified']:,}个公开基础特征值及{independent['counts']['public_action_values_verified']:,}个动作特征值。它调用原专业测量和告警函数，核对历史、时刻可行性、资格及C索引。额外用一个完整旧条件测试了有限遥测、余量和C索引篡改，均被拒绝。

本轮{resources['allocation_count']}项Slurm分配合计{resources['allocated_CPU_minutes']:.3f}分配CPU分钟，低于960上限；最多同时{resources['maximum_concurrent_allocations']}项，峰值RSS{resources['maximum_RSS_MiB']:.2f}MiB，均为cu、2CPU／8GB、无GPU。全量衍生数据{coverage['raw_derived_data_bytes']:,}字节，低于1GB。

| 作业 | 终态 | 墙钟秒 | 分配CPU分钟 |
| --- | --- | ---: | ---: |
{table}

初版漏掉了根时刻已公开在途的正常巡回完成，修订后重新测试与预审计。随后补强原件与B4-A锚点的来源链；全量没有使用初版数据。初版报告误累加视野元数据、复核范围未覆盖全部特征列，最终汇总与独立复核均另存修订版；旧版本保留，不作为最终验收。

这些是数据工程和验证修订，不是通过修改旧实验或搜索预测器提高分数。没有新物理过程生成，没有研究预测模型训练，PPO与Qwen调用均为零。
''')
    write('复现与读取说明.md', '''
# 读取B4-B0的公开样本与独立标签

完整128条件的数据位置和每份SHA保存在[evidence/report/raw-storage.json](evidence/report/raw-storage.json)。集群根为`/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b0/full`。本地保留全部A／B标签、事件、来源和统计，以及预定健康条件与耦合正例的完整公开特征；其他126份特征矩阵以原字节索引保留。

Python3.11与NumPy即可读取公开特征，不需要Torch或真实模型权重。例子如下，输出路径必须不存在：

```powershell
$env:PYTHONPATH = "$PWD\\src"
python scripts/read_opportunity_sample.py --condition research/2026-10-30-m12c1b4b0-data/evidence/full/development-0/condition-14 --root-time 735 --task 148 --horizon 60 --output sample-new.json
```

导出将公开特征和事后A／B标签置于不同字段；实际训练仅读`load_public()`返回的固定特征列，不能直接拼入整个导出对象。C只保留B4-A独立侧车引用。若额外检查的派单晚于预测根，C对应那个候选派单时刻，不等于预测根立刻检查的价值。

`labels-only.npz`保存物理／条件确认矩阵及事件归属；`targets-evaluator-only.npz`保存各根／视野的有效标签、合法动作数量和C连接索引。-1表示未定义，不能代为0；动作索引-1表示正常Tour、-2表示没有合法安排，非负值才连接B4-A额外检查标签。

完整独立复核通过Slurm运行`verify_opportunity_data.py`，需要B4-A全历史及B3.2原始数组／案例。便携复核入口与收据保存在evidence，历史结果保持原字节。新B4-B1清单仅为计划，不存在可下载的新训练集或新模型成绩。
''')
    write('阶段验收与下一步.md', '''
# B4-B0完成，B4-B1训练启动NO-GO

标签与数据工程接口已经建立，旧128条件的全量统计、因果审计、独立性核对、真实Torch测试和冻结协议均已交付。B0任务完成与下一轮训练可启动是两项独立判断。

训练当前不放行：电气任务没有异常正例，冷却30秒的独立事件很少，所有旧种子已暴露；64个新种子组的开发／校准／选择／最终划分只是冻结清单，还没有实际生成和通过覆盖门槛。

最小可执行修复是下一轮先运行固定的新过程清单和独立覆盖审计，补入原有电气接点热故障族；不足时报告具体专业／视野／划分缺口，不复制事件、不打散时间点、不在测试中过采样，也不临时改变阈值或扩大模型搜索。覆盖通过后再开始预注册经验频率、逻辑回归及固定估计工具的比较。

B3.2的NO-GO不变。本轮没有提前宣称CVAE、逻辑回归或新的检查策略有效，未启动PPO训练或RSI搜索。下一轮先解决“能否预测合格检查取得确认”，整回合净收益仍留给独立B4-C。
''')
    write('给导师的通俗说明.md', f'''
# 给第一次了解项目的人

可以把系统理解为一个巡检班组。A问“将来那里真的有问题吗”，B问“届时能合法到场并通过专业检查确认吗”，C问“为了查它绕一次路，整轮工作会不会更好”。这三个问题需要不同标签；本轮先准备前两个，不能用其中一个的好成绩替代另一个。

我们已经从旧记录建立了{coverage['total_sample_rows']:,}个预测位置的资料，逐项核查当时能看到的信息。但这并不意味着有这么多独立故障：资料只来自8个旧种子组、56种不同数值过程，同一故障在多个时间和噪声条件下会反复出现。

审计发现电气任务根本没有异常正例，冷却30秒也只有14个可覆盖的去重保留事件，它们仍来自8个种子组。现在直接训练，很容易得到“几乎总报正常、平均准确率很好”的模型，却学不会发现异常。因此我们把训练判为NO-GO，而不是靠复制正例制造足够数据。

下一步已经写实并冻结：使用64个新的种子组、五种受控情景、两档噪声，先核对独立正负例，再比较简单经验概率、逻辑回归、已有趋势／物理模型和冻结CVAE。此时尚未生成新数据或训练模型。只有能可靠预测有效检查机会，才有依据继续研究检查时机以及RSI怎样改进流程。
''')
    documents = {p.name: p.read_text(encoding='utf-8') for p in sorted(BASE.glob('*.md'))}
    old_path = ROOT / 'reports/2026-10-30-Zhinengti-RSI-项目全量汇总.json'
    old = read(old_path)
    project = {'文档信息': {'标题': 'Zhinengti-RSI B4-B0全量汇总：未来有效检查机会的数据与协议',
        '北京时间': datetime.now(timezone(timedelta(hours=8))).isoformat(), '最新阶段': 'C1-B4-B0',
        '阶段状态': '数据与协议准备完成，B4-B1训练启动NO-GO',
        '阅读入口': '先读当前结论与通俗说明；完整分层、协议及历史均内嵌。大规模特征矩阵以路径/SHA引用。'},
        '当前结论': readiness, '新手说明_markdown': documents['给导师的通俗说明.md'],
        '完整数据可用性与独立性统计': coverage,
        '全部128条件摘要': read(EV / 'report/condition-summaries.json'),
        '数据契约': read(EV / 'full/development-0/data-contract.json'),
        'B4-B1完整冻结协议': protocol, 'B4-B1冻结收据': freeze,
        'B4-B1新过程计划_尚未生成': read(BASE / 'B4-B1新数据清单.json'),
        '独立全特征与标签复核': independent, '真实Torch最终测试': gate,
        'Slurm成本': resources, '所有本轮文字材料': documents,
        '全量数据路径及SHA': read(EV / 'report/raw-storage.json'),
        '新种子碰撞核对': read(EV / 'new-seed-collision-review.json'),
        '本地特征篡改拒绝验证': read(EV / 'local-independent-feature-tamper-checks.json'),
        '历史_此前全部项目材料_原样': old,
        '历史快照SHA256': hashlib.sha256(old_path.read_bytes()).hexdigest()}
    project['数据传输收据'] = read(EV / 'delivery-receipt.json')
    project['远端最终原件核对'] = read(EV / 'remote-final-verification.json')
    project['四种真实读取示例'] = {name: read(EV / name) for name in
        ('positive-sample.json', 'inflight-Tour-sample.json', 'beyond-horizon-sample.json', 'physical-positive-no-legal-sample.json')
        if (EV / name).exists()}
    for name in ('portable-verification.json', 'portable-verification-final.json',
                 'sample-reader-review.json', 'github-ci-initial.json'):
        if (EV / name).exists():
            project[name] = read(EV / name)
    target = ROOT / 'reports/2026-10-30-B4-B0-项目全量汇总.json'
    save(target, project)
    save(BASE / '本轮汇总.json', {'阶段状态': readiness, '全部统计': coverage,
        '协议': protocol, '全部文字材料': documents, '集中阅读入口': target.relative_to(ROOT).as_posix()})
    human = ('# 当前项目进度与瓶颈：B4-B0数据准备\n\n' + documents['给导师的通俗说明.md'] + '\n' +
        documents['数据可用性与独立性结果.md'] + '\n' + documents['B4-B1最小基线与执行方案.md'])
    human = human.replace('(evidence/', '(../research/2026-10-30-m12c1b4b0-data/evidence/')
    (ROOT / 'reports/2026-10-30-B4-B0-新手说明.md').write_text(human, encoding='utf-8', newline='\n')
    print(json.dumps({'current_gate': 'NO-GO', 'samples': coverage['total_sample_rows'],
        'B_valid': totals['B_action_valid'], 'B_positive': totals['B_action_positive'],
        'new_project_JSON_bytes': target.stat().st_size, 'protocol_SHA': freeze['sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

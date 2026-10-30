# B4-B2：复用现有 Improver 的元层接口与对照设计

状态：**设计与工程验证准备**。本文件核对的代码基准为 `d55c1613317f9dfb520f5d6eaea1083e32b3ecb2`。下文“现有”均指这个基准；`rsi_controlled.py` 等新包装能力须由本轮实现和单独验收，不能倒写成基准已经支持。

本轮补充验收：新的`ControlledImprover`、按分区的不可变预算缓存、配对元更新、协议单独迁移、失败立即停止和重复计数已完成接口验证，最终源279项真实Torch／58项轻量CPU检查通过。可迁移指令收紧为三个公开模板，不能携带自由文本任务身份。下文作为基准审查与设计依据保留；最终字段、预算和开封规则以[已冻结JSON](RSI-Controlled-Meta-v1.json)及[验收收据](evidence/final-acceptance.json)为准。真实Qwen搜索与正式元迁移实验没有启动。

本轮可以恢复“多执行体巡检＋受控改进器更新”这条研究线。B1b 的概率预测 **NO-GO 保留**，不构成只做 RSI 接口、分区、预算与假调用验证的前置门；它也没有因此被转为概率工具有效的证据。B1b 的工程完成与科学 NO-GO 是两个状态，见[原验收](../2026-10-30-m12c1b4b1b-limited/evidence/final-acceptance.json) L2–3、[原结果](../../reports/2026-10-30-B4-B1b-Limited-完整进度与瓶颈.md) L97–109。本文件未读取旧 final 逐条记录、概率或标签，也未启动 Qwen、Slurm 或新物理过程。

## 1. 现有代码可直接复用到哪里

| 层 | 实际入口与返回 | 已实现语义 | 证据 |
|---|---|---|---|
| 元协议 | `Improver(diagnostics, memory, instruction)`；`from_dict()` / `to_dict()` / `version_id` | 不可变三字段协议；诊断工具 1–3 个且不重复；检索模式有界；指令 1–500 字符 | [improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L13–44 |
| 任务程序 | `PatrolProgram(refresh_s=30, cooldown_s=60, fallback='tour', rules=())`；同样支持字典往返和版本号 | 0–3 条有序 `TypedRule(signal, threshold, action)`；不是任意源码 | [improver_task.py](../../src/patrol_rsi/improver_task.py) L24–78 |
| 程序提案 | `program_prompt(program, views, history, protocol) -> dict`；`Caller.generate(user, seed) -> receipt`；`parse(raw, 'workflow') -> (PatrolProgram, envelope)` | 按协议取公开诊断和有限历史；每次生成一个程序；解析不补修字段 | [improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L104–146；[m12b.py](../../src/patrol_rsi/m12b.py) L80–116 |
| 协议提案 | `protocol_prompt(protocol, reports) -> dict`；`parse(raw, 'protocol') -> (Improver, envelope)` | 只看各轮是否保留及搜索计数，不给原始标签、最终结果或开发验证数值 | [improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L149–163；[m12b.py](../../src/patrol_rsi/m12b.py) L264–265 |
| 任务评估 | `TrialStore.evaluate(cases, program, *, estimator='history', actor_tool='ppo') -> {'validator', 'public', 'case_ids'}` | 独立保存有标注验证摘要与公开诊断；同一签名缓存；实际运行 `run_program()` | [m12b.py](../../src/patrol_rsi/m12b.py) L135–185 |
| 内层搜索 | `adapt(store, caller, protocol, tag, cases, seed, slots, *, random=False, checkpoints=(), initial=None)` | 从默认 Tour、空 history 和只含 Tour 的 seen 集开始；非法/重复消耗槽；严格变好才保留 | [m12b.py](../../src/patrol_rsi/m12b.py) L188–232 |
| 元更新 | `meta_update(store, caller, method, generation_seed, *, smoke=False) -> (protocol, rounds)` | 正式模式两轮；每轮旧/新各两个候选，同任务起点和生成种子；严格更优才保留协议 | [m12b.py](../../src/patrol_rsi/m12b.py) L235–269 |
| 多执行体与安全核 | `ImproverTask` → `WorkflowInspection` → `CoveredInspection`；`choose_program()` / `run_program()` | 128 rack 固定场景；18 位专业执行体，固定任务所有权、占用互斥、资格、完整返回证书、每所属巡回最多两次补查、统一末尾规则 | [aisles.py](../../src/patrol_rsi/aisles.py) L101–135；[improver_task.py](../../src/patrol_rsi/improver_task.py) L123–181、L209–291；[covered_inspection.py](../../src/patrol_rsi/covered_inspection.py) L108–167 |

18 位执行体来自 `12 thermal + 2 electrical + 2 cooling + 2 control`；统一协调器轮转派单并更新 reserved 集。当前不是 18 个语言模型分别商议，也没有跨属主接管、通信网络或碰撞模型。恢复多智能体任务层时，应沿用这些已经执行的资格、所属关系和派单接口；不能把新增自由文本角色名当成新增能力。[aisles.py](../../src/patrol_rsi/aisles.py) L101–135；[improver_task.py](../../src/patrol_rsi/improver_task.py) L132–136、L212–253；[M11.1 契约](../../src/patrol_rsi/m111.py) L39–50。

## 2. 能传给现有接口的精确字典

下面两个对象分别传给 `Improver.from_dict()` 与 `PatrolProgram.from_dict()`；它们不是同一个自修改对象。

```json
{
  "diagnostics": ["coverage", "professional_evidence", "novelty"],
  "memory": "best_and_failures",
  "instruction": "Compare a successful proposal with a failed proposal; propose one distinct typed program and preserve the protected tour."
}
```

`diagnostics` 允许 `coverage / professional_evidence / efficiency / thermal_disagreement / novelty`；`memory` 允许 `none / recent / best_and_failures`。这里的 `best_and_failures` 实际取**最后一个 accepted 加最后一个未 accepted**，没有对所有成功按分数求 best。`recent` 取末两个；`none` 不检索。失败包括未提升、重复和非法候选。[improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L13–40、L104–111；[m12b.py](../../src/patrol_rsi/m12b.py) L220–221。

```json
{
  "refresh_s": 30,
  "cooldown_s": 60,
  "fallback": "tour",
  "rules": [
    {"signal": "thermal_excess", "threshold": 0.05, "action": "signal"},
    {"signal": "age", "threshold": 0.75, "action": "return_tour"}
  ]
}
```

程序域固定为：

| 字段 | 域/行为 |
|---|---|
| `refresh_s` | 整数 `30 / 60 / 120`；历史 CVAE 的真实推理周期 |
| `cooldown_s` | 整数 `0 / 30 / 60 / 120`；只限制可选补查，不能挡住下一巡回义务 |
| `fallback` | `tour / ppo`；无规则命中时返回下一巡回项或用冻结演员 |
| `rules[].signal` | `age` 全专业；`thermal_excess / thermal_disagreement` 只热专业；`contact_heat` 电气；`flow_deficit` 冷却；`verified_sensor_bias` 控制 |
| `threshold` | age 为 `.25 / .5 / .75`；核验偏差只 `.5`；其他信号 `.05 / .25 / .5`，必须为数字 |
| `action` | `signal / urgency / ppo / return_tour`；按第一条有合格候选的规则执行；`return_tour` 选下一巡回义务，不要求该义务自身满足触发信号 |

证据：[improver_task.py](../../src/patrol_rsi/improver_task.py) L24–74、L184–254；[facility_workflow.py](../../src/patrol_rsi/facility_workflow.py) L98–126。`FacilityWorkflow` 是 M12-A 的旧三字段语法，规则是 `role/signal/threshold/ranker`；不能与 M12-B 的 `PatrolProgram` 字典混传。[facility_workflow.py](../../src/patrol_rsi/facility_workflow.py) L15–69。

现有模型响应封套精确为 `{'diagnosis': str, 'workflow': object, 'expected_effect': str}` 或把 `workflow` 换成 `protocol`。两个说明字符串各 1–500 字符。生成收据保存 `prompt / rendered_prompt / raw_response / input_tokens / output_tokens / generation_seconds`；Qwen3-8B 的 revision、采样参数和 6144/768 token 上限应沿用原固定契约，不在元协议中开放。[m12b.py](../../src/patrol_rsi/m12b.py) L36–53、L80–116。

## 3. 允许改动、固定核与能力缺口

“允许作为研究变量”与“基准已经能运行”必须分别登记。推荐新包装器复用原 `Improver`，将其他有限策略放在带版本的注册表/运行清单中；不向底层三字段字典塞入额外键。

| 用户允许的元层范围 | 基准实际支持 | B4-B2 最小处理 |
|---|---|---|
| 诊断工具选择与顺序 | 选择/排序 1–3 个预计算摘要，形成 prompt 中的列表；没有按协议动态重跑工具 | 可直接开放；收据记选中工具及顺序。不能声称排序改变了物理工具执行顺序或节省 CVAE 成本 |
| 成功/失败经验检索 | 三个固定检索器，最多两条；只在一次 `adapt()` 中有效 | 可直接开放；记录检索候选 ID、模式和空档原因，跨任务归零 |
| 候选产生 | Qwen 单次 JSON；`random_program(seed)` 是独立固定随机入口 | 包装器可选这两个已存在的有界函数；一次槽只调用一个生成器。模型和随机候选均验证同一 `PatrolProgram` 语法 |
| 候选去重 | `candidate.version_id in seen`；重复不评价、不免费补槽 | 保留 exact-version 去重。若新增规范化/语义去重或 mutation，必须先有实现、语义保持验证和版本收据；否则登记 `unsupported`，不能靠 stub 名称认定支持 |
| 预允许的接受 | 固定 `validation_rank(child) < validation_rank(parent)`；协议还要求有效且版本改变 | 固定验证器、安全门不开放。包装器最多选择既定“严格提升才采纳”或更保守的保留父项；任何新数值接受策略需预注册并证明不能绕过固定安全门 |
| 预允许的回退 | 程序的 `tour / ppo` 已存在；无效/重复候选保留父程序 | 生成失败回退、实体执行异常回退需显式定义和计费；区分“保持父项”与“重新执行安全 Tour”。不能将不安全执行记为普通低分后继续 |
| 提案分析方式 | diagnosis/expected_effect 字符串，公开诊断、novelty 计数和有限接受反馈 | 可选择固定的公开分析模板/有界摘要；分析结果不能成为验收裁判或增加未记账模型调用 |

证据：[improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L114–163；[m12b.py](../../src/patrol_rsi/m12b.py) L119–132、L188–269。诊断的计算发生在 `TrialStore.evaluate()`，早于协议选择；`novelty` 才是在 prompt 构造时由当前 history 计算。[m12b.py](../../src/patrol_rsi/m12b.py) L154–185；[improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L115–119。

建议运行清单显式登记以下固定策略 ID。ID 名字可由新模块统一，但应全部解析为已注册函数；这些字段在 v1 可以是单一可用值，不能据此声称它们已被 RSI 改进。

```json
{
  "schema": "B4B2-bounded-run-v1",
  "improver_schema": "m12b-three-fields",
  "candidate_generator": "qwen_json_once",
  "dedup_policy": "exact_version_consume_slot",
  "task_acceptance": "strict_validation_rank_v1",
  "meta_acceptance": "paired_strict_validation_rank_v1",
  "failed_proposal_fallback": "retain_parent",
  "proposal_analysis": "public_diagnostics_and_limited_feedback_v1"
}
```

固定核包括物理过程、观测通道和测量有效性、任务资格/所有权、工时/路程、返回证书、配额、末尾规则、模型权重、任务语法、评估器、分区和预算。候选程序先经过这些核，再由开发验证器排序。排序精确为 `(violating_tasks, 1-coverage, missed_segments, weighted_censored_delay_s or 0, mean_travel_m)`，逐项四舍五入到 8 位后严格比较；不能让元协议换成“自评高就接受”。[improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L87–101；[coverage_plan.py](../../src/patrol_rsi/coverage_plan.py) L32–73；[improver_task.py](../../src/patrol_rsi/improver_task.py) L272–291。

还需由新包装器补齐五个基准没有证明的契约：

1. `initial` 应绑定默认程序版本和 cases 指纹；基准 `adapt()` 会直接信任调用方给的 initial，没有一致性断言。[m12b.py](../../src/patrol_rsi/m12b.py) L188–192。
2. 公开诊断应按字段投影或拒绝未知字段；现有 `program_prompt()` 直接透传 `views[name]`，不能仅凭干净 fixture 证明恶意嵌套字段不会泄漏。[improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L114–122；[test_improver_task.py](../../tests/test_improver_task.py) L82–92。
3. 生成和执行异常应记录已占用槽、异常阶段和成本；现有 `caller.generate()` 在解析 try 外，`store.evaluate()` 也没有失败回退收据。[m12b.py](../../src/patrol_rsi/m12b.py) L202–217。
4. 新包装器须拒绝不支持的候选变异、工具名、策略 ID、源码/文件路径和越权字段；不能把工具白名单替换成动态导入或 `eval`。
5. 自由指令虽有长度限制，基准没有证明它不携带任务标识或跨任务内容。转移清单要记录协议来源和内容哈希；公开模板与测试须约束逐任务 ID、种子、起止标签等内容进入 instruction 的路径。

## 4. 元开发、后续适配和封存评价的隔离

建议 `ScopeTaskSplit` 用三种不同角色的清单绑定 `meta_development / transfer_adaptation / sealed_evaluation`。每个清单包含公开 task ID、物理过程 key、seed、输入来源哈希、任务语法版本；不把隐藏标签放入调用接口。独立性按物理过程分组，不能靠换 `phase`、噪声、estimator 或程序版本制造新任务。

物理 key 至少含 `(seed, scenario, duration_s, fault_start_s, plant_control, layout)`；同一个 seed 的各噪声/相位/程序副本必须留在同一分区。元开发与后续任务的 seed 集和物理 key 集均应不相交。跨场景/布局迁移是另外的科学问题，不能仅凭这里不相交就声称已经验证。

每个旧/候选协议臂以及每个后续新任务，执行以下重置：

```text
program = PatrolProgram()       # 默认 Tour，版本应核对
history = []
seen = {program.version_id}
retrieval_archive = []
per_task_budget = fresh_budget
```

只转移 `Improver` 或已验收有限包装协议的不可变字典和哈希；不转移元开发保留程序、候选库、case IDs、验证指标、经验历史、调参结果或模型权重。`adapt()` 内部重置已经存在；新增严格分区门和 transfer receipt 应把空历史、空 archive、默认程序版本、源/目的任务指纹和预算起点一起断言。[m12b.py](../../src/patrol_rsi/m12b.py) L188–192、L224–231、L251–265。

**不能直接拿现有 `--smoke` 入口证明分区隔离。**它在同一 `specs('smoke')` 上做 meta 和 transfer（[m12b.py](../../src/patrol_rsi/m12b.py) L294–299）。正式旧划分的 meta seeds 6400–6403、adapt 6500–6501、test 6600–6601 是分开的，但只存在构造约定，没有通用不相交验证器（同文件 L64–77）。当前 `MetaResetTests` 只证明程序/历史重置；后续要补分区负例。[test_improver_task.py](../../tests/test_improver_task.py) L102–139。

## 5. 四组元层对照：同时报告两种预算视角

四个组使用相同原子工具、程序语法、安全核、验证器、下游任务和生成种子规则。人工组预先固定为上文两条人工规则，不能在这次对照的结果出来后再选其阈值；所有搜索臂仍从默认 Tour 起步。人工组为固定规则参照，不伪造 13 次空调用来声称资源相等。

| 组 | 元准备 | 后续任务搜索 | 每生成种子的 LLM 上限 | 程序候选槽 |
|---|---:|---:|---:|---:|
| manual fixed | 无 | 固定人工程序直接评价 | 0 | 0 |
| random search | 无 | 既有 `random_program()`，13 槽并保存第 3 槽前缀 | 0 | 13 |
| fixed Qwen3-8B | `Improver()` 固定 | 13 槽，保存同一序列第 3 槽前缀 | 13 | 13 |
| controlled meta update | 两轮，每轮 1 次协议提案＋旧/新各 2 个程序候选 | 只转移协议；从默认 Tour/空历史起步，3 槽 | `2*(1+2+2)+3 = 13` | `2*(2+2)+3 = 11` |

每生成种子共 37 个程序候选槽、2 个协议槽，最多 26 次 LLM 调用；若做 3 个预先固定生成重复，为 111 程序槽、6 协议槽、78 次 LLM 上限。生成重复仍可能共用任务过程，不能冒充新的独立现场。这里只给运行封套，当前不启动正式调用。

- **等后续机会视角**：manual、random-prefix3、fixed-prefix3、meta-transfer3。主要问题是“转移来的协议是否使同样 3 次后续提案更有效”；元准备成本全部另列。meta 的 13 次调用不能写成只花 3 次。
- **等端到端模型调用上限视角**：fixed13 与 meta 的 `10+3`。问题是“把同样 13 次模型调用投在元准备上是否划算”；两组后续候选数不同，不能同时声称等后续搜索预算。random13 提供不用语言模型的搜索参照。
- 两个视角共享前缀，前 3 次不再执行、不再次收费，也不是独立重复。新有限策略若选择随机生成或保留父项导致少用资源，登记实际值；未使用额度不变成免费追加候选。

这延用 M12B 的公平预算经验，同时删去旧随机元协议作为独立第五个元层组；旧证据本身保留。M12B 曾为 direct、random-meta、model-meta 分别给 13 次上限，但实际 token 和评价耗时不等。[m12b.py](../../src/patrol_rsi/m12b.py) L36–53、L301–315；[旧方法复核](../2026-10-28-m12b-improver/代码与方法复核.md) L35–41。

## 6. 五臂行动小试验与四组元层实验分开

本轮另一个可准备的窄问题是：在现有开发观测/过程上，统一只允许一次全局补查后，程序选择是否改变可执行行动。它不产生新的训练样本，不证明 RSI 迁移增益，也不读取旧 final 行。按根任务预选的开发 seeds `20000 / 20001`，各保留 `9` 个唯一过程、各 `2` 档噪声，即 **18 个物理过程、36 个观测条件**；若实际源清单计数不同，应停止并修正清单，不能强填这个数。

| 行动臂 | 固定内容 | 共同机会约束 |
|---|---|---|
| Tour | 默认 `PatrolProgram()` | 必需巡回照常；无额外补查 |
| frozen PPO | 冻结 M11 actor 的合法候选排序 | 每回合全执行团队最多 1 次可选补查，其余返回 Tour |
| human | 预固定两条人工规则 | 同上 |
| fixed M12B retained | 在读取新开发行动结果前，按独立清单固定一个已有程序版本；不按旧 final 表现挑选 | 同上 |
| new revised | 新包装器产生且经开发门保留的固定程序版本 | 同上；评价期间不继续变异 |

五臂最多 `5*36 = 180` 个臂—观测条件引用；同一过程的两档噪声成组配对，不能把 180 当独立物理样本。每个动作都要重新核对当时资格、忙闲、占用、完成时刻、返回证书和任务预算；统一补查门只能收紧原有安全核。这里 frozen PPO 应命名为“冻结 PPO 排序＋共同 one-check 门”，不能混称旧 M12B 完整 PPO 部署（旧公开均值为每回合 220.5 次补查，见[旧公开结果](../2026-10-28-m12b-improver/运行与结果.md) L29–38）。

`fixed M12B retained` 的具体版本必须由根任务的源清单预定；本文件不凭旧测试成绩选择 trial 0/1/2。此小试验的阶段标签是 `development_action_probe`，无论是否出现正差异，都不能改写 B1b 的概率预测结论或当成后续封存任务的验收成绩。

## 7. 成本账本与必要工程验证

预算按向量报告，而不是一个“调用次数”。每一条 proposal/meta/transfer/action 收据至少包含：

```text
run_id, group, stage, task_split_hash, generation_seed, slot
parent_program_id, protocol_id, generator_id, dedup_policy_id
input_tokens, output_tokens, generation_seconds, llm_calls_attempted/completed
program_slots, protocol_slots, invalid, duplicate, accepted, reason
logical_case_references, cache_hits, unique_cases_executed, evaluation_seconds
ppo_score_calls, ppo_selected_dispatches, tour_dispatches, rule_dispatches
cvae_predict_calls, cvae_seconds, diagnostic_summary_calls
fallback_kind, fallback_executed_cases, error_stage, exception_type
source_code_hash, immutable_model_hashes, output_artifact_hashes
```

模型加载、元准备、失败尝试、回退执行、最终报告/独立审计的成本分别列出；不存在真实模型的 stub 字段应标 `simulated/not_measured`，不能写成已测的 0 token 推理。缓存命中同时记逻辑请求和真实执行；同源环境缓存可以共享算力，但不得共享另一组尚未公开的分数、候选和经验。

若元开发每轮有 `n_m` 个环境、各臂 `k` 候选，共 `r` 轮，后续有 `n_a` 个环境、`t` 候选，则最坏环境引用上限为 `r*n_m*(1+2*k) + n_a*(1+t)`，含每轮共享 Tour 起点；无效/重复可降低实际运行数但仍占 proposal 槽。这个数与 fixed 的 `n_a*(1+13)` 一般不同；调用上限相同不等于环境评价成本相同。报告 common cap、实际占用及分项差异即可，不能合称“总算力完全公平”。

推荐先完成以下 stub/纯接口检查，禁止借这些检查宣称真实科研效果：

| 检查 | 可判定的验收 | 现有基础 |
|---|---|---|
| 精确语法及拒绝越权 | 错类型、未知字段/工具、源码、第四规则/第四诊断和未实现策略均拒绝 | [test_improver_task.py](../../tests/test_improver_task.py) L15–23、L91–92、L136–139 |
| 成功/失败记忆 | none/recent/last-accepted-plus-last-failure 输出恰好符合定义，跨任务为空 | [improver_protocol.py](../../src/patrol_rsi/improver_protocol.py) L104–111；需补独立负例 |
| 配对与迁移 | 两臂同 case 指纹/初始程序/种子/槽；新任务不同分区且程序/archive/seen 重置；故意注入旧 archive 必须失败 | [test_improver_task.py](../../tests/test_improver_task.py) L102–139；需补严格分区和 initial 绑定 |
| 保留/回退 | 提升可留、相等/退化不留；非法/重复/生成异常都占槽；失败元候选不能替换旧协议 | [m12b.py](../../src/patrol_rsi/m12b.py) L205–232、L245–268；需补异常账本 |
| 公开边界 | 在嵌套诊断、history、instruction、reports 放入 final/label 哨兵；拒绝或删除，生成收据不含哨兵 | 现有提示测试仅用干净 fixture，需加强 |
| 真实执行影响的证据 | 选诊断只影响 prompt；选规则必须从选中程序走同一个解释器；改变策略 ID 没有实现时失败 | [improver_task.py](../../src/patrol_rsi/improver_task.py) L209–254 |
| 成本及前缀 | prefix3 只引用前 3 次原收据；不双计调用；计数可由逐槽收据和原始响应重新求和 | [m12b.py](../../src/patrol_rsi/m12b.py) L224–232、L303–315 |
| one-check 行动门 | 各臂同机会；全团队可选补查不超过 1；无合法候选/末尾不补查；必需 Tour 不受冷却和新门阻断 | [facility_workflow.py](../../src/patrol_rsi/facility_workflow.py) L116–144；新增全局门需单测 |

工程试跑可给现有 `adapt()` / `meta_update()` 注入只返回固定 JSON 的 caller 和按程序 ID 返回固定分数的 store。store 不创建 `ImproverTask`，不调用 `run_program()`，不加载模型，不读取过程数组；meta 与 transfer 使用不同 fixture 清单。它能证明状态转移、严格保留、分区与收据闭合，不能证明工具、CVAE/PPO、巡检收益或元更新迁移收益。既有 reset test 已有这种注入方式，但它仍不足以代替新增边界检查。[test_improver_task.py](../../tests/test_improver_task.py) L107–139。

当前可交付结论为：**现成三字段元协议与四字段任务程序足以承载受控 RSI 接口试验；新增有限策略必须逐项验收，真实下游增益仍未验证。**基准 M12B 的实际调用和负面迁移结果另见[接口调用证据](接口调用证据.md)。

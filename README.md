# Zhinengti-RSI

超算机房持续巡检：论文基线与智能体RSI。

当前研究以ESWA 2025论文为起点，研究VAE重要度估计、多执行者持续重访与智能体自改进。M1–M5使用共享DDQN，M6改为学习高层检查点分配，低层导航由共同控制器完成。

论文：[Variational model-based Deep Reinforcement Learning for Non-Homogeneous Patrolling aquatic environments with multiple unmanned surface vehicles](https://doi.org/10.1016/j.eswa.2025.126483)。

当前代码是缩小的原理参考实现。M6已完成3组×3种子的高层分配实验和66项测试：完整组新布局逾期率0.159%，风险年龄/距离启发式3.52%；无VAE组0.184%，允许本轮重复目标的学习对照为0%。结果支持继续研究高层学习，尚不支持VAE稳定增益、显式排重优势或RSI有效的结论。M6同时改变动作、表示和算法，不能与M5作单因素比较。尚未复现论文表格；源仓库及版本差异见[源代码审计](research/2026-10-26-paper-rsi/source_audit.json)。

最新M7已完成74项测试及27个冻结权重部署／规划任务：新压力条件下，完整学习组逾期4.875%、成对巡回4.324%。学习在相近机柜布局上更好，在交错通道布局上落后；VAE与排重的独立任务收益仍未证实。后续优先做新布局训练和表示对照。

M8现已完成82项测试和12项等预算从头训练。混合训练中，关系信息使三个配对种子的逾期均下降，均值由4.357%降至3.719%；但baffle仍落后成对巡回，且出现92步未重访。下一步优先研究长期漏检与重访约束，结果范围是同布局族的新巡检实例，未证明任意新墙体拓扑泛化。

M9按新的导师截图和用户JSON参考补充了供电—液冷—自控降阶模型：94项测试、150条机制轨迹、15次规模试验及完整数值重放通过。已生成风险与观测热点图。这里是合成模型内部验证，尚无现场标定或新的学习算法结论；B-PID保留为未校准的辅助观测接口。

## 研究说明

- [方向校准](research/2026-10-26-paper-rsi/方向校准与实施记录.md)
- [运行记录](research/2026-10-26-paper-rsi/运行结果.md)
- [固定基线与语言模型试运行设计](research/2026-10-26-fixed-baseline/实验设计.md)
- [本轮运行与结果：改善、退化和证据](research/2026-10-26-fixed-baseline/运行与结果.md)
- [M1复核与下一阶段修正](research/2026-10-26-m2-revisit/M1复核与修正.md)
- [M2补查技能与多种子比较设计](research/2026-10-26-m2-revisit/实验设计.md)
- [M2运行与结果：补查收益及改进器对照](research/2026-10-26-m2-revisit/运行与结果.md)
- [M2再次复核与后续研究安排](research/2026-10-26-m3-transfer/复核与后续方向.md)
- [M3流程迁移和完整配置空间实验设计](research/2026-10-26-m3-transfer/实验设计.md)
- [M3运行与结果：迁移无收益，后续方向调整](research/2026-10-26-m3-transfer/运行与结果.md)
- [M3复核与M4可执行工作流策划](research/2026-10-26-m4-workflow/M3复核与阶段策划.md)
- [M4冻结设计：工作流和改进器两层评价](research/2026-10-26-m4-workflow/实验设计.md)
- [M4运行与结果：程序执行、元更新与搜索对照](research/2026-10-26-m4-workflow/运行与结果.md)
- [M4复核与M5内层归因规划](research/2026-10-26-m5-attribution/复核与阶段规划.md)
- [M5冻结设计：时限感知学习与模块消融](research/2026-10-26-m5-attribution/实验设计.md)
- [M5运行与结果：内层贡献、配对消融与规划对照](research/2026-10-26-m5-attribution/运行与结果.md)
- [M5复核与M6高层分配规划](research/2026-10-27-m6-dispatch/M5复核与下一阶段规划.md)
- [M6冻结设计：共同导航和联合概率训练](research/2026-10-27-m6-dispatch/实验设计.md)
- [M6运行与结果：高层学习有效，VAE与排重增益待证](research/2026-10-27-m6-dispatch/运行与结果.md)
- [M6实现与解释复核](research/2026-10-27-m6-dispatch/代码与方法复核.md)
- [M6阶段验收与后续安排](research/2026-10-27-m6-dispatch/阶段验收.md)
- [M6再次复核与M7安排](research/2026-10-27-m7-stress/M6复核与M7规划.md)
- [M7冻结设计：固定权重压力迁移与部署交叉消融](research/2026-10-27-m7-stress/实验设计.md)
- [M7代码与方法复核](research/2026-10-27-m7-stress/代码与方法复核.md)
- [M7运行与结果：布局迁移、规划和部署交叉](research/2026-10-27-m7-stress/运行与结果.md)
- [M7阶段验收](research/2026-10-27-m7-stress/阶段验收.md)
- [M7复核与M8安排](research/2026-10-27-m8-representation/M7复核与M8规划.md)
- [M8冻结设计：布局训练与演员关系信息](research/2026-10-27-m8-representation/实验设计.md)
- [M8代码与方法复核](research/2026-10-27-m8-representation/代码与方法复核.md)
- [M8文献依据与使用边界](research/2026-10-27-m8-representation/文献依据.md)
- [M8运行与结果：关系输入、布局训练和长期漏检](research/2026-10-27-m8-representation/运行与结果.md)
- [M8阶段验收](research/2026-10-27-m8-representation/阶段验收.md)
- [M8复核与M9参考建议取舍](research/2026-10-27-m9-coupled/M8复核与建议取舍.md)
- [M9冻结设计：供电—液冷—自控](research/2026-10-27-m9-coupled/实验设计.md)
- [M9运行与结果：多系统故障传播和风险热点](research/2026-10-27-m9-coupled/运行与结果.md)
- [M9资料、国标及原VAE论文指标核对](research/2026-10-27-m9-coupled/资料与标准核对.md)
- [M9实现复核与验收](research/2026-10-27-m9-coupled/阶段验收.md)
- [项目术语](CONTEXT.md)
- 论文资料与已制作的组会报告分别保存在literature/与reports/。

## 运行

Python 3.11以上，环境与RSI版本接口测试只需要NumPy。神经网络测试和冒烟运行需要PyTorch 2.6。集群已检查到对应模块，计算在Slurm分配资源内执行。

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

本地没有PyTorch时，神经网络测试会明确跳过。集群验证必须运行全部测试，并检查实际CUDA设备。

```bash
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src"
sbatch --output="$PWD/slurm-%j.out" --error="$PWD/slurm-%j.err" cluster/patrol_smoke.sbatch "$PWD"
```

批处理分别运行paper_grid和supercomputer场景，训练12个小批量VAE更新及少量共享DDQN更新，保存事件日志、检查点、设备信息和summary.json。它不会下载任何预训练资源，也不会调用语言模型。结果目录拒绝覆盖。

## 原型结构

- environment.py：局部测量、估计输入、未检程度、重叠奖励、移动与协作合法性。
- models.py：条件prior/posterior VAE-UNet、共享Dueling Q、Double Q更新与优先回放。
- versions.py：有界修改提案与不可覆盖的版本记录。人工接口样例标为human_fixture，与真实语言模型输出分别记录。
- smoke.py：受限规模的工程执行，检查参数确实更新并保存证据。
- experiment.py / train.py：冻结数据划分、VAE预训练、3个种子的共享DDQN训练及4种固定控制器比较。
- improve.py：本地开源语言模型读取开发日志，生成有界配置及诊断指令，评价后决定保留。
- tests/：未检更新、重复测量、真值隔离、运动冲突、prior部署、学习更新与提案边界。

正式论文实验仍需核对原文的网络宽度、数据与训练预算，并扩展等预算搜索、多语言模型种子和新任务评价。当前没有将仿真重要度解释成真实故障概率。

## 分阶段训练与外层试运行

固定基线首先在Slurm上训练VAE，再提交策略数组；每一步都检查终态、日志和产物。VGG感知损失所需官方权重须提前下载到`assets/vgg16_bn-6c64b313.pth`。计算节点不下载资源，结果目录拒绝覆盖。

```bash
sbatch --output="$PWD/vae-%j.out" --error="$PWD/vae-%j.err" cluster/train_baseline.sbatch "$PWD" vae
# 确认VAE验证门槛通过并完成后，再提交：
sbatch --array=0-2%3 --output="$PWD/policy-%A_%a.out" --error="$PWD/policy-%A_%a.err" cluster/train_baseline.sbatch "$PWD" policy
```

VAE使用GroupNorm和向量潜变量融合，修正缩小工程版本的饱和及逐像素采样问题。训练只从训练划分取样，按验证指标选择检查点；最终测试结果不能用于重新挑选种子或检查点。

语言模型通过`.[neural,open-model]`依赖运行。本轮选择Qwen3-8B固定提交，本地FP16推理；不使用云端API，也不微调语言模型。先校验模型清单并完成GPU生成检查，再使用预先确定的策略种子0接入外层。模型只能修改刷新间隔、采样次数和执行者协调顺序，测试评价与观测边界固定。

```bash
sbatch --output="$PWD/rsi-%j.out" --error="$PWD/rsi-%j.err" cluster/improve_pilot.sbatch \
  "$PWD" /absolute/model-venv /absolute/verified-Qwen3-8B \
  /absolute/best-vae.pt /absolute/seed-0/best-policy.pt
```

`fixed_improver`保持诊断指令不变，`updated_improver`允许保留生成的诊断指令。每组3次提案只构成外层流程试运行；实际改善、退化或无变化均保留，不等同于已证明递归自改进有效。

## M2：长期漏检与等候选预算比较

M2继续使用冻结的VAE和3组共享DDQN权重。`revisit.py`提供人工编写的锁定目标补查技能，`m2.py`比较人工默认工具、等候选槽随机搜索、固定诊断工具改进器和可更新诊断工具改进器。工具收益与改进器收益分别报告，不能将人工规划带来的改善归为RSI。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m2_compare.sbatch \
  "$PWD" /absolute/model-venv /absolute/verified-Qwen3-8B /absolute/M1-results smoke
# 试运行只读取开发集，通过后提交3策略种子×3提案种子：
sbatch --array=0-8%3 --output="$PWD/compare-%A_%a.out" --error="$PWD/compare-%A_%a.err" cluster/m2_compare.sbatch \
  "$PWD" /absolute/model-venv /absolute/verified-Qwen3-8B /absolute/M1-results
```

每个优化方法6候选槽，非法和重复提案消耗槽位。选择时优先覆盖，再看固定24步逾期率和原加权未检程度。新测试集在全部选择结束后才使用，包含24种新布局。误差按3个策略种子分层汇总，9组配对不当作9次独立训练；等候选槽不代表相同总计算成本。

## M3：流程迁移检验

M3重新开始任务配置和历史，只从M2继承冻结诊断方案，比较人工固定覆盖、人工固定重访、迁移方案及随机搜索。提示词不含填好参数的答案示例，并对110种配置建立开发枚举锚点。枚举使用更多候选，不作为等预算方法；所有配置仍只在开发集选择。

```bash
# 先smoke；通过后，枚举3个策略种子：
sbatch --array=0-2%3 --output="$PWD/enumerate-%A_%a.out" --error="$PWD/enumerate-%A_%a.err" cluster/m3_transfer.sbatch \
  "$PWD" /absolute/model-venv /absolute/Qwen3-8B /absolute/M1-results /absolute/M2-results enumerate "$PWD/results/enumeration"
# 等枚举完成，再提交9组迁移比较；建议使用Slurm afterok依赖：
sbatch --array=0-8%3 --output="$PWD/transfer-%A_%a.out" --error="$PWD/transfer-%A_%a.err" cluster/m3_transfer.sbatch \
  "$PWD" /absolute/model-venv /absolute/Qwen3-8B /absolute/M1-results /absolute/M2-results transfer "$PWD/results/enumeration"
```

比较的是现有有界诊断流程在新布局上是否更会提出任务修改，不是开放式代码自修改。后续是否扩大技能与工作流空间，以固定正确工具和枚举对照的结果为依据。

## M4：可执行工作流和验证式改进器更新

`workflow.py`解释最多3个有条件的分配规则，支持目标排序、跨执行者分配、保持到观测完成、卡住释放和DDQN后备。模型生成的JSON经固定语法检查后真实执行，不把分析文字当作程序。

`m4.py`首先比较旧/候选改进器各2次内层工作流提案，候选严格更好才保留。冻结后的诊断/记忆流程用于另一批任务开发场景，任务程序和历史均重新开始。任务层另有随机工作流搜索、无记忆、仅记忆及人工规则对照；元训练的额外离线成本单列。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m4_workflow.sbatch \
  "$PWD" /absolute/model-venv /absolute/Qwen3-8B /absolute/M1-results smoke
# 只用开发集的试运行及完整测试通过后：
sbatch --array=0-8%3 --output="$PWD/workflow-%A_%a.out" --error="$PWD/workflow-%A_%a.err" cluster/m4_workflow.sbatch \
  "$PWD" /absolute/model-venv /absolute/Qwen3-8B /absolute/M1-results
```

本阶段加入公开的异质重访时限，VAE及共享DDQN仍冻结。工作流接管比例和随机后备动作对照用于检查神经策略的实际作用；它是受限语言中的流程合成，不代表开放式代码进化或已证明通用RSI。

## M5：重新训练并验证内层贡献

`deadline.py`提供公开时限输入、共同任务奖励和显式距离势函数；`m5.py`从头训练8通道共享DDQN，以同预算比较完整VAE组、无VAE、隐藏时限和无距离引导。主结果不使用规划接管，另列补查和人工流程组合。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m5_attribution.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt smoke
# 完整测试、CUDA参数更新与新读检查点重放通过后：
sbatch --array=0-11%6 --output="$PWD/train-%A_%a.out" --error="$PWD/train-%A_%a.err" cluster/m5_attribution.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt
```

12任务分别4组×3训练种子，各2,400回合；按独立验证集选择，最后测试配对同布局的时限与噪声条件。无VAE训练组是主要模块消融，部署时关闭VAE另列为敏感性检查。距离势函数是人工奖励引导，单独消融，不把它当作学习自主发现。M5不增加语言模型或RSI调用。

## M6：高层检查点分配和共同低层导航

`dispatch.py`让空闲执行者选择检查点并保持承诺直到被观测；所有方法使用同一公开地图控制器和周期避碰。`dispatch_model.py`使用共享实体策略、集中公开状态价值及联合条件概率的PPO更新，逐次回放实际采样掩码，避免独立Q备份与联合分配不一致。

它改变了M5动作、表示和算法，属于方法重构，不是原文DDQN逐项复现或去中心化部署。比较完整组、无VAE、独立重复目标选择，以及共同控制器下的启发式、随机目标和初始策略。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m6_dispatch.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt smoke
# 通过测试后，先只用固定小场景检查可学性，不读最终测试：
sbatch --output="$PWD/preflight-%j.out" --error="$PWD/preflight-%j.err" cluster/m6_dispatch.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt preflight
# 预检完整结束且验证优于初始模型后再提交主实验：
sbatch --array=0-8%6 --output="$PWD/train-%A_%a.out" --error="$PWD/train-%A_%a.err" cluster/m6_dispatch.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt
```

低层路径和不重复目标掩码均为明确人工规则，不能把其作用称为神经策略自主发现。每组3个配对训练种子，各1,200回合，验证后才读48个最终条件。

M6已完成，完整证据见上述结果报告。逐场轨迹无损保存在`summary.json.gz`，`scripts/m6_evidence_io.py`支持透明读取；不是删除字段后的摘要。实际权重保留在集群与本地私有目录，仓库提供对应SHA-256及参数审计。复核已保存的结果时无需重新训练：

```bash
python scripts/audit_m6.py --results research/2026-10-27-m6-dispatch/evidence --output /new/path/m6-audit.json
python scripts/report_m6.py --results research/2026-10-27-m6-dispatch/evidence --output /new/path/m6-report
python scripts/summarize_m6_supplement.py --results research/2026-10-27-m6-dispatch/evidence --output /new/path/m6-supplement.json
```

前两个命令仍需要PyTorch；最后一个只读取数据，使用Python标准库。

## M7：压力迁移与部署开关交叉

M6复核确认：静态场中目标测量永久保留，全部目标看过后，VAE不再影响这些目标的估计。完整组平均第10.625步看完目标，后续88.93%的决策步属于这种情况。M7先固定M6全部9份权重，检查更复杂布局和紧时限下的迁移，区分训练方式与部署工具／排重规则的作用。

`stress.py`复用相同低层控制，增加交错通道布局、时刻位置配对噪声、全队到达收益匹配和成对巡回规则。`m7.py`评价7种学习部署设置×3检查点种子与6种规划设置，共27任务、96条件、2,592条记录；不做新训练或检查点重选。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m7_stress.sbatch \
  "$PWD" /absolute/M6-results /absolute/frozen_m6_weights.json /absolute/M1-results/vae/best-vae.pt smoke
# 开发门槛完整通过后：
sbatch --array=0-26%6 --output="$PWD/eval-%A_%a.out" --error="$PWD/eval-%A_%a.err" cluster/m7_stress.sbatch \
  "$PWD" /absolute/M6-results /absolute/frozen_m6_weights.json /absolute/M1-results/vae/best-vae.pt
```

部署交叉是冻结权重的敏感性检验，不能当成重新训练消融；全队匹配仅对一次加性代理目标穷举最优，不是最优持续巡检。启动逾期下界与完整任务可行性分别报告。

M7全部2,592条记录已通过物理／规划重放，本地复算全部分层均值、样本标准差和工具／掩码交叉结果。四张结果图、原始压缩轨迹和Slurm终态见M7报告。没有增加新的语言模型或RSI调用。

## M8：从头训练与演员关系输入

M8以2种训练布局×2种演员输入，区分M7的分布迁移与表示限制。`relations.py`从公开地图与任务承诺构造点间及团队关系；`relation_model.py`使用共享MLP和集合均值。四组同为22,594参数、相同完整公开状态价值网络，local演员将两项关系上下文置零；不使用注意力名称。

每组3种子、1,200回合，VAE、PPO、公开奖励、任务承诺与人工低层固定。两种演员都先通过完整测试、CUDA更新和固定baffle场景种子0的可学性门槛，再提交12任务及独立规划评价。新划分与M7最终布局分离，不做检查点预训练迁移。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m8_representation.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt smoke
sbatch --array=0-1%2 --output="$PWD/preflight-%A_%a.out" --error="$PWD/preflight-%A_%a.err" cluster/m8_representation.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt preflight
# 确认两项可学性门槛和新读权重重放均通过后：
sbatch --array=0-11%6 --output="$PWD/train-%A_%a.out" --error="$PWD/train-%A_%a.err" cluster/m8_representation.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt
sbatch --output="$PWD/planners-%j.out" --error="$PWD/planners-%j.err" cluster/m8_representation.sbatch \
  "$PWD" /absolute/M1-results/vae/best-vae.pt planners
```

共同验证包含baffle，rack-only组也按其选检查点，因此不称完全未见baffle评价信息的零样本研究。主因子比较以M8相同最终条件为依据；M8与M6的网络仍不同，不跨阶段作单因素归因。

M8全部2,880条记录、12份实际选中权重和CPU首场景选择已通过独立检查，本地复算全部分层／配对因子／交互统计。完整轨迹保存在gzip中，模型权重留在集群及本地私有目录。复核已取回的证据无需重新训练：

```bash
python scripts/audit_m8.py --results research/2026-10-27-m8-representation/evidence --output /new/path/m8-audit.json
python scripts/report_m8.py --results research/2026-10-27-m8-representation/evidence --output /new/path/m8-report
python scripts/summarize_m8_diagnostics.py --results research/2026-10-27-m8-representation/evidence --output /new/path/m8-diagnostics.json
python scripts/diagnose_m8_tail.py --results research/2026-10-27-m8-representation/evidence --output /new/path/m8-tail.json
```

前两项需要项目NumPy及绘图环境，诊断汇总仅用Python标准库；尾部诊断另需NumPy，不需要PyTorch。`scripts/check_m8_checkpoints.py`须读取实际私有权重，并在PyTorch环境中执行。

## M9：多系统领域模型和观测基础

`facility.py`实现供电容量／损耗、并联泵管路、冷板/CDU/两回路热量交换与反馈规则；`facility_observation.py`区分过程真值、对外遥测、专业资格和检查完成。`m9.py`冻结150条情景和稀疏规模试验，没有新神经网络训练。

```bash
sbatch --output="$PWD/smoke-%j.out" --error="$PWD/smoke-%j.err" cluster/m9_coupled.sbatch "$PWD" smoke
# 94项测试及开发场景门槛成功结束后：
sbatch --output="$PWD/suite-%j.out" --error="$PWD/suite-%j.err" cluster/m9_coupled.sbatch "$PWD"
```

全部54,000步和150个完整数值NPZ已独立重放并核对电／热收支。标量轨迹、演示完整数组和四张静态图在研究目录，约290.8MB完整数值数组留在集群索引路径。查看[离线热点页面](research/2026-10-27-m9-coupled/evidence/report/risk_hotspots.html)可切换时间和机柜；脚本逻辑通过，浏览器外观因策略限制未完成预览。

严重度代理不是故障概率；检查更新信息不等于修复故障。M10再进行同观测、同预算的学习型估计比较，M11接入异质风险调度，M12单独验证RSI增量。具体任务见M9报告。

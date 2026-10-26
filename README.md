# Zhinengti-RSI

超算机房持续巡检：论文基线与智能体RSI。

当前研究以ESWA 2025论文为基础，保留VAE重要度估计、共享多智能体DDQL和持续重访，在超算机房场景外接智能体递归自改进。

论文：[Variational model-based Deep Reinforcement Learning for Non-Homogeneous Patrolling aquatic environments with multiple unmanned surface vehicles](https://doi.org/10.1016/j.eswa.2025.126483)。

当前代码是缩小的原理参考实现，已完成三种子基线训练、真实开源模型提案试运行，以及补查技能与多种子外层比较。补查明显减少长期漏检，递归改进优势仍未得到稳定证据。尚未复现论文表格；源仓库及版本差异见[源代码审计](research/2026-10-26-paper-rsi/source_audit.json)。

## 研究说明

- [方向校准](research/2026-10-26-paper-rsi/方向校准与实施记录.md)
- [运行记录](research/2026-10-26-paper-rsi/运行结果.md)
- [固定基线与语言模型试运行设计](research/2026-10-26-fixed-baseline/实验设计.md)
- [本轮运行与结果：改善、退化和证据](research/2026-10-26-fixed-baseline/运行与结果.md)
- [M1复核与下一阶段修正](research/2026-10-26-m2-revisit/M1复核与修正.md)
- [M2补查技能与多种子比较设计](research/2026-10-26-m2-revisit/实验设计.md)
- [M2运行与结果：补查收益及改进器对照](research/2026-10-26-m2-revisit/运行与结果.md)
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

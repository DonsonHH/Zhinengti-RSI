# Zhinengti-RSI

超算机房持续巡检：论文基线与智能体RSI。

当前研究以ESWA 2025论文为基础，保留VAE重要度估计、共享多智能体DDQL和持续重访，在超算机房场景外接智能体递归自改进。

论文：[Variational model-based Deep Reinforcement Learning for Non-Homogeneous Patrolling aquatic environments with multiple unmanned surface vehicles](https://doi.org/10.1016/j.eswa.2025.126483)。

当前新代码是缩小的原理参考实现。它验证学习和执行接口，尚未复现论文表格或验证RSI优势。源仓库及版本差异见[源代码审计](research/2026-10-26-paper-rsi/source_audit.json)。

## 研究说明

- [方向校准](research/2026-10-26-paper-rsi/方向校准与实施记录.md)
- [运行记录](research/2026-10-26-paper-rsi/运行结果.md)
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
- tests/：未检更新、重复测量、真值隔离、运动冲突、prior部署、学习更新与提案边界。

正式实验需另行核对原文的网络宽度、完整损失、数据、训练预算和多随机种子，并完成开源语言模型接入及独立场景评价。当前没有将仿真重要度解释成真实故障概率。

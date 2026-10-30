"""One Chinese handoff with prior project history, counts and evidence links."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from patrol_rsi.fresh_opportunity import need, read, sha, write


ROLE={'thermal':'机柜温度','electrical':'电气接点热','cooling':'冷却流量','control':'控制／传感偏差'}
SPLIT={'development':'开发','validation_calibration':'概率校准','validation_selection':'方法选择','final_test':'最终测试'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args();root=args.workspace;b=root/'research/2026-10-30-m12c1b4b1a-fresh';e=b/'evidence'
    acceptance=read(e/'final-acceptance.json');c=read(e/'report/coverage.json');cost=read(e/'slurm-cost-review.json')
    electrical=read(e/'report/electrical-nonfinal-diagnostics.json');clock=read(e/'report/development-clock-diagnostic.json')
    cells=c['coverage_cells'];failed=[v for v in cells if not v['gate']['passed']]
    domains=acceptance['kind_horizon_domains_passing_all_four_splits']
    domain_text='、'.join(ROLE[v['kind']]+' '+str(v['horizon_s'])+'秒' for v in domains) or '没有'
    total_a=sum(v['A_valid'] for v in cells);a_pos=sum(v['A_positive'] for v in cells)
    total_b=sum(v['B_action_valid'] for v in cells);b_pos=sum(v['B_action_positive'] for v in cells)
    missing=sum(v['B_action_invalid'] for v in cells)
    without=sum(v['A_positive_without_legal_B'] for v in cells)
    full=read(e/'gate/full-tests.json');cpu=read(e/'local-cpu-tests.json')
    alignment=read(e/'public-schedule-alignment-diagnostic.json')
    report=[f'# Zhinengti-RSI项目进度、瓶颈与下一步：B4-B1a', '',
        f'本轮实际生成与工程审计已完成。覆盖结论为 **{acceptance["coverage_verdict"]}**，通过{len(cells)-len(failed)}/48个单元。**全域B4-B1b拟合允许：{acceptance["B4_B1b_full_domain_fitting_allowed"]}**。', '',
        '## 1. 先把项目要解决的事情讲明白', '',
        '设想机房里有一个巡检班组。设备一直运行，人员只能在部分位置测量。固定巡回能守住检查义务，但有时会错过短暂异常。我们希望AI利用当时能看到的温度、流量、功率和检查历史，判断何时值得加一次检查，最终再研究智能体怎样改进技能与流程。', '',
        '本轮沿用的仿真规模为128个机柜、16个分区、18名专业执行者，额定IT负载32MW。物理参数和故障比例属于合成研究设定；这些结果不等于已经在真实机房部署，也不是本轮完成了GW规模验证。', '',
        '这里有三个不同的问题：A是“未来那个时刻，设备真的异常吗”；B是“当时有合格执行者能完成检查，并测得真实异常吗”；C是“为这次检查绕路后，整个班组这一轮是否更划算”。预测A不自动解决B，预测B也不自动证明C。当前阶段只准备并验收A／B数据。', '',
        '原实验的四类异常阈值为：机柜温度大于65°C、电气接点温度大于60°C、分区总冷却流量小于33.6kg/s、参考测得的传感偏差绝对值大于1°C。这些是本仿真的任务定义，沿用原值；现场适用性仍需校准。', '',
        '已有CVAE在这里相当于一件固定的状态估计工具，只运行已冻结权重。新生成数据和测试通过，说明接口与证据链可靠；它们没有证明CVAE更好，也没有证明RSI已取得收益。本轮没有拟合逻辑回归或GRU，没有重训CVAE／PPO，也没有Qwen调用或新策略搜索。', '',
        '## 2. 上一步到这一步，具体推进了什么', '',
        '此前正式策略B3.2在独立验证中未重复开发收益，按预设门槛回退固定巡回，NO-GO结论保留。后续审查发现需要先分清可用信息、合法行动时刻和整轮净收益，因此当前优先核对预测目标与数据。', '',
        'B4-B0只审查已看过的旧数据，并预先写好下一轮的清单和判定规则。此次真正执行了那份清单：64个新底层种子、五类固定情景、两个故障起点、两档观测噪声，共640个命名物理配置、1280个观测条件；没有换种子或加重故障补分。', '',
        f'实际数值过程去重后为{c["numeric_processes"]}个。每个过程有360步物理轨迹；每个观测条件保留359份5秒间隔公开历史，抽取59个30秒预测起点和30／60／120秒三个视野，共{c["sample_rows"]:,}条起点—任务—视野记录。', '',
        '开发、校准、方法选择、最终测试分别存放。最终测试逐样本标签、轨迹、特征和故障温度明细留在远端封存目录；本报告只展示授权的12个最终覆盖单元计数。最终测试没有模型评分，也没有用于挑选预测器。', '',
        '## 3. 为什么记录很多仍可能不够训练', '',
        f'A的有效端点共{total_a:,}条，其中物理异常{a_pos:,}条；合法B样本{total_b:,}条，其中正例{b_pos:,}条。另有{missing:,}条不满足端点或检查时刻条件，保留为-1，不能算“正常”。其中{without:,}个物理正例端点没有合法B检查。', '',
        '这些是相关记录数，不是独立故障数。同一故障每隔一段时间都会被再次记账；两档传感噪声也不会创造第二次物理故障。统计验收使用底层种子组、数值过程指纹和去重保留事件，三种计数分别达到门槛。不同预测视野仍共享事件，各行事件数不能相加当独立样本量。', '',
        '“种子”是控制随机仿真背景的编号，例如让机柜负载和关键度产生不同初始条件；同一种子下的情景统一归组。“数值指纹”用于识别两份轨迹是否实际上完全相同。“事件”是某项专业检查对应的真实状态连续处于异常范围的一段时间；持续至少15秒才计入本轮主要事件覆盖门槛。', '',
        '一个合法B样本说明该次候选检查可以单独按时完成。多个候选分别可行，并不代表在同一轮能全部执行；真正的整轮取舍留给后续C评价。', '',
        '例如开发集每专业／视野至少需要8个有正例的种子、12个有正例的数值过程、20个可预测且持续至少15秒的事件，以及12个有合法负例的种子；校准和选择各为4／6／8／6，最终为8／12／20／12。规则在生成前已冻结，没有看完结果再改。', '',
        '## 4. 电气故障：区分真的没过热与来不及查', '',
        '峰值按原离散模型的积分步划分：故障期间包含最后一个受故障影响的积分步末状态，结束后从下一积分步起算。每步5秒，连续时间峰值未另行插值；完整定义随逐过程诊断保存。', '']
    for split in ('development','validation_calibration','validation_selection'):
        values=[v for v in electrical if v['split']==split]
        during=[v['diagnostic']['peak_faulted_contact_during_c'] for v in values]
        after=[v['diagnostic']['peak_faulted_contact_after_c'] for v in values]
        if values:
            report.append(f'{SPLIT[split]}集的{len(values)}个接点热配置，故障期间峰值为{min(during):.2f}–{max(during):.2f}°C，故障结束后峰值为{min(after):.2f}–{max(after):.2f}°C。逐过程与起点明细见非最终诊断文件；最终明细保持封存。')
            report.append('')
    report+=['每个视野同时列出A物理正例、B合法正例和“有物理异常但无合法B”的数量。若温度超过60°C而30秒B仍无正例，定位为行动时刻覆盖，不能把设备说成正常。若事件不足15秒，也单独列出，不能将短事件偷偷抹掉或变成长事件。', '',
        '## 5. 完整48单元覆盖与所有缺口', '',
        '表中“种子／过程／事件／负种子”是实际独立覆盖；缺口是离冻结最低要求还差多少。正例比例的分母只用合法B。', '',
        '|划分|专业|视野秒|合法B／总候选|B正例|正例比例|种子／过程／事件／负种子|缺口|通过|',
        '|---|---|---:|---:|---:|---:|---|---|---|']
    for v in cells:
        a=v['gate']['actual'];d=v['gate']['deficits'];counts='/'.join(str(x) for x in a.values());deficit='/'.join(str(x) for x in d.values())
        ratio=(f"{v['B_action_positive']/v['B_action_valid']:.4%}" if v['B_action_valid'] else '无有效样本')
        report.append('|'+ '|'.join(map(str,[SPLIT[v['split']],ROLE[v['kind']],v['horizon_s'],
            f'{v["B_action_valid"]:,}/{v["total_samples"]:,}',v['B_action_positive'],ratio,counts,deficit,'是' if v['gate']['passed'] else '否']))+'|')
    report+=['',f'四个划分均满足数据门槛的专业／视野：{domain_text}。这只是限定覆盖范围，不自动授权另一个研究协议，更不是预测收益已通过。', '',
        '## 6. 时间先验：AI会不会只记住故障几点发生', '',
        '故障起点固定为335／705秒，因此钟表本身可能携带很强的合成数据先验。我们只在开发集做固定分箱的条件熵描述：先按专业和视野分组，再逐次加入时钟、排班和当时测量是否越过原阈值。没有搜索模型，也没有用最终测试挑选特征。', '',
        f'当前开发描述：时钟关联量{clock["clock_association_bits"]:.6f}比特，排班增加{clock["schedule_additional_association_bits"]:.6f}比特，当时读数再增加{clock["reading_additional_association_bits"]:.6f}比特。', '',
        '比特数来自同一开发数据的描述分组，更多分组会自然降低经验不确定性，不能当成AI的独立预测成绩或传感器的因果贡献。固定时间设计能否迁移到不同起点、未知故障类型和现场数据，仍不知道，需要后续独立协议验证；本轮不据此更改正式验收指标。', '',
        '## 7. 哪些问题已经定位，哪些仍不能下结论', '',
        '控制／传感偏差30秒的缺口已定位到时间对齐：固定预测根是30秒的倍数，但原控制专业Tour完成时刻只落在模30为5或20秒的位置；能对齐该网格的合法额外CHECK，从最近允许预测根起至少需要60秒。因此精确30秒B目标结构性没有合法样本，既无正例也无负例。控制检查服务本身为15秒；不能误说成服务时长超过30秒。增加相同排班下的随机种子无法修复这个对齐缺口。', '',
        '这也解释了为何B4-B0旧数据里控制30秒曾有合法样本，而本轮没有：B0的预测起点每5秒一个，本轮预先冻结为每30秒一个。检查服务和物理异常没有被改掉，是抽取预测起点后，对精确完成时刻的覆盖改变了。今后可单独讨论更细预测根或按下一次合法完成时刻定义目标；本轮保持原协议，不能现场改采样后宣布通过。', '',
        '已经能回答：每个受控故障是否真实发生、是否形成足够长的事件、原专业测量能否确认、指定未来端点是否有合法执行者，以及各划分的独立正负例是否够。它们在48单元表、物理／行动分解和原件审计里逐项保留。', '',
        '仍不能回答：这些公开信息能否稳定预测未来；新增预测是否优于简单经验概率和原物理／趋势工具；更好的预测是否能让整轮巡检净收益变好；RSI能否在相同预算下提出可重复的改进；合成场景的比例能否代表真实机房。因为本轮没有拟合或策略比较，也没有真实现场校准数据。', '',
        '覆盖不足的正确处理是先说明缺在哪里。若物理事件存在而行动时刻缺位，应考虑未来另立行动机会目标或任务安排研究；若只有某些视野可靠，则可讨论独立限定协议；若事件本身太少或公开信息不足，应分别讨论物理情景或传感设计。上述讨论不能回头改本轮阈值、时长、种子和最终集来追求通过。', '',
        '## 8. 原件、测试与资源', '',
        f'独立审计覆盖1280个条件的全部A／B标签、公开特征、原噪声测量、Tour完成时刻和完整返回资格；重新运行原物理方程，逐值对比640个物理配置。真实Torch测试{full["tests"]}项、轻量CPU测试{cpu["tests"]}项均通过，零跳过／失败／错误。', '',
        f'Slurm实际分配CPU时间共{cost["allocated_CPU_minutes"]:.2f}分钟，其中生成{cost["generation_allocated_CPU_minutes"]:.2f}、审计与开销{cost["audit_overhead_allocated_CPU_minutes"]:.2f}；GPU为0，最大并发{cost["maximum_concurrent_allocations"]}。取消记录仅为尚未运行的导出／审计版本排队项，实际样本未替换。', '',
        f'公开衍生数据{c["public_derived_data_bytes"]:,}字节，低于6,000,000,000上限；另存私有物理原件{c["private_physical_source_bytes"]:,}字节，总计{c["derived_and_source_data_bytes"]:,}字节。早期gzip和后续xz均保留，xz只改变无损存储，未改观测内容。', '',
        '协议、清单、源代码归档、冻结权重、每份原件和传输包均保留SHA。详细日志、账本、原始复算收据和复现命令见本阶段evidence；最终测试原件仍在远端私有目录。', '',
        '## 9. 下一步允许做什么', '',
        acceptance['next_step'], '',
        'B3.2正式策略NO-GO及B4-B0旧数据训练NO-GO作为历史结论原样保存。本轮完成的是新数据与覆盖验收，不回写旧结果。', '']
    text='\n'.join(report)
    report_path=root/'reports/2026-10-30-B4-B1a-进度与瓶颈.md'
    need(not report_path.exists(),'New final report path must not overwrite historical handoff')
    report_path.write_text(text,encoding='utf-8')
    prior=root/'reports/2026-10-30-B4-B0-项目全量汇总.json'
    package={'文档信息':{'阶段':'B4-B1a','基准':'567bab9f60b5882d6eff536c13b7817a8356b0af','用途':'单文档科研进度、瓶颈、未知与下一步；最终逐样本数据不嵌入'},
        '当前最终验收':acceptance,'给新手的完整说明_markdown':text,'全部48单元覆盖与缺口':c,
        '各非最终维度统计':read(e/'report/coverage-dimensions.json'),
        '非最终800条件计数':json.loads(gzip.decompress((e/'report/condition-counts-nonfinal.json.gz').read_bytes())),
        '电气故障非最终逐过程诊断':electrical,'时间排班遥测开发描述':clock,
        '公开固定排班与预测网格诊断':alignment,
        '独立计数复算':read(e/'report/counts-independent.json'),'原物理方程重放与全部来源':read(e/'report/source-replay.json'),
        '独立全特征标签测量时刻审计':read(e/'report/independent-audits.json'),
        '资源账本核验':cost,'真实Torch测试':full,'轻量CPU测试':cpu,
        '来源与传输清单':read(e/'delivery-manifest.json'),'原件存储目录及SHA':read(e/'report/raw-storage.json'),
        '历史_B4B0此前全部项目材料_原样':read(prior),'历史快照SHA256':sha(prior),
        '本轮文字材料':{p.name:p.read_text(encoding='utf-8') for p in b.glob('*.md')}}
    write(root/'reports/2026-10-30-B4-B1a-项目全量汇总.json',package)
    write(b/'阶段状态.json',acceptance)
    print('FRESH_REPORT_AND_SINGLE_JSON_WRITTEN',acceptance['coverage_verdict'],flush=True)


if __name__=='__main__':main()

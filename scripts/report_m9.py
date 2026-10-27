"""M9 data figures and a self-contained risk / observation inspection view."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from patrol_rsi.facility import SCENARIOS
from patrol_rsi.m9 import contract


def describe(values):
    return {'per_simulation_seed': values, 'mean': float(np.mean(values)),
            'sample_sd_across_simulation_seeds': float(np.std(values, ddof=1))}


def write_hotspot_html(output, arrays):
    severity = np.asarray(arrays['severity_proxy'])
    estimate = np.asarray(arrays['predicted_severity_proxy'])
    counts = np.asarray(arrays['thermal_inspection_counts'])
    stale = severity * np.minimum(np.asarray(arrays['inspection_age_s']) / 300, 4)
    data = {'time': arrays['time_s'], 'truth': severity.tolist(), 'estimate': estimate.tolist(),
            'counts': counts.tolist(), 'stale': stale.tolist(), 'temperature': arrays['rack_temperature_c'],
            'predictedTemperature': arrays['predicted_temperature_c'], 'flow': arrays['rack_flow_kg_s'],
            'age': arrays['inspection_age_s'], 'criticality': arrays['criticality']}
    payload = json.dumps(data, separators=(',', ':'))
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M9 多系统风险与观测图</title><style>body{font:15px system-ui,"Microsoft YaHei",sans-serif;margin:24px auto;padding:0 20px;max-width:1440px;color:#18344a;background:#f7fafc}h1{font-size:27px}p{line-height:1.65}.notice{background:#edf3f8;padding:12px;border-left:4px solid #2f688d}.controls{display:flex;gap:18px;align-items:center;margin:22px 0}input{flex:1}.panels{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}.grid{display:grid;grid-template-columns:repeat(8,1fr);gap:3px}.cell{border:0;min-height:19px;cursor:pointer;padding:0;border-radius:2px}.cell:focus{outline:3px solid #111}.cell.selected{outline:2px solid #111}.caption{font-size:12px;min-height:48px;color:#405d72}.panel h2{font-size:17px}#detail{margin-top:24px;border-top:1px solid #a9bfce;padding-top:12px}table{border-collapse:collapse;width:100%;max-width:700px}td{padding:7px;border-bottom:1px solid #d6e1e8}small{color:#4a6576}@media(max-width:1000px){.panels{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.panels{grid-template-columns:1fr}}
</style><h1>M9：供电—冷却—自控风险与观测</h1><p class="notice">合成联合仿真评估视图：种子0、128机柜、combined、反馈控制。颜色表示严重度代理，<b>不是故障概率</b>。真值只供评价；估计来自名义物理预测器，未运行VAE或B-PID。</p>
<div class="controls"><label for="time">时间</label><input id="time" type="range" min="0" max="359" value="179"><output id="clock"></output></div>
<div class="panels"><section class="panel"><h2>仿真严重度真值</h2><p class="caption">温度、电连接、服务与流量分量；仅评价器可见。</p><div id="truth" class="grid"></div></section><section class="panel"><h2>部分观测估计</h2><p class="caption">名义模型＋已取得的遥测和实测；与真值共享色标。</p><div id="estimate" class="grid"></div></section><section class="panel"><h2>有效热检查次数</h2><p class="caption">资格、路程和服务时间满足后才计数；不表示风险大小。</p><div id="counts" class="grid"></div></section><section class="panel"><h2>风险加权未核验指数</h2><p class="caption">真值严重度×检查陈旧度，只供评估；不是修复后残余故障概率。</p><div id="stale" class="grid"></div></section></div>
<p><small>每行是一个pod、每列为其机柜索引，属于拓扑排列示意，不是现场平面图。点击单元格查看依赖链。注入时段300–1200s；控制与热惯性可能使影响延续至恢复阶段。</small></p><div id="detail"></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),names=['truth','estimate','counts','stale'];let selected=0;
const fieldMax=(field,minimum)=>field.reduce((m,row)=>row.reduce((v,x)=>Math.max(v,x),m),minimum);
const maxima={truth:Math.max(fieldMax(data.truth,.01),fieldMax(data.estimate,.01)),estimate:0,counts:fieldMax(data.counts,1),stale:fieldMax(data.stale,.01)};maxima.estimate=maxima.truth;
function color(v,m){let x=Math.max(0,Math.min(1,v/m));let a=x<.5?[42,92,134]:[244,204,103],b=x<.5?[244,204,103]:[186,42,45],t=x<.5?x*2:(x-.5)*2;return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;}
for(const name of names){let grid=document.getElementById(name);for(let rack=0;rack<128;rack++){let cell=document.createElement('button');cell.className='cell';cell.dataset.rack=rack;cell.setAttribute('aria-label',`机柜 ${rack}`);cell.onclick=()=>{selected=rack;update()};grid.append(cell)}}
const slider=document.getElementById('time');slider.max=data.time.length-1;slider.oninput=update;
function update(){let frame=+slider.value,t=data.time[frame],pod=Math.floor(selected/8);document.getElementById('clock').textContent=`${t.toFixed(0)} s / ${(t/60).toFixed(1)} min`;
for(const name of names){document.querySelectorAll(`#${name} .cell`).forEach((cell,rack)=>{let v=data[name][frame][rack];cell.style.background=color(v,maxima[name]);cell.title=`rack ${rack}: ${v.toFixed(4)}`;cell.classList.toggle('selected',rack===selected)})}
let fields=[['严重度真值',data.truth[frame][selected].toFixed(4)],['严重度估计',data.estimate[frame][selected].toFixed(4)],['温度真值 / 名义预测 (°C)',`${data.temperature[frame][selected].toFixed(2)} / ${data.predictedTemperature[frame][selected].toFixed(2)}`],['流量 (kg/s)',data.flow[frame][selected].toFixed(3)],['距有效热检查 (s)',data.age[frame][selected].toFixed(0)],['有效热检查次数',data.counts[frame][selected]],['资产重要度',data.criticality[selected]]];
document.getElementById('detail').innerHTML=`<h2>机柜 ${selected} · pod ${pod}</h2><table>${fields.map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table><p><b>供电链：</b>10kV入口 → 变压/380V交流 → UPS → 直流变换 pod ${pod} → 机柜 ${selected}<br><b>冷却链：</b>冷却塔 ↔ 一次泵/一次环路 ↔ CDU ${pod} ↔ 冷板 ${selected}<br><b>自控链：</b>温度/流量传感 → 控制器 ${pod} → 二次泵及负载限制</p><p>当前B-PID接口仅接收未校准的图像参考，不提供本页温度或概率数值。有效检查更新信息，未模拟维修。</p>`;}
update();
</script></html>'''
    # Payload contains numeric arrays only, so no source strings enter script.
    (output / 'risk_hotspots.html').write_text(template.replace('__DATA__', payload), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.results / 'case-index.json').read_text())
    records = [json.loads(gzip.decompress((args.results / row['record']).read_bytes())) for row in index]
    groups = {}
    for scenario in SCENARIOS:
        groups[scenario] = {}
        for control in ('feedback', 'fixed_commands'):
            cases = sorted([row for row in records if row['scenario'] == scenario and row['control'] == control], key=lambda row: row['seed'])
            groups[scenario][control] = {metric: describe([row['metrics'][metric] for row in cases]) for metric in cases[0]['metrics']}
    paired = {}
    for scenario in SCENARIOS:
        left = sorted([r for r in records if r['scenario'] == scenario and r['control'] == 'fixed_commands'], key=lambda r:r['seed'])
        right = sorted([r for r in records if r['scenario'] == scenario and r['control'] == 'feedback'], key=lambda r:r['seed'])
        paired[scenario] = {metric: describe([b['metrics'][metric] - a['metrics'][metric] for a,b in zip(left,right)]) for metric in left[0]['metrics']}
    summary = json.loads((args.results / 'summary.json').read_text())
    aggregate = {'groups': groups, 'paired_feedback_minus_fixed': paired, 'scale_trials': summary['scale_trials'],
                 'cases': len(records), 'simulation_seeds_per_condition': 5, 'training_tasks': 0,
                 'scope': contract()['scope'], 'risk_probability': False,
                 'estimators': ['measurement_hold', 'nominal_physical_predictor']}
    (args.output / 'aggregate.json').write_text(json.dumps(aggregate, indent=2))
    arrays = json.loads(gzip.decompress((args.results / 'demo.json.gz').read_bytes()))
    times = np.asarray(arrays['time_s']); frame = int(np.flatnonzero(times == 900)[0])
    true = np.asarray(arrays['severity_proxy']); predicted = np.asarray(arrays['predicted_severity_proxy'])
    count = np.asarray(arrays['thermal_inspection_counts']); age = np.asarray(arrays['inspection_age_s'])
    stale = true * np.minimum(age / 300, 4)
    fig, axes = plt.subplots(1,4,figsize=(14,5.5),layout='constrained')
    vmax=max(float(true.max()),float(predicted.max()),.01)
    for ax, field, title, limit in zip(axes,[true,predicted,count,stale],['Simulated severity','Nominal observer severity','Completed thermal inspections','Severity x inspection staleness'],[vmax,vmax,max(float(count.max()),1),max(float(stale.max()),.01)]):
        im=ax.imshow(field[frame].reshape(16,8),aspect='auto',vmin=0,vmax=limit,cmap='YlOrRd')
        ax.set(title=title,xlabel='Rack within pod',ylabel='Pod index');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('M9 fixed display: combined / feedback / seed 0 / t=900 s; synthetic severity, not fault probability')
    fig.savefig(args.output/'m9-hotspots.png',dpi=170);fig.savefig(args.output/'m9-hotspots.svg');plt.close(fig)
    baseline=next(r for r in records if r['scenario']=='healthy' and r['control']=='feedback' and r['seed']==0)
    demo=next(r for r in records if r['scenario']=='combined' and r['control']=='feedback' and r['seed']==0)
    t=np.asarray([r['time_s'] for r in demo['scalar_trace']]);fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
    axes[0].plot(t,np.asarray(arrays['pump_command'])[:,0],label='Pump command');axes[0].plot(t,np.asarray(arrays['pump_speed'])[:,0],label='Actual pump speed');axes[0].set_ylabel('Relative speed')
    axes[1].plot(t,[r['max_rack_c'] for r in demo['scalar_trace']],label='Combined facility max T');axes[1].plot(t,[r['max_rack_c'] for r in baseline['scalar_trace']],label='Healthy facility max T');axes[1].set_ylabel('Temperature (C)')
    axes[2].plot(t,[r['impact_weighted_severity'] for r in demo['scalar_trace']],label='Severity proxy');axes[2].plot(t,[r['impact_weighted_severity'] for r in baseline['scalar_trace']],label='Healthy severity');axes[2].set(xlabel='Time (s)',ylabel='Severity index')
    for ax in axes:ax.axvspan(300,1200,color='#e6ba66',alpha=.15);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('M9 cross-system response; paired seed 0; highlighted injection interval; no repairs simulated')
    fig.savefig(args.output/'m9-propagation.png',dpi=170);fig.savefig(args.output/'m9-propagation.svg');plt.close(fig)
    values=np.asarray([[groups[s][c]['mean_impact_weighted_severity']['mean'] for c in ('fixed_commands','feedback')] for s in SCENARIOS])
    fig,ax=plt.subplots(figsize=(7,8),layout='constrained');im=ax.imshow(values,aspect='auto',cmap='YlOrRd');ax.set_xticks([0,1],['Fixed nominal commands','Feedback rules']);ax.set_yticks(range(15),SCENARIOS)
    for i in range(15):
        for j in range(2):ax.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='white' if values[i,j]>values.max()*.6 else '#222',fontsize=9)
    fig.colorbar(im,ax=ax,label='Mean impact-weighted severity proxy');ax.set_title('M9 15 synthetic scenarios; five paired seeds; neither column is a learned policy')
    fig.savefig(args.output/'m9-scenarios.png',dpi=170);fig.savefig(args.output/'m9-scenarios.svg');plt.close(fig)
    sizes=contract()['scale_racks'];median=[];memory=[]
    for size in sizes:
        trials=[r for r in summary['scale_trials'] if r['racks']==size];median.append(np.median([r['simulation_360_steps_seconds'] for r in trials]));memory.append(trials[0]['model_numeric_state_bytes']/1024**2)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained');axes[0].loglog(sizes,median,marker='o');axes[0].set(xlabel='Synthetic racks',ylabel='360-step simulator time (s)',title='Median of three CPU runs');axes[1].loglog(sizes,memory,marker='o');axes[1].set(xlabel='Synthetic racks',ylabel='Explicit numeric state (MiB)',title='Excludes topology Python objects and observer')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('M9 reduced-order sparse model scaling; no real-plant fidelity or FLOPS inference')
    fig.savefig(args.output/'m9-scaling.png',dpi=170);fig.savefig(args.output/'m9-scaling.svg');plt.close(fig)
    write_hotspot_html(args.output,arrays)
    print('M9_REPORT_COMPLETE',len(records),'cases; four figures and offline risk view',flush=True)


if __name__=='__main__':main()

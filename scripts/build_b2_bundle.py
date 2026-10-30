"""Finalize the bounded B2 evidence and one novice-readable project bundle."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-30-m12c1b4b2-alignment'
RUN=STAGE/'evidence/run'


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    gate=read(RUN/'reviewed-v6/gate/full-tests.json');cpu=read(STAGE/'evidence/local-cpu-tests.json')
    assert gate['tests']==279 and cpu['tests']==58 and all(v['success'] and not(v['skipped'] or v['errors'] or v['failures']) for v in (gate,cpu))
    results=read(STAGE/'evidence/b2-results.json');usage=read(STAGE/'evidence/interface-usage.json')
    cal=read(RUN/'calibration/calibration-diagnosis.json');action=[read(RUN/'actions'/f'seed-index-{i}'/'action-diagnosis.json') for i in (0,1)]
    ledger=read(RUN/'resource-ledger.json');storage=read(RUN/'storage.json')
    assert cal['new_calibration_fits']==42 and cal['replay_fits']==14 and cal['new_LR_fits']==0 and not cal['selection_or_final_access']
    assert len(list((RUN/'calibration').glob('*--start.json')))==42
    assert len(list((RUN/'calibration').glob('*--result.json')))==42
    assert len(list((RUN/'calibration').glob('*--replay-start.json')))==14
    assert len(list((RUN/'calibration').glob('*--replay-result.json')))==14
    for attempt in cal['new_calibrators']:
        assert attempt['calibrator']['status']=='completed_within_fixed_optimizer_budget' and attempt['calibrator']['closure_evaluations']<=300
    for original in cal['original_replays']:
        assert original['original']['a']==original['replayed']['a'] and original['original']['b']==original['replayed']['b']
    assert sum(v['conditions'] for v in action)==36 and all(v['phases_opened']==['development'] and not v['final_access'] for v in action)
    assert sum(v['independent_selected_schedule_checks'] for v in action)==70
    assert results['action_summary']['total_legal_current_candidates']==298152
    repair=read(RUN/'interface-smoke-v3/interface-smoke.json')
    assert repair['receipt_correctness_repair']['all10_action_and_measurement_hashes_unchanged']
    assert len(repair['rows'])==10 and all(v['usage']['global_optional_checks']<=1 and v['metrics']['tasks_violating_exact_gap']==0 for v in repair['rows'])
    meta=read(STAGE/'evidence/meta-interface-final/meta-interface-smoke.json')
    assert meta['real_LLM_calls']==meta['real_environment_episode_evaluations']==0 and meta['only_protocol_transferred']
    assert ledger['all_completed_exit0'] and ledger['max_concurrency']<=2 and ledger['GPU_allocations']==ledger['real_LLM_calls']==0
    assert storage['bytes_all_new_run_files']<2_000_000_000
    for snapshot in STAGE.glob('code-v*-manifest.json'):
        m=read(snapshot);assert sha(STAGE/'source'/m['archive'])==m['sha256']
    original_source=read(STAGE/'code-v1-manifest.json');final_source=read(STAGE/'code-v6-manifest.json')
    for name in ('src/patrol_rsi/b2_calibration.py','src/patrol_rsi/b2_actions.py','scripts/diagnose_b2_calibration.py','scripts/diagnose_b2_actions.py'):
        assert sha(ROOT/name)==original_source['files'][name]['sha256']
    for name,row in final_source['files'].items():
        if name.startswith(('src/','tests/')):assert sha(ROOT/name)==row['sha256'],name
    current=ROOT/'CURRENT_RESEARCH.md';report=ROOT/'reports/2026-10-30-B4-B2-目标对齐与RSI主线恢复.md'
    text=current.read_text(encoding='utf-8').replace('277项真实Torch与56项轻量CPU','279项真实Torch与58项轻量CPU')
    text=text.replace('9个Slurm分配全部正常退出，24.67分配CPU分钟',f"{ledger['allocation_count']}个Slurm分配全部正常退出，{ledger['allocated_CPU_minutes']:.2f}分配CPU分钟")
    current.write_text(text,encoding='utf-8',newline='\n')
    text=report.read_text(encoding='utf-8').replace('最终源277项真实Torch回归、56项轻量CPU','最终源279项真实Torch回归、58项轻量CPU')
    text=text.replace('本轮9个Slurm分配全部正常退出，累计24.67分配CPU分钟',f"本轮{ledger['allocation_count']}个Slurm分配全部正常退出，累计{ledger['allocated_CPU_minutes']:.2f}分配CPU分钟")
    text=text.replace('完整新运行目录约40.67MB',f"完整新运行目录约{storage['bytes_all_new_run_files']/1e6:.2f}MB")
    report.write_text(text,encoding='utf-8',newline='\n')
    protected=read(STAGE/'evidence/protected-baseline.json');changed=[]
    for name,row in protected['files'].items():
        if sha(ROOT/name)!=row['sha256']:
            assert name in ('CURRENT_RESEARCH.md','.github/workflows/cpu-checks.yml'),name
            changed.append({'path':name,'before_sha256':row['sha256'],'after_sha256':sha(ROOT/name)})
    protection={'baseline':protected['baseline'],'original_tracked_files_checked':len(protected['files']),
        'unchanged_tracked_files':len(protected['files'])-len(changed),'allowed_navigation_CI_changes':changed,'historical_experimental_changes':0,
        'B2_diagnostic_executed_source_unchanged':True,'final_interface_Torch_snapshot':'code-v6','final_Torch_gate_job':'130875'}
    write(STAGE/'evidence/protected-baseline-verification.json',protection)
    review=read(STAGE/'evidence/next-protocol-review.json')
    for key,name in (('paired_protocol_SHA256','MultiAgent-Patrol-Paired-v1.json'),('RSI_protocol_SHA256','RSI-Controlled-Meta-v1.json')):
        assert review[key]==sha(STAGE/name)
    review['final_source_actual_Torch_gate']={'job_id':'130875',**gate,'path':'evidence/run/reviewed-v6/gate/full-tests.json'}
    review['final_delivery_review']={'spec':'0 unresolved after incumbentboundaryfailure immediate-stop fix and negative regression.',
        'standards':'0 unresolved after noveltyreason normalization inprojectedprompt history with immutable receipts retained.',
        'approved_scope':'Diagnostic completion,interface verification andprotocolfreeze only;not scientificpositiveclaimorformallaunch.'}
    write(STAGE/'evidence/next-protocol-review.json',review)
    requirements=[
        ('A_original_goal_paper_code_evidence_contribution','原论文与当前贡献对照.md;CONTEXT andmandatoryhistory read'),
        ('A_preserve_old_NO_GO_and_sources','protected-baseline-verification.json;oldB3.2/B1bunchanged'),
        ('B1_prior_diagnostic_protocol_review','B4-B2-development-v1.json;protocol-review.json;initial274Torchbeforefits'),
        ('B1_probability_compression_and_limited_ablation','42new+14replayedfitreceipts;calibration-diagnosis;rawclassscores/ranges/weights/derivatives'),
        ('B2_current_action_actual_finish_and_constraints','298152currentcandidates;sourcegeometry rebuiltall359roots of36devconditions'),
        ('B2_public_budgeted_Top1_vs_Tour_full_outcomes','publicselectionreceiptsbeforelabels;70selectedindependentfullschedule/event/gapchecks'),
        ('C_actual_PPO_CVAE_workflow_usage','interface-usage.json;M11/M12Balreadyreleasedaggregates;10engineeringcombos/repairhashes'),
        ('C_five_arm_paired_frozen_protocol','MultiAgent-Patrol-Paired-v1.json;next-protocol-review;aligned_patrol'),
        ('D_bounded_reused_interfaces_and_allowed_meta_changes','rsi_controlled reuseImproverPatrolProgram;279Torch/58CPU;nohugeframework'),
        ('D_four_fair_groups_and_transfer_cost_protocol','RSI-Controlled-Meta-v1.json;counterfairness+78formal+1futurepreflightcaps;separatetasks/cache'),
        ('D_interface_only_protocol_transfer_verified','meta-interface-finalsmoke;reset/overlap/failure/novelty/budgetnegativecases'),
        ('E_mainline_decision_and_stopping_boundaries','reportLIMITEDGO;oldNO-GO;formalrunner/newsource/preflightnextstage;noBrieruniversalgate'),
        ('E_all_real_resources_scope','resource-ledger;allactualjobsexit0;read-onlyoriginals;0newphysics0training0realLLM0formaltests'),
        ('E_one_document_delivery','fullreport+newprotocols/diagnostics/usage/costs+exactpriorB1bfullJSON embedded')]
    acceptance={'status':'completed_goal_stage','decision':'LIMITED GO for mainline preparation;freeze_not_launch',
        'baseline':protected['baseline'],'requirements':[{'requirement':name,'status':'verified','evidence':evidence} for name,evidence in requirements],
        'engineering_tests_not_scientific_gain':True,'old_B3_2_and_B1b_NO_GO_unchanged':True,
        'formal_tests_launched':0,'real_LLM_calls':0,'new_physical_processes':0,'neural_retraining':0,
        'future_formal_orchestrator_and_model_preflight':'Explicit next execution gate,not performed in this goal stage.',
        'latest_Torch_gate':gate,'lightweight_CPU':cpu,'source_snapshot_SHA256':final_source['sha256'],
        'resources':{k:v for k,v in ledger.items() if k!='jobs'},'protected_originals':protection,
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    write(STAGE/'evidence/final-acceptance.json',acceptance)
    historical=ROOT/'reports/2026-10-30-B4-B1b-Limited-项目全量汇总.json'
    docs={p.name:p.read_text(encoding='utf-8') for p in STAGE.glob('*.md')}
    protocols={p.name:read(p) for p in (STAGE/'B4-B2-development-v1.json',STAGE/'MultiAgent-Patrol-Paired-v1.json',STAGE/'RSI-Controlled-Meta-v1.json')}
    attempt_receipts={p.name:read(p) for p in (RUN/'calibration').glob('*.json') if p.name!='calibration-diagnosis.json'}
    allpublicselections={p.name:read(p) for p in (RUN/'actions').rglob('*--public-selections.json')}
    bundle={'文件说明':{'标题':'Zhinengti-RSI全量项目汇总：B4-B2目标对齐与RSI主线恢复','日期':'2026-10-30','schema':'b4b2-mainline-bundle-v1',
        '阅读顺序':'先读当前结论与完整报告_markdown，再按需看新诊断、接口、协议和原始历史。历史状态不代表当前状态。',
        '内容':'本轮七项交付、全部开发诊断和公开选择收据、源码清单、审计、资源、前序全量历史。',
        '封存边界':'没有增加旧最终逐行数据、预测向量或物理数组；原件以来源SHA与聚合记录保留。'},
        '当前结论':{'阶段':'B4-B2目标对齐与RSI主线恢复','决定':acceptance['decision'],
            '研究主线':'多智能体巡检任务收益、开源改进器自身收益与独立任务迁移；B4预测为辅助。',
            '开发诊断':'同分数校准消融确认正则影响；2开发种子Top1公开净价值有描述性净益；无独立泛化结论。',
            '协议':'五臂巡检与四组RSI已独立审查冻结；正式运行器和新源/模型预检下一阶段验收。',
            '旧结论':'B3.2 NO-GO、B1b NO-GO及B1aLIMITED保留。','未证明':'PPO/CVAE/RSI独立收益、真实Qwen元更新迁移、新种子或新布局泛化、现场有效性。'},
        '完整报告_markdown':report.read_text(encoding='utf-8'),'原论文与接口等完整文档_markdown':docs,
        '本轮验收':acceptance,'诊断与下一轮冻结协议':protocols,'本轮诊断审查':read(STAGE/'evidence/protocol-review.json'),
        '下一轮协议与交付审查':review,'开发诊断汇总':results,'全部概率消融与分布':cal,'逐次校准尝试收据':attempt_receipts,
        '全部开发行动诊断':action,'先于离线评价的公开选择收据':allpublicselections,'真实工具调用统计':usage,
        '归档工程试跑最终收据':repair,'归档工程试跑首版_历史标签错误已修正':read(RUN/'interface-smoke/interface-smoke.json'),
        '元层纯接口试跑':meta,'全部实际资源账本':ledger,'新目录存储':storage,'历史原件保护':protection,
        '初始Torch门槛':read(RUN/'gate/full-tests.json'),'最终Torch门槛':gate,'轻量CPU检查':cpu,
        '执行源版本清单':{p.name:read(p) for p in STAGE.glob('code-v*-manifest.json')},
        '历史_B4B1b及此前全量汇总_原内容':read(historical),'历史全量文件SHA256':sha(historical),
        '复现入口':{'仓库':'https://github.com/DonsonHH/Zhinengti-RSI','当前入口':'CURRENT_RESEARCH.md',
            '本轮目录':str(STAGE.relative_to(ROOT)).replace('\\','/'),'远端本轮根':'/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b2',
            '报告重建':'python scripts/report_b2_alignment.py --stage research/2026-10-30-m12c1b4b2-alignment; python scripts/report_b2_usage.py --stage same_path; python scripts/build_b2_bundle.py',
            '说明':'报告重建仅读已发布证据；实际诊断源码与Slurm脚本由各snapshot绑定。正式阶段不在本轮执行。'}}
    output=ROOT/'reports/2026-10-30-B4-B2-项目全量汇总.json';write(output,bundle)
    loaded=read(output)
    assert loaded['完整报告_markdown']==report.read_text(encoding='utf-8') and loaded['历史_B4B1b及此前全量汇总_原内容']==read(historical)
    assert output.stat().st_size<95_000_000
    print(json.dumps({'bundle_bytes':output.stat().st_size,'bundle_SHA256':sha(output),'decision':acceptance['decision'],'all_requirement_groups':len(requirements),'actualTorch':gate['tests'],'CPU':cpu['tests'],'jobs':ledger['allocation_count']}))


if __name__=='__main__':main()

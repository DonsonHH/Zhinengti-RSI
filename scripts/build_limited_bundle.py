"""Bundle released B1b evidence and prior history; never reads remote raw data."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-30-m12c1b4b1b-limited'
RUN=STAGE/'evidence/run'


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    results=read(RUN/'report/results.json');selection=read(RUN/'selection/selection-receipt.json')
    audit=read(RUN/'independent-audit/independent-evidence-audit.json')
    alarms=read(RUN/'alarm-audit/alarm-probability-audit.json');ledger=read(RUN/'resource-ledger.json')
    protocol=STAGE/'B4-B1b-Limited-v1.json';review=STAGE/'evidence/protocol-freeze.json'
    gate=RUN/'reviewed/gate/full-tests.json';cpu=STAGE/'evidence/local-cpu-tests.json'
    assert sha(protocol)==results['protocol_SHA256']==selection['protocol_SHA256']==read(review)['protocol_SHA256']
    assert sha(RUN/'selection/selection-receipt.json')==results['selection_SHA256']==audit['selection_SHA256']==alarms['selection_SHA256']
    assert audit['status']==alarms['status']=='passed' and ledger['all_completed_exit0']
    assert read(gate)=={'tests':262,'skipped':0,'failures':0,'errors':0,'success':True,'torch':'2.6.0+cu124'}
    assert read(cpu)['success'] and read(cpu)['tests']==43
    opened=read(RUN/'cache/final-open.json')
    assert opened['opened_unix']>=selection['frozen_unix']
    fits={};started=[];fit_slots=cal_slots=0;candidate_count=0
    for ci in range(7):
        folder=RUN/'fits'/f'cell-{ci}';complete=read(folder/'complete.json');start=read(folder/'started.json')
        assert start['actualTorch_gate_SHA256']==sha(gate) and not start['final_split_access']
        assert start['started_unix']<selection['frozen_unix']
        fit_slots+=complete['LR_slots_consumed'];cal_slots+=complete['calibration_slots_consumed']
        candidates=[]
        for entry in complete['candidates']:
            path=folder/Path(entry['path']).name
            assert sha(path)==entry['sha256'] and entry['valid']
            candidate=read(path);model=candidate.get('model');cal=candidate.get('calibrator')
            if model:
                assert model['closure_evaluations']<=400 and model['max_feature_batch_rows']==16384
                assert model['status']=='completed_within_fixed_optimizer_budget'
            if cal:assert cal['closure_evaluations']<=300 and cal['status']=='completed_within_fixed_optimizer_budget'
            candidates.append(candidate);candidate_count+=1
        fits[str(ci)]={'start':start,'complete':complete,'frozen_candidates':candidates};started.append(start)
    assert (fit_slots,cal_slots,candidate_count)==(138,192,206)
    model_manifest=read(STAGE/'code-v3-manifest.json');critical=[]
    for name,row in model_manifest['files'].items():
        if name.startswith(('src/','tests/')) or name in ('scripts/prepare_limited_data.py','scripts/fit_limited_predictors.py','scripts/select_limited_predictors.py','scripts/evaluate_limited_final.py','cluster/m12c1b4b1b_stage.sbatch'):
            assert sha(ROOT/name)==row['sha256'],name
            critical.append(name)
    protected=read(STAGE/'evidence/protected-baseline.json');changes=[]
    for name,row in protected['files'].items():
        if sha(ROOT/name)!=row['sha256']:
            assert name in ('.github/workflows/cpu-checks.yml','CURRENT_RESEARCH.md'),name
            changes.append({'path':name,'before_sha256':row['sha256'],'after_sha256':sha(ROOT/name),
                'reason':'Add11CPUcontracts toCI' if name.startswith('.github/') else 'Update current navigation and result;historical experimental artifacts unchanged.'})
    checkpoint_path=STAGE/'evidence/checkpoint-live-recheck.json';checkpoint=read(checkpoint_path)
    assert checkpoint['sha256']==read(protocol)['original_checkpoint_SHA256'] and checkpoint['bytes']>0
    assert checkpoint['host']=='mu01' and checkpoint['user']=='renly'
    protection={'baseline':protected['baseline'],'tracked_files_checked':len(protected['files']),
        'unchanged_files':len(protected['files'])-len(changes),'allowed_navigation_or_CI_changes':changes,
        'historical_experimental_artifact_changes':0,'critical_executed_source_files_byte_equal':len(critical),
        'checkpoint_SHA256_live_rechecked':checkpoint['sha256'],'checkpoint_live_receipt_SHA256':sha(checkpoint_path),
        'checkpoint_live_recheck':checkpoint}
    write(STAGE/'evidence/protected-baseline-verification.json',protection)
    review_report={'before_fit_review':read(review),'actual_Torch_gate':read(gate),'gate_job':'130761',
        'first_fit_job':'130763','selection_job':'130767','final_cache_job':'130773','final_score_array':'130774',
        'selection_artifact_count':len(selection['artifacts']),'fit_gate_matches_all7_starts':True,
        'protocol_to_final_chronology_verified':True,'spec_axis':{'blocking_findings':0,'delivery_findings':1,
            'finding':'Per-noise/process eligible AND all physical event denominator aggregates missing in first report draft.',
            'resolution':'Added independent-evidence-audit and event-denominator-diagnostics;actual130807passed.'},
        'standards_axis':{'blocking_findings':0,'delivery_findings':1,'finding':'Same aggregate reporting gap;strengthen per-seedFPR and repeat-count independent audit.',
            'resolution':'Added alarm-probability-audit;actual130836passed114frozenmodels,without newinference.'},
        'implementation_correctness_repairs_before_fits':['CacheSHA/phase/cell validation','AandBeventdenominator fixedtoeligibleBconfirmation',
            'Nonfinite optimizer failure serializes finiteJSON','LBFGSbudget/actualfinalgradient verification'],
        'initial_protocol_draft_preserved':True,'post_final_model_or_calibration_or_threshold_change':False}
    write(STAGE/'evidence/independent-review-results.json',review_report)
    requirements=[
        ('source_and_four_facts','inherited-four-facts.json;protected-baseline-verification.json;independent-evidence-audit.json'),
        ('independent_new_seven_cell_protocol_before_fit','B4-B1b-Limited-v1.json;protocol-freeze.json;all7fitsstarted.json'),
        ('coverage_scope_selection_and_confirmation_limit','protocol.scope_selection;results.inference_scope;complete novice report'),
        ('all_fixed_baselines_LR_calibration_and_selection','138LR+192calibration+206candidates;selection-receipt.json'),
        ('A_B_missing_legality_and_no_future_inputs','262Torch+43CPU;all28source-cachecomparisons;originalpublicfeatureaudit'),
        ('all7final_views_frozen_once','114releasedmodels;selectionSHA;final-openchronology;zero refits'),
        ('metrics_uncertainty_time_CVAE_and_failures','results.json;prediction-diagnostics.json;event-denominator-diagnostics.json'),
        ('per_seed_FPR_and_alarm_counts_verified','alarm-probability-audit.json114models'),
        ('actual_development_action_indexed_all4roles','24seed480conditions;4659840candidates;action-indexed-feasibility.json'),
        ('action_grid_cooling_control_and_storage_interpretation','same30svsall5scontrasts;realfinishB_reference;storageestimates'),
        ('all_resources_and_originals_preserved','45allocationsCOMPLETED0;resource-ledger;storage-final;all13056originalSHA'),
        ('B4C_recommendation_and_no_extra_experiments','NO-GO;noCVAE/PPO/Qwen/newphysics/policysearch'),
        ('single_document_novice_progress_bottlenecks_unknowns','completeMarkdown embeddedplusallreleasednewandpriorJSONevidence')]
    acceptance={'engineering_status':'completed_and_independently_verified','scientific_recommendation':results['B4C_exploratory_recommendation'],
        'baseline':protected['baseline'],'protocol_SHA256':sha(protocol),'selection_SHA256':sha(RUN/'selection/selection-receipt.json'),
        'requirements':[{'requirement':name,'status':'verified','evidence':evidence} for name,evidence in requirements],
        'new_confirmation_or_transfer_not_claimed':True,'closed_loop_gain_not_claimed':True,
        'all_actual_jobs_complete':True,'new_training_after_final_open':0,'historical_experimental_changes':0,
        'independent_audit':audit,'alarm_audit_summary':{k:v for k,v in alarms.items() if k!='probability_compression'},
        'resources_summary':{k:v for k,v in ledger.items() if k!='jobs'},'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'publication_state_note':'Git commit/push and its exactHEAD CI are verified separately after artifact creation;thisreceipt attests experiment and local files.'}
    write(STAGE/'evidence/final-acceptance.json',acceptance)
    report=ROOT/'reports/2026-10-30-B4-B1b-Limited-完整进度与瓶颈.md'
    history=ROOT/'reports/2026-10-30-B4-B1a-项目全量汇总.json'
    summaries=[]
    for path in sorted((RUN/'selection').glob('cell-*.json')):
        value=read(path);summaries.append({'candidate_file':path.name,'sha256':sha(path),'summary':value['summary'],'seeds':value['seeds']})
    phases={name:read(RUN/'cache'/name/'complete.json') for name in ('development','validation_calibration','validation_selection','final_test')}
    action_seeds=[]
    for path in sorted((RUN/'action').glob('seed-index-*/complete.json')):
        value=read(path)
        action_seeds.append({**{k:v for k,v in value.items() if k not in ('statistics','sourcefiles')},
            'statistics':{key:{k:v for k,v in row.items() if k!='event_keys'} for key,row in value['statistics'].items()},'original_receipt_SHA256':sha(path)})
    bundle={
        '文件说明':{'标题':'Zhinengti-RSI全量项目汇总：截至B4-B1b-Limited','日期':'2026-10-30','schema_version':'b4b1b-limited-bundle-v1',
            '阅读顺序':'先读当前结论与新手版完整报告_markdown，再按需要展开指标、审计与历史。历史章节的旧状态不是当前状态。',
            '包含内容':'本报告全文、冻结协议、206候选与校准、114最终视图、选择摘要、各种子指标、事件覆盖、报警、行动诊断、45作业账本、审计、上一版完整历史。',
            '原件边界':'没有嵌入最终逐行标签/分数/事件ID/轨迹、模型权重二进制或大规模物理数组；以SHA、聚合和封存位置记录。'},
        '当前结论':{'已完成阶段':'B4-B1b-Limited','工程状态':'completed_and_independently_verified','对B4C建议':results['B4C_exploratory_recommendation'],
            '白话':'公开观测更会排序异常机会，但概率误差改善不足，CVAE额外收益不获支持；暂不进入依赖此概率的新闭环策略。',
            '完整观测相对时间先验':results['aggregate_comparisons']['time'],'最终宏误报率':results['macro_FPR'],
            '限定单元':read(protocol)['cells'],'原B1a状态':'LIMITED35/48','原B4B0训练':'NO-GO','原B3_2闭环':'NO-GO',
            '下一步建议':'先另立B4-B2行动目标与概率压缩的开发诊断协议；没有启动新训练或新物理生成。',
            '未证明':'新种子可靠性、未见起点/布局/现场迁移、整回合收益、RSI自改进收益。'},
        '新手版完整报告_markdown':report.read_text(encoding='utf-8'),
        '本轮验收与边界':acceptance,'独立审查与修正':review_report,'四项继承事实':read(STAGE/'evidence/inherited-four-facts.json'),
        '历史原件保护':protection,'冻结限定协议':read(protocol),'首次协议草稿_只作版本历史':read(STAGE/'B4-B1b-Limited-v1-initial-draft.json'),
        '各划分合法缓存聚合':phases,'全部拟合和校准候选':fits,'最终打开前选择收据':selection,'全部选择候选摘要':summaries,
        '最终七单元全部模型比较与区间':results,'预测与报警失败诊断':read(RUN/'diagnostics/prediction-diagnostics.json'),
        '噪声与过程事件分母汇总':read(RUN/'independent-audit/event-denominator-diagnostics.json'),
        '独立原件与指标复算':audit,'独立报警与概率范围复算':alarms,
        '行动时刻可行性':read(RUN/'action-report/action-indexed-feasibility.json'),'行动诊断各种子汇总':action_seeds,
        '全部实际资源账本':ledger,'存储实测':read(RUN/'storage-final.json'),'真实Torch门槛':read(gate),'本地CPU门槛':read(cpu),
        '执行源码快照清单':{path.name:read(path) for path in sorted(STAGE.glob('*-manifest.json'))},
        '历史_B4B1a及此前全量汇总_原内容':read(history),'历史汇总文件SHA256':sha(history),
        '复现入口':{'仓库':'https://github.com/DonsonHH/Zhinengti-RSI','当前入口':'CURRENT_RESEARCH.md',
            '本轮证据目录':str(STAGE.relative_to(ROOT)).replace('\\','/'),'远端封存根':'/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1b',
            '说明':'源码压缩包与manifest绑定实际作业；重建报告只读取已发布证据，不训练或重算最终模型。'}}
    path=ROOT/'reports/2026-10-30-B4-B1b-Limited-项目全量汇总.json';write(path,bundle)
    loaded=read(path)
    assert loaded['新手版完整报告_markdown']==report.read_text(encoding='utf-8')
    assert loaded['历史_B4B1a及此前全量汇总_原内容']==read(history)
    assert path.stat().st_size<95_000_000
    print(json.dumps({'bundle_bytes':path.stat().st_size,'bundle_SHA256':sha(path),'report_bytes':report.stat().st_size,
        'candidate_count':candidate_count,'final_models':audit['final_frozen_models_arithmetic_recomputed'],
        'baseline_files_checked':len(protected['files']),'allowed_changes':len(changes),'engineering_complete':True,
        'B4C':results['B4C_exploratory_recommendation']},ensure_ascii=True))


if __name__=='__main__':main()

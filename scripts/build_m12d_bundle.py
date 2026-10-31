"""Verify frozen M12D deliverables and assemble one JSON including exact history."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-30-m12d-formal'
RUN=STAGE/'evidence/run'
PRIOR=ROOT/'research/2026-10-30-m12c1b4b2-alignment'


def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def assert_public_final_boundary(value):
    if isinstance(value,dict):
        assert 'new_final_actual_cases' not in value,'Private final case index cannot enter bundle'
        spec=value.get('specification',{})
        if isinstance(spec,dict) and spec.get('phase')=='fresh_task_final':
            assert not(set(value)&{'metrics','anomaly_intervals','inspection_events','record_path','scalar_trace'}),'Private final per-case data cannot enter bundle'
        for child in value.values():assert_public_final_boundary(child)
    elif isinstance(value,list):
        for child in value:assert_public_final_boundary(child)


def main():
    expected={'MultiAgent-Patrol-Paired-v1.json':'dbea4a9b83cbbe4475d3343492646328c6a1c737f22ff79a785c8f2fb17f914a',
        'RSI-Controlled-Meta-v1.json':'3bd4e2478c50f152898ecf73d1359dbc246e5c3501425daa034b215328edbe76'}
    for name,digest in expected.items():assert sha(PRIOR/name)==digest
    formal=read(STAGE/'evidence/m12d-summary.json');rsi=read(STAGE/'evidence/rsi-results.json');cost=read(RUN/'resource-ledger.json')
    assert formal['formal_primary_episodes']==formal['formal_replay_episodes']==360
    assert not formal['five_arm_gate']['passed'] and formal['five_arm_final_episodes']==0 and not formal['five_arm_final_source_generated']
    dev=read(RUN/'development-complete.json');val=read(RUN/'validation-complete.json')
    for phase,receipt,seeds in (('development',dev,(31000,31001)),('validation',val,(31020,31021))):
        assert receipt['complete'] and receipt['formal_episodes']==receipt['replayed_episodes']==180 and receipt['all_five_arms_reported']
        assert receipt['unique_numeric_processes']==18 and not receipt['cross_phase_fingerprint_overlap'] and not receipt['final_sources_opened']
        assert sum(len(read(RUN/phase/f'seed-{seed}'/'complete.json')['rows']) for seed in seeds)==180
    assert read(RUN/'smoke/seed-30990/negative-executor-tests.json')['all_four_rejected']
    assert val['gate']==formal['five_arm_gate']
    independent=read(STAGE/'evidence/independent-aggregate-audit.json')
    assert independent['status']=='passed' and len(independent['phase_arm_checks'])==10 and independent['independently_rederived_gate'] is False
    actualgate=read(RUN/'rsi-regression-v3/gate/full-tests.json');fivegate=read(RUN/'reviewed/gate/full-tests.json');cpu=read(STAGE/'evidence/local-cpu-tests.json')
    for gate,n in ((actualgate,293),(fivegate,288),(cpu,72)):
        assert gate['success'] and gate['tests']==n and not(gate['errors'] or gate['failures'] or gate['skipped'])
    pre=read(RUN/'qwen-preflight/preflight.json');assert pre['valid_workflow_output'] and pre['input_tokens']==172 and pre['output_tokens']==28
    assert cost['within_caps'] and cost['actual_formal_model_calls']==78 and cost['separate_actual_preflight_model_calls']==1
    assert cost['formal_GPU_allocations']==3 and cost['formal_GPU_seconds']<=10800 and cost['actual_environment_attempts']<=6480
    assert cost['allocated_CPU_minutes_inclusive_all_M12D']<=2400 and cost['five_arm_allocated_CPU_minutes_including_regression_and_audit']<=960
    storage=read(RUN/'storage-after-export.json');assert storage['within_cap'] and storage['bytes_all_run_files_after_public_export']<4_000_000_000
    assert not cost['five_arm_final_generated']
    finalaudit=read(RUN/'rsi-independent-final-audit.json');assert finalaudit['passed'] and not finalaudit['raw_final_rows_exported']
    trials=read(RUN/'trial-summary.json')['trials'];assert len(trials)==3 and all(v['final_complete'] for v in trials)
    for i in range(3):
        assert read(RUN/f'rsi-trial-{i}/independent-search-audit.json')['passed']
        assert (RUN/f'rsi-trial-{i}/final-submission-claim/receipt.json').exists()
    assert len(rsi['primary_meta3_vs_fixed3'])==6 and rsi['scientific_decision']=='NO-GO_no_independent_RSI_increment'
    assert not rsi['all_six_primary_repeat_phase_checks_passed']
    sources={p.name:read(p) for p in STAGE.glob('*-manifest.json') if p.name.startswith(('code-v','rsi-v'))}
    for manifest in sources.values():
        archive=STAGE/'source'/manifest['archive'];assert sha(archive)==manifest['sha256'] and archive.stat().st_size==manifest['bytes']
    for name,row in sources['rsi-v3-manifest.json']['files'].items():
        if name.startswith(('src/','tests/')):assert sha(ROOT/name)==row['sha256'],name
    for name in ('scripts/run_m12d_paired.py','scripts/finalize_m12d_phase.py'):
        assert sha(ROOT/name)==sources['code-v3-manifest.json']['files'][name]['sha256'],name
    for name in ('scripts/run_m12d_rsi.py','scripts/run_m12d_rsi_final.py','scripts/seal_m12d_rsi.py'):
        assert sha(ROOT/name)==sources['rsi-v3-manifest.json']['files'][name]['sha256'],name
    old=read(STAGE/'evidence/protected-baseline.json');changes=[]
    for name,row in old['files'].items():
        if sha(ROOT/name)!=row['sha256']:
            assert name in ('CURRENT_RESEARCH.md','.github/workflows/cpu-checks.yml'),name
            changes.append({'path':name,'before_SHA256':row['sha256'],'after_SHA256':sha(ROOT/name)})
    protection={'baseline':old['baseline'],'old_tracked_files_checked':len(old['files']),
        'unchanged_old_files':len(old['files'])-len(changes),'allowed_navigation_and_CI_changes':changes,
        'historical_experiment_or_model_changes':0,'formal_and_RSI_executed_source_unchanged':True}
    write(STAGE/'evidence/protected-baseline-verification.json',protection)
    groups=[
        ('1_goal_and_protocol_exact_bytes','CONTEXT.md;目标与导师要求;原论文与当前贡献对照;two frozenSHAchecks'),
        ('2_five_arms_common_executor_and_attribution','360actualrows;288Torch;common1CHECK;byrolemetrics;actual/shadow/single/multiPPO counts'),
        ('3_independent_sources_and_pre_execution_gates','process-manifest;historicalnumericseedregistry;execution-identity;smoke4negativeaudits;allphasefingerprints'),
        ('4_five_arm_dev_validation_stop','180dev+180val;360replays;independentmacroaudit;NO-GO;no3104x sources/seal/jobs'),
        ('5_one_actual_Qwen_preflight','qwen-preflight actual172/28valid;failedstartupbeforegeneration;101GPUseconds;fixed13modelhashes'),
        ('6_separate_RSI_launch_approval','rsi-launch-review;actual293Torch;immutableidentity;firstrepeatresourcebeforeother2'),
        ('6_real_bounded_search_transfer_and_all_costs','3searchaudits;78formalLLMcalls;37taskslots/repeat;protocolonlyreset;fullledger'),
        ('6_validated_mappings_and_one_shot_final','3nominal/actualmappingseals;globalresource/source seal;heldSlurmscriptverification;3atomicclaims;independentfinalaudit'),
        ('7_task_and_meta_gain_separately','all6groupsvalidation/final;3repeatmeta3vsfixed3pairedmetrics;4physicalseedvalues;frozenNO-GO'),
        ('7_tool_contribution_and_negative_evidence','PPOearlycheckmechanism;commonCVAEcausalnotidentified;latestmatrix;hypothesesvsproofseparated'),
        ('7_history_preservation_and_single_document','6813oldfilehashes;exactpriorJSONnested;humanreadablefullreport;privatefinalrowsnotexported')]
    acceptance={'status':'completed_bounded_M12D_execution','baseline':old['baseline'],
        'scientific_decisions':{'five_arm':formal['five_arm_gate']['status'],'RSI':rsi['scientific_decision']},
        'requirements':[{'requirement':name,'status':'verified','evidence':evidence} for name,evidence in groups],
        'engineering_tests_are_not_scientific_gain':True,'five_arm_final_not_executed_due_to_validation_failure':True,
        'RSI_final_independent_audit_scope':finalaudit['scope'],'RSI_neural_execution_second_replay':False,
        'standalone_final_runner_requires_authorized_guarded_entry':True,'protected_originals':protection,
        'actual_Torch_293':actualgate,'formal_Torch_288':fivegate,'lightweight_CPU_72':cpu,
        'resources':{k:v for k,v in cost.items() if k!='jobs'},'storage_after_public_export':storage,'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    review_path=STAGE/'evidence/final-delivery-review.json'
    assert review_path.exists() and read(review_path)['approved_for_evidence_delivery']
    acceptance['independent_final_delivery_review']=read(review_path)
    write(STAGE/'evidence/final-acceptance.json',acceptance)
    report=ROOT/'reports/2026-10-30-M12-D-完整进度与瓶颈.md';assert report.exists()
    previous=ROOT/'reports/2026-10-30-B4-B2-项目全量汇总.json'
    published={}
    for p in STAGE.rglob('*.json'):
        if not p.is_file() or '/source/' in p.as_posix():continue
        assert p.name!='final-complete.json','Never embed a private final original'
        value=read(p);assert_public_final_boundary(value)
        published[str(p.relative_to(STAGE)).replace('\\','/')]=value
    markdown={p.name:p.read_text(encoding='utf-8') for p in STAGE.glob('*.md')}
    raw_text={str(p.relative_to(STAGE)).replace('\\','/'):p.read_text(encoding='utf-8',errors='replace') for p in RUN.rglob('*') if p.is_file() and p.suffix in ('.psv','.out','.err','.sbatch')}
    code_paths=sorted(list((ROOT/'scripts').glob('*m12d*.py'))+list((ROOT/'cluster').glob('m12d*.sbatch'))+
        [ROOT/'src/patrol_rsi/paired_campaign.py',ROOT/'src/patrol_rsi/rsi_campaign.py',ROOT/'tests/test_paired_campaign.py',ROOT/'tests/test_rsi_campaign.py'])
    code={str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':sha(p),'text':p.read_text(encoding='utf-8')} for p in code_paths}
    bundle={'文件说明':{'标题':'Zhinengti-RSI全量汇总：M12-D当前进度、结果与瓶颈','schema':'m12d-full-evidence-bundle-v1',
        '生成时间UTC':acceptance['checked_utc'],'阅读顺序':'先读当前结论和完整报告_markdown，再看实际结果、审计与成本；历史里的计划与状态属于当时。',
        '内容':'本文、全部已发布M12D汇总／非最终收据／模型调用／最终种子组统计、源码版本、审查、历史保护及此前完整JSON。',
        '封存边界':'不包含最终逐事件记录、逐样本预测或物理数组；其集群原件路径与SHA和聚合审计保留。',
        'JSON排版':'为容纳全部历史采用紧凑JSON，解析后可按键展开；完整报告_markdown与独立Markdown内容一致。'},
        '当前结论':{'阶段':'M12-D五臂验证与真实受控RSI执行已完成','五臂决定':formal['five_arm_gate']['status'],
            'RSI决定':rsi['scientific_decision'],'任务收益':'五臂新修订候选未胜Tour；RSI任务流程的各组独立结果与强基线比较见RSI实际结果，不将组合成绩归因于某个模型。',
            '改进器收益':'六次元提案全部语法非法，三次重复均沿用旧协议；matched3未满足冻结门槛，合法元更新的效果尚未实测。','未证明':'PPO/CVAE独立任务增量、合法递归元更新的独立增量、未知布局迁移、现场有效性。',
            '旧结论':'B3.2和B1b的NO-GO及B2历史证据保持原样。','下一步':'先核对合法决策机会、信息与动作收益是否匹配，再立新协议；本轮不再追加搜索或重选最终方案。'},
        '完整报告_markdown':report.read_text(encoding='utf-8'),'本轮其他完整说明_markdown':markdown,
        '验收与逐项证据':acceptance,'五臂实际结果与工具归因':formal,'RSI实际结果与正式判定':rsi,
        '全部已发布本轮JSON':published,'Slurm日志与启动脚本等原始文本':raw_text,
        '本轮执行与审计新增源码_完整文本':code,
        '冻结协议原内容':{name:read(PRIOR/name) for name in expected},'冻结协议原文件SHA256':expected,
        '历史_B4B2及此前全量汇总_原内容':read(previous),'历史全量原文件SHA256':sha(previous),
        '复现入口':{'仓库':'https://github.com/DonsonHH/Zhinengti-RSI','当前状态':'CURRENT_RESEARCH.md',
            '当前阶段目录':str(STAGE.relative_to(ROOT)).replace('\\','/'),'集群原件':'/share/home/renly/renly-patrol-rsi/runs/20261030-m12d',
            '报告重建':'python scripts/report_m12d_rsi.py --stage research/2026-10-30-m12d-formal; python scripts/build_m12d_bundle.py',
            '重要':'报告重建只读已发布证据。正式回合不得重跑到原目录；最终入口必须保留原子claim和Slurm实际脚本核验。'}}
    output=ROOT/'reports/2026-10-30-M12-D-项目全量汇总.json'
    output.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    assert output.stat().st_size<95_000_000
    loaded=read(output);assert loaded['完整报告_markdown']==report.read_text(encoding='utf-8')
    assert loaded['历史_B4B2及此前全量汇总_原内容']==read(previous)
    print(json.dumps({'bundle_bytes':output.stat().st_size,'bundle_SHA256':sha(output),'verified_requirement_groups':len(groups),
        'five_arm_final_episodes':0,'RSI_calls':cost['actual_formal_model_calls'],'RSI_actual_episodes':cost['actual_environment_attempts'],'decision':rsi['scientific_decision']}))


if __name__=='__main__':main()

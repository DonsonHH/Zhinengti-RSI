"""One complete M12-E document with lossless column tables for repetitive metadata."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-31-m12e-timing-schema'
RUN=STAGE/'evidence/run'
TAG_COLUMNS='__column_table_columns__'
TAG_ROWS='__column_table_rows__'


def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def encode_tables(value):
    """Keep every JSON value; write repeated row keys once, no numeric compression."""
    if isinstance(value,list):
        if len(value)>=2 and all(isinstance(v,dict) for v in value):
            keys=list(value[0])
            if keys and all(set(v)==set(keys) for v in value):
                return {TAG_COLUMNS:keys,TAG_ROWS:[[encode_tables(v[k]) for k in keys] for v in value]}
        return [encode_tables(v) for v in value]
    if isinstance(value,dict):
        assert set(value)!={TAG_COLUMNS,TAG_ROWS},'Reserved table marker collision'
        return {k:encode_tables(v) for k,v in value.items()}
    return value


def decode_tables(value):
    """Decode bundle['E1完整开发元数据_可逆列式表格'][seed]['data']."""
    if isinstance(value,dict):
        if set(value)=={TAG_COLUMNS,TAG_ROWS}:
            columns=value[TAG_COLUMNS];assert len(set(columns))==len(columns)
            assert all(len(row)==len(columns) for row in value[TAG_ROWS])
            return [{k:decode_tables(v) for k,v in zip(columns,row)} for row in value[TAG_ROWS]]
        return {k:decode_tables(v) for k,v in value.items()}
    if isinstance(value,list):return [decode_tables(v) for v in value]
    return value


def assert_no_private_final(value):
    if isinstance(value,dict):
        assert 'new_final_actual_cases' not in value
        spec=value.get('specification',{})
        if isinstance(spec,dict):assert not(spec.get('phase') in ('final_test','fresh_task_final') and any(k in value for k in ('metrics','anomaly_intervals','record_path')))
        for v in value.values():assert_no_private_final(v)
    elif isinstance(value,list):
        for v in value:assert_no_private_final(v)


def main():
    results=read(STAGE/'evidence/E-results.json');e0=read(STAGE/'evidence/E0-independent-audit.json');e1=read(RUN/'E1-independent-audit.json');ledger=read(RUN/'resource-ledger.json')
    assert e0['passed'] and e0['E0_passed'] and e1['passed'] and not e1['E1_opportunity_gate'] and not e1['E1_timing_gate']
    assert results['decisions']['E2']=='NO-GO' and not results['decisions']['E2_executed']
    assert ledger['allocation_count']==6 and ledger['all_allocations_completed_exit0'] and ledger['actual_new_model_calls']==24
    assert ledger['GPU_seconds']==195 and ledger['formal_GPU_allocations']==1 and ledger['actual_E1_environment_episodes']==220 and ledger['actual_E1_main_episodes']==180
    assert ledger['allocated_CPU_minutes_inclusive']<=960
    for name,n,ne in (('gate-v1',316,23),('gate-v2',317,24)):
        d=read(RUN/name/'full-tests.json');assert d['success'] and d['tests']==n and len(set(d['E_test_ids']))==ne and not(d['failures'] or d['errors'] or d['skipped'])
    cpu=read(STAGE/'evidence/local-cpu-tests.json');assert cpu['success'] and cpu['tests']==96 and not(cpu['failures'] or cpu['errors'] or cpu['skipped'])
    for review,source,protocol in (('E0',1,'E0-Schema-v1.json'),('E1',2,'E1-Timing-v1.json')):
        m=read(STAGE/f'code-v{source}-manifest.json');r=read(RUN/f'{review}-launch-review.json')
        assert sha(STAGE/'source'/m['archive'])==m['sha256']==r['source_archive_SHA256']
        assert sha(STAGE/protocol)==(r['E0_protocol_SHA256'] if review=='E0' else r['reviewed_input_SHA256']['E1'])
    source1=read(STAGE/'code-v1-manifest.json');source2=read(STAGE/'code-v2-manifest.json')
    for name,m in [('src/patrol_rsi/meta_schema.py',source1),('scripts/run_m12e_schema.py',source1),('src/patrol_rsi/patrol_timing.py',source2),
            ('src/patrol_rsi/e_stage.py',source2),('scripts/run_m12e_timing.py',source2),('tests/test_patrol_timing.py',source2),('tests/test_meta_schema.py',source2)]:
        assert sha(ROOT/name)==m['files'][name]['sha256'],name
    original=read(STAGE/'evidence/protected-baseline.json');changes=[]
    for name,row in original['files'].items():
        actual=sha(ROOT/name)
        if actual!=row['sha256']:
            assert name in ('CURRENT_RESEARCH.md','.github/workflows/cpu-checks.yml'),name
            changes.append({'path':name,'before_SHA256':row['sha256'],'after_SHA256':actual})
    protection={'baseline':original['baseline'],'old_tracked_files_checked':len(original['files']),'old_files_unchanged':len(original['files'])-len(changes),
        'allowed_navigation_CI_changes':changes,'old_experimental_or_test_file_changes':0,'old_M12D_reports_and_protocols_unchanged':True}
    write(STAGE/'evidence/protected-baseline-verification.json',protection)
    export=read(RUN/'export-inventory.json')
    for v in export['files']:
        p=RUN/v['path'];assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],p
    finalreview=read(STAGE/'evidence/final-delivery-review.json');assert finalreview['approved_for_evidence_delivery'] and finalreview['E2_execution_authorized'] is False
    requirements=[('A_latest_goal_history_and_immutable_boundary','required-evidence-read;old-sixrawdiagnosis;protected7384;liveboundaryreceipt'),
        ('B_six_full_structural_diagnoses','old-six-meta-diagnosis completeoriginaltext/parseerrors/enumchecks/SHA'),
        ('B_new_preregistered_E0_actual','2frozeninterfaces24actualcalls,3parents/exactprompts,separaterawvscompiled,failstop,budget,316actualTorch'),
        ('B_allowed_protocol_changes_and_transfer','24schema+differentparent+resetchecked;no hiddencontextoroldreplyrepair;no taskgainclaim'),
        ('C_explicit_preserve_use_and_target_ranking','TimingSelector sharesrun_episode,originalmasks/profession/return/oneCHECK;12actuallateCHECK,Tourprefixverified'),
        ('C_fresh_phases_processes_and_legal_roots','18newprocesses,36noiseconditions,12924roots,298152candidates;no3302x/3304xgeneration'),
        ('C_oracle_offline_and_actual_witnesses','40fixedwitnesses;sealedpublicTop1;all-eventgain/loss/wait/firstvectors;noafterresultreplacement'),
        ('C_five_arm_real_comparisons_attribution','180main,independenthierarchy;PPOsingle12/multi0;all5strongbaselines/cost/profession/revisit recorded'),
        ('D_joint_falsifiable_gates_and_negative_stop','E0PASS,E1opportunity4/1NO-GO,timingtieurgencyNO-GO;E2notrun,norelax/additionalcandidates'),
        ('D_reviewed_future_task_and_legal_meta_design','E2Task720+720/E2Meta48calls5292episodes;separatecodes/reviews/executiongates;no newcurrentmetasearch'),
        ('E_full_single_document_and_plain_report','Novicereport,oneJSON withexactpriorhistoryandlosslessE1metadata;rawarrays indexednotembedded')]
    acceptance={'status':'completed_M12E_E0_E1_and_design_stage','scientific_decisions':results['decisions'],'requirements':[{'requirement':k,'status':'verified','evidence':v} for k,v in requirements],
        'protected_history':protection,'resource_summary':{k:v for k,v in ledger.items() if k not in ('jobs','Slurm_memory_samples')},'final_review':finalreview,
        'E1_full_second_neural_replay':False,'old_private_final_access':False,'neural_retraining':0,'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    write(STAGE/'evidence/final-acceptance.json',acceptance)
    encoded={};skipped=set()
    for seed in (33000,33001):
        p=RUN/'E1'/f'seed-{seed}/complete.json';d=read(p);assert all(v['specification']['phase']=='development' for v in d['rows'])
        packed=encode_tables(d);assert decode_tables(packed)==d
        encoded[str(seed)]={'original_path':str(p.relative_to(ROOT)).replace('\\','/'),'original_file_SHA256':sha(p),'codec':'recursive_column_tables_v1','semantic_roundtrip_verified':True,'data':packed}
        skipped.add(p.resolve())
    published={}
    for p in STAGE.rglob('*.json'):
        if p.resolve() in skipped:continue
        d=read(p);assert_no_private_final(d);published[p.relative_to(STAGE).as_posix()]=d
    report=ROOT/'reports/2026-10-31-M12-E-完整进度与瓶颈.md';old=ROOT/'reports/2026-10-30-M12-D-项目全量汇总.json'
    scripts=sorted(list((ROOT/'scripts').glob('*m12e*.py'))+list((ROOT/'cluster').glob('m12e*.sbatch'))+
        [ROOT/'src/patrol_rsi/meta_schema.py',ROOT/'src/patrol_rsi/patrol_timing.py',ROOT/'src/patrol_rsi/e_stage.py',ROOT/'tests/test_meta_schema.py',ROOT/'tests/test_patrol_timing.py'])
    bundle={'文件说明':{'标题':'Zhinengti-RSI全量汇总：M12-E合法协议与补查时机','schema':'m12e-full-history-evidence-v1','时间UTC':acceptance['checked_utc'],
        '阅读顺序':'先读当前结论与完整报告_markdown。原始历史中的状态只代表当时。',
        '内容':'本轮所有发布JSON、完整新手说明、全部E0原始回复、E1开发元数据、实际审计/成本、未来设计与此前全量历史。',
        '列式编码':'E1完整开发元数据仅把重复字典键提取为列名，无数字有损压缩；用scripts/build_m12e_bundle.py的decode_tables还原，每个种子逐值验证。',
        '边界':'不新增旧最终逐行数据；大体积新物理数组、原始回合和公开向量留于集群，来源SHA/路径/已核对摘要保留。'},
        '当前结论':results['decisions'],'完整报告_markdown':report.read_text(encoding='utf-8'),'本轮逐项验收':acceptance,
        'E0与E1实际结果':results,'全部已发布本轮JSON':published,'E1完整开发元数据_可逆列式表格':encoded,
        '本轮新增执行和审计源码':{p.relative_to(ROOT).as_posix():{'sha256':sha(p),'text':p.read_text(encoding='utf-8')} for p in scripts},
        '全部Slurm日志与账本':{p.relative_to(STAGE).as_posix():p.read_text(encoding='utf-8',errors='replace') for p in RUN.rglob('*') if p.is_file() and p.suffix in ('.out','.err','.psv','.sbatch')},
        '历史_M12D及此前全量汇总_原内容':read(old),'历史全量原文件SHA256':sha(old),
        '复现入口':{'仓库':'https://github.com/DonsonHH/Zhinengti-RSI','当前状态':'CURRENT_RESEARCH.md','阶段目录':STAGE.relative_to(ROOT).as_posix(),
            '集群私有原件':'/share/home/renly/renly-patrol-rsi/runs/20261031-m12e','只读报告重建':'python scripts/report_m12e.py --stage research/2026-10-31-m12e-timing-schema; python scripts/build_m12e_bundle.py',
            '提示':'准备脚本为初始草案生成记录；最终经审查JSON为权威。正式E0/E1目录禁止重跑；当前E2为NO-GO，未来协议没有执行授权。'}}
    out=ROOT/'reports/2026-10-31-M12-E-项目全量汇总.json';out.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    assert out.stat().st_size<95_000_000
    reopened=read(out);assert reopened['完整报告_markdown']==report.read_text(encoding='utf-8') and reopened['历史_M12D及此前全量汇总_原内容']==read(old)
    for seed in (33000,33001):assert decode_tables(reopened['E1完整开发元数据_可逆列式表格'][str(seed)]['data'])==read(RUN/'E1'/f'seed-{seed}/complete.json')
    print(json.dumps({'bundle_bytes':out.stat().st_size,'bundle_SHA256':sha(out),'requirements':len(requirements),'old_files_checked':len(original['files']),'E0_calls':24,'E1_episodes':220,'E2':'NO-GO'}),flush=True)


if __name__=='__main__':main()

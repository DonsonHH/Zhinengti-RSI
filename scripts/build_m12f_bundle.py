"""One readable JSON document with lossless historical and execution archives."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib


def digest(data):return hashlib.sha256(data).hexdigest()


def packed_bytes(path,*,compress=True):
    data=path.read_bytes();payload=zlib.compress(data,9) if compress else data
    return {'文件名':path.name,'原始字节数':len(data),'原始SHA256':digest(data),
        '编码':'zlib+base64' if compress else 'base64','内容':base64.b64encode(payload).decode('ascii')}


def unpack(value):
    data=base64.b64decode(value['内容'],validate=True)
    if value['编码']=='zlib+base64':data=zlib.decompress(data)
    if len(data)!=value['原始字节数'] or digest(data)!=value['原始SHA256']:raise ValueError('Lossless embedded archive identity')
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--source-version',default='v6')
    p.add_argument('--replace-if-sha256',help='Replace only an own unpublished candidate matching this exact hash.');a=p.parse_args()
    output=a.output
    if output.exists():
        if a.replace_if_sha256 is None or digest(output.read_bytes())!=a.replace_if_sha256:raise ValueError('Existing artifact is not the declared unpublished candidate')
        target=output.with_suffix('.next.json')
    else:target=output
    root=Path(__file__).resolve().parents[1];stage=root/'research/2026-10-31-m12f-timing-skill'
    old=root/'reports/2026-10-31-M12-E-项目全量汇总.json'
    if digest(old.read_bytes())!='c3e3a2051de712ca3b84221140f70dd7834d15e05d78e8e257f20d3c3e29431e':raise ValueError('Historical82MB document changed')
    summary=json.loads(a.summary.read_bytes())
    if not summary.get('bounded_stage_complete'):raise ValueError('Actual F1/conditionalF2 completion required before final bundle')
    calls=[]
    for phase in ('F1','F2'):
        area=stage/'evidence/run'/phase
        if area.exists():calls.extend(json.loads(p.read_bytes()) for p in area.rglob('call-*.result.json') if p.parent.name=='calls')
    if len(calls)!=summary['actual_costs']['actual_model_calls']:raise ValueError('Complete actual generation cost receipts')
    summary['actual_costs'].update(model_input_tokens=sum(v['input_tokens'] for v in calls),known_output_tokens=sum(v['output_tokens'] for v in calls),
        unknown_output_calls=0,generation_call_wall_s=sum(v['generation_s'] for v in calls),
        token_scope='Source-measured successful generation receipts, with limits and charges independently checked; no second model/tokenizer execution.')
    json_evidence={};text_evidence={}
    for path in sorted((stage/'evidence').rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(root).as_posix();data=path.read_bytes()
        item={'SHA256':digest(data),'字节数':len(data)}
        if path.suffix=='.json':
            item['内容']=json.loads(data);json_evidence[relative]=item
        elif path.suffix in ('.md','.txt','.sbatch','.log','.err'):
            item['内容']=data.decode('utf-8',errors='strict');text_evidence[relative]=item
    # No raw old/new final arrays or undisclosed final event logs are exported.
    # Public result metadata and its record hashes remain fully inspectable above.
    archive=stage/'source'/f'code-{a.source_version}.tar.gz'
    files={}
    for path in sorted(stage.glob('*.json')):
        files[path.name]={'SHA256':digest(path.read_bytes()),'内容':json.loads(path.read_bytes())}
    for path in sorted(stage.glob('*.md')):text_evidence[path.relative_to(root).as_posix()]={'SHA256':digest(path.read_bytes()),'内容':path.read_text(encoding='utf-8')}
    review=stage/'source/user-review.txt'
    earlier_report=root/'reports/2026-10-31-Zhinengti-RSI-项目全景与瓶颈-单文件版.md'
    audit_sources={}
    for name in ('audit_m12f_actual.py','audit_m12f_prompts.py','finish_m12f_summary.py','build_m12f_bundle.py'):
        path=root/'scripts'/name;data=path.read_bytes()
        audit_sources[name]={'SHA256':digest(data),'字节数':len(data),'内容':data.decode('utf-8')}
    value={'文件说明':{'格式':'M12F-single-document-v1','阅读方式':'先看完整报告和当前实际结论；JSON证据供核查；压缩历史无需解码即可理解当前项目。',
        '完整历史':'此前82MB全量JSON按原字节无损内嵌，包含全部此前内容；源码tar.gz也按原字节内嵌。',
        '数据范围':'真实模型权重、物理数组和未公开原始回合不内嵌；保留来源、哈希、公开统计与复现边界。',
        '冻结清单':'方案里的审查前状态和planned_not_generated记录冻结时状态，不随执行改写；当前实际生成和停止状态以实际结论、完成收据及inventory为准。'},
        '当前实际结论':summary,'完整报告_markdown':a.report.read_text(encoding='utf-8'),
        '历史_20时28分进度快照':{'性质':'本轮正式执行之前的历史状态，当前结论以上方实际结果为准。','SHA256':digest(earlier_report.read_bytes()),'完整Markdown':earlier_report.read_text(encoding='utf-8')},
        '原始复核建议':review.read_text(encoding='utf-8'),'本轮方案与清单':files,
        '独立审计与交付源码':audit_sources,
        '全部本轮已发布JSON证据':json_evidence,'本轮文字与日志证据':text_evidence,
        '历史_M12E及此前全量JSON_无损归档':packed_bytes(old),'实际封存执行源码_tar_gz':packed_bytes(archive,compress=False),
        '内嵌归档解码示例_python':'import json,base64,zlib,hashlib,pathlib\nd=json.loads(pathlib.Path("本文件.json").read_bytes())\nfor key in ("历史_M12E及此前全量JSON_无损归档","实际封存执行源码_tar_gz"):\n v=d[key];b=base64.b64decode(v["内容"],validate=True)\n if v["编码"]=="zlib+base64": b=zlib.decompress(b)\n assert len(b)==v["原始字节数"] and hashlib.sha256(b).hexdigest()==v["原始SHA256"]\n with pathlib.Path(v["文件名"]).open("xb") as f: f.write(b)\n'}
    if unpack(value['历史_M12E及此前全量JSON_无损归档'])!=old.read_bytes():raise ValueError('Historical round trip')
    if unpack(value['实际封存执行源码_tar_gz'])!=archive.read_bytes():raise ValueError('Execution source round trip')
    with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    actual=json.loads(target.read_bytes())
    if actual!=value:raise ValueError('JSON serialization semantic round trip')
    if target.stat().st_size>=95000000:raise ValueError('Single JSON exceeds publication size limit')
    if target!=output:
        if digest(output.read_bytes())!=a.replace_if_sha256:raise ValueError('Unpublished candidate changed during rebuild')
        target.replace(output)
    print(json.dumps({'output':str(a.output.resolve()),'bytes':a.output.stat().st_size,'SHA256':digest(a.output.read_bytes()),
        'published_JSON_evidence':len(json_evidence),'published_text_evidence':len(text_evidence),
        'all_previous_history_exactly_preserved':True,'execution_archive_exactly_preserved':True}),flush=True)


if __name__=='__main__':main()

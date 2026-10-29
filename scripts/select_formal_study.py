"""Close an audited phase and freeze its selection before any next split."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from patrol_rsi.formal_study import load_protocol,conditions,method_definitions,development_selection,validation_selection

p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','validation'),required=True)
p.add_argument('--run-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path)
args=p.parse_args();protocol=load_protocol(args.protocol);all_rows=[];sources=[]
previous=json.loads(args.previous.read_bytes()) if args.previous else None
methods=method_definitions(protocol,args.phase,previous)
for chunk,seed in enumerate(protocol['physical_seeds'][args.phase]):
    folder=args.run_root/f'{args.phase}-{chunk}'
    index=folder/'case-index.json';rows=json.loads(index.read_bytes())
    audit_path=args.run_root/f'{args.phase}-{chunk}-audit.json';audit=json.loads(audit_path.read_bytes())
    digest=hashlib.sha256(index.read_bytes()).hexdigest()
    if audit['case_index_sha256']!=digest or audit['actual_cases_replayed']!=len(rows) or audit['exact_violations']:
        raise RuntimeError('Phase audit gate incomplete')
    expected=[(k,s) for k in methods for s in conditions(protocol,args.phase,chunk)]
    if [(r['alias'],r['specification']) for r in rows]!=expected:raise RuntimeError('Selection scope differs')
    all_rows.extend(rows);sources.append({'chunk':chunk,'seed':seed,'case_index_sha256':digest,
        'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'cases':len(rows)})
result=development_selection(all_rows) if args.phase=='development' else validation_selection(all_rows,previous['retained'])
result.update(frozen_unix=time.time(),protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
    previous_sha256=hashlib.sha256(args.previous.read_bytes()).hexdigest() if args.previous else None,
    audited_sources=sources,final_test_opened=False)
with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k in ('phase','ranked','retained','selected','fallback','rationale')},indent=2))

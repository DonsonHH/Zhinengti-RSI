"""Close an audited phase and freeze its selection before any next split."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from patrol_rsi.formal_study import load_protocol,development_selection,validation_selection,audited_phase_rows,verified_selection

p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','validation'),required=True)
p.add_argument('--run-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path)
args=p.parse_args();protocol=load_protocol(args.protocol);all_rows=[];sources=[]
previous=(verified_selection(args.previous,protocol,args.run_root,'development') if args.previous else None)
all_rows,sources=audited_phase_rows(args.run_root,protocol,args.phase,previous)
result=development_selection(all_rows) if args.phase=='development' else validation_selection(all_rows,previous['retained'])
result.update(frozen_unix=time.time(),protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
    previous_sha256=hashlib.sha256(args.previous.read_bytes()).hexdigest() if args.previous else None,
    audited_sources=sources,final_test_opened=False)
with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k in ('phase','ranked','retained','selected','fallback','rationale')},indent=2))

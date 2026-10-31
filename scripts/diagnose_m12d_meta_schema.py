"""Post-hoc reproduction of actual meta-output failures; no repair or new calls."""
import argparse
import hashlib
import json
from pathlib import Path

from patrol_rsi.rsi_controlled import ControlledImprover


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();r=a.run_root
    identity=read(r/'rsi-execution-identity.json');hashes={v['path']:v['sha256'] for v in identity['artifacts']}
    source=Path(identity['source_root']);assert source.resolve()==Path(__import__('patrol_rsi.rsi_controlled',fromlist=['x']).__file__).resolve().parents[2]
    for name in ('rsi_controlled.py','improver_protocol.py'):
        path=source/'src/patrol_rsi'/name;assert sha(path)==hashes[str(path)]
    rows=[]
    for trial in range(3):
        root=r/f'rsi-trial-{trial}';ad=read(root/'adapt-validation-complete.json');found=0
        for call in ad['model_call_receipts']:
            if not call['prompt'].get('task','').startswith('Propose one controlled improver protocol'):continue
            actual=read(root/'calls'/f"call-{call['number']:02d}.result.json");assert actual['raw_response']==call['raw_response']
            raw=actual['raw_response'];proposal=json.loads(raw[raw.index('{'):raw.rindex('}')+1])['protocol']
            try:ControlledImprover.from_dict(proposal);valid=True;error=None
            except (ValueError,TypeError,KeyError) as e:valid=False;error=type(e).__name__+': '+str(e)
            entry=ad['meta_rounds'][found];assert valid==entry['protocol_proposal_valid'];found+=1
            fields=proposal.get('protocol',proposal);diagnostics=fields.get('diagnostics',[])
            rows.append({'trial':trial,'call_number':call['number'],'actual_schema_valid':valid,'parser_error':error,
                'missing_inner_protocol_nesting':'protocol' not in proposal,'diagnostic_count':len(diagnostics),'diagnostic_count_exceeds_maximum':len(diagnostics)>3,
                'protocol_update_accepted':entry['accepted'],'actual_candidate_protocol_equals_incumbent':entry['candidate']['protocol']==entry['incumbent']['protocol'],
                'input_tokens':call['input_tokens'],'output_tokens':call['output_tokens'],'generation_s':call['generation_s'],
                'actual_response_SHA256':sha(root/'calls'/f"call-{call['number']:02d}.result.json"),
                'raw_response_text_SHA256':hashlib.sha256(raw.encode('utf-8')).hexdigest(),'original_adaptation_SHA256':sha(root/'adapt-validation-complete.json')})
        assert found==2
    assert len(rows)==6
    report={'completed':True,'diagnostic_scope':'Post-hoc format failure only; no parser tolerance, candidate repair, model call, new physical process or score reselection.',
        'meta_proposals':len(rows),'valid_meta_proposals':sum(v['actual_schema_valid'] for v in rows),'rows':rows,
        'model_calls_in_this_diagnosis':0,'environment_episodes_in_this_diagnosis':0,
        'interpretation':'All meta attempts consumed slots but no valid new protocol entered comparison. No evidence about effectiveness of legal meta updates.',
        'frozen_RSI_identity_SHA256':sha(r/'rsi-execution-identity.json')}
    with (r/'meta-schema-diagnosis.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print('M12D_META_SCHEMA_DIAGNOSIS',len(rows),report['valid_meta_proposals'],flush=True)


if __name__=='__main__':main()

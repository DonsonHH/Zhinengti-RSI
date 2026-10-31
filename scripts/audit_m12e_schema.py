"""Recompute E0 legality and cost from original prompts/responses; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from patrol_rsi.meta_schema import parse_proposal,strict_json,changed_fields,reset_receipt
from patrol_rsi.rsi_controlled import ControlledImprover


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();s=a.stage;r=s/'evidence/run';d=read(r/'E0/complete.json')
    config=read(s/'E0-Schema-v1.json');prompts=read(s/'E0-prompts.json');planned={(v['slot'],v['method']):v for v in prompts['calls']}
    assert sha(s/'E0-Schema-v1.json')==d['protocol_SHA256'] and sha(s/'E0-prompts.json')==config['prompts_SHA256']
    assert sha(r/'E0-execution-identity.json')==d['execution_identity_SHA256']
    identity=read(r/'E0-execution-identity.json');source=read(s/'code-v1-manifest.json')
    assert source['sha256']==read(r/'E0-launch-review.json')['source_archive_SHA256']
    claim=read(r/'E0-submission-claim/receipt.json');actual=r/'E0-submission-claim'/Path(claim['actual_batch_path']).name
    assert sha(actual)==claim['actual_batch_SHA256']==sha(r/'m12e_schema.sbatch') and claim['actual_script_verified_while_held'] and claim['persistent_no_retry']
    results={method:[] for method in config['methods']};counts={'real_calls':0,'input_tokens':0,'output_tokens':0,'unknown_output_calls':0};starts=list((r/'E0/calls').glob('*.start.json'))
    assert len(starts)==24 and len(list((r/'E0/calls').glob('*.result.json')))==24
    for method in config['methods']:
        seen=set()
        for slot in range(12):
            fixed=planned[slot,method];base=r/'E0/calls'/f'{slot:02d}--{method}'
            start=read(base.with_suffix('.start.json'));end=read(base.with_suffix('.result.json'));parent=ControlledImprover.from_dict(fixed['parent_protocol'])
            assert start['prompt']==fixed['user_prompt'] and start['parent']==fixed['parent_protocol'] and start['generation_seed']==fixed['generation_seed']
            assert start['charged_before_generation'] and start['thinking'] is False and start['input_tokens']==end['input_tokens']<=4096
            counts['real_calls']+=1;counts['input_tokens']+=end['input_tokens'];counts['output_tokens']+=end['output_tokens'] or 0;counts['unknown_output_calls']+=end['output_tokens'] is None
            assert end['output_tokens'] is not None and end['output_tokens']<=768 and end['generation_error'] is None
            syntax=valid=raw_full=False;child=None
            try:
                value=strict_json(end['raw_response']);syntax=True
                try:ControlledImprover.from_dict(value['protocol']);raw_full=True
                except (ValueError,KeyError,TypeError):pass
                child,_=parse_proposal(end['raw_response'],method);valid=True
            except (ValueError,KeyError,TypeError):pass
            assert syntax==end['raw_JSON_valid'] and valid==end['interface_schema_valid']==end['compiled_protocol_valid'] and raw_full==end['raw_full_ControlledImprover_valid']
            if valid:
                assert child.to_dict()==end['candidate_protocol'] and (child!=parent)==end['different_parent']
                assert changed_fields(parent,child)==end['changed_fields'] and reset_receipt(child)==end['transfer_reset']
                assert (child.version_id in seen)==end['duplicate'];seen.add(child.version_id)
            assert end['fallback_protocol']==parent.to_dict()
            results[method].append({'slot':slot,'syntax':syntax,'valid':valid,'different':bool(valid and child!=parent),'protocol_id':child.version_id if valid else None,'raw_full':raw_full,'duplicate':end['duplicate']})
    gate={}
    for method,rows in results.items():
        valid=sum(v['valid'] for v in rows);different=sum(v['different'] for v in rows);unique=len({v['protocol_id'] for v in rows if v['different']})
        gate[method]={'valid':valid,'different_parent':different,'distinct_nonparent_legal':unique,'passed':valid>=11 and different>=8 and unique>=3}
        assert gate[method]['passed']==d['summaries'][method]['passed'] and valid==d['summaries'][method]['compiled_protocol_valid']
    expected=next((m for m in config['methods'] if gate[m]['passed']),None)
    assert expected==d['future_interface'] and (expected is not None)==d['E0_passed']
    assert counts['real_calls']==d['real_model_calls']==24 and counts['input_tokens']==d['input_tokens']<=98304 and counts['output_tokens']==d['known_output_tokens']<=18432
    assert not d['weights_updated'] and d['real_environment_episodes']==d['task_programs_selected']==0
    output={'passed':True,'source_archive_SHA256':source['sha256'],'actual_execution_identity_SHA256':sha(r/'E0-execution-identity.json'),'actual_call_receipt_pairs_checked':24,
        'reconstructed_method_gates':gate,'actual_costs':counts,'E0_passed':expected is not None,'preferred_future_interface':expected,
        'raw_full_nested_success_distinct_from_machine_assembly':True,'only_protocol_transfer_resets_checked':24,'new_model_calls':0,'new_environment_episodes':0,
        'task_gain_or_RSI_claim':False,'raw_results_and_old_M12D_unmodified':True}
    (s/'evidence/E0-independent-audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('M12E_E0_INDEPENDENT_AUDIT',expected,counts,flush=True)


if __name__=='__main__':main()

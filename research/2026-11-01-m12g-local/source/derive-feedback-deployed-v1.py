"""Retained-public-record annotation, not an extra policy evaluation or repair."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--feedback-module',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('derived_feedback',a.feedback_module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=a.run_root;G=read(root/'G0-complete.json');area=root/'G1a-derived-feedback-v1.1';area.mkdir(exist_ok=False)
    totals={'cards':0,'earlier_CHECK_decision_cards':0,'same_CHECK_completion_cards':0,'later_Tour_followup_cards':0,'later_positive_with_direct_normal':0};rows=[]
    for result in G['results']:
        name=result['candidate']['id'];policyrows=[]
        for i,row in enumerate(result['rows']):
            path=Path(row['record_path']);assert sha(path)==row['record_SHA256']
            raw=json.loads(gzip.decompress(path.read_bytes()));index=module.OptionalResultIndex(raw['selected_certificates'],raw['inspection_events']);cards=[]
            for old in raw['g_difference_cards']:
                v=index.annotate(old);cards.append(v);totals['cards']+=1
                if old['decision']['executed_action']=='CHECK_THIS':
                    totals['earlier_CHECK_decision_cards']+=1
                    if v['follow_up_is_optional_completion']:totals['same_CHECK_completion_cards']+=1
                    else:totals['later_Tour_followup_cards']+=1
                    if old['actual_follow_up']['observed_alarm'] and not v['direct_optional_measurement']['observed_alarm']:totals['later_positive_with_direct_normal']+=1
            sample=module.select_refined_cards(cards);out=area/f'{name}-case-{i:02d}.json'
            value={'source_record_path':str(path),'source_record_SHA256':row['record_SHA256'],'feedback_contract':'g-difference-v1.1','cards_available':len(cards),'model_visible_sample':sample,
                'no_model_called':True,'scope':'Public observations only; earlier CHECK decisions and their direct completions distinguished from later Tour followups.'}
            with out.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
            policyrows.append({'source_record_SHA256':row['record_SHA256'],'path':str(out),'sha256':sha(out),'sample_cards':len(sample)})
        rows.append({'candidate_id':name,'cases':policyrows})
    value={'complete':True,'source_G0_SHA256':sha(root/'G0-complete.json'),'feedback_module_SHA256':sha(a.feedback_module),'totals':totals,'rows':rows,
        'G0_records_or_scores_modified':False,'new_environment_episodes':0,'new_model_calls':0,'new_physical_processes':0,
        'meaning':'No inference of earlier truth from later measurement. Optional success counts use exact completion event and are deduplicated by selected decision.'}
    with (area/'complete.json').open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
    print('G_DERIVED_FEEDBACK',totals,flush=True)


if __name__=='__main__':main()

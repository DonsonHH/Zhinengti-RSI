"""B1 outcomes, bounded opportunity references, timing and signal diagnostics."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(p.read_bytes())


def write(p,v):
    with p.open('x') as f:
        json.dump(v,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run;formal=root/'formal';out=args.output;out.mkdir(exist_ok=False)
    index=read(formal/'case-index.json');lookup={r['case']:r for r in index};groups=read(formal/'groups.json')
    atlas=read(formal/'atlas-before-branches.json');branches=read(formal/'branch-index.json')
    tests={k:v for k,v in atlas.items() if lookup[k]['specification']['phase']=='test'}
    process_index=read(formal/'process-index.json');arrays_by_process={}
    branch_by_base={key:[r for r in branches if r['base_case']==key] for key in atlas}
    event_rows=[];references=[]
    for key,a in tests.items():
        baseline=lookup[key];candidates=branch_by_base[key]
        options=[baseline]+[lookup[r['branch_case']] for r in candidates]
        def rank(r):
            m=r['metrics'];return m['missed_segments'],m['consequence_weighted_censored_delay_s'] or 0.,m['travel_m']
        best=min(options,key=rank)
        references.append({'base_case':key,'best_sample_case':best['case'],'sample_actions':len(candidates),
            'base_missed':baseline['metrics']['missed_segments'],'best_missed':best['metrics']['missed_segments'],
            'base_weighted_delay_sum':baseline['weighted_delay_sum'],'best_weighted_delay_sum':best['weighted_delay_sum'],
            'weight_sum':baseline['event_weight_sum']})
        for segment in a['segments']:
            samples=[r for r in candidates if any(s['segment_index']==segment['segment_index'] for s in r['sources'])]
            if baseline['process_key'] not in arrays_by_process:
                with np.load(formal/process_index[baseline['process_key']]['path'],allow_pickle=False) as pack:
                    arrays_by_process[baseline['process_key']]={k:pack[k] for k in pack.files}
            arrays=arrays_by_process[baseline['process_key']];interval=segment['interval']
            task_index=int(interval['task_id'].split(':')[-1]);endpoints=[]
            for tick in range(int(interval['onset_s']/5),int(interval['end_s']/5)):
                if interval['kind']=='thermal':
                    noise=np.random.default_rng(3000000+baseline['specification']['seed']*10000+task_index*361+tick).normal(0,.1)
                    positive=arrays['rack_temperature_c'][tick,task_index]+noise>65
                elif interval['kind']=='electrical':
                    positive=arrays['contact_temperature_c'][tick,task_index]>60
                elif interval['kind']=='cooling':
                    positive=arrays['rack_flow_kg_s'][tick].reshape(16,8)[task_index].sum()<33.6
                else:
                    error=arrays['public_sensor_temperature_c'][tick]-arrays['rack_temperature_c'][tick]
                    positive=abs(error.reshape(16,8)[task_index].mean())>1
                if positive:
                    endpoints.append((tick+1)*5.)
            event_rows.append({'base_case':key,'process_key':baseline['process_key'],'noise_c':baseline['specification']['noise_c'],
                'scenario':baseline['specification']['scenario'],'duration_s':baseline['specification']['duration_s'],
                'segment_index':segment['segment_index'],'interval':segment['interval'],
                'reference_positive_endpoints_s_evaluator_only':endpoints,
                'baseline_detected':segment['interval']['detected'],'legal_witness_instances':segment['legal_witness_instances'],
                'old_cued_witnesses':segment['old_cued_witnesses'],'new_cued_witnesses':segment['new_cued_witnesses'],
                'event_signal_first_s':segment['event_signal_first_s'],'sample_branches':[r['branch_case'] for r in samples],
                'best_total_missed_delta':min([r['labels']['missed_delta'] for r in samples],default=None),
                'target_recovered_and_total_misses_reduced':any(r['labels']['target_recovered_segments']>0 and r['labels']['missed_delta']<0 for r in samples)})
    event_summary={}
    for name,rows in [('all',event_rows)]+[(kind,[r for r in event_rows if r['interval']['kind']==kind]) for kind in ('thermal','electrical','cooling','control')]:
        missed=[r for r in rows if not r['baseline_detected']]
        event_summary[name]={'event_records':len(rows),'tour_missed_records':len(missed),
            'tour_missed_with_positive_reference_endpoint':sum(bool(r['reference_positive_endpoints_s_evaluator_only']) for r in missed),
            'tour_missed_with_legal_one_check_witness':sum(r['legal_witness_instances']>0 for r in missed),
            'tour_missed_with_old_cued_witness':sum(r['old_cued_witnesses']>0 for r in missed),
            'tour_missed_with_new_cued_witness':sum(r['new_cued_witnesses']>0 for r in missed),
            'tour_missed_target_recovered_and_net_misses_reduced_in_a_sample':sum(r['target_recovered_and_total_misses_reduced'] for r in missed)}
    reference={'cases':len(references),'sampled_actions':sum(r['sample_actions'] for r in references),
        'baseline_missed':sum(r['base_missed'] for r in references),'best_sample_missed':sum(r['best_missed'] for r in references),
        'baseline_weighted_wait_s':sum(r['base_weighted_delay_sum'] for r in references)/sum(r['weight_sum'] for r in references),
        'best_sample_weighted_wait_s':sum(r['best_weighted_delay_sum'] for r in references)/sum(r['weight_sum'] for r in references),
        'scope':'One realized branch or baseline per condition, selected offline from first/last/cheapest witnesses. Not a global bound.'}
    decisions={};timing_examples=[]
    for name,entry in groups.items():
        action_times=[];offsets=[];waits=Counter();new_confirmations=0
        for case in entry['case_ids']:
            row=lookup[case];raw=read_case(formal/row['path'])
            logs=[r for r in raw['workflow_decisions'] if r['task']!=r['baseline_next']]
            assert len(logs)<=1
            for r in logs:
                action_times.append(r['time_s'])
                if row['specification']['scenario']!='healthy':
                    offsets.append(r['time_s']-row['specification']['fault_start_s'])
            waits.update(r['decision'] for r in raw['timing_decisions'])
            new_confirmations+=sum(r['measurement_transition']=='new_measured_alert' for r in raw['inspection_events'])
            if row['specification']['scenario']=='coldplate_fouling' and row['specification']['duration_s']==180 and row['specification']['seed']==8200 and row['specification']['noise_c']==.2:
                timing_examples.append({'method':name,'case':case,'action_time_s':logs[0]['time_s'] if logs else None,
                    'decisions':[r for r in raw['timing_decisions'] if r['budget_available']],
                    'fault_start_s_evaluator_only':row['specification']['fault_start_s']})
        decisions[name]={'actions':len(action_times),'times_s':action_times,'action_min_s':min(action_times) if action_times else None,
            'action_max_s':max(action_times) if action_times else None,'actions_first_30s':sum(t<=30 for t in action_times),
            'offset_from_hidden_fault_start_s_evaluator_only':offsets,'decision_counts':dict(waits),
            'all_new_measured_alerts':new_confirmations}
    confusion={k:Counter() for k in next(iter(tests.values()))['confusion']}
    for a in tests.values():
        for name,counts in a['confusion'].items():
            confusion[name].update(counts)
    signals={}
    for name,c in confusion.items():
        event_flags=[r['event_signal_first_s'][name] for r in event_rows]
        leads=[r['interval']['onset_s']-r['event_signal_first_s'][name] for r in event_rows if r['event_signal_first_s'][name] is not None]
        signals[name]=dict(c)|{'precision':c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None,
            'recall':c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None,
            'event_records_with_legal_optional_trigger_in_window':sum(v is not None for v in event_flags),
            'event_records':len(event_flags),'lead_to_event_onset_s':leads,
            'association_window':'Event task, from onset-120s through event end; not a calibrated independent alarm'}
    points=[r for a in tests.values() for r in a['temperature']]
    y=np.array([r['truth_c'] for r in points]);fresh=np.array([r['fresh'] for r in points],dtype=bool);hot=y>65
    temperatures={}
    for name in ('hold','linear','nominal','history'):
        pred=np.array([r[name] for r in points]);error=pred-y;metrics={}
        for subset,mask in (('all',np.ones(y.shape,dtype=bool)),('hot',hot),('normal',~hot),('not_currently_observed_hot',hot&~fresh),('not_currently_observed',~fresh)):
            h,p=y[mask]>65,pred[mask]>65;tp=int((h&p).sum());fp=int((~h&p).sum());fn=int((h&~p).sum())
            metrics[subset]={'node_times':int(mask.sum()),'rmse_c':float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None,
                'mae_c':float(np.mean(abs(error[mask]))) if mask.any() else None,'tp':tp,'fp':fp,'fn':fn,
                'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                'estimate_min_c':float(pred[mask].min()) if mask.any() else None,'estimate_max_c':float(pred[mask].max()) if mask.any() else None}
        temperatures[name]=metrics
    paired={}
    baselines={condition(r):r for r in lookup.values() if r['tag']=='tour'}
    for name,entry in groups.items():
        cases=[]
        for case in entry['case_ids']:
            row=lookup[case];base=baselines[condition(row)]
            a,b=base['metrics'],row['metrics']
            cases.append({'case':case,'process_key':row['process_key'],'noise_c':row['specification']['noise_c'],
                'missed_delta':b['missed_segments']-a['missed_segments'],
                'weighted_wait_delta_s':b['consequence_weighted_censored_delay_s']-a['consequence_weighted_censored_delay_s'] if row['anomaly_intervals'] else None,
                'travel_delta_m':b['travel_m']-a['travel_m'],'service_delta_s':b['service_s']-a['service_s']})
        paired[name]={'fewer_misses':sum(r['missed_delta']<0 for r in cases),'more_misses':sum(r['missed_delta']>0 for r in cases),'same_misses':sum(r['missed_delta']==0 for r in cases),'cases':cases}
    audits={k:read(root/(k+'-audit.json')) for k in ('smoke','validation','formal')}
    result={'groups':groups,'paired_to_tour':paired,'decisions':decisions,'opportunity_reference':reference,
        'event_catalogue':event_rows,'event_summary':event_summary,'signal_quality':signals,'temperature_quality':temperatures,
        'validation_selection':read(root/'validation/selection-frozen.json'),'formal_summary':read(formal/'summary.json'),
        'validation_summary':read(root/'validation/summary.json'),'audits':audits,
        'notes':['No new neural training or LLM calls','Shared processes/noise and frozen old initializations are dependent',
            'Future physical truth is evaluator-only','Public horizon omits future observations and uses fixed surrogate utility',
            'Limited branch reference cannot establish global optimality or resource impossibility']}
    write(out/'aggregate.json',result);write(out/'reference-cases.json',references);write(out/'timing-examples.json',timing_examples)
    (out/'compact-index.json.gz').write_bytes(gzip.compress((json.dumps(index)+'\n').encode(),mtime=0))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    order=list(groups);labels=['Tour','Cue\nnow','Net 0','Net 1','Net 2','Cue\ngate','Evidence\nnow','Horizon\n120s','Horizon\nnominal']
    fig,axes=plt.subplots(1,2,figsize=(16,4.8));x=np.arange(len(order))
    for ax,field,title in ((axes[0],'missed','Missed anomaly records'),(axes[1],'weighted_wait_s','Weighted censored delay (s)')):
        values=[groups[k]['aggregate'][field] for k in order]
        bars=ax.bar(x,values,color=['#8894a5']*5+['#bd955d','#5c9b9e','#28699c','#9a79aa'])
        for bar,v in zip(bars,values):
            ax.annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=labels,ylabel=title);ax.tick_params(axis='x',labelsize=9)
        ax.axhline(groups['tour']['aggregate'][field],color='#334155',ls='--',lw=1)
        ax.margins(y=.15)
    fig.suptitle('One optional global check, same 32 observation conditions / 16 physical processes')
    fig.text(.05,-.015,'Net 0/1/2 are frozen B0 scorers. No new training. Nominal substitution is a deployment sensitivity check.',fontsize=10)
    fig.tight_layout();save(fig,'b1-timing-outcomes')
    fig,axes=plt.subplots(1,2,figsize=(14,5.))
    axes[0].bar(['Tour','Best sampled\nevaluator branch','Public horizon\n120s'],[reference['baseline_missed'],reference['best_sample_missed'],groups['lookahead']['aggregate']['missed']],color=['#8894a5','#ceac6b','#28699c'])
    axes[0].set(ylabel='Missed anomaly records',title='Finite sampled feasible reference; no global bound')
    timing_order=['cue_immediate','cue_gate','evidence_now','lookahead','old_net_0','old_net_1','old_net_2']
    for i,name in enumerate(timing_order):
        values=decisions[name]['times_s']
        axes[1].scatter(values,[i]*len(values),s=24,alpha=.45)
    axes[1].set(yticks=np.arange(len(timing_order)),yticklabels=timing_order,xlabel='Actual optional dispatch time (s)',title='No action means no point');axes[1].invert_yaxis()
    fig.text(.05,-.015,'Branches are separate one-action interventions. Their gains are never added into one combined schedule.',fontsize=10)
    fig.tight_layout();save(fig,'b1-opportunity-and-time')
    fig,axes=plt.subplots(1,2,figsize=(14,4.8));names=list(signals);x=np.arange(len(names))
    axes[0].bar(x-.17,[signals[k]['precision'] or 0 for k in names],.34,label='Candidate precision')
    axes[0].bar(x+.17,[signals[k]['recall'] or 0 for k in names],.34,label='Candidate recall')
    axes[0].set(xticks=x,xticklabels=['Old cue','New\nevidence','Trend','Residual','Power gap'],ylabel='Repeated candidate-instance fraction',ylim=(0,1.08));axes[0].legend(fontsize=9)
    tnames=list(temperatures);x=np.arange(4)
    axes[1].bar(x-.17,[temperatures[k]['normal']['rmse_c'] for k in tnames],.34,label='Normal node-times')
    axes[1].bar(x+.17,[temperatures[k]['hot']['rmse_c'] for k in tnames],.34,label='Hot node-times')
    axes[1].set(xticks=x,xticklabels=['Hold','Linear','Nominal','CVAE'],ylabel='Conditional current-temperature RMSE (C)');axes[1].legend(fontsize=9)
    fig.text(.05,-.015,'Identical tour histories. Event-window metrics and freshness-specific recall are separate in the JSON report.',fontsize=10)
    fig.tight_layout();save(fig,'b1-signal-and-conditional-error')
    print('B1_REPORT_COMPLETE',len(index),flush=True)


def read_case(p):
    return json.loads(gzip.decompress(p.read_bytes()))


def condition(r):
    s=r['specification'];return tuple(s[k] for k in ('scenario','duration_s','seed','noise_c','fault_start_s'))


if __name__=='__main__':
    main()

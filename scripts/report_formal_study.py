"""Complete frozen phase tables and explanatory paired diagnostics, no tuning."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from patrol_rsi.formal_study import aggregate,load_protocol


def read(path):return json.loads(path.read_bytes())
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def coordinate(r):
    return tuple(r['specification'][k] for k in ('seed','scenario','duration_s','fault_start_s','noise_c'))


def paired(method,baseline):
    pairs=[]
    for r in method:
        b=baseline[coordinate(r)];a=r['anomaly_intervals'];before=b['anomaly_intervals']
        if [(w['task'],w['onset_s'],w['end_s'],w['weight']) for w in a]!=[(w['task'],w['onset_s'],w['end_s'],w['weight']) for w in before]:
            raise RuntimeError('Unpaired hidden evaluator windows')
        missed=r['metrics']['missed_segments']-b['metrics']['missed_segments']
        delay=(r['metrics']['consequence_weighted_censored_delay_s']-b['metrics']['consequence_weighted_censored_delay_s']) if a else None
        lost=[w['task_id'] for x,w in zip(before,a) if x['detected'] and not w['detected']]
        gained=[w['task_id'] for x,w in zip(before,a) if not x['detected'] and w['detected']]
        outcome=('harmful' if missed>0 or missed==0 and delay is not None and delay>1e-9 else
            'helpful' if missed<0 or missed==0 and delay is not None and delay < -1e-9 else 'unchanged')
        q=r.get('public_queue_comparison');score=r['selected_scores'][0] if r['selected_scores'] else None
        pairs.append({'case':r['case'],'specification':r['specification'],'missed_delta':missed,'weighted_wait_delta_s':delay,
            'travel_delta_m':r['metrics']['travel_m']-b['metrics']['travel_m'],
            'extra_check_executed':r['metrics']['selected_detours']==1,'outcome':outcome,
            'lost_confirmations':lost,'gained_confirmations':gained,'action':r.get('optional_action'),
            'target_lead_s':q['target_lead_s'] if q else None,
            'affected_task_count':len(q['affected_tasks']) if q else 0,
            'first_round_elapsed_delta_s':q['extra_full_round_elapsed_s'] if q else 0.,
            'full_delay_exposure_s':sum(v['mission_delay_exposure_s'] for v in q['affected_tasks']) if q else 0.,
            'predicted_score':score['net_score'] if score else None,
            'predicted_target_benefit':score['target_benefit'] if score else None,
            'predicted_first_collateral':score['first_round_collateral_cost'] if score else None,
            'predicted_future_risk':score['future_risk_correction'] if score else None})
    return {'fewer_misses':sum(r['missed_delta']<0 for r in pairs),'equal_misses':sum(r['missed_delta']==0 for r in pairs),
        'more_misses':sum(r['missed_delta']>0 for r in pairs),
        'actual_checks':sum(r['extra_check_executed'] for r in pairs),
        'harmful_checks':sum(r['extra_check_executed'] and r['outcome']=='harmful' for r in pairs),
        'helpful_checks':sum(r['extra_check_executed'] and r['outcome']=='helpful' for r in pairs),
        'check_lost_any_other_confirmation':sum(r['extra_check_executed'] and bool(r['lost_confirmations']) for r in pairs),
        'cases':pairs}


def phase_report(rows):
    aliases=list(dict.fromkeys(r['alias'] for r in rows));base={coordinate(r):r for r in rows if r['alias']=='tour'}
    result={'actual_cases':len(rows),'groups':{},'by_seed':{},'by_fixture':{},'by_start':{},'by_noise':{},'paired_to_tour':{},'sources':{},'waits':{}}
    for alias in aliases:
        part=[r for r in rows if r['alias']==alias];result['groups'][alias]=aggregate(part)
        result['by_seed'][alias]={str(seed):aggregate([r for r in part if r['specification']['seed']==seed])
            for seed in sorted({r['specification']['seed'] for r in part})}
        result['by_fixture'][alias]={s+'--'+str(d):aggregate([r for r in part if r['specification']['scenario']==s and r['specification']['duration_s']==d])
            for s,d in sorted({(r['specification']['scenario'],r['specification']['duration_s']) for r in part})}
        result['by_start'][alias]={str(start):aggregate([r for r in part if r['specification']['fault_start_s']==start]) for start in (335,705)}
        result['by_noise'][alias]={str(noise):aggregate([r for r in part if r['specification']['noise_c']==noise]) for noise in (.2,1.)}
        result['paired_to_tour'][alias]=paired(part,base)
        source=Counter();cache=Counter()
        for r in part:
            source.update(r['source_attribution']['target_source_occurrences']);cache.update(r['source_attribution']['history_cache_types'])
        result['sources'][alias]={'target_source_occurrences':dict(source),'history_cache_types':dict(cache),
            'interpretation':'Source of observed score terms, not causal benefit or a new estimator ablation.'}
        result['waits'][alias]={'cases_with_positive_now_but_never_checked':sum(r['metrics']['selected_detours']==0 and r['waiting_positive_current_frames']>0 for r in part),
            'positive_now_wait_frames':sum(r['waiting_positive_current_frames'] for r in part)}
    return result


def forecast_diagnostics(root,rows):
    """Evaluator-only comparison at saved forecast points; no new rollout."""
    result={};arrays={}
    for row in rows:
        if row['definition']['method']!='revised':continue
        group=result.setdefault(row['specification']['phase']+':'+row['alias'],
            {'future_points':0,'predicted_positive_but_reference_zero':0,'sum_error':0.,'sum_absolute_error':0.,'sum_squared_error':0.,
                'positive_score_term_sources':{},'nonzero_topology_terms':{}})
        key=(row['folder'],row['process_key'])
        if key not in arrays:
            folder=root/row['folder'];meta=read(folder/'process-index.json')[row['process_key']]
            with np.load(folder/meta['path']) as data:arrays[key]={k:data[k].copy() for k in data.files}
        a=arrays[key];path=root/row['folder']/row['path']
        record=json.loads(gzip.decompress(path.read_bytes()))
        for d in record['stopping_decisions']:
            for kind,point in (('actual_CHECK',d.get('selected')),('best_future',d.get('future_selected'))):
                if point:
                    counts=group['positive_score_term_sources']
                    name=kind+':target:'+point['target_source'];counts[name]=counts.get(name,0)+1
                    for term in point['task_costs']:
                        if term['risk_signal'] is not None and term['risk_signal']>0:
                            name=kind+':collateral:'+term['risk_source'];counts[name]=counts.get(name,0)+1
                        if term['topology_correction']>0:
                            group['nonzero_topology_terms'][kind]=group['nonzero_topology_terms'].get(kind,0)+1
            future=d.get('future_selected')
            if not future:continue
            j=future['task'];tick=min(359,int(future['finish_s']/5)-1)
            if j<128:actual=max(a['rack_temperature_c'][tick,j]-65.,0.)/20
            elif j<144:actual=max(a['contact_temperature_c'][tick,j-128]-60.,0.)/30
            elif j<160:actual=max(1-a['rack_flow_kg_s'][tick].reshape(16,8).sum(1)[j-144]/33.6,0.)
            else:actual=float(abs((a['public_sensor_temperature_c'][tick]-a['rack_temperature_c'][tick]).reshape(16,8).mean(1)[j-160])>1)
            error=future['target_signal']-actual;group['future_points']+=1
            group['predicted_positive_but_reference_zero']+=int(actual<=0 and future['target_signal']>0)
            group['sum_error']+=float(error);group['sum_absolute_error']+=abs(float(error));group['sum_squared_error']+=float(error)**2
    for group in result.values():
        n=group['future_points']
        group['mean_signal_error']=group['sum_error']/n if n else None
        group['signal_MAE']=group['sum_absolute_error']/n if n else None
        group['signal_RMSE']=(group['sum_squared_error']/n)**.5 if n else None
    return {'groups':result,'scope':'Repeated saved best-future target points against instantaneous process reference signal; evaluator only. Positive source terms include target and collateral. Topology source not isolated. Not distinct samples, event recall, or realized counterfactual value; no estimator intervention.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False);protocol=load_protocol(args.protocol)
    phases={};all_rows=[];audits={};numeric={}
    for phase in ('development','validation','test'):
        rows=[]
        for i,seed in enumerate(protocol['physical_seeds'][phase]):
            folder=args.run_root/f'{phase}-{i}'
            part=read(folder/'case-index.json');rows.extend(part);all_rows.extend(dict(r,folder=folder.name) for r in part)
            audits[folder.name]=read(args.run_root/(folder.name+'-audit.json'))
            if audits[folder.name]['actual_cases_replayed']!=len(part):raise RuntimeError('Incomplete replay gate')
            numeric.update({k:v['numeric_digest'] for k,v in read(folder/'process-index.json').items()})
        phases[phase]=phase_report(rows)
    selection=read(args.run_root/'validation-selection.json');test=phases['test']
    tour=test['groups']['tour'];revised=test['groups']['revised'];look=test['groups']['lookahead']
    worst=max(test['by_seed']['revised'][s]['missed']-test['by_seed']['tour'][s]['missed'] for s in test['by_seed']['tour'])
    strict=revised['missed']<tour['missed'] or revised['weighted_wait_s']<tour['weighted_wait_s']-1e-10
    overall=(not selection['fallback'] and worst<=0 and revised['missed']<=tour['missed'] and
        revised['weighted_wait_s']<=tour['weighted_wait_s']+1e-10 and strict)
    vs_look=(revised['missed']<=look['missed'] and revised['weighted_wait_s']<=look['weighted_wait_s']+1e-10)
    limited=[]
    test_rows=[r for r in all_rows if r['specification']['phase']=='test']
    if not selection['fallback']:
        for fixture,g in test['by_fixture']['revised'].items():
            if g['weighted_wait_s'] is None:continue
            scenario,duration=fixture.rsplit('--',1)
            subset=[r for r in test_rows if r['specification']['scenario']==scenario and r['specification']['duration_s']==int(duration)]
            other={a:test['by_fixture'][a][fixture] for a in ('tour','lookahead','conservative')}
            dominance=all(g['missed']<=v['missed'] and g['weighted_wait_s']<=v['weighted_wait_s']+1e-10 for v in other.values())
            improved=0;safe=True
            for seed in protocol['physical_seeds']['test']:
                own=aggregate([r for r in subset if r['alias']=='revised' and r['specification']['seed']==seed])
                baseline=aggregate([r for r in subset if r['alias']=='tour' and r['specification']['seed']==seed])
                safe=safe and own['missed']<=baseline['missed']
                if own['missed']<baseline['missed'] or own['missed']==baseline['missed'] and own['weighted_delay_sum']<baseline['weighted_delay_sum']-1e-10:improved+=1
            if dominance and safe and improved>=2:limited.append({'predeclared_fixture':fixture,'strictly_improved_physical_seeds':improved})
    verdict='GO' if overall and vs_look else 'LIMITED GO' if limited else 'NO-GO'
    base={coordinate(r):r for r in all_rows if r['specification']['phase']=='test' and r['alias']=='tour'}
    lookup={coordinate(r):r for r in all_rows if r['specification']['phase']=='test' and r['alias']=='revised'}
    opportunities=[]
    for row in [r for r in all_rows if r['specification']['phase']=='test' and r['alias']=='lookahead']:
        b=base[coordinate(row)];rev=lookup[coordinate(row)]
        if row['metrics']['missed_segments']<b['metrics']['missed_segments']:
            opportunities.append({'specification':row['specification'],'tour_missed':b['metrics']['missed_segments'],
                'lookahead_missed':row['metrics']['missed_segments'],'revised_missed':rev['metrics']['missed_segments']})
    result={'stage':'C1-B3.2','phases':phases,'development_selection':read(args.run_root/'development-selection.json'),
        'validation_selection':selection,'audits':audits,'physical_configurations':len(numeric),
        'distinct_numeric_fingerprints':len(set(numeric.values())),'actual_formal_cases':len(all_rows),
        'verdict':verdict,'verdict_facts':{'validation_fallback':selection['fallback'],'worst_test_seed_missed_delta':worst,
            'strict_improvement_vs_tour':strict,'pooled_pass_and_worst_seed':overall,'no_worse_than_lookahead_both_primary':vs_look},
        'limited_scope_evidence':limited,
        'helpful_lookahead_conditions':opportunities,
        'forecast_point_diagnostics':forecast_diagnostics(args.run_root,all_rows),
        'new_training':False,'LLM_calls':0,'estimator_ablation_executed':False,
        'notes':['All fixed protocols unchanged; results include negative, fallback and equal outcomes.',
            'Predicted net score is dimensionless; realized missed counts and censored seconds are reported separately, not combined as a calibrated error.',
            'Subset effects are descriptive, not a newly validated application scope.',
            'Physical process seeds are shared across noise, phase and method repetitions.']}
    write(args.output/'aggregate.json',result)
    (args.output/'complete-case-index.json.gz').write_bytes(gzip.compress((json.dumps(all_rows)+'\n').encode(),mtime=0))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(args.output/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(args.output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    labels=['Tour','Evidence','H120','Conservative','Revised'];aliases=list(test['groups']);colors=['#8b98a8','#6a9e9f','#2c6da2','#b8965c','#615b9e']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(5)
    for ax,field,ylabel in ((axes[0],'missed','Missed anomaly records'),(axes[1],'weighted_wait_s','Weighted censored delay (s)')):
        vals=[test['groups'][a][field] for a in aliases];bars=ax.bar(x,vals,color=colors)
        for bar,v in zip(bars,vals):ax.annotate(f'{v:.2f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center')
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel);ax.margins(y=.18)
    fig.suptitle('One frozen final test: 64 observation conditions / 4 underlying seeds')
    fig.text(.04,-.015,'Revised fallback = '+str(selection['selected'])+'; all actual runs retained. Noise/method/phase repeats are dependent.')
    fig.tight_layout();save(fig,'b32-final-outcomes')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8));shown=('evidence','lookahead','conservative','revised')
    for ax,kind,keys in ((axes[0],'by_seed',[str(s) for s in protocol['physical_seeds']['test']]),
        (axes[1],'by_start',['335','705']),(axes[2],'by_noise',['0.2','1.0'])):
        data=np.array([[test[kind][a][k]['missed']-test[kind]['tour'][k]['missed'] for k in keys] for a in shown])
        bound=max(1,float(abs(data).max()));ax.imshow(data,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        ax.set(xticks=range(len(keys)),xticklabels=keys,yticks=range(4),yticklabels=labels[1:],title=kind.replace('by_',''))
        for i in range(4):
            for j in range(len(keys)):ax.text(j,i,f'{data[i,j]:g}',ha='center',va='center',color='white' if abs(data[i,j])>.65*bound else 'black')
    fig.suptitle('Miss difference vs matched Tour; negative = fewer misses')
    fig.text(.04,-.01,'Strata from one frozen final test; not independent replications.');fig.tight_layout();save(fig,'b32-final-strata')
    fig,axes=plt.subplots(1,2,figsize=(14,4.9));names=['Development','Validation','Final test']
    for ax,field,title in ((axes[0],'actual_checks','Executed optional checks'),(axes[1],'harmful_checks','Checks worse than paired Tour (miss first, then delay)')):
        for i,phase in enumerate(('development','validation','test')):
            part=phases[phase];tags=list(part['groups']);vals=[part['paired_to_tour'][a][field] for a in tags]
            positions=np.arange(len(tags))+i*10
            bars=ax.bar(positions,vals,color='#2c6da2' if field=='actual_checks' else '#ba775b')
            for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.1,str(v),ha='center',fontsize=8)
            ax.text(i*10+(len(tags)-1)/2,.96,names[i],transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
        ticks=[];ticklabels=[]
        for i,phase in enumerate(('development','validation','test')):
            tags=list(phases[phase]['groups']);ticks.extend(np.arange(len(tags))+i*10);ticklabels.extend(tags)
        maximum=max(phases[p]['paired_to_tour'][a][field] for p in phases for a in phases[p]['groups'])
        ax.set(xticks=ticks,xticklabels=ticklabels,title=title,ylim=(0,max(1.,maximum*1.25)))
        ax.tick_params(axis='x',labelrotation=65,labelsize=7)
    fig.suptitle('All stages and candidates; disjoint seed splits, different group counts')
    fig.tight_layout();save(fig,'b32-action-diagnostics')
    print('FORMAL_REPORT_DONE',len(all_rows),verdict,flush=True)


if __name__=='__main__':main()

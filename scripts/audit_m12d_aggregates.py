"""Independent arithmetic from released formal per-condition metadata."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from patrol_rsi.limited_prediction import read,write,need


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();s=a.stage;r=s/'evidence/run';checks=[]
    for phase,seeds in (('development',(31000,31001)),('validation',(31020,31021))):
        phase_report=read(r/(phase+'-complete.json'));allrows=[]
        for seed in seeds:
            d=read(r/phase/f'seed-{seed}'/'complete.json')
            need(d['formal_episodes']==90 and len(d['rows'])==90,'Exactly five arms×18conditions')
            allrows.extend(d['rows'])
        for arm,reported in phase_report['aggregate'].items():
            rows=[v for v in allrows if v['arm']==arm];seedmeans=[];waitmeans=[]
            for seed in seeds:
                own=[v for v in rows if v['seed']==seed];groups=defaultdict(list)
                for row in own:groups[row['process_fingerprint']].append(row)
                need(len(groups)==9 and all(len(v)==2 for v in groups.values()),'Independent seed/process/noise denominator')
                missing=[sum(v['metrics']['missed_segments'] for v in pair)/2 for pair in groups.values()]
                waits=[sum(v['metrics']['consequence_weighted_censored_delay_s'] for v in pair)/2 for pair in groups.values() if pair[0]['metrics']['consequence_weighted_censored_delay_s'] is not None]
                m=sum(missing)/9;w=sum(waits)/len(waits)
                target=next(v for v in reported['seeds'] if v['seed']==seed)
                need(np.isclose(m,target['macro_missed_count'],atol=1e-12) and np.isclose(w,target['macro_wait_s'],atol=1e-12),'Independent per-seed macro arithmetic')
                seedmeans.append(m);waitmeans.append(w)
            need(np.isclose(sum(seedmeans)/2,reported['macro_missed_count'],atol=1e-12),'Independent phase macro missed')
            need(np.isclose(sum(waitmeans)/2,reported['macro_wait_s'],atol=1e-12),'Independent phase macro wait')
            checks.append({'phase':phase,'arm':arm,'per_seed_missed':seedmeans,'per_seed_wait_s':waitmeans})
    report=read(r/'validation-complete.json');v=report['aggregate'];candidate=v['revised_PPO_evidence'];verdict=True
    for name in ('Tour','human_professional'):
        baseline=v[name];no_increase=all(x['macro_missed_count']<=y['macro_missed_count'] for x,y in zip(candidate['seeds'],baseline['seeds']))
        delta=candidate['macro_missed_count']-baseline['macro_missed_count']
        fraction=(baseline['macro_wait_s']-candidate['macro_wait_s'])/baseline['macro_wait_s']
        passed=no_increase and (delta<0 or delta==0 and fraction>=.05)
        need(passed==report['gate']['comparators'][name]['passed'],'Independent unchanged gate inequality')
        verdict &= passed
    need(bool(verdict)==report['gate']['passed'] and not verdict,'Independent five-arm NO-GO verified')
    write(s/'evidence/independent-aggregate-audit.json',{'status':'passed','phase_arm_checks':checks,'independently_rederived_gate':False,
        'not_using_production_aggregate_or_validation_gate':True,'whole_record_replays_and_independent_route_event_checks':'Recorded actualauditpercase inread-only source files;separate fromthisarithmeticpass.',
        'final_data_or_models_opened':False})
    print('M12D_INDEPENDENT_AGGREGATE_AUDIT_PASSED',len(checks))


if __name__=='__main__':main()

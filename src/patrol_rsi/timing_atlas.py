"""Evaluator-only candidate/event atlas, passive observations and bounded labels."""
from __future__ import annotations

from collections import Counter
import numpy as np

from .active_inspection import measured_anomaly
from .inspection_metrics import task_labels, anomaly_intervals
from .inspection_timing import public_packet, strength


class Atlas:
    def __init__(self):
        self.intervals = None
        self.segments, self.at_completion, self.temperature = [], {}, []
        self.confusion = {k: Counter(tp=0, fp=0, fn=0, tn=0) for k in ('old_cue', 'new_evidence', 'trend', 'residual', 'power_gap')}

    def observe(self, env, groups):
        if self.intervals is None:
            labels, excess = task_labels(env)
            self.intervals = anomaly_intervals(labels, excess, env.tasks, env.asset_weights)
            for i, r in enumerate(self.intervals):
                for tick in range(int(r['onset_s']/5), int(r['end_s']/5)):
                    self.at_completion[r['task'], (tick+1)*5.] = i
                self.segments.append({'segment_index': i, 'legal_witness_instances': 0,
                    'old_cued_witnesses': 0, 'new_cued_witnesses': 0, 'first': None, 'last': None, 'cheapest': None,
                    'event_signal_first_s': {k: None for k in self.confusion}})
        packet = public_packet(env)
        raw_gap = 1 - env.current_packet['public_delivered_it_kw'] / np.maximum(env.current_packet['public_requested_it_kw'], 1e-12)
        for g in groups:
            for r in g['candidates']:
                if not r['is_optional']:
                    continue
                j, finish = r['task'], r['candidate_finish_s'];row = packet['tasks'][j];task = env.tasks[j]
                linear = np.clip(row['last_temperature_c'] + row['velocity_c_s'] * min(120., row['temperature_age_s']+finish-env.clock),35,110)
                signals = {'old_cue': r['cue_strength'], 'new_evidence': strength(row,finish-env.clock),
                    'trend': max(float(linear)-65,0)/20 if task.kind=='thermal' else 0.,
                    'residual': max(row['last_temperature_c']-row['nominal_temperature_c'],0)/20 if task.kind=='thermal' and row['temperature_age_s']<=120 else 0.,
                    'power_gap': float(raw_gap[task.index] if task.kind=='thermal' else raw_gap.reshape(16,8)[task.index].mean())}
                flags = {k: bool(v>=.05 and not r['known_measured_alarm']) for k,v in signals.items()}
                segment = self.at_completion.get((j,finish))
                truth = segment is not None and measured_anomaly(task,env.trace.measurement(task,int(finish/5)-1))
                for key,flag in flags.items():
                    self.confusion[key]['tp' if truth and flag else 'fp' if flag else 'fn' if truth else 'tn'] += 1
                for i,v in enumerate(self.intervals):
                    if v['task']==j and v['onset_s']-120<=env.clock<=v['end_s']:
                        for key,flag in flags.items():
                            if flag and self.segments[i]['event_signal_first_s'][key] is None:
                                self.segments[i]['event_signal_first_s'][key]=env.clock
                if not truth:
                    continue
                target = self.segments[segment]
                witness = {'intervention': {'dispatch_time_s':g['time_s'],'agent':g['agent'],'task':j},
                    'finish_s':finish,'completion_cost_s':finish-g['time_s'],'minimum_return_slack_s':r['minimum_return_slack_s'],
                    'old_cued':flags['old_cue'],'new_cued':flags['new_evidence']}
                target['legal_witness_instances'] += 1
                target['old_cued_witnesses'] += int(flags['old_cue']);target['new_cued_witnesses'] += int(flags['new_evidence'])
                identity = lambda w:(w['intervention']['dispatch_time_s'],w['intervention']['agent'])
                if target['first'] is None or identity(witness)<identity(target['first']):
                    target['first']=witness
                if target['last'] is None or identity(witness)>identity(target['last']):
                    target['last']=witness
                costkey=lambda w:(w['completion_cost_s'],*identity(w))
                if target['cheapest'] is None or costkey(witness)<costkey(target['cheapest']):
                    target['cheapest']=witness
        if env.clock%30==0:
            obs=env.state.observer;age=np.where(obs.temperature_stamp>=0,env.clock-obs.temperature_stamp,env.clock)
            self.temperature.append({'time_s':env.clock,'truth_c':env.trace.arrays['rack_temperature_c'][int(env.clock/5)-1].tolist(),
                'fresh':(obs.temperature_stamp==env.clock).tolist(),'hold':obs.last_temperature.tolist(),
                'linear':np.clip(obs.last_temperature+env.state.velocity*np.minimum(age,120),35,110).tolist(),
                'nominal':obs.predicted_temperature.tolist(),'history':env.current_history.tolist(),
                'age_s':age.tolist(),'velocity_c_s':env.state.velocity.tolist()})

    def finish(self, record):
        selections = {}
        for s,v in zip(self.segments,record['anomaly_intervals']):
            s['interval']=v
            for name in ('first','last','cheapest'):
                witness=s[name]
                if witness is not None:
                    key=tuple(witness['intervention'].values())
                    selections.setdefault(key,{'intervention':witness['intervention'],'sources':[]})['sources'].append(
                        {'segment_index':s['segment_index'],'kind':name})
        return {'segments':self.segments,'samples':list(selections.values()),
            'confusion':{k:dict(v) for k,v in self.confusion.items()},'temperature':self.temperature,
            'scope':'Three predeclared witnesses per segment on original tour states; separate branches, not global optimum or impossibility proof'}

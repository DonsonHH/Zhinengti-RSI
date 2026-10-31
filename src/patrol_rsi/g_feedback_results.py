"""Derived feedbackv1.1: exact optional completion versus later Tour observations.

This only annotates retained public records; it never changes a policy/score.
"""
import copy


class OptionalResultIndex:
    def __init__(self,certificates,events):
        by_key={};names={}
        for event in events:
            key=(event['task'],event['agent'],event['time_s'])
            if key in by_key:raise ValueError('Duplicate qualified event identity')
            by_key[key]=event;names[event['task']]=event['task_id']
        self.results={}
        for cert in certificates:
            if cert['task']==cert['baseline_next']:continue
            key=(cert['task'],cert['agent'],cert['certificate']['candidate_finish_s'])
            if key not in by_key:raise ValueError('Optional completion has no corresponding actual measurement')
            event=by_key[key]
            if not event['calibrated'] or event['role']!=event['kind']:raise ValueError('Qualified optional measurement required')
            field={'thermal':'temperature_c','electrical':'contact_temperature_c','cooling':'pod_flow_kg_s','control':'controller_sensor_error_c'}[event['kind']]
            self.results[(cert['clock_s'],cert['agent'],names[cert['task']])]={'time_s':event['time_s'],'kind':event['kind'],
                'value':event['measurement'][field],'field':field,'qualified_role':event['role'],'calibrated':True,
                'observed_alarm':event['alarm'],'transition':event['measurement_transition']}

    def annotate(self,card):
        value=copy.deepcopy(card);d=card['decision'];selected=d['executed_action']=='CHECK_THIS'
        direct=self.results.get((d['time_s'],d['agent'],d['task_public_id'])) if selected else None
        if selected and direct is None:raise ValueError('Earlier selected CHECK decision lacks its actual completion')
        value.update(feedback_contract='g-difference-v1.1',direct_optional_measurement=copy.deepcopy(direct),
            follow_up_is_optional_completion=bool(direct and card['actual_follow_up']['time_s']==direct['time_s']),
            direct_optional_positive=bool(direct and direct['observed_alarm']),
            is_direct_optional_positive_card=bool(direct and direct['observed_alarm'] and card['actual_follow_up']['time_s']==direct['time_s']),
            result_scope='direct_optional_measurement is that CHECK completion; actual_follow_up may be a later Tour reading, not earlier truth or a counterfactual.')
        return value


def select_refined_cards(cards):
    selected=[];tie=lambda v:(v['decision']['time_s'],v['actual_follow_up']['time_s'],v['decision']['task_public_id'])
    for kind in ('thermal','electrical','cooling','control'):
        pool=[v for v in cards if v['actual_follow_up']['kind']==kind]
        if pool:selected.append(min(pool,key=tie))
    for pool in ([v for v in cards if v['admission_differs'] or v['team_choice_differs']],
                 [v for v in cards if v['is_direct_optional_positive_card']],
                 [v for v in cards if not v['actual_follow_up']['observed_alarm']],
                 [v for v in cards if v['public_effect']=='admitted_not_selected']):
        value=next((v for v in sorted(pool,key=tie) if v not in selected),None)
        if value is not None:selected.append(value)
    return selected[:8]

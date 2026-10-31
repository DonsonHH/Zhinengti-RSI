"""Editable timing over original hard-legal actions; no E1 heuristic prefilter."""
from __future__ import annotations
import copy
import numpy as np

from .causal_observation import capture_public
from .timing_skill_features import PublicTimingView


class TimingSkillSelector:
    def __init__(self,skill,*,ranker='urgency',capture_tape=None):
        if ranker not in ('urgency','ppo'):raise ValueError('Frozen existing ranking tools only')
        self.skill,self.ranker,self.capture_tape=skill,ranker,capture_tape
        self.frames=[];self.cards=[];self.latest={};self.processed=0;self.network_calls=0;self.checked_ownership=False

    def drain_events(self,env):
        for event in env.events[self.processed:]:
            j=event['task'];old=self.latest.get(j)
            if old is not None:
                if old['time_s']>event['time_s']:raise ValueError('Future decision snapshot in experience')
                kind=event['kind'];field={'thermal':'temperature_c','electrical':'contact_temperature_c','cooling':'pod_flow_kg_s','control':'controller_sensor_error_c'}[kind]
                self.cards.append({'decision':copy.deepcopy(old),'measurement':{
                    'time_s':float(event['time_s']),'kind':kind,'value':float(event['measurement'][field]),
                    'field':field,'unit':'kg/s' if kind=='cooling' else 'degC','qualified_role':event['role'],'calibrated':bool(event['calibrated']),
                    'observed_alarm':bool(event['alarm']),'transition':event['measurement_transition']},
                    'elapsed_to_actual_measurement_s':float(event['time_s']-old['time_s'])})
        self.processed=len(env.events)

    def __call__(self,env,scorer):
        self.drain_events(env)
        if not self.checked_ownership:
            owned=[j for p in env.plans for j in p.tour]
            if len(owned)!=len(set(owned)):raise ValueError('This study requires original disjoint task ownership')
            self.checked_ownership=True
        used=any(v['task']!=v['baseline_next'] for v in env.selected_certificates)
        choices={};logs=[];groups=[];reserved=env.reserved()
        for raw in np.roll(np.arange(len(env.agents)),-(env.tick%len(env.agents))):
            i=int(raw)
            if env.agents[i]['task'] is not None:continue
            ids,f,_=env.options(i,reserved);base=int(env.plans[i].tour[env.protected_cursor[i]])
            if not len(ids):choices[i]=None;continue
            if base not in ids:raise ValueError('Original mission-tail mask must keep baseline or expose no action')
            choices[i]=base
            groups.append({'agent':i,'ids':ids,'features':f,'baseline':base})
        public_view=None;candidates=[];selected=None
        if not used and groups:
            public_view=PublicTimingView(capture_public(env))
            for group in groups:
                i=group['agent'];base=group['baseline'];plan=env.plans[i];pos=env.agents[i]['position'];cursor=int(env.protected_cursor[i])
                fallback=plan.check(env.clock,pos,env.last_inspection,cursor)
                if not fallback['ok']:raise ValueError('Protected continuation invalid before timing decision')
                arrivals=dict(zip(fallback['fallback_order'],fallback['fallback_completion_s']));local=[]
                for k,raw_j in enumerate(group['ids']):
                    j=int(raw_j)
                    if j==base:continue
                    cert=env.candidate_certificate(i,j)
                    if not cert['ok']:raise ValueError('Hard mask exposed uncertified action')
                    finish=float(cert['candidate_finish_s'])
                    geometry={'agent':i,'task':j,'finish_s':finish,'target_lead_s':max(0.,min(arrivals[j],1800.)-finish),
                        'shift_s':max(0.,finish+plan.cost[j,base]-(env.clock+plan.travel_service(pos,base)))}
                    features=public_view.features(geometry);decision=self.skill.decide(features);admit=decision['action']=='CHECK'
                    item={'time_s':float(env.clock),'agent':i,'task':j,'task_public_id':env.tasks[j].task_id,'features':features,
                        'baseline_next':base,'candidate_finish_s':finish,'Tour_target_finish_s':float(arrivals[j]),
                        'skill_action':'CHECK' if admit else 'CONTINUE_TOUR','executed_action':'CONTINUE_TOUR',
                        'matched_rule_index':decision['matched_rule'],
                        'rank_urgency':float((group['features'][:,0]+group['features'][:,1])[k]),'admitted':admit,'feature_index':k}
                    candidates.append(item);local.append(item)
                group['local']=local;pool=[v for v in local if v['admitted']]
                if selected is None and pool:
                    if self.ranker=='ppo':
                        at=np.asarray([v['feature_index'] for v in pool]);values=np.asarray(scorer(group['features'][at]));self.network_calls+=1
                        if values.shape!=(len(pool),) or not np.isfinite(values).all():raise ValueError('Frozen PPO score shape/finite contract')
                        for v,value in zip(pool,values):v['rank_value']=float(value)
                    else:
                        for v in pool:v['rank_value']=v['rank_urgency']
                    selected=max(pool,key=lambda v:(v['rank_value'],-v['feature_index']));selected['executed_action']='CHECK'
                    selected['ranking_candidates']=len(pool);choices[i]=selected['task']
            for item in candidates:
                is_selected=selected is item
                item['team_action']='CHECK' if selected is not None else 'CONTINUE_TOUR'
                item['team_selected_pair']={'agent':selected['agent'],'task':selected['task']} if selected else None
                item['executed_action']='CHECK_THIS' if is_selected else 'CHECK_OTHER' if selected and selected['agent']==item['agent'] else 'CONTINUE_TOUR'
                item['selection_reason']='global_first_admitted' if is_selected else 'another_task_selected' if selected else 'skill_declined'
                if is_selected:item['check_intent']='recheck_known_alarm' if item['features']['already_alarm'] else 'acquire_or_first_confirm'
                self.latest[item['task']]=copy.deepcopy({k:v for k,v in item.items() if k!='feature_index'})
            if self.capture_tape is not None:self.capture_tape(env,candidates)
        for group in groups:
            i=group['agent'];base=group['baseline'];actual=selected if selected is not None and selected['agent']==i else None
            logs.append({'time_s':float(env.clock),'agent':i,'task':choices[i],'baseline_next':base,'actor_task':choices[i],
                'tool':self.ranker if actual is not None else 'f_tour_after_budget' if used else 'return_tour',
                'rule_index':actual['matched_rule_index'] if actual is not None else None,'legal_candidates':actual['ranking_candidates'] if actual is not None else len(group['ids']),
                'hard_legal_candidates':len(group['ids']),'detours_used_before':int(env.detours_used[i]),'baseline_completed_before':int(env.baseline_completed[i])})
        self.frames.append({'time_s':float(env.clock),'quota_used_before':used,'safe_extra_candidates':len(candidates),
            'admitted_candidates':sum(v['admitted'] for v in candidates),'decision':'CHECK' if selected else 'TOUR_BUDGET_SPENT' if used else 'CONTINUE_TOUR',
            'selected':copy.deepcopy({k:v for k,v in selected.items() if k!='feature_index'}) if selected else None})
        return choices,logs

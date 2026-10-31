"""G sidecar: original F execution kernel plus recorded same-state shadows."""
import copy
from unittest.mock import patch

from patrol_rsi.g_policy import bridge_program
from patrol_rsi.g_feedback import difference_card,retrieve_difference_cards
from patrol_rsi.g_timing_patch import digest
from patrol_rsi.timing_skill import TimingSkill
from timing_skill_execution import run_policy,policy_id


def execute_candidate(case,candidate,trace,predictor,scorer):
    policy=candidate['policy'];shadows={};counts={'candidate_rows':0,'admission_differences':0,'team_choice_differences':0,'first_difference':None}
    parent=TimingSkill.initial_e1()
    def observe(env,rows):
        first_agent=None;pool=[]
        for row in rows:
            if parent.action(row['features'])!='CHECK':continue
            if first_agent is None:first_agent=row['agent']
            if row['agent']==first_agent:pool.append(row)
        selected=max(pool,key=lambda v:(v['rank_urgency'],-v['feature_index'])) if pool else None
        parent_pair={'agent':selected['agent'],'task_public_id':selected['task_public_id']} if selected else None
        for row in rows:
            chosen=row['team_selected_pair'];child_pair={'agent':chosen['agent'],'task_public_id':env.tasks[chosen['task']].task_id} if chosen else None
            a=parent.action(row['features']);counts['candidate_rows']+=1;counts['admission_differences']+=a!=row['skill_action'];counts['team_choice_differences']+=parent_pair!=child_pair
            if counts['first_difference'] is None and (a!=row['skill_action'] or parent_pair!=child_pair):
                counts['first_difference']={'time_s':row['time_s'],'agent':row['agent'],'task_public_id':row['task_public_id'],'features':copy.deepcopy(row['features']),
                    'parent_admission':a,'child_admission':row['skill_action'],'parent_team_choice':copy.deepcopy(parent_pair),'child_team_choice':copy.deepcopy(child_pair),'recorded_before_execution':True}
            shadows[(row['time_s'],row['agent'],row['task'])]={'parent_team_choice':parent_pair,'child_team_choice':child_pair}
    if policy['executor']=='g-human-urgency-v1':
        actual={'executor':'original-human-v1','ranker':'rules','skill':None}
        with patch('timing_skill_execution.arm_programs',lambda:{'human_professional':bridge_program()}):
            record,env,audit=run_policy(case['specification'],actual,trace,predictor,scorer)
        record.update(policy_spec=copy.deepcopy(policy),policy_version_id=policy_id(policy),g_shadow_counts=None,g_difference_cards=[])
        record['physical_shell']['meaning']='Original execution configuration with explicit H01 ranking policy; rules are not the physical safety kernel.'
    else:
        child=TimingSkill.from_dict(policy['skill'])
        record,env,audit=run_policy(case['specification'],policy,trace,predictor,scorer,capture_tape=observe)
        cards=[];run_id=digest({'policy':policy,'observed_packet_sequence':record['public_telemetry_sequence_SHA256']})
        for raw in record['causal_experience_cards']:
            d=raw['decision'];key=(d['time_s'],d['agent'],d['task'])
            if key not in shadows:raise ValueError('Actual causal card has no pre-execution shadow')
            cards.append(difference_card(parent,child,raw,shadows[key],run_id,patch=candidate['patch']))
        record.update(g_shadow_counts=counts,g_difference_cards=cards,g_feedback_sample=retrieve_difference_cards(cards))
    record['g_candidate_id']=candidate['id']
    record['g_applied_patch']=candidate['patch']
    record['g_shadow_scope']='Parent decision on the child actual current public state before quota use; never an unexecuted parent outcome.'
    return record,audit

"""One physical executor for F timing policies, with original independent audits."""
import hashlib
import json
from unittest.mock import patch

from patrol_rsi.aligned_patrol import arm_programs
from patrol_rsi.causal_observation import ObservedTask
from patrol_rsi.facility_workflow import public_outcome
from patrol_rsi.improver_task import PatrolProgram,choose_program
from patrol_rsi.timing_skill import TimingSkill,information_baseline
from patrol_rsi.timing_skill_control import TimingSkillSelector
from run_m12d_paired import InputAuditedTask,execute,audit_record,usage_details


class SkillObservedTask(ObservedTask,InputAuditedTask):
    """Existing pure-output timestamp recorder plus existing packet hash recorder."""


def policy_id(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]


def skill_policy(skill,ranker='urgency'):
    return {'executor':'timing-skill-v1','ranker':ranker,'skill':skill.to_dict()}


def continue_skill():return TimingSkill.from_dict({'schema_version':'timing-skill-v1','rules':[],'default_action':'CONTINUE_TOUR'})


def canonical_actions(record):return hashlib.sha256(json.dumps(record['action_log'],sort_keys=True).encode()).hexdigest()


def run_policy(spec,policy,trace,predictor,scorer,*,capture_tape=None):
    if policy['executor']=='timing-skill-v1':
        skill=TimingSkill.from_dict(policy['skill']);selector=TimingSkillSelector(skill,ranker=policy['ranker'],capture_tape=capture_tape);program=PatrolProgram()
    elif policy=={'executor':'original-human-v1','ranker':'rules','skill':None}:
        selector=None;program=arm_programs()['human_professional']
    else:raise ValueError('Only frozen skill or original human execution policies')
    def choose(env,actor):
        if selector is not None:return selector(env,actor)
        # This reference has no PPO fallback/rule. Scores are unconsumed placeholders,
        # not a neural intervention or a claimed PPO execution.
        return choose_program(env,lambda f:__import__('numpy').zeros(len(f)))
    with patch('run_m12d_paired.InputAuditedTask',SkillObservedTask),patch('patrol_rsi.aligned_patrol.choose_program',choose):
        record,env=execute(spec,program,trace,predictor,scorer)
    if selector is not None:selector.drain_events(env)
    audit=audit_record(record,env);old_estimate=record['usage']['total_actor_scoring_calls'];usage=usage_details(record)
    calls=selector.network_calls if selector is not None else 0
    usage.update(legacy_scoring_call_estimate=old_estimate,total_actor_scoring_calls=calls,
        exclusive_shadow_actor_scoring_calls=max(0,calls-usage['actual_PPO_dispatches']))
    record.update(policy_spec=policy,policy_version_id=policy_id(policy),timing_frames=selector.frames if selector else [],
        causal_experience_cards=selector.cards if selector else [],usage=usage,
        physical_shell={'meaning':'Immutable environment configuration, not the editable skill','program':program.to_dict()})
    record['public_selection']=public_outcome(record,env.asset_weights)
    # Only actual admitted-pool scores are evaluated. A separate ungated
    # actor counterfactual for every dispatch was not observed in any F arm.
    record['public_selection']['different_from_actor']=None
    record['public_selection']['actor_counterfactual_observed']=False
    record['action_log_SHA256']=canonical_actions(record)
    return record,env,audit

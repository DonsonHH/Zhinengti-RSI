"""High-level task assignment and a common, fixed low-level controller."""
from __future__ import annotations

import numpy as np

from .deadline import measured_mean_estimate,target_deadlines
from .environment import MOVES,PatrolEnv
from .experiment import room
from .revisit import geometry

FEATURE_NAMES=('age_ratio','deadline','importance','unseen','travel_distance','nearest_other_distance',
               'reserved','target_row','target_col','executor_row','executor_col','task_time',
               'own_assignment','other_assignment_fraction')


class DispatchEnv:
    """Select a target when free, keep it until anyone observes that target.

    Policies and heuristic controls share target eligibility, unit-step shortest
    paths, commitment release, cyclic conflict priority and sensor semantics.
    """
    def __init__(self,case):
        self.case=case.copy();self.env=PatrolEnv(room(case['layout_seed'],case['noise']),case['field_seed'])
        self.targets,self.costs=geometry(case['layout_seed'])
        self.deadlines=target_deadlines(case['layout_seed'],case['profile'])
        self.assigned=np.full(4,-1,dtype=int)
        self.assignments=0;self.duplicate_assignments=0

    def prepare(self):
        for i,j in enumerate(self.assigned):
            if j>=0 and self.env.raw_age[self.targets[j]]==0:
                self.assigned[i]=-1
        free=self.assigned<0
        reserved=np.zeros(8,dtype=bool)
        for j in self.assigned[self.assigned>=0]:reserved[j]=True
        features=np.zeros((4,8,len(FEATURE_NAMES)),dtype=np.float32)
        allowed=np.zeros((4,8),dtype=bool)
        for i,p in enumerate(self.env.positions):
            for j,g in enumerate(self.targets):
                distance=max(0,float(self.costs[j][tuple(p)])-1)
                others=min(max(0,float(self.costs[j][tuple(q)])-1) for k,q in enumerate(self.env.positions) if k!=i)
                features[i,j]=[min(4,self.env.raw_age[g]/self.deadlines[j]),self.deadlines[j]/32,
                    self.env.estimate[g],float(not self.env.ever_measured[g]),distance/22,others/22,
                    float(reserved[j]),g[0]/23,g[1]/23,p[0]/23,p[1]/23,self.env.t/96,
                    float(self.assigned[i]==j),float(sum(k!=i and goal==j for k,goal in enumerate(self.assigned)))/3]
                # Do not treat zero-length repeated sensing as a new trip.
                allowed[i,j]=not self.env.neighborhood(p)[g] and not reserved[j]
        if any(free[i] and not allowed[i].any() for i in range(4)):
            raise RuntimeError('No eligible next target; review controller contract')
        order=np.roll(np.arange(4),-(self.env.t%4))
        return {'features':features,'free':free.copy(),'allowed':allowed,'reserved':reserved,'order':order}

    def assign(self,choices,state,joint=True):
        choices=np.asarray(choices)
        if choices.shape!=(4,) or not np.issubdtype(choices.dtype,np.integer):raise ValueError('Four integer target choices required')
        reserved=state['reserved'].copy()
        proposed=self.assigned.copy();new_assignments=0;new_duplicates=0
        for i in state['order']:
            if not state['free'][i]:
                if choices[i]!=-1:raise ValueError('Busy executor cannot be reassigned')
                continue
            j=int(choices[i])
            if not 0<=j<8 or not state['allowed'][i,j]:raise ValueError('Illegal high-level assignment')
            if joint and reserved[j]:raise ValueError('Joint selector cannot reserve same target twice')
            new_duplicates+=int(reserved[j]);reserved[j]=True
            proposed[i]=j;new_assignments+=1
        # Rejecting a later executor must not leave earlier choices committed.
        self.assigned=proposed
        self.duplicate_assignments+=new_duplicates;self.assignments+=new_assignments

    def low_level_scores(self):
        if (self.assigned<0).any():raise RuntimeError('All free executors must receive targets before moving')
        scores=np.full((4,8),-1e6)
        for i,p in enumerate(self.env.positions):
            costs=self.costs[self.assigned[i]]
            for a,move in enumerate(MOVES):
                if self.env.valid_move(p,a):scores[i,a]=-float(costs[tuple(p+move)])
        return scores

    def step(self,choices,state,joint=True):
        self.assign(choices,state,joint)
        before=self.env.ever_measured[self.env.scenario.targets].copy()
        scores=self.low_level_scores();actions=self.env.safe_actions(scores,'cyclic');event=self.env.step(actions)
        ages=self.env.raw_age[self.env.scenario.targets]/self.deadlines
        weights=.1+measured_mean_estimate(self.env)[self.env.scenario.targets]
        # Reward uses public measured/fixed-fill importance, not hidden truth or
        # VAE predictions. There is no map-distance reward shaping in M6.
        public_weighted=float((self.env.idleness[self.env.scenario.targets]*weights).sum()/weights.sum())
        new=float((self.env.ever_measured[self.env.scenario.targets]&~before).mean())
        violation=float((ages>1).mean())
        reward=1.0*new-violation-.1*float(np.minimum(ages,4).mean())-.25*public_weighted
        return reward,{'low_level_scores':scores.tolist(),'actions':actions.tolist(),'assigned_targets':self.assigned.tolist(),
            'public_reward':reward,'new_coverage_gain':new,'deadline_violation':violation,'public_weighted_idleness':public_weighted,
            'movement_distance':event['movement_distance']}


def heuristic_choices(state,method,rng=None):
    choices=np.full(4,-1,dtype=int);reserved=state['reserved'].copy()
    for i in state['order']:
        if not state['free'][i]:continue
        legal=state['allowed'][i]&~reserved
        if not legal.any():raise RuntimeError('No joint target choice')
        f=state['features'][i]
        if method=='random_targets':
            if rng is None:raise ValueError('RNG required')
            score=rng.random(8)
        elif method=='slack':
            # Select most urgent target using its own trip time.
            score=f[:,0]*(f[:,1]*32)+f[:,4]*22-f[:,1]*32
        elif method=='risk_age_distance':
            score=((f[:,0]+.05)*(.1+f[:,2])+f[:,3])/(1+f[:,4]*22)
        else:raise ValueError('Unknown high-level heuristic')
        j=int(np.argmax(np.where(legal,score,-1e9)));choices[i]=j;reserved[j]=True
    return choices

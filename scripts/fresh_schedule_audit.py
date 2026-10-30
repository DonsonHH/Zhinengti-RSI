"""Independent deterministic Tour and one-CHECK complete-return arithmetic."""
from __future__ import annotations

from functools import lru_cache
import math
from types import SimpleNamespace

import numpy as np

from patrol_rsi.active_inspection import ProcessTrace, graph_for, measured_anomaly
from patrol_rsi.aisles import build_agents, DEADLINES
from patrol_rsi.fresh_opportunity import need


def verify_fixed_task_metadata(bundle):
    tasks=graph_for(128,'open').tasks
    need(len(bundle['tasks'])==len(tasks)==176,'Original176-task metadata')
    for j,(row,task) in enumerate(zip(bundle['tasks'],tasks)):
        pod=task.index//8 if task.kind=='thermal' else task.index
        need((row['task'],row['kind'],row['pod'],row['service_s'],row['deadline_s'],row['access'],row['required_role'])==
            (j,task.kind,pod,task.service_s,DEADLINES[task.kind],task.access,task.required_role),
            'Original fixed task topology/service/deadline/access/role metadata')


@lru_cache(maxsize=1)
def original_schedule():
    graph=graph_for(128,'open'); agents=build_agents(graph,'provisioned')
    rows=[]; edges={}; initial={}
    for agent in agents:
        i=agent['id'];tour=agent['tour']
        edges[i]={(j,k):math.ceil((graph.path(graph.tasks[j].position_m,graph.tasks[k].position_m)[0]+
            graph.tasks[k].service_s)/5-1e-10)*5 for j in tour for k in tour}
        initial[i]={j:math.ceil((graph.path(agent['position'],graph.tasks[j].position_m)[0]+
            graph.tasks[j].service_s)/5-1e-10)*5 for j in tour}
        at=5;position=agent['position'];cursor=0
        while at<1800:
            j=tour[cursor]; task=graph.tasks[j]
            distance,route=graph.path(position,task.position_m)
            finish=at+math.ceil((distance+task.service_s)/5-1e-10)*5
            if finish>1800:break
            rows.append({'time_s':at,'finish_s':finish,'agent':i,'task':j,'source':list(position),
                'target':list(task.position_m),'distance_m':distance,'route':[list(v) for v in route],
                'service_s':task.service_s})
            position=task.position_m;at=finish;cursor=(cursor+1)%len(tour)
    return graph,agents,rows,edges,initial


def verify_record(record, arrays, seed):
    graph,agents,expected,_,_=original_schedule()
    actual={(v['agent'],v['time_s']):v for v in record['assignments']}
    need(len(actual)==len(record['assignments'])==len(expected),'Complete fixed Tour assignment count')
    for row in expected:
        need({k:v for k,v in row.items() if k!='finish_s'}==actual[row['agent'],row['time_s']], 'Original map/service/protected next task and exact dispatch')
    events={(v['agent'],v['time_s']):v for v in record['inspection_events']}
    need(len(events)==len(record['inspection_events'])==len(expected),'Every original assignment completes once')
    trace=SimpleNamespace(arrays=arrays,config=graph.config,seed=seed)
    for row in expected:
        event=events[row['agent'],row['finish_s']]; task=graph.tasks[row['task']]
        measurement=ProcessTrace.measurement(trace,task,int(row['finish_s']//5-1))
        need(event['task']==row['task'] and event['role']==agents[row['agent']]['role']==task.required_role and
            event['access']==task.access and event['calibrated'] and event['service_s']==task.service_s,
            'Independent actual completion/profession/access qualification')
        need(measurement==event['measurement'] and bool(measured_anomaly(task,measurement))==event['alarm'],
            'Every original professional measurement, deterministic0.1C noise and alarm')
    return len(expected)


class GeometryAudit:
    def __init__(self,record):
        self.graph,self.agents,self.expected,self.edges,self.initial=original_schedule()
        self.records=record;self.rows=[]
        self.by_agent={i:sorted((r for r in self.expected if r['agent']==i),key=lambda r:r['time_s']) for i in range(18)}

    def append(self,bundle):
        verify_fixed_task_metadata(bundle)
        now=bundle['time_s']; tasks=bundle['tasks']
        for agent in bundle['agents']:
            i=agent['agent'];expected=self.agents[i];tour=expected['tour']
            completed=[r for r in self.by_agent[i] if r['finish_s']<=now]
            busy=next((r for r in self.by_agent[i] if r['time_s']<now<r['finish_s']),None)
            cursor=len(completed)%len(tour)
            need(agent['tour']==tour and agent['cursor']==cursor and agent['detours_used']==0,
                'Public tour and cursor derived independently from original route')
            need((agent['busy_target'],agent['busy_finish_s'])==((busy['task'],busy['finish_s']) if busy else (None,None)), 'Known in-flight task and exact remaining finish')
            edge=self.edges[i]
            need(all(agent['costs_s'][str(j)][str(k)]==edge[j,k] for j in tour for k in tour) and
                agent['period_s']==sum(edge[j,k] for j,k in zip(tour[-1:]+tour[:-1],tour)), 'Public travel/service matrix and cycle from original map')
            if busy:continue
            costs={j:edge[completed[-1]['task'],j] if completed else self.initial[i][j] for j in tour}
            need(all(agent['first_costs_s'][str(j)]==costs[j] for j in tour),'Free-state travel/service from known completed position')
            if now+costs[tour[cursor]]>1800:continue
            last={j:max((r['finish_s'] for r in completed if r['task']==j),default=0.) for j in tour}
            need(all(max(0.,tasks[j]['last_check_s'])==last[j] for j in tour),'Public prior checks from exact original Tour')
            order=tour[cursor:]+tour[:cursor]
            arrival={};at=now;prev=None
            for j in order:
                at+=costs[j] if prev is None else edge[prev,j];arrival[j]=at;prev=j
            for j in tour:
                finish=now+costs[j]
                if j==tour[cursor] or finish>1800 or now-last[j]<60 and (last[j]>0 or tasks[j]['last_check_s']>=0):continue
                virtual=dict(last);slack=virtual[j]+tasks[j]['deadline_s']-finish
                if slack< -1e-9:continue
                virtual[j]=finish;at=finish;prev=j;valid=True
                for target in order:
                    at+=edge[prev,target];gap=virtual[target]+tasks[target]['deadline_s']-at
                    if gap< -1e-9:valid=False;break
                    slack=min(slack,gap);virtual[target]=at;prev=target
                if valid:self.rows.append((now,i,j,finish,arrival[j],float(slack)))

    def finish(self,geometry):
        fields=('dispatch_time_s','agent','task','finish_s','tour_target_finish_s','return_slack_s')
        need(len(self.rows)==len(geometry['task']),'Every hard-legal optional completion counted')
        for col,name in enumerate(fields):
            need(np.array_equal(np.asarray([v[col] for v in self.rows]),geometry[name]),
                'Independent whole-round return geometry: '+name)
        return len(self.rows)

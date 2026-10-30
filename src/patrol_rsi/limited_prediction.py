"""Closed public feature views and bounded read-only limited prediction data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .opportunity_data import ACTION_FEATURE_NAMES, FEATURE_NAMES

CELLS=(('thermal',30),('thermal',60),('thermal',120),('electrical',60),('electrical',120),('control',60),('control',120))
HEADS=('A_physical','B_action')
RAW_NAMES=tuple(FEATURE_NAMES)+tuple(ACTION_FEATURE_NAMES)
VIEW_NAMES=('time_lr','public_lr','public_raw_lr','public_assimilated_lr','public_all_lr')
SCALARS=('constant','professional_hold','bounded_trend','nominal_physical','frozen_CVAE_raw','frozen_CVAE_assimilated')
TIME_BASE=('root_time_s','pod','cdu_branch','power_branch','service_s','deadline_s','last_check_available','last_check_s','inspection_age_s')+tuple(
    f'qualified_history_{i}.{field}' for i in range(3) for field in ('available','time_s','age_s'))
TIME_NAMES=TIME_BASE+tuple(v for v in ACTION_FEATURE_NAMES if v!='bounded_trend_at_completion_c')
EXTRA_NAMES=tuple(f'clock_root_{v}' for v in range(30,1800,30))+tuple(f'public_pod_{v}' for v in range(16))
COL={name:i for i,name in enumerate(RAW_NAMES)}


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def load_reviewed(protocol_path,review_path):
    p=read(protocol_path);review=read(review_path)
    need(p['protocol']=='B4-B1b-Limited-v1' and tuple(map(tuple,p['cells']))==CELLS,'Exactly seven reviewed cells')
    need(review['status']=='approved_for_fixed_budget_limited_fitting' and review['protocol_SHA256']==sha(protocol_path),
        'Independent approved protocol SHA before fitting')
    need(p['candidates']['L2']==[.1,1.,10.] and p['candidates']['LR_fit_slots']==138 and p['candidates']['GRU_CVAE_PPO_training']==0,
        'Fixed fitting budget, no neural retraining')
    return p,review


def view_indices(view,kind):
    need(view in VIEW_NAMES,'Fixed model view')
    need(kind=='thermal' or view in ('time_lr','public_lr'),'CVAE views thermal only')
    if view=='time_lr':names=TIME_NAMES
    else:
        names=tuple(n for n in RAW_NAMES if not n.startswith(('cvae_raw.','history_assimilated.')) or
            view=='public_all_lr' or view=='public_raw_lr' and n.startswith('cvae_raw.') or
            view=='public_assimilated_lr' and n.startswith('history_assimilated.'))
    return np.asarray([COL[n] for n in names],dtype=int)


def expanded_features(raw,view,kind):
    """The only float64 feature expansion is this bounded batch."""
    need(raw.ndim==2 and raw.shape[1]==98 and np.isfinite(raw).all(),'Closed finite98 public columns')
    selected=view_indices(view,kind)
    root=((raw[:,COL['root_time_s']]-30)/30).astype(int)
    pod=raw[:,COL['pod']].astype(int)
    need(np.all((root>=0)&(root<59)) and np.all((pod>=0)&(pod<16)),'Original clock and public topology')
    result=np.zeros((len(raw),len(selected)+75),np.float64)
    result[:,:len(selected)]=raw[:,selected]
    result[np.arange(len(raw)),len(selected)+root]=1.
    result[np.arange(len(raw)),len(selected)+59+pod]=1.
    return result


def scalar_score(raw,family,kind):
    need(family in SCALARS and family!='constant','Fixed public scalar family')
    if kind!='thermal':
        need(family not in ('frozen_CVAE_raw','frozen_CVAE_assimilated'),'No fictitious nonthermal CVAE')
        if kind=='electrical':
            return np.maximum((raw[:,COL['contact.value']]-60)/30,0.)*raw[:,COL['contact.available']]
        need(kind=='control','Limited nonthermal professions')
        return (np.abs(raw[:,COL['sensor_error.value']])>1).astype(float)*raw[:,COL['sensor_error.available']]
    if family=='bounded_trend':
        value=raw[:,COL['bounded_trend_at_completion_c']];available=raw[:,COL['temperature_measurement.available']]
    else:
        channel={'professional_hold':'temperature_measurement','nominal_physical':'nominal',
            'frozen_CVAE_raw':'cvae_raw','frozen_CVAE_assimilated':'history_assimilated'}[family]
        value=raw[:,COL[channel+'.value']];available=raw[:,COL[channel+'.available']]
    return np.maximum((value.astype(float)-65)/20,0.)*available


class HeadData:
    """Read-only float32 mmap, one cohort shared by two distinct target heads."""
    def __init__(self,folder,*,expected_phase,expected_cell,expected_protocol):
        self.folder=Path(folder);self.info=read(self.folder/'complete.json');self.n=self.info['rows']
        need(self.info['phase']==expected_phase and self.info['cell_index']==expected_cell and
            self.info['protocol_SHA256']==expected_protocol and
            (self.info['kind'],self.info['horizon_s'])==CELLS[expected_cell],'Exact expected cache phase/protocol/cell')
        expected={'features.npy','weights.npy','labels.npy','evaluator-meta.npy'}
        need(set(self.info['files'])==expected,'Exactly four closed cache arrays')
        for name,row in self.info['files'].items():
            path=self.folder/name
            need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Cache array byte/SHA provenance: '+name)
        self.x=np.load(self.folder/'features.npy',mmap_mode='r')
        self.w=np.load(self.folder/'weights.npy',mmap_mode='r')
        self.y=np.load(self.folder/'labels.npy',mmap_mode='r')
        self.meta=np.load(self.folder/'evaluator-meta.npy',mmap_mode='r')
        capacity=self.info['capacity']
        need(type(self.n) is int and 0<self.n<=capacity and self.x.dtype==np.float32 and self.x.shape==(capacity,98)
            and self.w.dtype==np.float64 and self.w.shape==(capacity,) and self.y.dtype==np.int8 and self.y.shape==(capacity,2)
            and self.meta.dtype==np.int32 and self.meta.shape==(capacity,8),'Bounded exact float32 design and separate targets')
        need(np.isin(self.y[:self.n],(0,1)).all() and
            not np.any((self.y[:self.n,1]==1)&(self.y[:self.n,0]!=1)),'Binary separate labels and legal confirmation impliesphysical anomaly')
        need(np.isfinite(self.w[:self.n]).all() and (self.w[:self.n]>0).all() and
            abs(float(self.w[:self.n].sum())-1)<1e-9,'Positive normalized hierarchy weights')
        for start in range(0,self.n,16384):
            need(np.isfinite(self.x[start:min(self.n,start+16384)]).all(),'Finite public cache values')

    def batches(self,batch_rows=16384):
        need(1<=batch_rows<=16384,'Frozen maximum batch rows')
        for start in range(0,self.n,batch_rows):
            end=min(self.n,start+batch_rows)
            yield slice(start,end),self.x[start:end],self.w[start:end],self.y[start:end]


def weighted_scaler(data,view,kind):
    first=expanded_features(data.x[:1],view,kind);total=0.;s=np.zeros(first.shape[1]);ss=s.copy()
    for _,raw,w,_ in data.batches():
        x=expanded_features(raw,view,kind);w=np.asarray(w,dtype=np.float64)
        total+=float(w.sum());s+=np.einsum('i,ij->j',w,x);ss+=np.einsum('i,ij,ij->j',w,x,x)
    mean=s/total;scale=np.sqrt(np.maximum(ss/total-mean*mean,0.));scale[scale<1e-8]=1.
    return {'mean':mean.tolist(),'scale':scale.tolist(),'weight_sum':total,'development_only':True,
        'names':[RAW_NAMES[i] for i in view_indices(view,kind)]+list(EXTRA_NAMES)}

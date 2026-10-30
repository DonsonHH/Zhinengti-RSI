"""Bounded development calibration diagnosis, separate from frozen B1b."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .limited_model import FitBudgetExceeded,finite_number,optimizer,verify_termination
from .limited_prediction import need


def class_weights(y,w,balanced=False):
    y=np.asarray(y);w=np.asarray(w,float)
    need(np.isin(y,(0,1)).all() and np.isfinite(w).all() and (w>0).all(),'Legal weighted binary cohort')
    w=w/w.sum();prevalence=float(w@y)
    need(0<prevalence<1,'Both classes required for prespecified diagnosis')
    if balanced:w=w*np.where(y==1,.5/prevalence,.5/(1-prevalence))
    return w,prevalence


def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-700,700)))


def fit(scores,y,w,rho,balanced=False):
    need(rho in (0.,.0001),'Only two new slope ridge strengths')
    torch.set_num_threads(1);weights,prevalence=class_weights(y,w,balanced)
    score=torch.as_tensor(np.asarray(scores),dtype=torch.float64)
    target=torch.as_tensor(np.asarray(y,dtype=float),dtype=torch.float64)
    weight=torch.as_tensor(weights,dtype=torch.float64)
    parameter=torch.tensor([math.log(math.expm1(1.)),0.],dtype=torch.float64,requires_grad=True)
    opt=optimizer(parameter,200,300);count=0;last=None;termination=None;started=time.perf_counter()
    def closure():
        nonlocal count,last
        if count>=300:raise FitBudgetExceeded('B2 diagnostic300closure cap')
        count+=1;opt.zero_grad();a=F.softplus(parameter[0])
        value=(F.binary_cross_entropy_with_logits(a*score+parameter[1],target,reduction='none')*weight).sum()+rho*a*a/2
        need(torch.isfinite(value).item(),'Finite bounded calibration objective')
        value.backward();last=float(value.detach());return value
    try:
        opt.step(closure);closure();termination=verify_termination(opt.state[parameter],parameter,count,last,200,300)
        status='completed_within_fixed_optimizer_budget';error=None
    except (ValueError,RuntimeError) as exc:status='failed_consumes_diagnostic_slot';error=type(exc).__name__+': '+str(exc)
    a=finite_number(F.softplus(parameter[0]).detach());b=finite_number(parameter[1].detach())
    return {'status':status,'error':error,'a':a,'b':b,'rho':rho,'balanced':balanced,'original_calibration_prevalence':prevalence,
        'closure_evaluations':count,'termination':termination,'wall_s':time.perf_counter()-started,
        'prior_correction_logit':math.log(prevalence/(1-prevalence)) if balanced else 0.,'fit_phase':'validation_calibration'}


def weighted_auc(y,scores,w):
    y=np.asarray(y);s=np.asarray(scores);w=np.asarray(w,float);w=w/w.sum()
    order=np.argsort(s,kind='stable');s=s[order];y=y[order];w=w[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    pos=np.cumsum(w*y)[ends];neg=np.cumsum(w*(1-y))[ends]
    p=np.diff(np.r_[0.,pos]);n=np.diff(np.r_[0.,neg])
    return float(np.sum(p*(neg-n+.5*n))/(pos[-1]*neg[-1])) if pos[-1] and neg[-1] else None


def distribution(values,w):
    values=np.asarray(values,float);w=np.asarray(w,float)
    if not len(values):return {'n':0}
    w=w/w.sum();order=np.argsort(values,kind='stable');cdf=np.cumsum(w[order])
    mean=float(w@values)
    return {'n':len(values),'mean':mean,'std':float(np.sqrt(w@((values-mean)**2))),
        'minimum':float(values.min()),'maximum':float(values.max()),'range':float(np.ptp(values)),
        'weighted_quantiles_0_01_25_50_75_99_100':np.interp([0,.01,.25,.5,.75,.99,1],cdf,values[order]).tolist()}


def derivative(scores,y,w,a,b,rho,balanced=False):
    weights,_=class_weights(y,w,balanced);p=sigmoid(a*np.asarray(scores)+b)
    data=float(weights@((p-y)*scores));penalty=rho*a
    return {'BCE_gradient_wrt_a':data,'ridge_gradient_wrt_a':penalty,'sum_gradient_wrt_a':data+penalty,
        'gradient_wrt_b':float(weights@(p-y)),'logit_std_raw':distribution(scores,weights)['std'],
        'logit_std_after_calibration':a*distribution(scores,weights)['std'],
        'sigmoid_global_sensitivity_upper_bound':a/4}

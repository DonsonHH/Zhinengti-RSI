"""Fixed float64 LBFGS with one bounded feature batch per gradient step."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .limited_prediction import expanded_features,need,weighted_scaler


class FitBudgetExceeded(RuntimeError):pass


def optimizer(parameter,max_iter,max_eval):
    return torch.optim.LBFGS([parameter],lr=1.,history_size=10,max_iter=max_iter,max_eval=max_eval,
        tolerance_grad=1e-6,tolerance_change=1e-9,line_search_fn='strong_wolfe')


def finite_number(value):
    return float(value) if value is not None and np.isfinite(float(value)) else None


def verify_termination(state,parameter,evaluations,last,max_iter,max_eval):
    need(torch.isfinite(parameter).all().item(),'Nonfinite fitted parameter')
    gradient=finite_number(parameter.grad.detach().abs().max()) if parameter.grad is not None else None
    change=finite_number(abs(last-float(state['prev_loss']))) if last is not None and state.get('prev_loss') is not None else None
    step=finite_number((state['d']*state['t']).detach().abs().max()) if 'd' in state and 't' in state else None
    converged=(gradient is not None and gradient<=1e-6) or (change is not None and change<=1e-9) or (step is not None and step<=1e-9)
    reached=int(state.get('n_iter',0))>=max_iter or evaluations>=max_eval or int(state.get('func_evals',0))>=max_eval
    if reached and not converged:raise FitBudgetExceeded('Optimizer returned at fixedlimit withoutgradient/change termination')
    return {'gradient':gradient,'objective_change':change,'step_change':step,'budget_reached':reached,'termination_tolerance_satisfied':converged}


def fit_lr(data,view,kind,label_index,l2,scaler=None):
    torch.set_num_threads(1);scaler=scaler or weighted_scaler(data,view,kind)
    mean=np.asarray(scaler['mean']);scale=np.asarray(scaler['scale']);width=len(mean)
    parameter=torch.zeros(width+1,dtype=torch.float64,requires_grad=True)
    opt=optimizer(parameter,300,400);evaluations=0;started=time.perf_counter();last_value=None
    def closure():
        nonlocal evaluations,last_value
        if evaluations>=400:raise FitBudgetExceeded('Fixed400LRclosure budget exhausted')
        evaluations+=1;opt.zero_grad();total=0.
        for _,raw,w,y in data.batches():
            array=(expanded_features(raw,view,kind)-mean)/scale
            x=torch.as_tensor(array,dtype=torch.float64)
            weights=torch.as_tensor(np.array(w,copy=True),dtype=torch.float64)/scaler['weight_sum']
            target=torch.as_tensor(np.array(y[:,label_index],copy=True),dtype=torch.float64)
            value=(F.binary_cross_entropy_with_logits(x@parameter[:-1]+parameter[-1],target,reduction='none')*weights).sum()
            value.backward();total+=float(value.detach())
        regularizer=l2/2*parameter[:-1].square().sum();regularizer.backward();total+=float(regularizer.detach())
        need(np.isfinite(total) and torch.isfinite(parameter.grad).all().item(),'Finite fixed-budget LR objective/gradient')
        last_value=total
        return torch.tensor(total,dtype=torch.float64)
    termination=None
    try:
        opt.step(closure)
        # Strong-Wolfe may leave .grad from another trial point. Recompute at
        # the actual final parameter, charging it to the same400call budget.
        closure();termination=verify_termination(opt.state[parameter],parameter,evaluations,last_value,300,400)
        status='completed_within_fixed_optimizer_budget';error=None
    except (FitBudgetExceeded,ValueError,RuntimeError) as exc:
        status='failed_consumes_candidate_slot';error=type(exc).__name__+': '+str(exc)
    need(evaluations<=400,'No closure budget overrun')
    state=opt.state[parameter]
    gradient=finite_number(parameter.grad.detach().abs().max()) if parameter.grad is not None else None
    return {'status':status,'error':error,'view':view,'kind':kind,'label_index':label_index,'L2':l2,
        'coefficient':parameter.detach().numpy().tolist() if status=='completed_within_fixed_optimizer_budget' else None,'scaler':scaler,
        'closure_evaluations':evaluations,'optimizer_iterations':int(state.get('n_iter',0)),
        'objective':finite_number(last_value),'max_absolute_gradient':gradient,'termination':termination,'wall_s':time.perf_counter()-started,
        'max_feature_batch_rows':16384,'no_full_float64_design_matrix':True,
        'termination_note':'LBFGS fixed gradient/change tolerances and caps;finite early termination is recorded,not a claim of proven global optimum.'}


def lr_scores(data,model):
    need(model['status']=='completed_within_fixed_optimizer_budget','Failed fit cannot be used')
    result=np.empty(data.n,dtype=np.float64)
    coef=np.asarray(model['coefficient']);mean=np.asarray(model['scaler']['mean']);scale=np.asarray(model['scaler']['scale'])
    for sl,raw,_,_ in data.batches():
        x=(expanded_features(raw,model['view'],model['kind'])-mean)/scale
        result[sl]=x@coef[:-1]+coef[-1]
    need(np.isfinite(result).all(),'Finite model scores')
    return result


def fit_calibrator(scores,y,weights):
    """Only three one-dimensional vectors, never a large feature matrix."""
    torch.set_num_threads(1)
    score=torch.as_tensor(np.asarray(scores),dtype=torch.float64)
    target=torch.as_tensor(np.asarray(y,dtype=np.float64),dtype=torch.float64)
    weight=torch.as_tensor(np.array(weights,copy=True),dtype=torch.float64);weight=weight/weight.sum()
    parameter=torch.tensor([math.log(math.expm1(1.)),0.],dtype=torch.float64,requires_grad=True)
    opt=optimizer(parameter,200,300);evaluations=0;started=time.perf_counter();last=None
    def closure():
        nonlocal evaluations,last
        if evaluations>=300:raise FitBudgetExceeded('Fixed300calibration closure budget exhausted')
        evaluations+=1;opt.zero_grad();a=F.softplus(parameter[0])
        value=(F.binary_cross_entropy_with_logits(a*score+parameter[1],target,reduction='none')*weight).sum()+a*a/2
        need(torch.isfinite(value).item(),'Finite monotone calibration')
        value.backward();last=float(value.detach());return value
    termination=None
    try:
        opt.step(closure);closure();termination=verify_termination(opt.state[parameter],parameter,evaluations,last,200,300)
        status='completed_within_fixed_optimizer_budget';error=None
    except (FitBudgetExceeded,ValueError,RuntimeError) as exc:status='failed_consumes_candidate_slot';error=type(exc).__name__+': '+str(exc)
    a=finite_number(F.softplus(parameter[0]).detach());b=finite_number(parameter[1].detach())
    return {'status':status,'error':error,'a':a,'b':b,'monotone':a is not None and a>=0,
        'closure_evaluations':evaluations,'optimizer_iterations':int(opt.state[parameter].get('n_iter',0)),
        'objective':finite_number(last),'termination':termination,'wall_s':time.perf_counter()-started,'fit_split':'validation_calibration'}


def calibrated(scores,calibrator):
    need(calibrator['status']=='completed_within_fixed_optimizer_budget' and calibrator['a']>=0,'Only valid fixed monotone calibration')
    z=np.clip(calibrator['a']*scores+calibrator['b'],-700,700)
    return 1/(1+np.exp(-z))

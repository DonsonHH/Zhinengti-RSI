"""Offline selection prescribed before pilot; never called by online policy."""
from __future__ import annotations


def select_validation(candidates,tour):
    """Inputs are per-seed validation aggregates, never final-test metrics."""
    if set(tour)!={'8600','8601'}:raise ValueError('Only frozen validation seed aggregates accepted')
    eligible=[]
    for identity,seeds in candidates.items():
        if set(seeds)!=set(tour) or identity not in ('C0','C1','C2','C3'):
            raise ValueError('Unexpected candidate or split')
        if any(seeds[s]['violations'] or seeds[s]['missed']>tour[s]['missed'] for s in tour):continue
        if sum(seeds[s]['weighted_delay_sum'] for s in tour)>sum(tour[s]['weighted_delay_sum'] for s in tour)+1e-10:continue
        misses=sum(seeds[s]['missed'] for s in tour);delays=sum(seeds[s]['weighted_delay_sum'] for s in tour)
        if misses==sum(tour[s]['missed'] for s in tour) and abs(delays-sum(tour[s]['weighted_delay_sum'] for s in tour))<=1e-10:continue
        eligible.append((misses,delays,sum(seeds[s]['travel_m'] for s in tour),identity))
    return min(eligible)[-1] if eligible else 'TOUR'

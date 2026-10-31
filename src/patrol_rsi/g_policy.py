"""Explicit policy change for the human ranking bridge, not an actor-tool alias."""
from .improver_task import PatrolProgram,TypedRule


def bridge_program():
    return PatrolProgram(rules=(TypedRule('thermal_excess',.05,'urgency'),TypedRule('age',.75,'return_tour')))

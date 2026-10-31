import unittest
from types import SimpleNamespace
import numpy as np
from patrol_rsi.improver_task import choose_program,PatrolProgram,TypedRule
from patrol_rsi.g_policy import bridge_program


class BridgeTests(unittest.TestCase):
    def test_actual_signal_to_urgency_change_preserves_baseline_participation(self):
        def env(program):
            f=np.zeros((2,16),dtype=np.float32);f[:,0]=(.1,.9);f[:,1]=.1;f[:,6]=.5
            return SimpleNamespace(workflow=program,agents=[{'task':None}],tick=0,clock=100.,reserved=lambda:set(),
                options=lambda i,res:(np.array([0,1]),f,None),plans=[SimpleNamespace(tour=[0,1])],protected_cursor=[0],
                tasks=[SimpleNamespace(kind='thermal',index=j) for j in range(2)],mode='history',current_history=np.array([85.,67.]),
                state=SimpleNamespace(observer=SimpleNamespace()),detours_used=[0],baseline_completed=[0])
        original=PatrolProgram(rules=(TypedRule('thermal_excess',.05,'signal'),TypedRule('age',.75,'return_tour')))
        a,_=choose_program(env(original),lambda f:np.zeros(len(f)),actor_tool='urgency')
        b,_=choose_program(env(bridge_program()),lambda f:np.zeros(len(f)),actor_tool='urgency')
        self.assertEqual(a,{0:0});self.assertEqual(b,{0:1})
        self.assertEqual(bridge_program().to_dict()['rules'][1],original.to_dict()['rules'][1])


if __name__=='__main__':unittest.main()

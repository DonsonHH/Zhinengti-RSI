import unittest
import numpy as np
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.workflow import Workflow, WorkflowExecutor, dispatch, human_workflow


class WorkflowTests(unittest.TestCase):
    def test_code_and_unknown_tools_cannot_execute(self):
        for steps in [[{'op':'python','code':'print(1)'}], [{'op':'fallback','policy':'random'}]]:
            with self.assertRaises(ValueError):Workflow.from_dict({'steps':steps})
        with self.assertRaises(ValueError):Workflow.from_dict({'steps':human_workflow().to_dict()['steps'],'evaluator':'replace'})

    def test_program_order_changes_assignment(self):
        env=PatrolEnv(room(10000));goals=list(map(tuple,np.argwhere(env.scenario.targets)))
        env.ever_measured[env.scenario.targets]=True
        env.ever_measured[goals[0]]=False
        env.raw_age[goals[1]]=40  # Two goals compete for the same nearest executor.
        due=dispatch('overdue','oldest');unseen=dispatch('unseen','oldest');fallback={'op':'fallback','policy':'ddqn'}
        a=WorkflowExecutor(env,10000,np.full(8,24),Workflow.from_dict({'steps':[due,unseen,fallback]}))
        b=WorkflowExecutor(env,10000,np.full(8,24),Workflow.from_dict({'steps':[unseen,due,fallback]}))
        a.scores(env,np.zeros((4,8)));b.scores(env,np.zeros((4,8)))
        self.assertNotEqual(a.assigned,b.assigned)

    def test_stationary_steps_count_as_stalled_history(self):
        env=PatrolEnv(room(10003));x=WorkflowExecutor(env,10003,np.full(8,24),human_workflow())
        for _ in range(4):
            env.step(np.full(4,-1));x.stalled(env)
        self.assertEqual(x.stalled(env),set(range(4)))

    def test_workflow_inference_never_reads_hidden_field(self):
        s=room(10001);a=PatrolEnv(s,truth=np.zeros((24,24)));b=PatrolEnv(s,truth=np.ones((24,24)))
        a.estimate[:]=.5;b.estimate[:]=.5
        p=human_workflow()
        one=WorkflowExecutor(a,10001,np.full(8,24),p);two=WorkflowExecutor(b,10001,np.full(8,24),p)
        np.testing.assert_array_equal(one.scores(a,np.zeros((4,8))),two.scores(b,np.zeros((4,8))))

    def test_human_program_reaches_targets_and_executes_branches(self):
        env=PatrolEnv(room(10002));x=WorkflowExecutor(env,10002,np.full(8,24),human_workflow())
        for _ in range(96):env.step(env.safe_actions(x.scores(env,np.zeros((4,8)))))
        self.assertEqual(env.evaluation()['target_coverage'],1)
        self.assertGreater(sum(x.executed_rules),0)
        self.assertGreater(x.interventions,0)

    def test_workflow_snapshot_is_immutable(self):
        d=human_workflow().to_dict();p=Workflow.from_dict(d);before=p.version_id
        d['steps'][1]['when']='always'
        self.assertEqual(p.version_id,before)
        self.assertNotEqual(p.to_dict(),d)

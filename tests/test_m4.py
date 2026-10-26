import json
from dataclasses import asdict
import unittest
from patrol_rsi.m4 import (ImproverProtocol, contract, deadlines, meta_accept,
                           model_message, rank, select_memory, snapshot, workflow_prompt)
from patrol_rsi.workflow import human_workflow


def metrics(coverage=1.0,violation=.1):
    return {'target_coverage':coverage,'mean_deadline_violation':violation,
            'weighted_idleness':.3,'intervention_fraction':.4}


class M4Tests(unittest.TestCase):
    def test_model_message_separates_input_evidence_from_output_request(self):
        message=model_message({'task':'synthesize_patrol_workflow','metrics':{'x':1}})
        self.assertIn('BEGIN READ-ONLY INPUT EVIDENCE',message)
        self.assertIn('exactly diagnosis',message)
        self.assertTrue(message.endswith('Do not reproduce the input evidence.'))
        with self.assertRaises(ValueError):model_message({'task':'run_python'})

    def test_protocol_update_requires_better_inner_outcome(self):
        before={'metrics':metrics()}
        self.assertFalse(meta_accept(before,{'metrics':metrics(violation=.2)}))
        self.assertFalse(meta_accept(before,before))
        self.assertTrue(meta_accept(before,{'metrics':metrics(violation=.05)}))

    def test_coverage_cannot_be_traded_for_smaller_violation(self):
        self.assertLess(rank(metrics()),rank(metrics(coverage=.9,violation=0)))

    def test_protocol_rejects_eval_edits_and_unknown_tools(self):
        for d in [{'diagnostics':['oracle'],'memory':'recent','procedure':'x'},
                  dict(asdict(ImproverProtocol()))|{'evaluator':'change'}]:
            with self.assertRaises(ValueError):ImproverProtocol.from_dict(d)

    def test_actual_tools_and_memory_change_workflow_prompt(self):
        views=[{'layout_seed':10000,'profile':'balanced','coverage':{'view':'coverage'},'deadlines':{'view':'ages'}}]
        history=[{'generation':0,'accepted':False,'error':'bad'}]
        a=workflow_prompt(human_workflow(),metrics(),views,ImproverProtocol(('coverage',),'none','x'),history,1)
        b=workflow_prompt(human_workflow(),metrics(),views,ImproverProtocol(('deadlines',),'recent','x'),history,1)
        self.assertIn('coverage',a['tool_results'][0]);self.assertNotIn('deadlines',a['tool_results'][0])
        self.assertIn('deadlines',b['tool_results'][0]);self.assertEqual(a['memory'],[]);self.assertEqual(len(b['memory']),1)

    def test_model_request_snapshot_cannot_gain_future_meta_feedback(self):
        history=[];user={'previous_meta_feedback':history};saved=snapshot(user)
        history.append({'round':0,'candidate':'future'})
        self.assertEqual(saved['previous_meta_feedback'],[])

    def test_public_deadlines_vary_and_noise_conditions_pair(self):
        c=contract()
        for a,b in zip(c['test'][:12],c['test'][12:]):
            self.assertEqual(a['layout_seed'],b['layout_seed']);self.assertEqual(deadlines(a).tolist(),deadlines(b).tolist())
            self.assertNotEqual(a['noise'],b['noise'])
        self.assertGreater(len(set(deadlines(c['test'][0]))),1)
        all_meta={r['layout_seed'] for fold in c['meta_development'] for r in fold}
        dev={r['layout_seed'] for r in c['development']};test={r['layout_seed'] for r in c['test']}
        self.assertFalse(all_meta&dev or all_meta&test or dev&test)

    def test_memory_preserves_best_and_last_failure(self):
        history=[{'generation':0,'accepted':True,'metrics':metrics(violation=.01)},
                 {'generation':1,'accepted':False,'metrics':metrics(violation=.5)},
                 {'generation':2,'accepted':True,'metrics':metrics(violation=.05)}]
        selected=select_memory(history,'best_and_failures')
        self.assertEqual([r['generation'] for r in selected],[0,1])

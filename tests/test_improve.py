import unittest
from patrol_rsi.improve import parse_proposal,prompt_snapshot
from patrol_rsi.versions import AgentVersion


class ProposalTests(unittest.TestCase):
    def test_saved_prompt_cannot_acquire_future_feedback(self):
        history=[{'generation':0,'accepted':True,'parsed':{'patch':{'estimator_samples':4}}}]
        saved=prompt_snapshot({'history':history})
        history[0]['parsed']['patch']['estimator_samples']=8
        history.append({'generation':1,'accepted':False})
        self.assertEqual(len(saved['history']),1)
        self.assertEqual(saved['history'][0]['parsed']['patch']['estimator_samples'],4)

    def test_real_output_parser_rejects_evaluator_change(self):
        with self.assertRaises(ValueError):parse_proposal('{"patch":{"evaluator":"always accept"}}',AgentVersion(),False)

    def test_meta_instruction_changes_only_in_updated_group(self):
        text='{"patch":{"estimator_samples":2},"next_diagnostic_instruction":"Analyze redundant sensing before changing coordination."}'
        fixed,_=parse_proposal(text,AgentVersion(),False)
        updated,_=parse_proposal(text,AgentVersion(),True)
        self.assertEqual(fixed.diagnostic_instruction,AgentVersion().diagnostic_instruction)
        self.assertNotEqual(updated.diagnostic_instruction,AgentVersion().diagnostic_instruction)
        self.assertEqual(updated.estimator_samples,2)


if __name__=='__main__':unittest.main()

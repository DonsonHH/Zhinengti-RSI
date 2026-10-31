"""Public local-edit contract for TimingSkill; no model or simulator execution."""
import copy
import unittest

from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.g_timing_patch import patch_view,compile_patch


class TimingPatchTests(unittest.TestCase):
    def test_threshold_edit_preserves_parent_and_rejects_stale_parent(self):
        parent=TimingSkill.initial_e1();before=parent.to_dict();view=patch_view(parent)
        patch={'schema_version':'timing-patch-v1','parent_id':view['parent_id'],'edits':[
            {'op':'replace_threshold','target':'r0.p4','expected_hash':view['predicates']['r0.p4']['hash'],'value':.01}]}
        child=compile_patch(parent,patch)
        self.assertEqual(child.to_dict()['rules'][0]['all'][4]['value'],.01)
        self.assertEqual(parent.to_dict(),before)
        invalid=copy.deepcopy(patch);invalid['parent_id']='0'*64
        with self.assertRaises(ValueError):compile_patch(parent,invalid)

    def test_deleted_predicate_does_not_shift_second_target_and_failure_is_atomic(self):
        parent=TimingSkill.initial_e1();view=patch_view(parent)
        patch={'schema_version':'timing-patch-v1','parent_id':view['parent_id'],'edits':[
            {'op':'delete_predicate','target':'r0.p0','expected_hash':view['predicates']['r0.p0']['hash']},
            {'op':'replace_threshold','target':'r0.p1','expected_hash':view['predicates']['r0.p1']['hash'],'value':.1}]}
        self.assertEqual(compile_patch(parent,patch).to_dict()['rules'][0]['all'][0],{'field':'professional_signal','op':'ge','value':.1})
        invalid=copy.deepcopy(patch);invalid['edits'][1]['target']='r0.p0';invalid['edits'][1]['expected_hash']=view['predicates']['r0.p0']['hash']
        with self.assertRaises(ValueError):compile_patch(parent,invalid)
        self.assertEqual(parent,TimingSkill.initial_e1())

    def test_independent_missingness_branch_and_illegal_edits(self):
        parent=TimingSkill.initial_e1();v=patch_view(parent)
        rule={'all':[{'field':'direct_age_s','op':'is_missing','value':None},{'field':'lead_s','op':'ge','value':60}]}
        patch={'schema_version':'timing-patch-v1','parent_id':v['parent_id'],'edits':[{'op':'add_rule','rule':rule}]}
        public={'kind':'electrical','professional_available':False,'professional_signal':None,'direct_available':False,'direct_age_s':None,'lead_s':100.}
        self.assertEqual(parent.action(public),'CONTINUE_TOUR')
        self.assertEqual(compile_patch(parent,patch).action(public),'CHECK')
        bad=copy.deepcopy(patch);bad['edits'][0]['rule']['all']*=4
        with self.assertRaises(ValueError):compile_patch(parent,bad)
        boolean={'schema_version':'timing-patch-v1','parent_id':v['parent_id'],'edits':[{'op':'replace_threshold','target':'r0.p0','expected_hash':v['predicates']['r0.p0']['hash'],'value':False}]}
        with self.assertRaises(ValueError):compile_patch(parent,boolean)


if __name__=='__main__':unittest.main()

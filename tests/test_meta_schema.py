import json
import unittest
from patrol_rsi.meta_schema import (strict_json,compile_compact,parse_proposal,prompt,format_parents,reset_receipt,changed_fields)
from patrol_rsi.rsi_controlled import ControlledImprover


class MetaSchemaTests(unittest.TestCase):
    def leaf(self):return {'diagnostic_ids':[0,1],'memory_id':2,'instruction_id':1,'generator':'qwen_once','dedup':'exact_version','acceptance':'strict_improvement','failure_fallback':'keep_parent'}
    def test_full_nested_example_is_legal(self):
        v=prompt('full_nested',ControlledImprover(),0)['complete_valid_example'];pg,_=parse_proposal(json.dumps(v),'full_nested')
        self.assertNotEqual(pg,ControlledImprover())
    def test_compact_example_compiles_unchanged_DSL(self):
        v=prompt('compact_leaves',ControlledImprover(),0)['complete_valid_example'];pg,_=parse_proposal(json.dumps(v),'compact_leaves')
        self.assertEqual(pg,ControlledImprover.from_dict(pg.to_dict()))
    def test_six_old_flattened_shapes_rejected(self):
        v=prompt('full_nested',ControlledImprover(),0)['complete_valid_example'];p=v['protocol'];v['protocol']={**p['protocol'],**{k:x for k,x in p.items() if k!='protocol'}}
        with self.assertRaises(ValueError):parse_proposal(json.dumps(v),'full_nested')
    def test_duplicate_JSON_keys_rejected(self):
        with self.assertRaises(ValueError):strict_json('{"protocol":{},"protocol":{}}')
    def test_all_nonfinite_forms_rejected(self):
        for v in ('NaN','Infinity','-Infinity','1e309','-1e309'):
            with self.subTest(v=v),self.assertRaises(ValueError):strict_json('{"x":'+v+'}')
    def test_markdown_and_trailing_prose_rejected(self):
        for v in ('prefix{}','{} suffix'):
            with self.assertRaises(ValueError):strict_json(v)
    def test_unknown_seed_field_rejected(self):
        v=self.leaf();v['seed']=33000
        with self.assertRaises(ValueError):compile_compact(v)
    def test_diagnostic_count_and_duplicates_rejected(self):
        for ids in ([],[0,0],[0,1,2,3]):
            v=self.leaf();v['diagnostic_ids']=ids
            with self.assertRaises(ValueError):compile_compact(v)
    def test_ids_strict_integer_literals(self):
        for value in (True,1.0,-1,5):
            v=self.leaf();v['diagnostic_ids']=[value]
            with self.assertRaises(ValueError):compile_compact(v)
    def test_all_enum_bounds_rejected(self):
        for key,value in (('memory_id',True),('instruction_id',3),('generator','other_model'),('dedup','reorder_rules'),('acceptance','accept_any'),('failure_fallback','retry')):
            v=self.leaf();v[key]=value
            with self.assertRaises(ValueError):compile_compact(v)
    def test_instruction_cannot_hide_task_identity(self):
        v=ControlledImprover().to_dict();v['protocol']['instruction']='wait until seed33000 fault'
        with self.assertRaises(ValueError):ControlledImprover.from_dict(v)
    def test_only_protocol_transfers(self):
        for pg in format_parents():
            reset=reset_receipt(pg);self.assertTrue(reset['only_protocol_transferred']);self.assertEqual(reset['history_length'],0)
    def test_changed_fields_within_allowed_meta_space(self):
        keys=changed_fields(ControlledImprover(),compile_compact(self.leaf()))
        self.assertTrue(keys);self.assertTrue(set(keys)<={'protocol.diagnostics','protocol.memory','protocol.instruction','generator','dedup','acceptance','failure_fallback'})


if __name__=='__main__':unittest.main()

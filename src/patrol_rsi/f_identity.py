"""Immutable F-study source/model/protocol and substage boundary."""
from pathlib import Path
from .limited_prediction import read,sha,need

BASE='39d06ce5edb4dc7f7fff18bedc76909350a946d4'


def verify_identity(path,required,scope):
    identity=read(path);root=Path(identity['source_root']);manifest=read(Path(identity['bound_inputs']['source_manifest']))
    need(identity['baseline']==BASE and root.resolve()==Path(__file__).resolve().parents[2],'Actual F imported source/baseline')
    indexed={v['path']:v['sha256'] for v in identity['artifacts']}
    need(len(indexed)==len(identity['artifacts']) and set(identity['bound_inputs'].values())<=set(indexed),'Unique complete frozen inputs')
    mandatory={'study','cases','source_manifest','source_archive','review','gate','actor_manifest','CVAE_manifest','actor0','actor1','actor2','CVAE'}
    need(mandatory<=set(identity['bound_inputs']),'Complete F runtime identity')
    for key,value in required.items():need(str(Path(value).resolve())==identity['bound_inputs'][key],'Actual F arguments/weights/input paths bound')
    for name in ('src','scripts','tests'):
        for file in (root/name).rglob('*.py'):
            relative=file.relative_to(root).as_posix()
            need(str(file.resolve()) in indexed and relative in manifest['files'] and sha(file)==manifest['files'][relative]['sha256'],'Every executed source/test equals reviewed archive')
    for file,digest in indexed.items():need(sha(Path(file))==digest,'F source/input/model bytes unchanged')
    review=read(Path(identity['bound_inputs']['review']));gate=read(Path(identity['bound_inputs']['gate']))
    need(review['approved'] and scope in review['approved_scopes'],'Independent review authorizes this F substage')
    need(review['source_archive_SHA256']==manifest['sha256']==sha(Path(identity['bound_inputs']['source_archive'])),'Actual reviewed archive bytes')
    need({'study','cases','source_manifest'}<=set(review['reviewed_input_SHA256']),'Review binds actual study, cases and archive manifest')
    for name,digest in review['reviewed_input_SHA256'].items():need(sha(Path(identity['bound_inputs'][name]))==digest,'Reviewed study/case/input matches actual runtime')
    needed={'tests/test_timing_skill.py','tests/test_timing_semantics.py','tests/test_timing_skill_features.py'}
    if scope!='F0':needed.add('tests/test_timing_learning.py')
    minimum=329 if scope=='F0' else 341
    need(gate['success'] and not(gate['errors'] or gate['failures'] or gate['skipped']) and gate['tests']>=minimum,'Real fullTorch plus F tests execution gate')
    need(gate['source_root']==str(root) and set(gate['F_test_sources'])==needed and len(set(gate['F_test_ids']))>=minimum-317,'Actual F test modules executed at reviewed source')
    for name,digest in gate['F_test_sources'].items():need(sha(root/name)==digest,'Actual F test sources match gate')
    if scope!='F0':
        need({'F0_complete','F0_manifest','F0_audit'}<=set(identity['bound_inputs']),'Actual F0 source and parity evidence bound')
        proof=read(Path(identity['bound_inputs']['F0_complete']));old=read(Path(identity['bound_inputs']['F0_manifest']))
        need(proof['complete'] and proof['predicate_mismatches']==0 and proof['initial_E1_candidate_parity']==298152 and proof['charged_known_closed_loop_episodes']==20 and proof['all_action_hashes_and_metrics_equal'],'Actual complete F0 parity before search')
        audit=read(Path(identity['bound_inputs']['F0_audit']))
        need(audit['passed'] and audit['complete_receipt_SHA256']==sha(Path(identity['bound_inputs']['F0_complete'])) and audit['source_archive_SHA256']==old['sha256'],'Independent F0 audit joins exact completed source and receipt')
        core=('src/patrol_rsi/timing_skill.py','src/patrol_rsi/timing_skill_features.py','src/patrol_rsi/timing_skill_control.py','scripts/timing_skill_execution.py')
        need(all(manifest['files'][v]['sha256']==old['files'][v]['sha256'] for v in core),'F0-validated timing interpreter unchanged in actual search')
    return identity

"""M12-E immutable execution boundary, separate from all historical protocols."""
from pathlib import Path
from .limited_prediction import read,sha,need

BASE='6174f2ab3ae227b8bec6f83b05fdca512bb84192'


def verify_stage_identity(path,required,*,scope):
    value=read(path);root=Path(value['source_root'])
    need(value['baseline']==BASE and root.resolve()==Path(__file__).resolve().parents[2],'Actual imported E source tree and baseline')
    entries={v['path']:v['sha256'] for v in value['artifacts']}
    need(len(entries)==len(value['artifacts']),'Unique immutable artifact paths')
    need(set(value['bound_inputs'].values())<=set(entries),'All required E inputs hashed')
    for name,p in required.items():need(str(Path(p).resolve())==value['bound_inputs'][name],'Actual protocol, review, model and split arguments are bound')
    for group in ('src','scripts','tests'):
        files=list((root/group).rglob('*.py'));need(files and all(str(p.resolve()) in entries for p in files),'Every imported source and test file bound')
    for file,digest in entries.items():need(sha(Path(file))==digest,'E source/input/model bytes unchanged')
    review=read(Path(value['bound_inputs']['review']));gate=read(Path(value['bound_inputs']['gate']))
    need(review['approved_before_real_execution'] and gate['success'] and not(gate['failures'] or gate['errors'] or gate['skipped']),'Independent review and real Torch execution gate')
    need(scope in review['approved_scopes'],'Independent approval covers the actual E substage')
    need(gate['tests']>=317 and len(set(gate.get('E_test_ids',[])))==24,'All293historical plus13schema and11timing tests executed')
    expected={'tests/test_meta_schema.py','tests/test_patrol_timing.py'}
    need(gate.get('source_root')==str(root) and set(gate.get('E_test_sources',{}))==expected,'Actual E tests required; historical293-only gate cannot pass')
    for name,digest in gate['E_test_sources'].items():need(sha(root/name)==digest,'Current reviewed E test source executed in gate')
    need(all(any(name in v for v in gate.get('E_test_ids',[])) for name in ('test_meta_schema','test_patrol_timing')),'Both E test modules executed')
    manifest=read(Path(value['bound_inputs']['source_manifest']))
    need(review['source_archive_SHA256']==manifest['sha256'],'Review bound to actual executed source archive')
    need(sha(Path(value['bound_inputs']['source_archive']))==manifest['sha256'],'Actual reviewed source archive bytes')
    for group in ('src','scripts','tests'):
        for file in (root/group).rglob('*.py'):
            name=file.relative_to(root).as_posix()
            need(name in manifest['files'] and sha(file)==manifest['files'][name]['sha256'],'Executed source/test content equals reviewed archive member')
    need(({'E0','E0_prompts'} if scope=='E0' else {'E1','manifest','history'})<=set(review['reviewed_input_SHA256']),'Actual substage critical inputs explicitly reviewed')
    for name,digest in review['reviewed_input_SHA256'].items():need(sha(Path(value['bound_inputs'][name]))==digest,'Reviewed protocol/prompt/manifest equals actual runtime input')
    return value

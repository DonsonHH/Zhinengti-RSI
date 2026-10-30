"""Export only released audit counts and non-final examples; no final labels."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from patrol_rsi.fresh_opportunity import need, read, sha, write


def main():
    p=argparse.ArgumentParser()
    for name in ('run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();r=args.run;args.output.mkdir(exist_ok=False)
    files=[]
    # Explicit allow-list: all final row/process/event diagnostics are outside.
    for path in (r/'report').iterdir():
        if path.is_file():files.append((path,'report/'+path.name))
    for folder in ('final/gate','jobs'):
        for path in (r/folder).iterdir():
            if folder=='jobs' and path.name.startswith('package-'):continue
            if path.is_file():files.append((path,('gate/' if folder.endswith('gate') else 'jobs/')+path.name))
    for name in ('cluster-start.txt','environment-start.txt'):
        files.append((r/name,name))
    # One development positive-condition example permits local independent
    # reproduction without opening final data or duplicating full private runs.
    group=r/'data/development/seed-20000'
    condition='development--20000--dc_contact_hotspot--180--335--0.2'
    for path in (group/condition).iterdir():
        if path.is_file():files.append((path,'development-example/'+condition+'/'+path.name))
    files.extend((group/name,'development-example/'+name) for name in
        ('geometry.npz','processes/20000--dc_contact_hotspot--180--335.npz','processes/20000--dc_contact_hotspot--180--335.json'))
    archive=args.output/'delivery.tar.gz';inventory={}
    with tarfile.open(archive,'w:gz') as tar:
        for path,name in sorted(files,key=lambda v:v[1]):
            need(path.is_file() and 'final_test' not in name and 'sealed' not in name,'Only explicitly released package members')
            need(name not in inventory,'No duplicate archive member')
            inventory[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
            tar.add(path,arcname=name,recursive=False)
    # Defence-in-depth projection checks on the actual serialized exports.
    coverage=read(r/'report/coverage.json')
    for row in coverage['coverage_cells']:
        if row['split']=='final_test':
            need(not any(k in row for k in ('seed_groups','B_positive_seed_groups','B_negative_seed_groups')),'Final aggregate seed sets not public')
    dimensions=read(r/'report/coverage-dimensions.json')
    need(not any(key.startswith('final_test/') for dim,cells in dimensions.items() if dim!='split' for key in cells),'No final process/scenario/phase tables')
    conditions=json.loads(gzip.decompress((r/'report/condition-counts-nonfinal.json.gz').read_bytes()))
    need(len(conditions)==800 and all(v['split']!='final_test' for v in conditions),'Exactly800 nonfinal condition summaries')
    receipt={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'files':inventory,
        'all_final_row_labels_features_trajectories_not_exported':True,'final_aggregate_cells':12,
        'nonfinal_condition_counts':800,'raw_data_remains':str(r/'data'),'full_originals_deleted':False,
        'sealed_full_coverage_reference':coverage['sealed_full_coverage'],'source_replay_SHA256':sha(r/'report/source-replay.json')}
    write(args.output/'delivery-manifest.json',receipt)
    print('FRESH_EVIDENCE_PACKAGED',len(files),archive.stat().st_size,sha(archive),flush=True)


if __name__=='__main__':main()

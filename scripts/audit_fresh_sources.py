"""Replay original plant equations, bind every source byte and final seal."""
from __future__ import annotations

import argparse
from pathlib import Path
import stat

import numpy as np

from patrol_rsi.fresh_opportunity import WEIGHTS_SHA, frozen_inputs, need, read, sha, write
from patrol_rsi.improver_task import PhasedTrace


def main():
    p=argparse.ArgumentParser()
    for n in ('root','protocol','manifest','storage','weights','output'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();protocol,manifest=frozen_inputs(args.protocol,args.manifest)
    need(sha(args.weights)==WEIGHTS_SHA,'Original frozen CVAE checkpoint')
    storage=read(args.storage); size=0
    for file in storage['files']:
        path=Path(file['path']);need(path.is_file(),'All recorded sources still present')
        need(path.stat().st_size==file['bytes'] and sha(path)==file['sha256'],'Every raw source byte matches released inventory')
        size+=file['bytes']
        if file['final_test_sealed']:
            need(stat.S_IMODE(path.stat().st_mode)&0o077==0,'Final raw file permissions sealed')
    need(size==storage['bytes'] and size<=6*1000**3,'Unique storage accounting cap')
    seen=set();replayed=values=0
    for item in manifest:
        key=item['physical_key']
        if key in seen:continue
        seen.add(key);spec=item['specification']
        group=args.root/item['split']/f"seed-{item['group_seed']}"
        path=group/'processes'/(key+'.npz')
        # Fresh original physical simulation; no observer, changed parameters,
        # noise, trained predictor, summary label or condition metadata as truth.
        fresh=PhasedTrace(spec['seed'],spec['scenario'],spec['duration_s'],spec['fault_start_s'])
        with np.load(path) as arrays:
            need(set(arrays.files)==set(fresh.arrays),'Original physical array keys')
            for name,value in fresh.arrays.items():
                need(np.array_equal(value,arrays[name]),'Every original M9 phased physical value replayed')
                values+=int(value.size)
        metadata=read(group/'processes'/(key+'.json'))
        need(metadata==fresh.metadata,'Original physical parameters, phase, accounting and fingerprint')
        if item['split']=='final_test':
            need(stat.S_IMODE(group.stat().st_mode)==0o700,'Final group directory sealed700')
        replayed+=1
    need(replayed==640 and len(storage['files'])==len({v['path'] for v in storage['files']}),'All640 processes and no duplicate storage paths')
    write(args.output,{'status':'passed','original_processes_replayed':replayed,'physical_values_compared':values,
        'raw_files_SHA_verified':len(storage['files']),'all_data_bytes':size,'final_permissions_verified':True,
        'frozen_weights_SHA':WEIGHTS_SHA,'source_coverage_SHA':sha(args.storage),
        'no_new_model_training':True})
    print('FRESH_SOURCE_REPLAY_DONE',replayed,values,size,flush=True)


if __name__=='__main__':main()

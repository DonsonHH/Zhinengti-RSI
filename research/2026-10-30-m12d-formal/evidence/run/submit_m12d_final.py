"""Hold each one-shot job until Slurm's actual batch bytes match the seal."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--trial',type=int,required=True);a=p.parse_args()
    r=a.run_root.resolve();assert a.trial in (0,1,2);root=r/f'rsi-trial-{a.trial}'
    seal=r/'rsi-final-seal.json';assert read(seal)['all_three_adaptations_frozen']
    guard=read(r/'rsi-resource-before-final.json')['guarded_final_wrapper'];script=Path(guard['path']);assert sha(script)==guard['sha256']
    assert not(root/'final-started.json').exists() and not(root/'final-complete.json').exists() and not(root/'final-one-shot-claim').exists()
    pending_or_running=subprocess.check_output(['squeue','-h','-u','renly','-o','%i|%T'],text=True).strip().splitlines()
    assert len(pending_or_running)<2,'At most two concurrent study allocations'
    claim=root/'final-submission-claim';claim.mkdir(mode=0o700,exist_ok=False)
    job=subprocess.check_output(['sbatch','--hold','--parsable',f'--output={r}/jobs/rsi-final{a.trial}-%j.out',f'--error={r}/jobs/rsi-final{a.trial}-%j.err',str(script),str(r/'rsi-v3'),str(r),str(a.trial)],text=True).strip()
    assert job.isdigit();spool=claim/f'Slurm-{job}.sbatch'
    subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True,text=True)
    assert sha(spool)==guard['sha256'],'Actual held Slurm batch source matches frozen resource seal; never release mismatch'
    receipt={'trial':a.trial,'Slurm_job_id':job,'resolved_run_root':str(r),'seal_SHA256':sha(seal),
        'actual_batch_script_path':str(spool),'actual_batch_script_SHA256':sha(spool),'approved_wrapper_SHA256':guard['sha256'],
        'script_verified_while_job_held_before_release':True,'no_resubmission':True}
    with (claim/'receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
    subprocess.run(['scontrol','release',job],check=True)
    print('M12D_VERIFIED_FINAL_SUBMITTED',a.trial,job,flush=True)


if __name__=='__main__':main()

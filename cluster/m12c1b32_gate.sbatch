#!/usr/bin/env bash
#SBATCH --job-name=b32-gate
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
python scripts/test_causal_torch.py --output "$1/gate"
python - <<'PY'
from pathlib import Path
from patrol_rsi.formal_study import load_protocol
p=load_protocol(Path('research/2026-10-29-m12c1b31-interface/正式实验冻结方案.json'))
print('FROZEN_PROTOCOL_OK',p['protocol'],flush=True)
PY

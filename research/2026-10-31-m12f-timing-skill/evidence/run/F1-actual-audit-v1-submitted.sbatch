#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12f-independent-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
audit_script="$1"
run_root="$2"
source_root="$3"
output="$4"
expected_sha="$5"
accounting="$6"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$run_root"
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python - "$audit_script" "$expected_sha" <<'PY'
import hashlib,pathlib,sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()==sys.argv[2], 'Exact independent audit tool bytes'
PY
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python "$audit_script" --run-root "$run_root" --source-root "$source_root" --phase F1 --accounting "$accounting" --output "$output"

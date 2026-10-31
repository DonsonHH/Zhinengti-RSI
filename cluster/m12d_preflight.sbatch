#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d-qwen-preflight
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
set -euo pipefail
source_root="$1"
run_root="$2"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$source_root"
prior="$source_root/research/2026-10-30-m12c1b4b2-alignment"
python scripts/preflight_m12d_qwen.py --paired "$prior/MultiAgent-Patrol-Paired-v1.json" --rsi "$prior/RSI-Controlled-Meta-v1.json" --model /share/home/renly/renly-patrol-rsi/models/Qwen3-8B-b968826d --output "$run_root/qwen-preflight"

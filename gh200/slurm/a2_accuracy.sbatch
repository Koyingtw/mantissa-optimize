#!/bin/bash
# A2: BF16-truncation vs FP8 accuracy (CPU only).
#SBATCH -J a2-acc
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 00:40:00
#SBATCH -o logs/a2acc_%j.out
#SBATCH -e logs/a2acc_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "job=$SLURM_JOB_ID host=$(hostname)"
.venv/bin/python analysis/a2_accuracy.py --out "results/a2_accuracy.csv"

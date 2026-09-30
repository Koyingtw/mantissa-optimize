#!/bin/bash
# E4: operand-mantissa error model for GEMM (CPU only).
#SBATCH -J e4-error
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o logs/e4_%j.out
#SBATCH -e logs/e4_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "job=$SLURM_JOB_ID host=$(hostname)"
.venv/bin/python analysis/e4_error.py --out "logs/e4_${SLURM_JOB_ID}.csv" --m 256 --trials 3 --ks 256,1024,4096,16384
echo "E4 done"

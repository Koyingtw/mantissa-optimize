#!/bin/bash
# Range/precision statistics of HPL's DGEMM operands (baseline: uniform random matrix).
#SBATCH -J range-hpl
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -o logs/range/hpl_%j.out
#SBATCH -e logs/range/hpl_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO=$(cd .. && pwd)
module purge; module load nvhpc/26.5
NH=/software/nvhpc/Linux_aarch64/26.5
export LD_LIBRARY_PATH="$NH/comm_libs/nccl/lib:$NH/math_libs/13.2/lib64:$NH/cuda/13.2/lib64:${LD_LIBRARY_PATH:-}"
export HPL_QUANT_A_BITS=53 HPL_QUANT_B_BITS=53 HPL_QUANT_C_BITS=53
export MANT_STATS=$PWD/logs/range/hpl MANT_STATS_SAMPLE=262144
LD_PRELOAD=$PWD/shim/libmantshim.so "$REPO/rocHPL/build-gh200/bin/rochpl" -P 1 -Q 1 -p 1 -q 1 -N 65536 -NB 1024 -f 0.05 -it 1 | grep -E "Final Score|Residual"

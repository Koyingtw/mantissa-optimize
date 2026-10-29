#!/bin/bash
# DF-MP2 前處理（只用 CPU）：PySCF DF-RHF + 三中心張量 + CPU 參考相關能 → $MP2_DIR（預設 /tmp/mp2_<分子>，節點本地）。
#   sbatch -w <node> --export=ALL,SCRIPT=mp2_prep.sh,MOL=113_Taxol mi350/slurm/run.sbatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MOL=${MOL:-113_Taxol}
MP2_DIR=${MP2_DIR:-/tmp/mp2_$MOL}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-128}
mkdir -p logs
.venv/bin/python apps/mp2_prep.py "../apps/gpu4pyscf/benchmarks/molecules/organic/$MOL.xyz" "$MP2_DIR" \
  > "logs/mp2prep_${MOL}_$(date +%m%d_%H%M).out" 2>&1
cat "$MP2_DIR/meta.txt"

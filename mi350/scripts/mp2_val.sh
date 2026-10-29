#!/bin/bash
# DF-MP2 GPU 實作的正確性檢查：苯（def2-TZVP）的 PySCF CPU 參考相關能 vs bin/dfmp2（原生與截斷）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
D=/tmp/mp2_benzene
OMP_NUM_THREADS=64 .venv/bin/python apps/mp2_prep.py apps/benzene.xyz $D > logs/mp2val_prep.out 2>&1
cat $D/meta.txt
bin/dfmp2 --dir $D --device 0 --variants "native t36n t20n native"
LD_PRELOAD=$SW/gemmul8/lib/libgemmul8.so GEMMUL8_NUM_MOD_D_GEMM=14 GEMMUL8_FASTMODE_D_GEMM=1 \
  bin/dfmp2 --dir $D --device 0 --variants "native" --tag g8m14

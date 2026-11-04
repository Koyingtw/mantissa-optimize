#!/bin/bash
# 診斷：原版 rocBLAS 跑 stride_2e32 gtest 的完整輸出，加 ROCBLAS_LAYER=2（bench log）看實際呼叫的參數。
set -uo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
B=$W/rocblas-build
T=$B/clients/staging/rocblas-test
O=$W/mi350/logs/rocblas_gtest_diag_${SLURM_JOB_ID:-x}
mkdir -p "$O"
cd "$(dirname "$T")"
env LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib ROCBLAS_LAYER=2 ROCBLAS_LOG_BENCH_PATH=$O/bench_stock.log \
  "$T" --gtest_filter='*stride_2e32*i8*' > $O/gtest_stock.txt 2>&1
echo "rc=$?"
head -60 $O/gtest_stock.txt
echo "--- bench log"
head -20 $O/bench_stock.log

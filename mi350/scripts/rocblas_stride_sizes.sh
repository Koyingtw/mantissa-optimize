#!/bin/bash
# 原版 rocBLAS（/opt/rocm-7.2.3）在 C/D stride = 2^32 時哪些大小會錯：小到大、NN 與 TN、兩種後端。
set -uo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
BIN=$W/mi350/bin/rocblas_stride_sizes
CASES="NN 64 64 64  TN 64 64 64  TN 256 256 256  TN 1024 1024 1024  NN 1024 1024 1024  TN 4096 4096 1024  NN 4096 4096 1024  TN 8192 8192 1024  TN 16384 16384 1024  TN 65536 64 1024  TN 64 65536 1024"
for lt in 0 1; do
  for s in 4294967296 4294967297; do
    ROCBLAS_USE_HIPBLASLT=$lt $BIN --stride $s --cases "$CASES"
  done
done

#!/bin/bash
# 找出 rocblas_stride_sizes 從哪個 C/D stride 開始算錯（NN 64³，原版 rocBLAS）。
set -uo pipefail
BIN=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/bin/rocblas_stride_sizes
for s in 4096 1048576 268435456 1073741824 2147483647 2147483648 4294967295 4294967296; do
  $BIN --stride $s --cases "NN 64 64 64"
done

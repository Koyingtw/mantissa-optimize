#!/bin/bash
# 只重編 rocblas-test（gtest client），16 核。要在 n01 跑（n02 沒有 gtest/OpenBLAS 開發套件，cmake 會失敗）。
set -uo pipefail
export PATH=/opt/rocm-7.2.3/bin:/opt/rocm-7.2.3/llvm/bin:$PATH
echo "host=$(hostname) start=$(date)"
cmake --build /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/rocblas-build --target rocblas-test -j 16 > /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/rocblas_test_rebuild.log 2>&1; grep -m5 -A3 "error:" /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/rocblas_test_rebuild.log; tail -3 /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/rocblas_test_rebuild.log
echo "rc=done end=$(date)"

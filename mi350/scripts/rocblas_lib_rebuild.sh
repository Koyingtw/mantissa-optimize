#!/bin/bash
# 重編修正版 librocblas 與 rocblas-test（16 核，在 n01 跑），把新的 librocblas.so.5.2 複製到 rocblas-build/install/lib。
# Tensile 函式庫檔沒有變，所以只換 .so。
set -uo pipefail
B=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/rocblas-build
LOG=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/rocblas_lib_rebuild.log
export PATH=/opt/rocm-7.2.3/bin:/opt/rocm-7.2.3/llvm/bin:$PATH
echo "host=$(hostname) start=$(date)"
cmake --build $B --target rocblas rocblas-test -j 16 > $LOG 2>&1
rc=$?
grep -m5 -A3 "error:" $LOG; tail -3 $LOG
if [ $rc = 0 ]; then
  cp $B/library/src/librocblas.so.5.2 $B/install/lib/librocblas.so.5.2.new
  mv $B/install/lib/librocblas.so.5.2.new $B/install/lib/librocblas.so.5.2
  md5sum $B/library/src/librocblas.so.5.2 $B/install/lib/librocblas.so.5.2
fi
echo "rc=$rc end=$(date)"

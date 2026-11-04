#!/bin/bash
# 跑 rocblas-test 的 gemm_strided_batched_ex_stride_2e32（stride ≥ 2^32 回歸測試；yaml 內含 2^32-1 對照與
# use_hipblaslt 0/1），原版 /opt/rocm-7.2.3 與修正版各一次。原版應在 2^32、2^32+1 失敗，修正版全過。
# rocblas-test 連到 apt 的 OpenBLAS，只有 n01 有裝。
set -uo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
B=$W/rocblas-build
T=$B/clients/staging/rocblas-test
D=$(dirname "$T")
for lib in stock fix; do
  if [ $lib = fix ]; then L=$B/install/lib; else L=/opt/rocm-7.2.3/lib; fi
  echo "=== RUN lib=$lib"
  echo "loaded: $(env LD_LIBRARY_PATH=$L ldd $T | grep librocblas)"
  ( cd "$D" && env LD_LIBRARY_PATH=$L "$T" --gtest_filter='*stride_2e32*' 2>&1 \
      | grep -E "^\[ *(OK|FAILED|PASSED|SKIPPED|RUN) *\]|tests ran|max_error|error|Error" | tail -80 )
  echo "=== END lib=$lib"
done

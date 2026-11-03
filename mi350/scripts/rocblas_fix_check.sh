#!/bin/bash
# 以修過的 rocBLAS（rocblas-build/install）重跑 int8_stride_layers：hipBLAS 與 rocBLAS 兩種後端，比較原版 /opt/rocm-7.2.3。
set -uo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
BIN=$W/mi350/bin/int8_stride_layers
FIX=$W/rocblas-build/install/lib
for lib in stock fix; do
  for lt in 0 1; do
    for api in rocblas hipblas; do
      for s in 65535 65536 65537; do
        if [ $lib = fix ]; then pre="env LD_LIBRARY_PATH=$FIX:${LD_LIBRARY_PATH:-}"; else pre=env; fi
        t0=$(date +%s.%N)
        out=$($pre ROCBLAS_USE_HIPBLASLT=$lt $BIN --api $api --sizes "$s" 2>&1 | grep RESULT_LAYER)
        t1=$(date +%s.%N)
        echo "lib=$lib hipblaslt=$lt $out wall_s=$(echo "$t1 - $t0" | bc)"
      done
    done
  done
done
echo "loaded: $(env LD_LIBRARY_PATH=$FIX ldd $BIN | grep rocblas)"

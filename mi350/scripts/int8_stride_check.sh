#!/bin/bash
# ROCm strided-batched INT8 GEMM 在 batch stride／偏移量達到 2^32 個元素時的正確性（GEMMul8 2^32 錯誤的根源）。
# 用 mi350/tools/int8_gemm_check.hip（make: hipcc ... -o mi350/bin/int8_gemm_check -lhipblas）。
#   srun -w p01-r01-n02 -N1 mi350/scripts/int8_stride_check.sh "<batch> <size> [<batch> <size> ...]"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null 2>&1
read -ra a <<<"${1:-2 65535 2 65536}"
for ((i = 0; i < ${#a[@]}; i += 2)); do
  bin/int8_gemm_check --batch "${a[i]}" --sizes "${a[i+1]}" ${EXTRA:-}
done

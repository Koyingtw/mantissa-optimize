#!/bin/bash
# 2^32 batch stride 截斷在哪一層：hipBLAS、rocBLAS（Tensile 或 hipBLASLt 後端）、hipBLASLt 直接呼叫（bin/int8_stride_layers）。
#   srun -w p01-r01-n02 -N1 mi350/scripts/int8_layers.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null 2>&1
SIZES=${SIZES:-"65535 65536 65537"}
bin/int8_stride_layers --api hipblas --sizes "$SIZES"
for lt in 0 1; do
  echo "ROCBLAS_USE_HIPBLASLT=$lt"
  ROCBLAS_USE_HIPBLASLT=$lt bin/int8_stride_layers --api rocblas --sizes "$SIZES"
done
bin/int8_stride_layers --api lt --sizes "$SIZES"

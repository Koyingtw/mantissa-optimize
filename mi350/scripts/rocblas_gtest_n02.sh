#!/bin/bash
# 在 n02 跑 stride_2e32 gtest（n02 沒裝 OpenBLAS，用 rocblas-build/extlib 裡從 n01 複製的 .so）。
# LD_LIBRARY_PATH 要在 job 裡面設：srun --export 傳進來的值會被覆蓋。
# LIB=stock（預設，/opt/rocm-7.2.3/lib）或 LIB=fix（rocblas-build/install/lib）。
set -uo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
B=$W/rocblas-build
T=$B/clients/staging/rocblas-test
if [ "${LIB:-stock}" = fix ]; then L=$B/install/lib; else L=/opt/rocm-7.2.3/lib; fi
export LD_LIBRARY_PATH=$L:$B/extlib
echo "host=$(hostname) lib=${LIB:-stock} rocblas=$(ldd $T | grep librocblas)"
cd "$(dirname "$T")"
"$T" --gtest_filter="${FILTER:-*stride_2e32*i8*}" 2>&1 | grep -v -E "^(Device ID|with |maxGrid|---)" | head -${LINES_MAX:-80}
echo "--- tool, same env"
$W/mi350/bin/rocblas_stride_sizes --beta 1 --stride 4294967296 --cases "NN 64 64 64"

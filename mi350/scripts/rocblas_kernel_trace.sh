#!/bin/bash
# 比較 stride_2e32 gtest 與 rocblas_stride_sizes 實際啟動的 kernel（原版 rocBLAS，NN 64³，stride 2^32，hipBLASLt 0）。
set -uo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
B=$W/rocblas-build
T=$B/clients/staging/rocblas-test
O=$W/mi350/logs/rocblas_kernel_trace_${SLURM_JOB_ID:-x}
mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib:$B/extlib
export ROCBLAS_USE_HIPBLASLT=0
cd "$(dirname "$T")"
/opt/rocm-7.2.3/bin/rocprofv3 --kernel-trace --output-format csv -d $O/gtest -o trace -- \
  "$T" --gtest_filter='*stride_2e32*i8*' > $O/gtest.txt 2>&1
echo "gtest rc=$?"; grep -E "PASSED|FAILED|tests" $O/gtest.txt | head
/opt/rocm-7.2.3/bin/rocprofv3 --kernel-trace --output-format csv -d $O/tool -o trace -- \
  $W/mi350/bin/rocblas_stride_sizes --beta 1 --stride 4294967296 --cases "NN 64 64 64" > $O/tool.txt 2>&1
echo "tool rc=$?"; grep RESULT $O/tool.txt
for d in gtest tool; do
  echo "--- $d kernels"
  f=$(find $O/$d -name "*kernel_trace.csv" | head -1)
  python3 -c "
import csv,sys,collections
c=collections.Counter()
for r in csv.DictReader(open('$f')): c[(r['Kernel_Name'][:150], r.get('Grid_Size_X',''), r.get('Grid_Size_Z',r.get('Grid_Size','')))]+=1
for k,v in c.most_common(15): print(v,k)
"
done

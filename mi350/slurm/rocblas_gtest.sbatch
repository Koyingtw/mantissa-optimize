#!/bin/bash
# rocBLAS stride ≥ 2^32 回歸測試（gtest），原版 vs 修正版：sbatch -w p01-r01-n01 mi350/slurm/rocblas_gtest.sbatch
# 先重新產生 rocblas_gtest.data（yaml 改過），再跑測試。GPU 不由 Slurm 追蹤，所以用 --exclusive；只需幾分鐘到十幾分鐘。
#SBATCH -J rocblas-gtest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/slurm_%x_%j.out
cd /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350
source env.sh
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
export PATH=/opt/rocm-7.2.3/bin:/opt/rocm-7.2.3/llvm/bin:$PATH
cmake --build /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/rocblas-build --target rocblas-test -j 16 2>&1 | tail -3
grep -a -o "stride_2e32" /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/rocblas-build/clients/staging/rocblas_gtest.data | wc -l
${GTEST_SCRIPT:-scripts/rocblas_gtest_stride.sh}
echo "end=$(date)"

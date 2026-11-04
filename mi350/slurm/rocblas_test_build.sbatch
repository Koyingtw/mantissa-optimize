#!/bin/bash
# 在 rocblas-build（7.2.3 + 修正）裡加編 rocblas-test（gtest client），供 stride ≥ 2^32 回歸測試用。
# 不佔滿節點（48 CPU、不要 GPU、不 exclusive）：sbatch -w p01-r01-n01 mi350/slurm/rocblas_test_build.sbatch
#SBATCH -J rocblas-test-build
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=192G
#SBATCH -o /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/slurm_%x_%j.out
set -eo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
SRC=$W/rocm-libraries
B=$W/rocblas-build
ROCM=/opt/rocm-7.2.3
echo "job=$SLURM_JOB_ID host=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date)"
export PATH=$ROCM/bin:$ROCM/llvm/bin:$PATH
cmake -S "$SRC/projects/rocblas" -B "$B" -DBUILD_CLIENTS_TESTS=ON -DLINK_BLIS=OFF -DBUILD_FORTRAN_CLIENTS=OFF
cmake --build "$B" --target rocblas-test -j "$SLURM_CPUS_PER_TASK"
find "$B" -name rocblas-test -type f
find "$B" -name "rocblas_gtest.data" -type f
echo "end=$(date)"

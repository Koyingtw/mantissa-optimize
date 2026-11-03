#!/bin/bash
# 驗證修過的 rocBLAS（stride ≥ 2^32 時逐 batch）：sbatch -w <node> mi350/slurm/rocblas_fix.sbatch
# GPU 不是 Slurm 追蹤的資源，所以用 --exclusive 以免和別人的量測共用 GPU；整個 job 只需幾分鐘。
#SBATCH -J rocblas-fix
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -o /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/slurm_%x_%j.out
cd /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350
source env.sh
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
scripts/rocblas_fix_check.sh
echo "end=$(date)"

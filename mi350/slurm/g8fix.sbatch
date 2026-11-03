#!/bin/bash
# GEMMul8 2^32 修正的驗證（worktree g8-overflow）：sbatch -w p01-r01-n02 mi350/slurm/g8fix.sbatch
#SBATCH -J g8fix
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/slurm_%x_%j.out
cd /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
scripts/int8_layers.sh
scripts/int8_stride_check.sh "1 65536 2 46341 3 50000 2 65535 2 65536 2 65537"
scripts/g8_fix_check.sh
echo "rc=$? end=$(date)"

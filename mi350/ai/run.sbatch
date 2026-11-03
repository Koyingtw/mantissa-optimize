#!/bin/bash
# AI 論文（第二篇）的實驗入口：在送出 sbatch 的那個 checkout（例如 worktree）裡執行 mi350/scripts/ 的腳本，
# 不像 mi350/slurm/run.sbatch 寫死主 checkout 的路徑。從 checkout 的根目錄送出：
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=ai_gemm_wo.sh mi350/ai/run.sbatch
#SBATCH -J mi350ai
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o mi350/logs/slurm_%x_%j.out
cd "$SLURM_SUBMIT_DIR/mi350"
echo "job=$SLURM_JOB_ID host=$(hostname) script=$SCRIPT dir=$PWD start=$(date)"
"scripts/${SCRIPT:?SCRIPT=<scripts 下的檔名>}"
echo "rc=$? end=$(date)"

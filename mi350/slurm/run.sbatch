#!/bin/bash
# 在 Slurm 節點上執行 mi350/scripts/ 裡的一支實驗腳本（環境變數照常傳入）。各節點家目錄不共用：
# 程式要先複製到該節點（見實驗日誌 05:00），log 寫在該節點的 mi350/logs/，結束後再取回。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=e1_gemm_tdp.sh,CAPS="1000 500" mi350/slurm/run.sbatch
#SBATCH -J mi350
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o /home/cust_nthu/koying/mantissa-optimize/mi350/logs/slurm_%x_%j.out
cd /home/cust_nthu/koying/mantissa-optimize/mi350
echo "job=$SLURM_JOB_ID host=$(hostname) script=$SCRIPT start=$(date)"
"scripts/${SCRIPT:?SCRIPT=<scripts 下的檔名>}"
echo "rc=$? end=$(date)"

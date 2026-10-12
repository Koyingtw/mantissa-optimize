#!/bin/bash
#SBATCH --job-name=fp8attn-measure
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --exclude=head
#SBATCH --output=/nfs/home/koying/mantissa-optimize/gh200/fp8attn/logs/fp8attn-measure_%j.log
# 離線量測（容器內 PyTorch）。用法：
#   ARGS="/nfs/home/koying/fp8attn-dumps/L*_step6.pt" sbatch gh200/fp8attn/slurm/measure.sbatch
set -euo pipefail
SIF=/nfs/home/koying/sc26-mlperf-build/containers/llama2_70b_lora_ngc25.04.sif
HERE=/nfs/home/koying/mantissa-optimize/gh200/fp8attn
echo "節點 $(hostname) job $SLURM_JOB_ID 開始 $(date -u +%FT%TZ)  ARGS=${ARGS:-}"
apptainer exec --nv --bind /nfs/home/koying "$SIF" python -u "$HERE/residual_measure.py" ${ARGS:-/nfs/home/koying/fp8attn-dumps/L*_step6.pt}
echo "結束 $(date -u +%FT%TZ)"

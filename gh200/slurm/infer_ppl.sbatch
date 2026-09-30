#!/bin/bash
# AI inference accuracy: WikiText-2 perplexity of open LLMs with their linear layers in BF16,
# truncated BF16, FP8 E4M3 (per-tensor / per-channel) and FP8 with fewer mantissa bits.
#   MODEL=Llama-2-7b-hf sbatch slurm/infer_ppl.sbatch
# Env: MODEL (dir under /nfs/home/koying/models, or an absolute path), SPECS (see infer/ppl_eval.py),
#      SCRIPT (infer/ppl_eval.py, or infer/ppl_eval_layerwise.py for Llama2-70B), EXTRA (more args).
#SBATCH -J infer-ppl
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o logs/infer/ppl_%j.out
#SBATCH -e logs/infer/ppl_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
SIF=/nfs/home/koying/sc26-mlperf-build/containers/llama2_70b_lora_ngc25.04.sif
MODELS_DIR=/nfs/home/koying/models
MODEL=${MODEL:-Llama-2-7b-hf}
[[ $MODEL == /* ]] && MPATH=$MODEL || MPATH=$MODELS_DIR/$MODEL
SCRIPT=${SCRIPT:-infer/ppl_eval.py}
SPECS=${SPECS:-"bf16 bf16:5 bf16:4 bf16:3 bf16:2 bf16:1 bf16:0 bf16:3:trunc bf16:0:trunc fp8 fp8c fp8:2 fp8:1 fp8:0 fp8c:2 fp8c:1 fp8c:0 fp8:2;wa=w fp8:0;wa=w"}
echo "job=$SLURM_JOB_ID host=$(hostname) model=$MODEL"
echo "specs=[$SPECS]"
nvidia-smi --query-gpu=name,module.power.limit,clocks.max.sm --format=csv,noheader
read -ra specs <<<"$SPECS"
apptainer exec --nv --bind /nfs,/data "$SIF" env HF_HOME=$MODELS_DIR/hf_home HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$MODELS_DIR/pyextra_infer \
  python "$SCRIPT" --model "$MPATH" --tag "job=$SLURM_JOB_ID" --specs "${specs[@]}" ${EXTRA:-}
echo "infer-ppl done"

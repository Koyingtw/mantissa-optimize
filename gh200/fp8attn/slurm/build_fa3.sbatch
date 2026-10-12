#!/bin/bash
#SBATCH -J fp8attn-fa3build
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 72
#SBATCH --time=60
#SBATCH --exclude=head
#SBATCH -o /nfs/home/koying/mantissa-optimize/gh200/fp8attn/logs/%x_%j.log
# CPU-only FlashAttention-3 (hopper/) build: hdim128, bf16 fwd/bwd + e4m3 fwd, sm90a only.
set -e
SIF=/nfs/home/koying/sc26-mlperf-build/containers/llama2_70b_lora_ngc25.04.sif
cd /nfs/home/koying/fp8attn-build/fa3src/hopper
date -u; hostname; ls -la .. ; cat ../README.md >/dev/null
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_CLUSTER=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export MAX_JOBS=8 NVCC_THREADS=4 TORCH_CUDA_ARCH_LIST=9.0
apptainer exec --bind /nfs/home/koying/fp8attn-build $SIF python setup.py build_ext --inplace 2>&1 | grep -v "^ptxas info\|bytes stack frame\|Function properties\|Compiling entry\|Used .* registers" 
ls -la *.so; date -u

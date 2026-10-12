#!/bin/bash
# 第 2 關計時：CUDA 版 / Triton / cuDNN FP8 / FA3 bf16 各一個程序。LOAD=1 時每次 bwd 前插入 GEMM 負載。
cd /nfs/home/koying/mantissa-optimize/gh200/fp8attn/cuda_dm
export FP8DM_NO_BUILD=1
nvidia-smi -q -d POWER | grep -iE "power limit|Module Power" | head -8
for impl in ${IMPLS:-cuda triton cudnn fa3}; do IMPL=$impl timeout 300 python bench_cuda_dm.py 2>&1 | grep -vE "Warning|warn"; done

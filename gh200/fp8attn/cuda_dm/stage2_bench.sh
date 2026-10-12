#!/bin/bash
# 第 2 關：同一節點、同一 job 內依序量 CUDA 版（兩個變體）/ Triton / cuDNN FP8 / FA3 bf16，
# 先單獨（無 GEMM 負載），再 LOAD=1（每次 bwd 前插入 bf16 GEMM 負載，讓模組撞 575 W 上限）。
cd /nfs/home/koying/mantissa-optimize/gh200/fp8attn/cuda_dm
export FP8DM_NO_BUILD=1
nvidia-smi -q -d POWER | grep -A3 "Module Power" | grep "Current Power Limit"
for load in ${LOADS:-0 1}; do
  echo "######## LOAD=$load"
  for impl in ${IMPLS:-cuda cuda_v3 triton cudnn fa3}; do
    case $impl in
      cuda_v3) FP8DM_BUILD=/nfs/home/koying/fp8attn-build/cuda_dm_v3 LOAD=$load IMPL=cuda timeout 400 python bench_cuda_dm.py 2>&1 | grep -E "RESULT|kernel  *void fp8dm::d" | sed 's/^/[v3] /' ;;
      *) LOAD=$load IMPL=$impl timeout 400 python bench_cuda_dm.py 2>&1 | grep -E "RESULT|kernel  *void fp8dm::d" ;;
    esac
  done
done

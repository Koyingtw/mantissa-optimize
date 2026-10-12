#!/bin/bash
# 單節點 steady-state 對照：attention 三種設定（A FP8 fwd+bwd、B FP8 fwd + cuDNN bf16 bwd、C FA3 bf16 fwd+bwd）。
# 用 offload 設定、575 W、風扇不動。FA3 的 shim 套件複製到 run dir（容器內 /results）避免改 launch.sh 的 bind。
set -uo pipefail
MLP=/nfs/home/koying/mantissa-optimize/SC26-MLPerf
export WORK_ROOT=/nfs/home/koying/mlperf-llama2-70b-lora BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
BASE='export MAX_STEPS=40 VAL_CHECK_INTERVAL=2 LIMIT_VAL_BATCHES=0 WARMUP=True WARMUP_VALIDATION_STEPS=0'
declare -A OVR=(
  [A]="NVTE_FP8_DPA_BWD=1"
  [B]="NVTE_FP8_DPA_BWD=0"
  [C]="FP8_DPA=0 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 PYTHONPATH=/results/fa3pkg:\$PYTHONPATH"
)
for v in ${VARIANTS:-A B C}; do
  rd=$WORK_ROOT/runs/attn-smoke-$v-${TAG:-1}
  rm -rf "$rd"; mkdir -p "$rd"
  [ $v = C ] && cp -rL /nfs/home/koying/fp8attn-build/fa3pkg "$rd/"
  NNODES=1 SEED=1234 TIME=00:30:00 EXCLUDE_NODES=head,work2 POWER_CAP=575 FAN_SPEED= GPUMON=1 RUN_DIR=$rd \
    OVERRIDES="$BASE ${OVR[$v]}" $MLP/nvidia/launch.sh config_GH200_8x1x1xtp1pp1cp1_offload.sh 2>&1 | grep -E "Submitted|run dir"
done

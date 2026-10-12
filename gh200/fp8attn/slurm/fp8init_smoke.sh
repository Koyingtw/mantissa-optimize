#!/bin/bash
# FP8 delayed-scaling 冷啟動修正的單節點 smoke（A 設定：FP8 fwd + FP8 bwd，不設 ATTN_IMPL）。
# 同一節點依 ORDER 依序跑；575 W（比賽規定），風扇不動；用工作副本 nvidia_impl。
#   NODE=work12 ORDER="R K" DIAG=1 gh200/fp8attn/slurm/fp8init_smoke.sh
# DIAG=1：開 FP8SAT_LOG 監看 + dump L0/L40/L79 的 step 1,2,3,30（計時會被 dump 拖慢，只看數值）
# DIAG=0：純計時（不 dump、不監看）
set -uo pipefail
MLP=/nfs/home/koying/mantissa-optimize/SC26-MLPerf
export WORK_ROOT=/nfs/home/koying/mlperf-llama2-70b-lora BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
export IMPL=/nfs/home/koying/mlperf-fp8attn-work/nvidia_impl
RUNS=/nfs/home/koying/mlperf-fp8attn-work/runs
NODE=${NODE:?NODE}
DIAG=${DIAG:-0}
BASE="export MAX_STEPS=40 VAL_CHECK_INTERVAL=2 LIMIT_VAL_BATCHES=0 WARMUP=True WARMUP_VALIDATION_STEPS=0 NVTE_FP8_DPA_BWD=1 ${EXTRA_BASE:-}"
[ "$DIAG" = 1 ] && BASE="$BASE FP8SAT_LOG=1 FP8SAT_LOG_FILE=/results/fp8sat.jsonl FP8ATTN_DUMP=1 FP8ATTN_DUMP_DIR=/results/dumps FP8ATTN_DUMP_LAYERS=0,40,79 FP8ATTN_DUMP_STEPS=1,2,3,30"
declare -A OVR=(
  [R]=""                                            # 原始：warmup 後 reset（scale=1）
  [K]="RESET_FP8_STATS_AFTER_WARMUP=0"              # (a) 不 reset，沿用 warmup（合成資料）的 amax/scale
  [B]="FP8_BOOTSTRAP_PASSES=2"                      # (b') 第 1 步先在同一 batch 跑 2 次校準 fwd+bwd（丟梯度）
  [B3]="FP8_BOOTSTRAP_PASSES=3"
  [M2]="FP8_MARGIN=2"
  [B3M2]="FP8_BOOTSTRAP_PASSES=3 FP8_MARGIN=2"
  [KB2]="RESET_FP8_STATS_AFTER_WARMUP=0 FP8_BOOTSTRAP_PASSES=2"
  [B5]="FP8_BOOTSTRAP_PASSES=5"
  [KB2M2]="RESET_FP8_STATS_AFTER_WARMUP=0 FP8_BOOTSTRAP_PASSES=2 FP8_MARGIN=2"
  [KM2]="RESET_FP8_STATS_AFTER_WARMUP=0 FP8_MARGIN=2"
  [KB2M2d]="RESET_FP8_STATS_AFTER_WARMUP=0 FP8_BOOTSTRAP_PASSES=2 FP8_MARGIN=2 FP8ATTN_DUMP_STEPS=3,4,5,32"   # dump 計數含 2 次校準：3,4,5,32 = 實際 step 1,2,3,30
  [KB]="RESET_FP8_STATS_AFTER_WARMUP=0 FP8_BOOTSTRAP_PASSES=1"
  [H]="FP8_AMAX_HISTORY=16"                         # (d) history 4 -> 16
  [M]="FP8_MARGIN=1"                                # (d) margin 0 -> 1
  [KM]="RESET_FP8_STATS_AFTER_WARMUP=0 FP8_MARGIN=1"
  [C]="FP8_RECIPE=tensorwise RESET_FP8_STATS_AFTER_WARMUP=0"                     # (c) 全程 current scaling（Megatron 不設 fp8_dpa -> attention 變 BF16）
  [CD]="FP8_RECIPE=tensorwise FP8_CS_DPA=1 RESET_FP8_STATS_AFTER_WARMUP=0"         # (c) current scaling + 強制 FP8 attention
)
mkdir -p $RUNS
i=0
for v in ${ORDER:-R K}; do
  i=$((i+1)); rd=$RUNS/fp8init${TAG:-}-d$DIAG-$NODE-$i$v
  rm -rf "$rd"; mkdir -p "$rd"
  out=$(NNODES=1 NODELIST=$NODE EXCLUDE_NODES=none SEED=${SEED:-1234} TIME=00:40:00 POWER_CAP=575 POWER_RESTORE=575 FAN_SPEED= GPUMON=1 RUN_DIR=$rd \
    OVERRIDES="$BASE ${OVR[$v]}" $MLP/nvidia/launch.sh config_GH200_8x1x1xtp1pp1cp1_offload.sh 2>&1)
  jid=$(awk '/Submitted batch job/{print $4}' <<<"$out" | head -1)
  echo "$(date -u +%T) $i$v job=$jid dir=$rd"
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 20; done
  tr '\r' '\n' < "$rd/slurm-$jid.out" | grep '^:::MLLOG' | python3 -c "
import sys,json
ev=[json.loads(l[l.index('{'):]) for l in sys.stdin]
t={e['key']:e['time_ms'] for e in ev if e['key'] in('run_start','run_stop')}
print('  $i$v', 's/step=%.3f' % ((t['run_stop']-t['run_start'])/40000) if len(t)==2 else 'no timing')
"
  tr '\r' '\n' < "$rd/slurm-$jid.out" | grep '"train_loss"' | sed 's/.*"value": \([0-9.e-]*\).*"samples_count": \([0-9]*\).*/  loss \1 @ \2/' | head -20
  grep -h "fp8_scale_fix: bootstrap\|fp8_scale_fix: FP8\|forced True" "$rd/slurm-$jid.out" | head -3
  grep -h -m3 -E "Traceback|Error|out of memory|OOM" "$rd/slurm-$jid.out"
done

#!/bin/bash
# 同一節點依序跑 attention 三種設定（順序 C A B B A C），575 W、風扇不動，用工作副本 nvidia_impl（多 ATTN_BACKEND 開關）。
#   A FP8 fwd + FP8 bwd（目前提交設定）  B FP8 fwd + cuDNN bf16 bwd（arm）  C FA3 bf16 fwd + bwd（TE→FA3）
set -uo pipefail
MLP=/nfs/home/koying/mantissa-optimize/SC26-MLPerf
export WORK_ROOT=/nfs/home/koying/mlperf-llama2-70b-lora BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
export IMPL=/nfs/home/koying/mlperf-fp8attn-work/nvidia_impl
NODE=${NODE:-work5}
BASE="export MAX_STEPS=40 VAL_CHECK_INTERVAL=2 LIMIT_VAL_BATCHES=0 WARMUP=True WARMUP_VALIDATION_STEPS=0 ${EXTRA_BASE:-}"
declare -A OVR=(
  [A]="NVTE_FP8_DPA_BWD=1"
  [B]="NVTE_FP8_DPA_BWD=0"
  [C]="FP8_DPA=0 ATTN_BACKEND=flash NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 PYTHONPATH=/results/fa3pkg:\$PYTHONPATH"
  [D]="FP8_DPA=0 ATTN_IMPL=fp8dm PYTHONPATH=/workspace/ft-llm/_extra:\$PYTHONPATH TRITON_CACHE_DIR=/tmp/triton-cache-fp8dm"
)
i=0
for v in ${ORDER:-D A}; do
  i=$((i+1)); rd=$WORK_ROOT/runs/attn-d${TAG:-}-$NODE-$i$v
  rm -rf "$rd"; mkdir -p "$rd"
  [ $v = C ] && cp -rL /nfs/home/koying/fp8attn-build/fa3pkg "$rd/"
  out=$(NNODES=1 NODELIST=$NODE EXCLUDE_NODES=none SEED=1234 TIME=00:40:00 POWER_CAP=575 POWER_RESTORE=575 FAN_SPEED= GPUMON=1 RUN_DIR=$rd \
    OVERRIDES="$BASE ${OVR[$v]}" $MLP/nvidia/launch.sh config_GH200_8x1x1xtp1pp1cp1_offload.sh 2>&1)
  jid=$(awk '/Submitted batch job/{print $4}' <<<"$out" | head -1)
  echo "$(date -u +%T) $i$v job=$jid"
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 20; done
  while squeue -h -u "$USER" -n nv-lora-restore -w $NODE 2>/dev/null | grep -q .; do sleep 10; done
  tr '\r' '\n' < "$rd/slurm-$jid.out" | grep '^:::MLLOG' | python3 -c "
import sys,json
ev=[json.loads(l[l.index('{'):]) for l in sys.stdin]
t={e['key']:e['time_ms'] for e in ev if e['key'] in('run_start','run_stop')}
print('  $i$v', 's/step=%.3f' % ((t['run_stop']-t['run_start'])/40000) if len(t)==2 else 'no timing')
"
  tr '\r' '\n' < "$rd/slurm-$jid.out" | grep '"train_loss"' | sed 's/.*"value": \([0-9.e-]*\).*"samples_count": \([0-9]*\).*/  loss \1 @ \2/' | head -20
  grep -h "fp8dm_attn" "$rd/slurm-$jid.out" | sort | uniq -c | head -5; grep -h -m3 -E "Traceback|Error|out of memory|OOM" "$rd/slurm-$jid.out"
  grep -h -m1 -o "Selected backend = [A-Za-z0-9 ().]*" "$rd/slurm-$jid.out"; grep -h -m2 -E "Error|out of memory" "$rd/slurm-$jid.out"
done

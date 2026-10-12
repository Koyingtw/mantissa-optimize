#!/bin/bash
# 575 W 下 attention 三種設定的每步時間（單節點、40 步 steady state），每個節點跑一輪 ABCCBA（或輪換順序），多節點並行。
#   D 自寫 Triton FP8 fwd + Delta-Matching bwd（ATTN_IMPL=fp8dm，工作副本 fp8dm_attn.py；triton 3.8 在 IMPL/_extra）
#   A FP8 fwd + FP8 bwd（目前提交設定）  B FP8 fwd + cuDNN bf16 bwd  C FA3 bf16 fwd + bwd（TE→FA3）
# POWER_RESTORE=575：job 結束不回 1000 W（比賽固定 575 W），也避免還原 job 晚到、讓下一個 run 沒有上限的競態。
set -uo pipefail
MLP=/nfs/home/koying/mantissa-optimize/SC26-MLPerf
export WORK_ROOT=/nfs/home/koying/mlperf-llama2-70b-lora BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
export IMPL=/nfs/home/koying/mlperf-fp8attn-work/nvidia_impl
NODES=${NODES:-"work3 work4 work5 work6"}
LOG=/nfs/home/koying/mantissa-optimize/gh200/fp8attn/speed_abc
mkdir -p $LOG
BASE='export MAX_STEPS=40 VAL_CHECK_INTERVAL=2 LIMIT_VAL_BATCHES=0 WARMUP=True WARMUP_VALIDATION_STEPS=0'
declare -A OVR=(
  [A]="NVTE_FP8_DPA_BWD=1"
  [B]="NVTE_FP8_DPA_BWD=0"
  [C]="FP8_DPA=0 ATTN_BACKEND=flash PYTHONPATH=/results/fa3pkg:\$PYTHONPATH"
  [D]="FP8_DPA=0 ATTN_IMPL=fp8dm PYTHONPATH=/workspace/ft-llm/_extra:\$PYTHONPATH TRITON_CACHE_DIR=/tmp/triton-cache-fp8dm"
)
ORDERS=("A B C D D C B A" "B C D A A D C B" "C D A B B A D C" "D A B C C B A D")
# 先等所有舊的還原 job（會把上限設回 1000 W）跑完。
while squeue -h -u "$USER" -n nv-lora-restore,nv-lora 2>/dev/null | grep -q .; do sleep 15; done
k=0
for node in $NODES; do
  order=${ORDERS[$((k % 4))]}; k=$((k+1))
  (
    i=0
    for v in $order; do
      i=$((i+1)); rd=$WORK_ROOT/runs/speed-$node-$i$v
      rm -rf "$rd"; mkdir -p "$rd"
      [ $v = C ] && cp -rL /nfs/home/koying/fp8attn-build/fa3pkg "$rd/"
      out=$(NNODES=1 NODELIST=$node EXCLUDE_NODES=none SEED=1234 TIME=00:40:00 POWER_CAP=575 POWER_RESTORE=575 FAN_SPEED= GPUMON=1 RUN_DIR=$rd \
        OVERRIDES="$BASE ${OVR[$v]}" $MLP/nvidia/launch.sh config_GH200_8x1x1xtp1pp1cp1_offload.sh 2>&1)
      jid=$(awk '/Submitted batch job/{print $4}' <<<"$out" | head -1)
      rj=$(awk '/Submitted batch job/{print $4}' <<<"$out" | sed -n 2p)
      while squeue -h -j "$jid${rj:+,$rj}" 2>/dev/null | grep -q .; do sleep 15; done
      r=$(tr '\r' '\n' < "$rd/slurm-$jid.out" | grep '^:::MLLOG' | python3 -c "
import sys,json
ev=[json.loads(l[l.index('{'):]) for l in sys.stdin]
t={e['key']:e['time_ms'] for e in ev if e['key'] in('run_start','run_stop')}
print('%.4f' % ((t['run_stop']-t['run_start'])/40000) if len(t)==2 else 'NA')")
      p=$(awk -F, 'NR>1 && $8+0>90 {n++; c+=$9; m+=$15; if($15+0>mx) mx=$15} END{if(n) printf "%.0f %.0f %.0f", c/n, m/n, mx; else print "NA NA NA"}' "$rd"/gpumon-$node.csv 2>/dev/null)
      echo "$node $i $v $jid $r $p" | tee -a $LOG/results.txt
    done
  ) &
done
wait
echo "全部完成"

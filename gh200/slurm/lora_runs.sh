#!/bin/bash
# A3: MLPerf Llama2-70B LoRA with the frozen FP8 base weights' mantissa reduced.
# Runs the variants one after another on the same 8 nodes with the same seed
# (paired comparison: same data order, only the weight bits differ).
#   nohup gh200/slurm/lora_runs.sh > gh200/logs/lora/driver.log 2>&1 &
# Env: VARIANTS (list of bits:round:match, e.g. "3 2:nearest 2:trunc:mlp"; round =
# trunc|nearest|stochastic, match = regex on parameter names), or BITS (plain list),
# SEED, POWER (module cap, W), NODES (Slurm nodelist), TAG (suffix for run dirs and result files, e.g. the
# node set, so paired series on different nodes do not overwrite each other).
set -uo pipefail
ROOT=/nfs/home/koying/mantissa-optimize
MLP=$ROOT/SC26-MLPerf
export WORK_ROOT=/nfs/home/koying/mlperf-llama2-70b-lora BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
BITS=${BITS:-"3 2 1"}
SEED=${SEED:-1234}
POWER=${POWER:-575}
NODES=${NODES:-work[4-7,10-13]}
HOSTS=$(scontrol show hostnames "$NODES" | paste -sd,)
OUT=$ROOT/gh200/logs/lora
mkdir -p "$OUT"

# Power caps are applied inside each job by launch.sh (POWER_CAP), so they take
# effect when the job actually starts and are restored by its dependent restore
# job, whatever else ran on the nodes while it was queued.

VARIANTS=${VARIANTS:-$BITS}
for v in $VARIANTS; do
  IFS=: read -r b round match <<<"$v"
  round=${round:-trunc}
  label="bits$b"; [ "$b" -lt 3 ] && label="bits$b-$round${match:+-$match}"
  run_dir=$WORK_ROOT/runs/a3-$label-seed$SEED-p$POWER${TAG:+-$TAG}
  rm -rf "$run_dir"
  out=$(NNODES=8 NODELIST="$NODES" POWER_CAP="$POWER" FAN_SPEED= SEED=$SEED TIME=${TIME_LIMIT:-00:50:00} GPUMON=1 GPUMON_INTERVAL=2 RUN_DIR="$run_dir" \
        OVERRIDES="export FP8_WEIGHT_MANT_BITS=$b FP8_WEIGHT_MANT_ROUND=$round ${match:+FP8_WEIGHT_MANT_MATCH=$match}" \
        "$MLP/nvidia/launch.sh" config_GH200_8x1x1xtp1pp1cp1_offload.sh 2>&1)
  jid=$(awk '/Submitted batch job/{print $4}' <<<"$out" | head -1)
  echo "$(date -u +%T) variant=$label job=$jid run_dir=$run_dir"
  [ -n "$jid" ] || { echo "$out"; continue; }
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 30; done
  # launch.sh's dependent restore job resets the caps to 1000 W; let it finish before
  # the next variant is submitted, or it could land in the middle of that run.
  while squeue -h -u "$USER" -n nv-lora-restore 2>/dev/null | grep -q .; do sleep 10; done
  res="$OUT/result_${label}_seed${SEED}_p${POWER}${TAG:+_$TAG}.txt"
  tr '\r' '\n' < "$run_dir/slurm-$jid.out" | grep '^:::MLLOG' > "$res"
  python3 - "$res" "$label" <<'PY'
import json, sys
ev = [json.loads(l[l.index("{"):]) for l in open(sys.argv[1])]
t0 = next((e["time_ms"] for e in ev if e["key"] == "run_start"), None)
stop = next((e for e in ev if e["key"] == "run_stop"), None)
evals = [(e["metadata"]["samples_count"], round(e["value"], 4)) for e in ev if e["key"] == "eval_accuracy"]
print(f"variant={sys.argv[2]} status={stop['metadata']['status'] if stop else 'no run_stop'}",
      f"converged_at={next((s for s, v in evals if v <= 0.925), '-')}",
      f"ttt_min={(stop['time_ms'] - t0) / 60000:.2f}" if stop and t0 else "ttt=-", f"evals={evals}")
PY
  for f in "$run_dir"/gpumon-*.csv; do cp "$f" "$OUT/${label}_p${POWER}_seed${SEED}${TAG:+_$TAG}_$(basename "$f")"; done 2>/dev/null
done
echo "$(date -u +%T) all variants done"

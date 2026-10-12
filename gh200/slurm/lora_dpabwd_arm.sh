#!/bin/bash
# MLPerf Llama2-70B LoRA：bf16 attention backward（NVTE_FP8_DPA_BWD=0）的收斂／RCP 驗證 arm。
# 與已採用設定（LR 5.5e-4、600 步 cosine、14 節點、575 W）相同，只把 attention backward 從 FP8 換成 bf16，
# 種子與 15-seed LR arm 相同（逐種子配對比較收斂點）。逐一執行，每次佔滿 14 節點。
#   nohup gh200/slurm/lora_dpabwd_arm.sh > gh200/logs/lora/dpabwd0_driver.log 2>&1 &
# Env: SEEDS（預設 LR arm 的 15 個）、POWER（模組上限 W，預設 575）、FP8_BWD（設 1 則跑 FP8 backward 對照）。
# 風扇不動（FAN_SPEED 留空）：只有使用者要求時才改風扇。
set -uo pipefail
ROOT=/nfs/home/koying/mantissa-optimize
MLP=$ROOT/SC26-MLPerf
export WORK_ROOT=/nfs/home/koying/mlperf-llama2-70b-lora BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
SEEDS=${SEEDS:-"2468 4680 1234 5678 5791 3579 6802 7913 8024 9135 1111 2222 3333 4444 5555"}
POWER=${POWER:-575}
BWD=$([ "${FP8_BWD:-0}" = 1 ] && echo 1 || echo 0)
OUT=$ROOT/gh200/logs/lora/dpabwd$BWD
mkdir -p "$OUT"
for s in $SEEDS; do
  run_dir=$WORK_ROOT/runs/dpabwd$BWD-seed$s-p$POWER
  if [ -s "$OUT/result_seed$s.txt" ]; then echo "seed $s 已完成，略過"; continue; fi
  rm -rf "$run_dir"
  echo "$(date -u +%FT%T) seed=$s 送出 run_dir=$run_dir"
  SEED=$s RUN_DIR="$run_dir" POWER_CAP=$POWER FAN_SPEED= EXTRA_OVR="NVTE_FP8_DPA_BWD=$BWD" \
    "$MLP/nvidia/run_final.sh" 2>&1 | grep -v '^\s*$'
  cp "$run_dir/result_seed$s.txt" "$OUT/" 2>/dev/null
  for f in "$run_dir"/gpumon-*.csv; do cp "$f" "$OUT/seed${s}_$(basename "$f")"; done 2>/dev/null
  # 等 launch.sh 的還原 job（功耗上限回 1000 W）結束再送下一個。
  while squeue -h -u "$USER" -n nv-lora-restore 2>/dev/null | grep -q .; do sleep 10; done
done
echo "$(date -u +%FT%T) 全部完成"

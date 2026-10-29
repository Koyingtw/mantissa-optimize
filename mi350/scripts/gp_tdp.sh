#!/bin/bash
# E5（MI350X）：GP likelihood（apps/gp_chol.hip）× PPT0 上限 × panel 保留位元。8 張 GPU 各跑一個獨立行程
# （同資料、同變體順序），每個上限一次執行 VARIANTS 清單（預設已是 ABBA：native t36n t28n t28n t36n native）。
#   nohup mi350/scripts/gp_tdp.sh >/dev/null 2>&1 &
# 環境變數：CAPS、VARIANTS、GP_N（80000）、REPEAT（每個上限重複幾次整串清單）、START_TEMP、COOL_MAX、TAG
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
CAPS=${CAPS:-"1000 800 650 500"}
VARIANTS=${VARIANTS:-"native t36n t28n t28n t36n native"}
GP_N=${GP_N:-80000}
REPEAT=${REPEAT:-2}
START_TEMP=${START_TEMP:-63}
COOL_MAX=${COOL_MAX:-60}
ID=${TAG:-gp}_$(date +%m%d_%H%M)
mkdir -p logs
exec >"logs/$ID.out" 2>&1
restore() { tools/pcap.sh 1000 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!
hot_max() { local m=0 t; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp2_input; do
  t=$(( $(cat "$f") / 1000 )); (( t > m )) && m=$t; done; echo $m; }
echo "id=$ID n=$GP_N caps=[$CAPS] variants=[$VARIANTS] repeat=$REPEAT"
for cap in $CAPS; do
  tools/pcap.sh "$cap"
  for ((r = 0; r < REPEAT; r++)); do
    waited=0
    while (( $(hot_max) > START_TEMP && waited < COOL_MAX )); do sleep 5; waited=$((waited + 5)); done
    echo "=== RUN cap=$cap round=$r cool=${waited}s"
    pids=()
    for g in 0 1 2 3 4 5 6 7; do
      bin/gp_chol --device "$g" --n "$GP_N" --variants "$VARIANTS" --tag "cap=$cap,round=$r" &
      pids+=($!)
    done
    wait "${pids[@]}"
  done
done
echo "gp done"

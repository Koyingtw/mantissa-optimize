#!/bin/bash
# E5（MI350X）：GP likelihood（apps/gp_chol.hip）× PPT0 上限 × panel 保留位元。8 張 GPU 各跑一個獨立行程
# （同資料、同變體順序），每個上限一次執行 VARIANTS 清單（預設已是 ABBA：native t36n t28n t28n t36n native）。
#   nohup mi350/scripts/gp_tdp.sh >/dev/null 2>&1 &
# 環境變數：CAPS、VARIANTS、GP_N（80000）、REPEAT（每個上限重複幾次整串清單）、START_TEMP、COOL_MAX、TAG、
#   EMU（例如 "12 14"：每輪之後再以 GEMMul8 k moduli 跑 native 兩次，只計第二次；冷啟動見 DF-MP2）、
#   GP_HIPBLAS=1（trailing DGEMM 走 hipBLAS；有 EMU 時自動開啟，基準行程也走同一路徑）
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
CAPS=${CAPS:-"1000 800 650 500"}
VARIANTS=${VARIANTS:-"native t36n t28n t28n t36n native"}
EMU=${EMU:-}
GP_NB=${GP_NB:-1024}
HB=()
[[ ${GP_HIPBLAS:-0} == 1 || -n $EMU ]] && HB=(--hipblas)
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
      bin/gp_chol --device "$g" --n "$GP_N" --nb "$GP_NB" --variants "$VARIANTS" --tag "cap=$cap,round=$r,emu=none" \
        "${HB[@]}" &
      pids+=($!)
    done
    wait "${pids[@]}"
    for m in $EMU; do
      waited=0
      while (( $(hot_max) > START_TEMP && waited < COOL_MAX )); do sleep 5; waited=$((waited + 5)); done
      echo "=== RUN cap=$cap round=$r emu=g8m$m cool=${waited}s"
      pids=()
      for g in 0 1 2 3 4 5 6 7; do
        env LD_PRELOAD="$SW/gemmul8/lib/libgemmul8.so" GEMMUL8_NUM_MOD_D_GEMM="$m" GEMMUL8_FASTMODE_D_GEMM=1 \
          GEMMUL8_MAX_NUM_MOD_GEMM="$m" GEMMUL8_MAX_M_GEMM="$((GP_N - GP_NB))" GEMMUL8_MAX_N_GEMM="$((GP_N - GP_NB))" \
          GEMMUL8_MAX_K_GEMM="$GP_NB" bin/gp_chol --device "$g" --n "$GP_N" --nb "$GP_NB" --variants "native native" \
          --tag "cap=$cap,round=$r,emu=g8m$m" --hipblas &
        pids+=($!)
      done
      wait "${pids[@]}"
    done
  done
done
echo "gp done"

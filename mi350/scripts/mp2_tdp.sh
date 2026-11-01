#!/bin/bash
# E6（MI350X）：DF-MP2（apps/dfmp2.hip，taxol def2-TZVP）× PPT0 上限 × 三中心張量 B 的保留位元，及 GEMMul8 emulation。
# 需要先在同一節點跑過 scripts/mp2_prep.sh（B 存在節點本地 $MP2_DIR）。8 張 GPU 各跑一個行程，行程內依
# VARIANTS（預設 ABBA：native t32n t20n t20n t32n native）；EMU 列出的 moduli 各以一個獨立行程（LD_PRELOAD
# GEMMul8）跑 native，接在同一輪之後。
#   sbatch -w p01-r01-n01 --export=ALL,SCRIPT=mp2_tdp.sh mi350/slurm/run.sbatch
# 環境變數：CAPS、VARIANTS、EMU（"12 14"）、MOL、MP2_DIR、REPEAT、START_TEMP、COOL_MAX、TAG、
#   BIN（預設 bin/dfmp2；bin/dfmp2_354 = 連結 GEMMul8 v3.5.4）、EXTRA（傳給 dfmp2，例如 --percall）
#   VARIANTS 也可用直接呼叫 GEMMul8 的 e<M> 與逐次選擇的 s<M>t<k>n@<N>（見 apps/dfmp2.hip）
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
CAPS=${CAPS:-"1000 800 650 500"}
VARIANTS=${VARIANTS:-"native t32n t20n t20n t32n native"}
EMU=${EMU-"12 14"}   # 設成空字串可略過 emulation
MOL=${MOL:-113_Taxol}
MP2_DIR=${MP2_DIR:-/tmp/mp2_$MOL}
REPEAT=${REPEAT:-2}
START_TEMP=${START_TEMP:-63}
COOL_MAX=${COOL_MAX:-60}
ID=${TAG:-mp2}_$(date +%m%d_%H%M)
mkdir -p logs
exec >"logs/$ID.out" 2>&1
[[ -f $MP2_DIR/B.bin ]] || { echo "missing $MP2_DIR/B.bin (run scripts/mp2_prep.sh on this node first)"; exit 1; }
restore() { tools/pcap.sh 1000 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!
hot_max() { local m=0 t; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp2_input; do
  t=$(( $(cat "$f") / 1000 )); (( t > m )) && m=$t; done; echo $m; }
cool() { local waited=0
  while (( $(hot_max) > START_TEMP && waited < COOL_MAX )); do sleep 5; waited=$((waited + 5)); done; echo "$waited"; }
echo "id=$ID mol=$MOL dir=$MP2_DIR caps=[$CAPS] variants=[$VARIANTS] emu=[$EMU] repeat=$REPEAT"
cat "$MP2_DIR/meta.txt"
eval "$(tr ' ' '\n' < "$MP2_DIR/meta.txt" | grep -E '^(nocc|nvir|naux)=' | tr a-z A-Z)"
run8() {  # $1 = tag, 其餘 = 傳給 dfmp2 的環境變數
  local tag=$1; shift
  local pids=()
  for g in 0 1 2 3 4 5 6 7; do
    env "$@" "${BIN:-bin/dfmp2}" --dir "$MP2_DIR" --device "$g" --variants "${RUN_VARIANTS}" --tag "$tag" ${EXTRA:-} &
    pids+=($!)
  done
  wait "${pids[@]}"
}
for cap in $CAPS; do
  tools/pcap.sh "$cap"
  for ((r = 0; r < REPEAT; r++)); do
    echo "=== RUN cap=$cap round=$r cool=$(cool)s"
    RUN_VARIANTS=$VARIANTS run8 "cap=$cap,round=$r,emu=none"
    for m in $EMU; do
      echo "=== RUN cap=$cap round=$r emu=g8m$m cool=$(cool)s"
      # 每個新形狀第一次呼叫 GEMMul8 有很大的冷啟動成本（taxol 第一輪 141 s、第二輪 12.9 s），所以同一行程跑兩次，
      # 分析只取第二次（pass B）；workspace 以最大形狀與 moduli 預先配置（GEMMUL8_MAX_NUM_MOD 預設只有 2）
      RUN_VARIANTS="native native" run8 "cap=$cap,round=$r,emu=g8m$m" LD_PRELOAD="$SW/gemmul8/lib/libgemmul8.so" \
        GEMMUL8_NUM_MOD_D_GEMM="$m" GEMMUL8_FASTMODE_D_GEMM=1 GEMMUL8_MAX_NUM_MOD_GEMM="$m" \
        GEMMUL8_MAX_M_GEMM="$NVIR" GEMMUL8_MAX_N_GEMM="$((NOCC * NVIR))" GEMMUL8_MAX_K_GEMM="$NAUX"
    done
  done
done
echo "mp2 done"

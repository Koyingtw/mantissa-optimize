#!/bin/bash
# E1（MI350X）：GEMM 吞吐量與功耗 × PPT0 上限 × 運算元保留的尾數位元。
# 8 張 GPU 同時跑同一個設定（每張一個行程），所以每個量測窗得到 8 個 GPU 的配對值。
# 每個上限把設定清單先正向再反向各跑一次（ABBA）。
#   nohup mi350/scripts/e1_gemm_tdp.sh > /dev/null 2>&1 &      （從 repo 根目錄或任何地方）
# 環境變數：CAPS（W）、CONFIGS（dtype:frac[:low]，low = zero|ones|const；g8:<moduli> = FP64 GEMM 改走
#   GEMMul8 INT8 emulation）、SHAPES（"mxnxk" 清單）、WARMUP、SECONDS_PER、
#   START_TEMP（每次量測前所有 GPU 的 junction 溫度須降到此值以下，°C）、COOL_MAX（最多等幾秒）、
#   TAG（log 名稱前綴，預設 e1）、EXTRA_ARGS（傳給 bench）
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
CAPS=${CAPS:-"1000 800 650 500"}
CONFIGS=${CONFIGS:-"fp64:52 fp64:36 fp64:27 fp64:12 fp64:0 bf16:7 bf16:3 bf16:0 fp8:3 fp8:1 fp8:0"}
WARMUP=${WARMUP:-8}
SECONDS_PER=${SECONDS_PER:-20}
START_TEMP=${START_TEMP:-62}
COOL_MAX=${COOL_MAX:-120}
ID=${TAG:-e1}_$(date +%m%d_%H%M)
OUT=logs/$ID.out
mkdir -p logs
exec >"$OUT" 2>&1

restore() { tools/pcap.sh 1000 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!

hot_max() { local m=0 t; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp2_input; do
  t=$(( $(cat "$f") / 1000 )); (( t > m )) && m=$t; done; echo $m; }

echo "id=$ID host=$(hostname) caps=[$CAPS] configs=[$CONFIGS] warmup=$WARMUP seconds=$SECONDS_PER start_temp=$START_TEMP"
amd-smi version 2>/dev/null | head -1
read -ra cfg <<<"$CONFIGS"
# SHAPES：選用的 "mxnxk" 清單（用 x 分隔，sbatch --export 會在逗號處切開）；每個設定在每個形狀都跑
read -ra shapes <<<"${SHAPES:-default}"
rev=()
for ((i = ${#cfg[@]} - 1; i >= 0; i--)); do rev+=("${cfg[i]}"); done

for cap in $CAPS; do
  tools/pcap.sh "$cap"
  echo "CAP $cap set: $(tools/pcap.sh show | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
  for pass in A B; do
    list=("${cfg[@]}")
    [[ $pass == B ]] && list=("${rev[@]}")
    for shape in "${shapes[@]}"; do
    for c in "${list[@]}"; do
      IFS=: read -r dtype frac low ifrac <<<"$c"
      low=${low:-zero}
      extra=()
      envs=()
      [[ $dtype == bf16 || $dtype == fp8 ]] && extra=(--k 8192)
      if [[ $shape != default ]]; then IFS=x read -r sm sn sk <<<"$shape"; extra=(--m "$sm" --n "$sn" --k "$sk"); fi
      # g8:<moduli>[:<low>:<輸入位元>]：FP64 GEMM 交給 GEMMul8（Ozaki-II，INT8）以 LD_PRELOAD 攔截 hipblasDgemm；
      # 運算元預設保留全部 52 個小數位元，第四欄可先截斷輸入（截斷 + emulation 疊加）
      emu=none
      if [[ $dtype == g8 ]]; then
        envs=(LD_PRELOAD="$SW/gemmul8/lib/libgemmul8.so" GEMMUL8_NUM_MOD_D_GEMM="$frac" GEMMUL8_FASTMODE_D_GEMM=1)
        emu=g8m$frac; dtype=fp64; frac=${ifrac:-52}
      fi
      bench=bin/gemm_mant_bench
      # 非 GEMM 的 FP64 核心（向量 FMA 迴圈、n-body、STREAM copy/triad）
      if [[ $dtype == dfma || $dtype == nbody || $dtype == copy || $dtype == triad ]]; then
        bench=bin/fp64_kernel_bench; extra=()
      fi
      waited=0
      while (( $(hot_max) > START_TEMP && waited < COOL_MAX )); do sleep 5; waited=$((waited + 5)); done
      echo "COOL waited=${waited}s hot_max=$(hot_max)C"
      # 其他工具可能改掉上限：每次量測前確認
      tools/pcap.sh show | awk -v w="${cap}W" '$2 != w {bad=1} END {exit bad}' || { echo "CAP RESET detected -> re-applying $cap"; tools/pcap.sh "$cap"; }
      pids=()
      for g in 0 1 2 3 4 5 6 7; do
        env "${envs[@]}" "$bench" --device "$g" --dtype "$dtype" --frac "$frac" --low "$low" \
          --warmup "$WARMUP" --seconds "$SECONDS_PER" --tag "cap=$cap,pass=$pass,emu=$emu" "${extra[@]}" \
          ${EXTRA_ARGS:-} &
        pids+=($!)
      done
      wait "${pids[@]}"
    done
    done
  done
done
echo "E1 done"

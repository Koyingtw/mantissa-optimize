#!/bin/bash
# DF-MP2 與 GP 的受控熱實驗（第八輪審查：先前截斷版本排在每輪最熱的中段，且只用迴歸做溫度校正）。
# 每輪先以 8 張 GPU 跑 PRECOND 秒原生 BF16 GEMM 預熱（與 HPL-MxP 的熱控制相同，不等冷卻），再以 ABBA 或 BAAB
# 交替的順序執行一個程式（每張 GPU 一個行程）：截斷版本在 ABBA 位於中段、在 BAAB 位於兩端，兩種順序平均後抵消位置效應。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=thermal_apps.sh mi350/slurm/run.sbatch
# 環境變數：CAP（W，預設 1000）、ROUNDS（每種順序幾輪，預設 3）、PRECOND（秒，預設 180）、APPS（mp2 gp）
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
CAP=${CAP:-1000}
ROUNDS=${ROUNDS:-3}
PRECOND=${PRECOND:-180}
APPS=${APPS:-"mp2 gp"}
MP2=${MP2:-/tmp/mp2_113_Taxol}
ID=${TAG:-thermal}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
restore() { tools/pcap.sh 1000 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!
tools/pcap.sh "$CAP"
echo "id=$ID host=$(hostname) cap=$CAP rounds=$ROUNDS precond=$PRECOND apps=[$APPS]"
r=0
for app in $APPS; do
  for ((i = 0; i < ROUNDS; i++)); do
    for order in ABBA BAAB; do
      if [[ $app == mp2 ]]; then a=native; b=t20n; else a=native; b=t28n; fi
      [[ $order == ABBA ]] && vs="$a $b $b $a" || vs="$b $a $a $b"
      pids=()
      for g in 0 1 2 3 4 5 6 7; do
        bin/gemm_mant_bench --device "$g" --dtype bf16 --k 8192 --warmup 0 --seconds "$PRECOND" --tag precond >/dev/null &
        pids+=($!)
      done
      wait "${pids[@]}"
      echo "=== RUN app=$app cap=$CAP round=$r order=$order"
      pids=()
      for g in 0 1 2 3 4 5 6 7; do
        if [[ $app == mp2 ]]; then
          bin/dfmp2 --dir "$MP2" --device "$g" --variants "$vs" --tag "cap=$CAP,round=$r,order=$order" &
        else
          bin/gp_chol --device "$g" --variants "$vs" --tag "cap=$CAP,round=$r,order=$order" &
        fi
        pids+=($!)
      done
      wait "${pids[@]}"
      r=$((r + 1))
    done
  done
done
echo "thermal done"

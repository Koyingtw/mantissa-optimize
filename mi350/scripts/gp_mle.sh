#!/bin/bash
# 高斯過程的完整最大概似估計（審稿第五輪 Minor 7）：bin/gp_chol --mle，每個變體一張 GPU、同時執行，
# 估計 Matérn-3/2 的三個參數（變異數解析 profile，range 與 nugget 以 Nelder-Mead），比較原生與截斷的估計值。
#   sbatch -w p01-r01-n01 --export=ALL,SCRIPT=gp_mle.sh mi350/slurm/run.sbatch
# 環境變數：VARIANTS（每個一張 GPU，預設 "native t36n t28n"）、N、SEEDS（每個 seed 一組資料）、CAP、TAG
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
VARIANTS=${VARIANTS:-"native t36n t28n"}
N=${N:-80000}
SEEDS=${SEEDS:-"1 2"}
CAP=${CAP:-1000}
ID=${TAG:-gpmle}_$(date +%m%d_%H%M)
mkdir -p logs
exec >"logs/$ID.out" 2>&1
restore() { tools/pcap.sh 1000 || true; }
trap restore EXIT
tools/pcap.sh "$CAP"
echo "id=$ID n=$N variants=[$VARIANTS] seeds=[$SEEDS] cap=$CAP"
g=0
pids=()
for s in $SEEDS; do
  for v in $VARIANTS; do
    bin/gp_chol --mle --device "$g" --n "$N" --seed "$s" --variants "$v" --tag "seed=$s" &
    pids+=($!)
    g=$((g + 1))
  done
done
wait "${pids[@]}"
echo "gpmle done"

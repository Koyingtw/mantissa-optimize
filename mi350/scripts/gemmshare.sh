#!/bin/bash
# 截斷的 GEMM 佔整支程式時間的比例 f（第七輪審查 P2 的 kernel 層級校準用）：在 gfxclk 上限鎖 SCLK MHz、1000 W 下，
# 8 張 GPU 各跑原生版本兩次（只計第二次）。GP：GP_PROF=1 以 HIP event 計每次 trailing update；DF-MP2：--percall。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=gemmshare.sh mi350/slurm/run.sbatch
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
SCLK=${SCLK:-1300}
MP2=${MP2:-/tmp/mp2_113_Taxol}
ID=${TAG:-gemmshare}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
restore_clk() { for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; }
trap restore_clk EXIT
for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done
echo "id=$ID host=$(hostname) sclk=$SCLK"
for g in 0 1 2 3 4 5 6 7; do GP_PROF=1 bin/gp_chol --device "$g" --variants "native native" --tag gp & done; wait
for g in 0 1 2 3 4 5 6 7; do bin/dfmp2 --dir "$MP2" --device "$g" --variants "native native" --percall --tag mp2 & done; wait
echo "gemmshare done"

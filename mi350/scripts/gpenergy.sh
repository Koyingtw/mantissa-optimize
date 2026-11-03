#!/bin/bash
# GP 各階段的能量（第九輪審查 minor 5）：在 gfxclk 上限鎖 SCLK MHz（預設 1300）、1000 W 下，以 GP_ENERGY=1 在每次 trailing
# update 前後同步並讀硬體能量計數器，分開 GEMM 階段與其他階段（POTRF、TRSM、共變異數產生、似然值）的能量與時間，
# 得到程式內 GEMM 的 R 與它佔動態能量的比例（對照 lockclk_apps.sh 的整支程式 R 與 kcal.sh 的面板 R）。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=gpenergy.sh mi350/slurm/run.sbatch
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
SCLK=${SCLK:-1300}
ID=${TAG:-gpenergy}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
restore_clk() { for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; tools/pcap.sh 1000 || true; }
trap restore_clk EXIT
tools/pcap.sh 1000
for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done
echo "id=$ID host=$(hostname) sclk=$SCLK"
for r in 0 1; do
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    GP_ENERGY=1 bin/gp_chol --device "$g" --variants "native t36n t28n t28n t36n native" --tag "round=$r" &
    pids+=($!)
  done
  wait "${pids[@]}"
done
echo "gpenergy done"

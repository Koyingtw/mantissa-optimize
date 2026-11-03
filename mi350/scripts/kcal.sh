#!/bin/bash
# 功耗模型的 kernel 層級校準（第七輪審查 P2）：不跑整支截斷程式，只拿應用程式自己的 GEMM 運算元，
# 在 gfxclk 上限鎖 SCLK MHz（預設 1300）、1000 W 下用 GEMM 微基準量原生與截斷的功耗、時脈，得到 R、σ，
# 再和整支程式鎖頻量到的 R、σ（scripts/lockclk_apps.sh）比較，並預測各上限的加速（analysis/kernel_cal.py）。
#   DF-MP2：A = B[150]（k×m，--ta），B = B[0..150]（k×n），即 i = 150 那次呼叫（m=2002, n=151·2002, k=5477，
#           約為以 flop 加權的平均呼叫）；位元數 32、20（round to nearest，同 t32n、t20n）。
#   GP：gp_chol --dump-panel 寫出的 L21（第 J 欄那一步，(n−J−nb)×nb），trailing update 是 L21·L21ᵀ（--tb）；
#       J ∈ JS、range ∈ GP_RANGES 各一組；位元數 36、28。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=kcal.sh mi350/slurm/run.sbatch
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
SCLK=${SCLK:-1300}
MP2=${MP2:-/tmp/mp2_113_Taxol}
JS=${JS:-"16384 40960"}
GP_RANGES=${GP_RANGES:-"0.02 0.08"}
GPDIR=/tmp/gp_panels
restore_clk() { for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; }
trap restore_clk EXIT

# GP 面板（原生 FP64，GPU 0；先在預設時脈下寫出，不影響量測）
mkdir -p "$GPDIR"
for J in $JS; do
  [[ -f $GPDIR/j${J}_range0.08.bin ]] || bin/gp_chol --device 0 --variants native --dump-panel "$J:$GPDIR/j$J" | grep -E "DUMP|RESULT"
done

for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done
read -r nocc nvir naux < <(sed -E 's/.*nocc=([0-9]+) nvir=([0-9]+) naux=([0-9]+).*/\1 \2 \3/' "$MP2/meta.txt")
I=${MP2_I:-150}
CAPS=1000 TAG=kcalmp2 CONFIGS="fp64:52 fp64:32 fp64:20" SHAPES="${nvir}x$(( (I + 1) * nvir ))x$naux" \
  EXTRA_ARGS="--round nearest --ta --load-a $MP2/B.bin@$(( I * nvir * naux )) --load-b $MP2/B.bin" \
  scripts/e1_gemm_tdp.sh
for J in $JS; do
  for rg in $GP_RANGES; do
    f=$GPDIR/j${J}_range$rg.bin
    r=$(( $(stat -c %s "$f") / 8 / 1024 ))
    CAPS=1000 TAG=kcalgp_j${J}_r$rg CONFIGS="fp64:52 fp64:36 fp64:28" SHAPES="${r}x${r}x1024" \
      EXTRA_ARGS="--round nearest --tb --load-a $f --load-b $f" scripts/e1_gemm_tdp.sh
  done
done
echo "kcal done"

#!/bin/bash
# 事後 FP64 抽查在「真的會失敗」的資料上（第八輪審查 Q1）：
#   1. 纈胺黴素 DFT 的真實運算元（apps/valino_prep.py，CPU 上 DF-B3LYP／def2-TZVP，寫到節點本地 $VDIR）：
#      xc4096、xc76800（XC 矩陣 ao^T (w v ao)）與 mo131072（ao C_occ），GEMMul8 10／12／14 模數，隨機抽樣與依跨度分層抽樣，
#      並算逐元素誤差 |dC|/(|A||B|) 與截斷 36／32 位的對照；
#   2. m·n ≥ 2^32 的尺寸錯誤（ROCm strided-batched INT8 GEMM）：均勻資料 65536×65536×256，對照 65535；
#   3. GP 面板與 DF-MP2 用同樣的新指標重跑（對照：寬跨度但模擬正確）。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=spot2.sh mi350/slurm/run.sbatch
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
VDIR=${VDIR:-/tmp/valino}
MP2=${MP2:-/tmp/mp2_113_Taxol}
ID=${TAG:-spot2}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
echo "id=$ID host=$(hostname)"
if [[ ! -f $VDIR/mo131072/meta.txt ]]; then
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-128} .venv/bin/python apps/valino_prep.py \
    ../apps/gpu4pyscf/benchmarks/molecules/organic/168_Valinomycin.xyz "$VDIR"
fi
for c in xc4096 xc76800 mo131072; do
  read -r mm nn kk lay < <(sed -E 's/m=([0-9]+) n=([0-9]+) k=([0-9]+) layout=(..)/\1 \2 \3 \4/' "$VDIR/$c/meta.txt")
  t=(); [[ $lay == TN ]] && t=(--ta)
  for st in "" --stratify; do
    bin/g8_spot --m "$mm" --n "$nn" --k "$kk" "${t[@]}" --a "$VDIR/$c/A.bin" --b "$VDIR/$c/B.bin" \
      --moduli "10 12 14" --rows 64 $st --cw --trunc "36 32" --tag "valino_$c"
  done
done
for nm in 65535 65536; do
  bin/g8_spot --m "$nm" --n "$nm" --k 256 --synthetic benign:0 --moduli 10 --rows 64 --tag "size_$nm"
done
f=/tmp/gp_panels/j16384_range0.02.bin
if [[ -f $f ]]; then
  r=$(( $(stat -c %s "$f") / 8 / 1024 ))
  for st in "" --stratify; do
    bin/g8_spot --m "$r" --n "$r" --k 1024 --tb --a "$f" --b "$f" --moduli "10 12" --rows 64 $st --cw --trunc 36 \
      --tag gp_j16384_range0.02
  done
fi
read -r nocc nvir naux < <(sed -E 's/.*nocc=([0-9]+) nvir=([0-9]+) naux=([0-9]+).*/\1 \2 \3/' "$MP2/meta.txt")
I=150
bin/g8_spot --m "$nvir" --n $(( (I + 1) * nvir )) --k "$naux" --ta --a "$MP2/B.bin@$(( I * nvir * naux ))" \
  --b "$MP2/B.bin" --moduli "10 12" --rows 64 --cw --trunc 32 --tag dfmp2_i$I
echo "spot2 done"

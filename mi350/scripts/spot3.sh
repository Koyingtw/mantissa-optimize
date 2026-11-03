#!/bin/bash
# 纈胺黴素 XC 運算元的逐元素誤差分布（第八輪審查 Q1 補充）：spot2.sh 的 cw_full 是最大值，可能被 |A||B| 接近下溢的元素主導；
# 這裡再數出逐元素誤差 > 1e-9／1e-6／1e-3 的元素數與 |A||B| < 1e-290 的元素數（g8_spot --cw 的 cw9／cw6／cw3／tiny）。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=spot3.sh mi350/slurm/run.sbatch
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
VDIR=${VDIR:-/tmp/valino}
ID=${TAG:-spot3}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
echo "id=$ID host=$(hostname)"
for c in xc4096 xc76800 mo131072; do
  read -r mm nn kk lay < <(sed -E 's/m=([0-9]+) n=([0-9]+) k=([0-9]+) layout=(..)/\1 \2 \3 \4/' "$VDIR/$c/meta.txt")
  t=(); [[ $lay == TN ]] && t=(--ta)
  bin/g8_spot --m "$mm" --n "$nn" --k "$kk" "${t[@]}" --a "$VDIR/$c/A.bin" --b "$VDIR/$c/B.bin" \
    --moduli "10 12 14" --rows 16 --cw --trunc "36 32" --tag "valino_$c"
done
f=/tmp/gp_panels/j16384_range0.02.bin
r=$(( $(stat -c %s "$f") / 8 / 1024 ))
bin/g8_spot --m "$r" --n "$r" --k 1024 --tb --a "$f" --b "$f" --moduli "10 12" --rows 64 --cw --trunc 36 --tag gp_j16384_range0.02
echo "spot3 done"

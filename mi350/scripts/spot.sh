#!/bin/bash
# 事後 FP64 抽查（第七輪審查 P3）：GEMMul8 v3.5.4（INT8，直接 API）算完 C 之後，用原生 DGEMM 重算隨機 ROWS 列
# （n > m 時改抽 ROWS 欄），
# 抽查誤差和整個 C 的誤差比較，並記下跨列範圍檢查（span check）的統計量與抽查的時間成本（apps/g8_spot.hip）。
# 資料：GP 的 trailing-update 面板（kcal.sh 寫在 /tmp/gp_panels）、DF-MP2 的 i = 150 呼叫、兩組合成資料
# （hidden：每一項乘積都在列最大值以下 2^-40，emulation 會整個丟掉；benign：同樣一半元素很小，但對 C 的貢獻也小）。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=spot.sh mi350/slurm/run.sbatch
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh
ROWS=${ROWS:-64}
CASES=${CASES:-"synth mp2 gp"}  # 只跑其中幾組：synth | mp2 | gp
MP2=${MP2:-/tmp/mp2_113_Taxol}
ID=${TAG:-spot}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
echo "id=$ID host=$(hostname) rows=$ROWS"
read -r nocc nvir naux < <(sed -E 's/.*nocc=([0-9]+) nvir=([0-9]+) naux=([0-9]+).*/\1 \2 \3/' "$MP2/meta.txt")
I=150
# 每個 moduli 數各跑一個行程，工作空間放不下時只丟掉那一筆
for M in 10 12 14; do
  if [[ $CASES == *synth* ]]; then
  bin/g8_spot --m 16384 --n 16384 --k 2048 --synthetic hidden:40 --moduli "$M" --rows "$ROWS" --tag synth_hidden
  bin/g8_spot --m 16384 --n 16384 --k 2048 --synthetic benign:40 --moduli "$M" --rows "$ROWS" --tag synth_benign
  fi
  [[ $CASES == *mp2* ]] && bin/g8_spot --m "$nvir" --n $(( (I + 1) * nvir )) --k "$naux" --ta --a "$MP2/B.bin@$(( I * nvir * naux ))" \
    --b "$MP2/B.bin" --moduli "$M" --rows "$ROWS" --tag dfmp2_i$I
  [[ $CASES == *gp* ]] && for f in /tmp/gp_panels/j*_range0.0[28].bin; do
    r=$(( $(stat -c %s "$f") / 8 / 1024 ))
    bin/g8_spot --m "$r" --n "$r" --k 1024 --tb --a "$f" --b "$f" --moduli "$M" --rows "$ROWS" \
      --tag "gp_$(basename "$f" .bin)"
  done
done
echo "spot done"

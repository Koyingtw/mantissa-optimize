#!/bin/bash
# AI 論文 W5：整台節點同一設定的 LoRA 每步速度（ABBA）。job 352 顯示 4 張原生＋4 張截斷混跑時，原生行程會出現
# 週期性的 GPU 閒置（鎖頻也有、只開 2 張卡就沒有），兩組互相影響；實際部署時整台機器是同一設定，所以比較
# 「8 張都原生」與「8 張都截斷」。順序 A B B A（A＝原生 7 位，B＝BITS 位），每段 STEPS 步、不評估。
#   sbatch --export=ALL,SCRIPT=ai_lora_abba.sh mi350/ai/run.sbatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BITS=${BITS:-3}
STEPS=${STEPS:-12}
all() { local r=""; for s in 1 2 3 4 5 6 7 8; do r+="$1:$s "; done; echo "$r"; }
for p in A B B A; do
  b=7
  [[ $p == B ]] && b=$BITS
  STEPS=$STEPS EXTRA="--no-eval" TAG="loraabba_${p}${b}" RUNS="$(all "$b")" ./ai_lora.sh
done
echo "ai_lora_abba done"

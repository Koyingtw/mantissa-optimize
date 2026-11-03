#!/bin/bash
# AI 論文 W5 診斷：LoRA 每步快 16–22% 的來源。原生每個 micro-batch 有約 0.2 s GPU 閒置（功耗 < 600 W、busy < 50），
# 截斷幾乎沒有。三段，各 12 步、不評估（ai_lora.sh）：
#   lock：gfxclk 上限鎖 1300 MHz（功耗遠低於上限、PPT 不作用），8 張 GPU 4 seed × 原生／3 位。閒置消失 → 功耗管理造成。
#   pair：不鎖頻、只開 GPU 0／1（一張原生、一張 3 位），排除 8 個行程搶 CPU。
#   pairswap：同上、兩張互換。
#   sbatch -w p01-r01-n01 --export=ALL,SCRIPT=ai_lora_diag.sh mi350/ai/run.sbatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STEPS=12 EXTRA="--no-eval" TAG=loradiag_lock SCLK=1300 ./ai_lora.sh
STEPS=12 EXTRA="--no-eval" TAG=loradiag_pair RUNS="7:1 3:1" ./ai_lora.sh
STEPS=12 EXTRA="--no-eval" TAG=loradiag_pairswap RUNS="3:1 7:1" ./ai_lora.sh
echo "ai_lora_diag done"

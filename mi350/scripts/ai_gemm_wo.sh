#!/bin/bash
# AI 論文 W1：LLM 形狀的 BF16／FP8 GEMM 在 MI350X 上「只截一個運算元」與「兩個都截」的差別。推論時權重可以在載入時
# 一次截好、不花執行時間，活化值則每一步都要截，所以只截權重有沒有效決定了推論端值不值得做。
# 形狀為 Llama-3-8B 的 MLP（prefill 8192 個 token）：up 8192×14336×4096、down 8192×4096×14336。
# 三種模式各跑一次 e1_gemm_tdp.sh（ABBA、8 張 GPU）：both、only-a、only-b。PyTorch 的 F.linear 在 hipBLASLt 的
# column-major 視角下權重是 A（op T），FP8 bench 也是 TN、A 為 k×m，所以 FP8 的 only-a 對應「只截權重」；
# BF16 bench 是 NN，A 為 m×k（活化值），only-b 才是權重。兩邊都量，分析時再對應。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=ai_gemm_wo.sh mi350/ai/run.sbatch
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CAPS=${CAPS:-"1000 700"}
SHAPES=${SHAPES:-"8192x14336x4096 8192x4096x14336"}
CONFIGS=${CONFIGS:-"bf16:7 bf16:4 bf16:3 bf16:0 fp8:3 fp8:2 fp8:1 fp8:0"}
for mode in both only-a only-b; do
  extra=""
  [[ $mode != both ]] && extra="--$mode"
  CAPS="$CAPS" SHAPES="$SHAPES" CONFIGS="$CONFIGS" WARMUP=6 SECONDS_PER=15 TAG="aiwo_${mode/-/}" EXTRA_ARGS="$extra" \
    scripts/e1_gemm_tdp.sh
done
echo "ai_gemm_wo done"

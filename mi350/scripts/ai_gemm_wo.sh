#!/bin/bash
# AI 論文 W1：LLM 形狀的 BF16／FP8 GEMM 在 MI350X 上「只截權重」與「兩個運算元都截」的差別。推論時權重可以在載入時
# 一次截好、不花執行時間，活化值則每一步都要截，所以只截權重有沒有效決定了推論端值不值得做。
# 形狀為 Llama-3-8B 的 MLP（prefill 8192 個 token）：up 8192×14336×4096、down 8192×4096×14336。
# 哪一個運算元是權重：PyTorch 的 F.linear 在 hipBLASLt 的 column-major 視角下權重是 A（op T）。FP8 bench 是 TN、A 為 k×m，
# 所以 FP8 的「只截權重」= --only-a；BF16 bench 是 NN、A 為 m×k（活化值），BF16 的「只截權重」= --only-b。
# 三段：兩邊都截（BF16＋FP8）、BF16 只截 B、FP8 只截 A，各一次 e1_gemm_tdp.sh（8 GPU × ABBA）。
# job 293 的第一版每個量測窗都等滿 120 s 冷卻（n02 滿載後降不到 62 °C），三種模式要 7 h 以上，改為 START_TEMP 65、
# COOL_MAX 45，並只量對應權重的那一邊。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=ai_gemm_wo.sh mi350/ai/run.sbatch
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CAPS=${CAPS:-"1000 700"}
SHAPES=${SHAPES:-"8192x14336x4096 8192x4096x14336"}
BF=${BF:-"bf16:7 bf16:4 bf16:3 bf16:0"}
F8=${F8:-"fp8:3 fp8:2 fp8:1 fp8:0"}
run() {  # $1 = TAG 後綴, $2 = 設定, $3 = bench 額外參數
  CAPS="$CAPS" SHAPES="$SHAPES" CONFIGS="$2" WARMUP=6 SECONDS_PER=15 START_TEMP=${START_TEMP:-65} COOL_MAX=${COOL_MAX:-45} \
    TAG="aiwo_$1" EXTRA_ARGS="$3" scripts/e1_gemm_tdp.sh
}
run both "$BF $F8" ""
run bf16w "$BF" "--only-b"
run fp8w "$F8" "--only-a"
echo "ai_gemm_wo done"

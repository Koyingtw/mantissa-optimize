#!/bin/bash
# AI 論文：bench 在 LLM 形狀、單一運算元截斷下的快速煙霧測試（每個設定 2 秒，GPU 0）。
#   srun -p gpu -w p01-r01-n02 -t 3 mi350/scripts/ai_smoke.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
for a in "--dtype bf16 --frac 7" "--dtype bf16 --frac 3" "--dtype bf16 --frac 3 --only-b" "--dtype fp8 --frac 3" \
         "--dtype fp8 --frac 1 --only-a"; do
  # shellcheck disable=SC2086
  bin/gemm_mant_bench $a --m 8192 --n 14336 --k 4096 --warmup 1 --seconds 2 |
    grep -o "dtype=[^ ]* frac=[^ ]* only_a=[^ ]* only_b=[^ ]*\|tflops=[^ ]*" | tr "\n" " "
  echo
done

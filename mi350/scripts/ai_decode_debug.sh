#!/bin/bash
# AI 論文 W4：decode_bench 的穩定性與 Python 開銷檢查。job 383：各元件單獨正常；job 384：整個 pass 錄成一個 HIP graph
# 在 replay 時 segfault（單一行程與 8 行程都是）。這段在 GPU 0 依序跑三種模式（eager／layergraph／graph），
# 每種都是 bf16:7、fp8:3 × batch 1／64，比較每次 pass 的時間與是否崩潰。
#   sbatch --export=ALL,SCRIPT=ai_decode_debug.sh mi350/ai/run.sbatch
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1
PY=$HOME/koying/sw/ai-venv/bin/python
for mode in eager layergraph graph; do
  echo "== mode $mode"
  "$PY" -X faulthandler ai/decode_bench.py --device 0 --t0 $(( $(date +%s) + 40 )) --mode "$mode" \
    --model meta-llama/Llama-2-7b-hf --formats "bf16:7 fp8:3" --batches "1 64" --warmup 1 --seconds 3 --gap 2 2>&1 |
    grep -v -i "warn\|Latency not found" | grep -E "^(RESULT|LATE|END|Fatal)|decode_bench.py" | cut -c1-260
  echo "rc=${PIPESTATUS[0]}"
done

#!/bin/bash
# AI 論文 W4：decode_bench 的檢查。job 383：各元件單獨正常；job 384、387：整個 pass 一個 HIP graph 與每層一個 graph
# 都在 replay 時 segfault，eager 正常但 FP8 每次呼叫量化活化值使其比 BF16 慢 4 倍。這段在 GPU 0 以 eager、
# 活化值預先量化、--profile 跑 bf16:7／fp8:3 × batch 1／64／256，比較 GPU kernel 時間與 wall time（是否 GPU-bound）。
#   sbatch --export=ALL,SCRIPT=ai_decode_debug.sh mi350/ai/run.sbatch
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1
PY=$HOME/koying/sw/ai-venv/bin/python
"$PY" -X faulthandler ai/decode_bench.py --device 0 --t0 $(( $(date +%s) + 40 )) --mode eager --profile \
  --model meta-llama/Llama-2-7b-hf --formats "bf16:7 fp8:3" --batches "1 64 256" --warmup 1 --seconds 3 --gap 4 2>&1 |
  grep -v -i "warn\|Latency not found" | grep -E "^(RESULT|PROF|LATE|END|Fatal)|decode_bench.py" | cut -c1-200
echo "rc=${PIPESTATUS[0]}"

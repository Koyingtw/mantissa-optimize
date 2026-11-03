#!/bin/bash
# AI 論文 G1：在 MI350X 上 dump Llama 2 70B（與 GH200 同模型、同 GovReport 資料、同層 0/20/40/79）的 attention q/k/v/dO
# （ai/attn_dump.py，GPU 0），再用 ai/fp8attn_trunc.py 跑朝零截斷與四捨五入（亂數輸入＋四層真實張量）。
#   sbatch --export=ALL,SCRIPT=ai_attn_dump.sh mi350/ai/run.sbatch
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1
PY=$HOME/koying/sw/ai-venv/bin/python
DUMPS=${DUMPS:-$HOME/koying/sw/fp8attn-dumps}
MODEL=${MODEL:-meta-llama/Llama-2-70b-hf}
name=$(basename "$MODEL")
HIP_VISIBLE_DEVICES=0 "$PY" -X faulthandler ai/attn_dump.py --model "$MODEL" --layers 0 20 40 79 --seq 2048 --out "$DUMPS" 2>&1 |
  grep -E "^DUMP|Error|error|Fatal"
echo "dump rc=${PIPESTATUS[0]}"
inputs="randn,${name}_L00_s0,${name}_L20_s0,${name}_L40_s0,${name}_L79_s0"
for r in trunc nearest; do
  ROUND=$r INPUTS=$inputs DUMPS=$DUMPS HIP_VISIBLE_DEVICES=0 "$PY" ai/fp8attn_trunc.py 2>&1 | grep -E "^\||^ROUND|Error|error"
  echo "rc=${PIPESTATUS[0]}"
done

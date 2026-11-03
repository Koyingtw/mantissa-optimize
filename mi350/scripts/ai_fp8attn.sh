#!/bin/bash
# AI 論文 G1：GH200 的 Triton FP8 flash attention（gh200/fp8attn/fp8attn2.py）在 MI350X 上重現「截尾數 × stale delta」，
# 並比較朝零截斷與四捨五入（ai/fp8attn_trunc.py）。GPU 0，INPUTS 預設只有亂數輸入。
#   sbatch --export=ALL,SCRIPT=ai_fp8attn.sh mi350/ai/run.sbatch
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
PY=$HOME/koying/sw/ai-venv/bin/python
for r in trunc nearest; do
  ROUND=$r INPUTS=${INPUTS:-randn} HIP_VISIBLE_DEVICES=0 "$PY" -X faulthandler ai/fp8attn_trunc.py 2>&1 | grep -v -i "warn"
  echo "rc=${PIPESTATUS[0]}"
done

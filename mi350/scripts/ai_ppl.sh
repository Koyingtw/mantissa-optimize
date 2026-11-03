#!/bin/bash
# AI 論文 W3：MI350X 上的 WikiText-2 perplexity（沿用 gh200/infer/ppl_eval.py，格式清單同 GH200 的 infer_ppl.sbatch），
# 每個模型一張 GPU、同時跑。MODELS 用 Hugging Face 名稱（在 HF_HOME 的快取裡，離線讀取）。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=ai_ppl.sh mi350/ai/run.sbatch
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/koying/sw/ai-venv/bin/python
MODELS=${MODELS:-"Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.1 facebook/opt-6.7b Qwen/Qwen3-8B"}
SPECS=${SPECS:-"bf16 bf16:5 bf16:4 bf16:3 bf16:2 bf16:1 bf16:0 bf16:3:trunc bf16:0:trunc fp8 fp8c fp8:2 fp8:1 fp8:0 fp8c:2 fp8c:1 fp8c:0 fp8:2;wa=w fp8:0;wa=w"}
ID=${TAG:-ppl}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
read -ra specs <<<"$SPECS"
read -ra models <<<"$MODELS"
echo "id=$ID host=$(hostname) models=[${models[*]}] specs=[${specs[*]}]"
pids=()
for g in "${!models[@]}"; do
  HIP_VISIBLE_DEVICES=$g "$PY" ../gh200/infer/ppl_eval.py --model "${models[g]}" --tag "id=$ID,gpu=$g" \
    --data-cache "$HF_HOME/datasets" --specs "${specs[@]}" ${EXTRA:-} > "logs/${ID}_gpu$g.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "^RESULT" logs/"${ID}"_gpu*.log
echo "ai_ppl done rc=$rc"

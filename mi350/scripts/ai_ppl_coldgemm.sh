#!/bin/bash
# cold-gemm 交接的補跑（2026-11-03）：7 個模型 × cold-gemm 的整數量化器（ai/ppl_coldgemm.py，複製自 cold-gemm/llm/ppl.py，
# 只改 OPT 的 decoder 層位置）的 WikiText-2 perplexity，每個模型一張 GPU（0–6）；GPU 7 跑 INT8 GEMM 可用性探測（ai/int8_probe.py）。
# 輸出：results/ppl_coldgemm/<模型>/ppl.csv 與 results/ppl_coldgemm/all.csv。
#   sbatch --export=ALL,SCRIPT=ai_ppl_coldgemm.sh mi350/ai/run.sbatch
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/koying/sw/ai-venv/bin/python
SCHEMES=${SCHEMES:-bf16,w8sym,w7sym,w7asym,w8sym-a8sym,w7sym-a8sym,w7asym-a7asym}
MODELS=${MODELS:-"meta-llama/Llama-3.1-8B meta-llama/Llama-2-7b-hf Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.1 facebook/opt-6.7b Qwen/Qwen3-8B meta-llama/Llama-2-70b-hf"}
ID=${TAG:-pplcg}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
read -ra models <<<"$MODELS"
echo "id=$ID host=$(hostname) schemes=$SCHEMES models=[${models[*]}]"
pids=()
for g in "${!models[@]}"; do
  m=${models[g]}
  HIP_VISIBLE_DEVICES=$g "$PY" ai/ppl_coldgemm.py --model "$m" --schemes "$SCHEMES" \
    --out "results/ppl_coldgemm/$(basename "$m")" > "logs/${ID}_gpu$g.log" 2>&1 &
  pids+=($!)
done
HIP_VISIBLE_DEVICES=7 "$PY" ai/int8_probe.py > "logs/${ID}_int8probe.log" 2>&1 &
pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
.venv/bin/python - <<'EOF'
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/ppl_coldgemm/*/ppl.csv"))])
b = d[d.scheme == "bf16"].set_index("model").ppl
d["delta_pct"] = 100 * (d.ppl / d.model.map(b) - 1)
d.to_csv("results/ppl_coldgemm/all.csv", index=False)
print(d.pivot_table(index="scheme", columns="model", values="delta_pct").round(2).to_string())
EOF
grep -v -i warn "logs/${ID}_int8probe.log"
echo "ai_ppl_coldgemm done"

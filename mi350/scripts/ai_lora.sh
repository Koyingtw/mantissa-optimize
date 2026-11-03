#!/bin/bash
# AI 論文 W5：凍結權重截斷的 LoRA 微調（ai/lora.py），每張 GPU 一個獨立訓練，8 張同時跑，功耗取樣器記錄全程。
# RUNS：每張 GPU 一組「位元:seed」，依 GPU 0..7 順序（預設 4 個 seed × 原生 7／截斷 3 位，同 seed 的兩者放在相鄰 GPU；
# 第二次執行時用 SWAP=1 交換原生與截斷的 GPU，抵銷個別 GPU 的差異）；給少於 8 組就只用前幾張 GPU。
# 其他環境變數：MODEL、STEPS、TARGET、EXTRA、TAG、CAP、SCLK（設定時把 8 張 GPU 的 gfxclk 上限鎖在 SCLK MHz，結束時還原）。
#   sbatch -w p01-r01-n01 --export=ALL,SCRIPT=ai_lora.sh,MODEL=meta-llama/Llama-2-7b-hf mi350/ai/run.sbatch
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/koying/sw/ai-venv/bin/python
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
RUNS=${RUNS:-"7:1 3:1 7:2 3:2 7:3 3:3 7:4 3:4"}
STEPS=${STEPS:-1024}
CAP=${CAP:-1000}
SCLK=${SCLK:-}
ID=${TAG:-lora}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
restore() {
  tools/pcap.sh 1000 || true
  if [[ -n $SCLK ]]; then for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; fi
  kill "${SMI_PID:-0}" 2>/dev/null || true
}
trap restore EXIT
tools/pcap.sh "$CAP"
if [[ -n $SCLK ]]; then for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done; fi
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!
read -ra runs <<<"$RUNS"
if [[ ${SWAP:-0} == 1 ]]; then  # 相鄰兩張 GPU 互換
  for ((i = 0; i + 1 < ${#runs[@]}; i += 2)); do t=${runs[i]}; runs[i]=${runs[i+1]}; runs[i+1]=$t; done
fi
echo "id=$ID host=$(hostname) model=$MODEL cap=$CAP sclk=${SCLK:-none} steps=$STEPS runs=[${runs[*]}] extra=[${EXTRA:-}]"
pids=()
for g in "${!runs[@]}"; do
  IFS=: read -r bits seed <<<"${runs[g]}"
  "$PY" ai/lora.py --model "$MODEL" --bits "$bits" --seed "$seed" --device "$g" --max-steps "$STEPS" \
    --target "${TARGET:-0}" --tag "gpu=$g,bits=$bits,seed=$seed" ${EXTRA:-} > "logs/${ID}_gpu$g.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h -E "^(START|CONVERGED|END)" logs/"${ID}"_gpu*.log
echo "ai_lora done rc=$rc"

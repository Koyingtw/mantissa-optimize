#!/bin/bash
# AI 論文 W4：decode 形態的權重串流（ai/decode_bench.py）的每秒 token 數與每 token 能量。8 張 GPU 各一個行程，
# 依共同的起始時間 T0 走同一份時間表，所以整台節點在每個量測窗都是同一種格式；每個 batch 內格式先正序再反序（ABBA）。
# 環境變數：MODEL、FORMATS、BATCHES、CAPS（每個上限跑一次整份時間表）、WARMUP、MEAS（每窗量測秒數）、GAP、TAG、
# LEAD（載入權重的預留秒數）、EXTRA（傳給 decode_bench，例如 --profile）。
#   sbatch --export=ALL,SCRIPT=ai_decode.sh mi350/ai/run.sbatch
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source env.sh >/dev/null
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf} HF_HUB_OFFLINE=1
PY=$HOME/koying/sw/ai-venv/bin/python
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
FORMATS=${FORMATS:-"bf16:7 bf16:4 bf16:3 fp8:3 fp8:2"}
BATCHES=${BATCHES:-"1 16 64 256"}
CAPS=${CAPS:-"1000 600"}
WARMUP=${WARMUP:-5}
SECONDS_PER=${MEAS:-15}  # 不用 SECONDS：那是 bash 的內建變數
GAP=${GAP:-10}
LEAD=${LEAD:-90}
ID=${TAG:-decode}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
restore() { tools/pcap.sh 1000 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!
echo "id=$ID host=$(hostname) model=$MODEL formats=[$FORMATS] batches=[$BATCHES] caps=[$CAPS]"
for cap in $CAPS; do
  tools/pcap.sh "$cap"
  t0=$(( $(date +%s) + LEAD ))
  echo "CAP $cap t0=$t0"
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    "$PY" ai/decode_bench.py --model "$MODEL" --device "$g" --t0 "$t0" --formats "$FORMATS" --batches "$BATCHES" \
      --warmup "$WARMUP" --seconds "$SECONDS_PER" --gap "$GAP" --tag "cap=$cap,gpu=$g" ${EXTRA:-} > "logs/${ID}_cap${cap}_gpu$g.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  grep -h "^LATE" logs/"${ID}"_cap"${cap}"_gpu*.log | head -5 || true
done
echo "ai_decode done"

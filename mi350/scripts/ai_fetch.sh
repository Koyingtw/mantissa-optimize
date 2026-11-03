#!/bin/bash
# AI 論文：下載開放模型與 WikiText-2 到 HF_HOME（預設 ~/koying/sw/hf，兩節點共用家目錄）。
# Llama 系列是 gated，需要先 `hf auth login`；MODELS 可以覆寫清單。
#   mi350/scripts/ai_fetch.sh
set -euo pipefail
export HF_HOME=${HF_HOME:-$HOME/koying/sw/hf}
HF=$HOME/koying/sw/ai-venv/bin/hf
MODELS=${MODELS:-"Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.1 facebook/opt-6.7b Qwen/Qwen3-8B"}
for m in $MODELS; do
  # 只取 safetensors；OPT 只有 PyTorch .bin（另有 Flax、TF 權重，不取）
  if [[ $m == facebook/opt* ]]; then skip=("*.msgpack" "*.h5"); else skip=("*.bin" "*.pth" "*.gguf" "original/*"); fi
  args=()
  for s in "${skip[@]}"; do args+=(--exclude "$s"); done  # 每個樣式各一個 --exclude（多值會被當成檔名）
  "$HF" download "$m" "${args[@]}" >/dev/null && echo "ok $m"
done
"$HF" download --repo-type dataset Salesforce/wikitext --include "wikitext-2-raw-v1/*" >/dev/null && echo "ok wikitext-2"
du -sh "$HF_HOME"

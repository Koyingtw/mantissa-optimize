#!/bin/bash
# 建立「獨立」的 nvidia_impl 工作副本並加上 attention dump hook。
# 生產用的 /nfs/home/koying/mlperf-llama2-70b-lora/nvidia_impl 與 SC26-MLPerf/nvidia/ 一律不動
# （15-seed nv-lora 驗證實驗正在用它們）。
#
#   gh200/fp8attn/prepare_work.sh            # 預設 WORK=/nfs/home/koying/mlperf-fp8attn-work
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-/nfs/home/koying/mlperf-llama2-70b-lora/nvidia_impl}"
WORK="${WORK:-/nfs/home/koying/mlperf-fp8attn-work}"
DST="$WORK/nvidia_impl"

mkdir -p "$WORK/runs"
rsync -a --delete --exclude '__pycache__' --exclude 'nemo_experiments' --exclude 'outputs' "$SRC/" "$DST/"
cp "$HERE/attn_dump.py" "$DST/attn_dump.py"

python3 - "$DST/train.py" <<'EOF'
import sys
p = sys.argv[1]
t = open(p).read()
anchor = "    trainer.callbacks.append(custom_callback)\n"
block = (anchor +
         "    if os.environ.get(\"FP8ATTN_DUMP\", \"0\") == \"1\":  # fp8attn dump hook (gh200/fp8attn)\n"
         "        import attn_dump\n"
         "        trainer.callbacks.append(attn_dump.make_callback())\n")
if "attn_dump" not in t:
    assert t.count(anchor) == 1, "anchor not found exactly once"
    t = t.replace(anchor, block, 1)
    open(p, "w").write(t)
    print("patched", p)
else:
    print("already patched", p)
EOF
diff -r -q "$SRC" "$DST" -x __pycache__ -x nemo_experiments -x outputs || true

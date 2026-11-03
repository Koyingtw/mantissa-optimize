#!/bin/bash
# 驗證 GEMMul8 的 2^32 修正（branch fix/hip-strided-batched-2e32，clone 在 worktree 的 gemmul8-fix/）：
# 原版 3.5.4 與修正版，(1) 直接 API（bin/g8_repro_354／g8_repro_fix），(2) LD_PRELOAD hook（gemm bench --tb --check），
# 兩者都在 m=n 跨過 65536（m·n 跨過 2^32）。hook 的 TFLOP/s 同時看修正是否變慢。1000 W，單張 GPU。
#   srun -w p01-r01-n02 -N1 mi350/scripts/g8_fix_check.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null 2>&1
STOCK=$HOME/koying/sw/GEMMul8-v3.5.4/lib
FIX=$(cd .. && pwd)/gemmul8-fix/lib
SIZES=${SIZES:-"65535 65536 65537 78976"}
DEV=${DEV:-0}
mkdir -p logs
OUT=logs/g8fix_$(date +%m%d_%H%M).out
{
echo "== direct API, stock 3.5.4"
bin/g8_repro_354 --device "$DEV" --moduli 14 --k 1024 --sizes "$SIZES"
echo "== direct API, fixed"
bin/g8_repro_fix --device "$DEV" --moduli 14 --k 1024 --sizes "$SIZES"
for lib in stock fix; do
  L=$STOCK; [[ $lib == fix ]] && L=$FIX
  for k in 1024 4096; do
    for m in 65535 65536; do
      echo "== hook $lib m=n=$m k=$k"
      LD_PRELOAD=$L/libgemmul8.so GEMMUL8_BACKEND_GEMM=INT8 GEMMUL8_NUM_MOD_D_GEMM=12 GEMMUL8_FASTMODE_D_GEMM=1 \
        bin/gemm_mant_bench --device "$DEV" --m "$m" --n "$m" --k "$k" --tb --warmup 3 --seconds 10 --check --tag "hook_$lib" \
        2>&1 | grep -E "^RESULT|rror" | grep -o -E "tag=[^ ]+|m=[0-9]+|k=[0-9]+|tflops=[^ ]+|rel_err=[^ ]+" | tr '\n' ' '
      echo
    done
  done
done
} > "$OUT" 2>&1
cat "$OUT"

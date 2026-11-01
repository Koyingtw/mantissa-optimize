#!/bin/bash
# GEMMul8 失效定位：C = A B^T（k=1024，14 moduli），m = n 跨過 65536（m*n 跨過 2^32）。
# (1) 直接呼叫 gemmul8::gemm（bin/g8_repro，不經 hook）；(2) 經 LD_PRELOAD hook 的 hipblasDgemm（gemm bench --tb --check）。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
SIZES=${SIZES:-"63976 65000 65535 65536 65537 66000 70000 78976"}
# G8V=354：改測 GEMMul8 v3.5.4（bin/g8_repro_354 與 ~/koying/sw/GEMMul8-v3.5.4/lib）
REPRO=bin/g8_repro; G8LIB=$SW/gemmul8/lib
if [[ ${G8V:-} == 354 ]]; then REPRO=bin/g8_repro_354; G8LIB=$HOME/koying/sw/GEMMul8-v3.5.4/lib; fi
mkdir -p logs
OUT=logs/g8thr${G8V:-}_$(date +%m%d_%H%M).out
{
echo "== direct API"
$REPRO --moduli 14 --k 1024 --sizes "$SIZES"
echo "== hook (hipblasDgemm N,T)"
for m in $SIZES; do
  LD_PRELOAD=$G8LIB/libgemmul8.so GEMMUL8_BACKEND_GEMM=INT8 GEMMUL8_NUM_MOD_D_GEMM=14 GEMMUL8_FASTMODE_D_GEMM=1 \
    bin/gemm_mant_bench --m "$m" --n "$m" --k 1024 --tb --warmup 0 --seconds 1 --check --tag hook 2>&1 | grep -E "^RESULT|rror"
done
} > "$OUT" 2>&1
cat "$OUT"

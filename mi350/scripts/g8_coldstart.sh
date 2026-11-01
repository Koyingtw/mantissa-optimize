#!/bin/bash
# GEMMul8 每個新形狀的一次性成本，是版本還是呼叫方式造成的？DF-MP2（taxol，226 種形狀）在 1000 W 下，同一行程跑兩次
# （pass A 含首次呼叫、pass B 為重複）：GPU 0 = 3.2.0 直接 API（bin/dfmp2 e12），GPU 1 = 3.2.0 hook，
# GPU 2 = 3.5.4 直接 API（bin/dfmp2_354 e12），GPU 3 = 3.5.4 hook。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
D=${MP2_DIR:-/tmp/mp2_113_Taxol}
eval "$(tr ' ' '\n' < "$D/meta.txt" | grep -E '^(nocc|nvir|naux)=' | tr a-z A-Z)"
OUT=logs/g8cold_$(date +%m%d_%H%M).out
tools/pcap.sh 1000 >/dev/null
HK=(GEMMUL8_BACKEND_GEMM=INT8 GEMMUL8_NUM_MOD_D_GEMM=12 GEMMUL8_FASTMODE_D_GEMM=1 GEMMUL8_MAX_NUM_MOD_GEMM=12
    GEMMUL8_MAX_M_GEMM=$NVIR GEMMUL8_MAX_N_GEMM=$((NOCC * NVIR)) GEMMUL8_MAX_K_GEMM=$NAUX)
{
bin/dfmp2 --dir "$D" --device 0 --variants "e12 e12" --tag v320direct &
env LD_PRELOAD=$SW/gemmul8/lib/libgemmul8.so "${HK[@]}" bin/dfmp2 --dir "$D" --device 1 --variants "native native" --tag v320hook &
bin/dfmp2_354 --dir "$D" --device 2 --variants "e12 e12" --tag v354direct &
env LD_PRELOAD=$HOME/koying/sw/GEMMul8-v3.5.4/lib/libgemmul8.so "${HK[@]}" bin/dfmp2 --dir "$D" --device 3 --variants "native native" --tag v354hook &
wait
} > "$OUT" 2>&1
grep RESULT_MP2 "$OUT"

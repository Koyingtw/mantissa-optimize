#!/bin/bash
# FP64 HPL 三方比較（原生／截斷 36／GEMMul8 10 moduli）的小尺寸冒煙測試：NB=1024 P1Q8 與 NB=4096 P4Q2，1000 W
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BENCH=hpl CAPS=1000 REPEAT=1 VARIANTS="53:53 36:36 g8m10" N=131072 NB=1024 P=1 Q=8 TAG=hpl3smoke1024 scripts/hpl_tdp.sh
BENCH=hpl CAPS=1000 REPEAT=1 VARIANTS="53:53 36:36 g8m10" N=131072 NB=4096 P=4 Q=2 TAG=hpl3smoke4096 scripts/hpl_tdp.sh
for f in logs/hpl3smoke*.out; do echo "== $f"; grep -E "=== RUN|^WC|PASSED|FAILED" "$f"; done

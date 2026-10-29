#!/bin/bash
# 第一批正式實驗的佇列（依序執行，任一失敗不擋後面）：E1b → E2 HPL-MxP → E3 FP64 HPL → E5 GP
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "queue start $(date)"
scripts/e1b_fixedclk.sh;                                         echo "e1b rc=$? $(date)"
BENCH=mxp VARIANTS="native 4 3" REPEAT=2 scripts/hpl_tdp.sh;    echo "mxp rc=$? $(date)"
BENCH=hpl VARIANTS="53:53 36:36" REPEAT=2 scripts/hpl_tdp.sh;   echo "hpl rc=$? $(date)"
scripts/gp_tdp.sh;                                               echo "gp rc=$? $(date)"
echo "queue done $(date)"

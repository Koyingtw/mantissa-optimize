#!/bin/bash
# E1b（MI350X）機制對照：把 gfxclk 上限鎖在 SCLK MHz（低於任何設定在 1000 W 下的受限時脈），讓每個設定都跑在
# 同一個時脈、都不受 PPT 限制，功耗差就直接是切換能量的差。被丟掉的位元填 0（截斷）、填 1、或填固定圖樣：
# 填 1／固定圖樣同樣讓那些位元不再翻轉、但值不是 0，用來分開「不翻轉」與「值為零」兩種效應。
#   nohup mi350/scripts/e1b_fixedclk.sh >/dev/null 2>&1 &
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCLK=${SCLK:-1300}
restore_clk() { for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; }
trap restore_clk EXIT
for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done
CAPS=1000 TAG=${TAG:-e1b_clk$SCLK} \
CONFIGS=${CONFIGS:-"fp64:52 fp64:27 fp64:27:ones fp64:27:const fp64:0 fp64:0:ones fp64:0:const bf16:7 bf16:0 bf16:0:ones bf16:0:const fp8:3 fp8:0 fp8:0:ones fp8:0:const"} \
  "$ROOT/scripts/e1_gemm_tdp.sh"

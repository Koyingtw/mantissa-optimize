#!/bin/bash
# 取樣器與程式內能量計數器的交叉驗證（第十輪審查 S1、S4）：在 gfxclk 上限鎖 SCLK MHz（預設 1300）、1000 W 下，同一次執行
# 同時有 100 ms 的 amdsmi 取樣器（各腳本自己開）與程式內在窗口／階段邊界讀的硬體能量計數器：
#   GEMM（e1_gemm_tdp.sh，bench --energy）：FP64 52／36／27／0 位；
#   GP（gp_tdp.sh，GP_ENERGY=1）與 DF-MP2（mp2_tdp.sh，DFMP2_ENERGY=1）：整支程式與 GEMM 階段的能量。
#   sbatch -w p01-r01-n02 --export=ALL,SCRIPT=xval.sh mi350/slurm/run.sbatch
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SCLK=${SCLK:-1300}
restore_clk() { for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; }
trap restore_clk EXIT
for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done
CAPS=1000 TAG=xvalgemm CONFIGS="fp64:52 fp64:36 fp64:27 fp64:0" EXTRA_ARGS="--energy" scripts/e1_gemm_tdp.sh
GP_ENERGY=1 CAPS=1000 REPEAT=1 TAG=xvalgp scripts/gp_tdp.sh
DFMP2_ENERGY=1 CAPS=1000 REPEAT=1 EMU="" VARIANTS="native t32n t20n t20n t32n native" TAG=xvalmp2 scripts/mp2_tdp.sh
echo "xval done"

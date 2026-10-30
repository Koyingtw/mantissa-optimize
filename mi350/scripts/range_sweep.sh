#!/bin/bash
# E8（MI350X）指數範圍：運算元每個元素（elem）或每列／每欄（row）的指數散佈 S binades 時，截斷與 GEMMul8
# （Ozaki-II，逐列縮放成整數）的精度。只看精度，1000 W、短量測；每個 SPREADS 項目跑一次 e1_gemm_tdp.sh。
#   sbatch -w <node> --export=ALL,SCRIPT=range_sweep.sh mi350/slurm/run.sbatch
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SPREADS=${SPREADS:-"0/elem 16/elem 32/elem 64/elem 32/row 64/row"}
for spr in $SPREADS; do
  CAPS=1000 TAG=range_${spr%/*}${spr#*/} SHAPES="8192x8192x4096" WARMUP=1 SECONDS_PER=3 COOL_MAX=0 \
    CONFIGS="fp64:52 fp64:35 fp64:27 g8:10 g8:12 g8:14" \
    EXTRA_ARGS="--round nearest --check --spread ${spr%/*} --spread-mode ${spr#*/}" scripts/e1_gemm_tdp.sh
done

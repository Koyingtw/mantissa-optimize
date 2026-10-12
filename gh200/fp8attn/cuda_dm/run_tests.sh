#!/bin/bash
# 第 1 關測試：randn + 4 個真實 dump（S=2048、8q/1kv）
cd /nfs/home/koying/mantissa-optimize/gh200/fp8attn/cuda_dm
export FP8DM_NO_BUILD=1
DU=/nfs/home/koying/fp8attn-dumps
echo "== randn"; timeout 240 python test_cuda_dm.py 2>&1 | grep -E '^\||PASS|FAIL|Error|error'
for f in ${DUMPS:-L40_step6 L00_step10 L79_step30 L20_step3}; do echo "== $f"; INPUT=dump:$DU/$f.pt timeout 240 python test_cuda_dm.py 2>&1 | grep -E '^\||PASS|FAIL|Error|error'; done

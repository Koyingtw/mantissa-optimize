#!/bin/bash
# 模型的外部驗證：DF-MP2 與 GP 在 gfxclk 上限鎖 SCLK MHz（預設 1300）、1000 W 下的功耗與有效時脈，算出 R、σ，
# 預測它們在各上限的加速（analysis/model_apps.py），與 mp2_tdp／gp_tdp 的實測比較。兩者都沒有參與模型的建立。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SCLK=${SCLK:-1300}
restore_clk() { for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi reset -g "$g" -c >/dev/null 2>&1 || true; done; }
trap restore_clk EXIT
for g in 0 1 2 3 4 5 6 7; do sudo -n amd-smi set -g "$g" -L sclk max "$SCLK" >/dev/null; done
CAPS=1000 REPEAT=1 EMU="" VARIANTS="native t32n t20n t20n t32n native" TAG=mp2lock$SCLK scripts/mp2_tdp.sh
CAPS=1000 REPEAT=1 TAG=gplock$SCLK scripts/gp_tdp.sh

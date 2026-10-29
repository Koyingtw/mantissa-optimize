#!/bin/bash
# 設定 MI350X 的 PPT0（socket）功耗上限，並讀回確認。
#   pcap.sh <W> [gpu ...]     不給 gpu 就設 0–7 全部；W 介於 1–1000（預設上限 1000 W）
#   pcap.sh show              印出每張卡目前的上限
set -euo pipefail
show() { for c in /sys/class/drm/card*/device; do [[ -d $c/hwmon ]] || continue
  echo "$(basename "$(readlink -f "$c")") $(( $(cat "$c"/hwmon/hwmon*/power1_cap) / 1000000 ))W"; done | sort; }
if [[ ${1:-} == show ]]; then show; exit 0; fi
W=${1:?usage: pcap.sh <W> [gpu ...]}
shift
GPUS=("$@")
(( ${#GPUS[@]} )) || GPUS=(0 1 2 3 4 5 6 7)
(( W >= 1 && W <= 1000 )) || { echo "W 必須在 1–1000" >&2; exit 2; }
for g in "${GPUS[@]}"; do sudo -n amd-smi set -g "$g" -o PPT0 "$W" >/dev/null; done
# 讀回：sysfs power1_cap 與 BDF 排序後的 amd-smi 序一致
mapfile -t caps < <(show | awk '{print $2}')
for g in "${GPUS[@]}"; do [[ ${caps[$g]} == "${W}W" ]] || { echo "GPU $g 上限讀回 ${caps[$g]}，預期 ${W}W" >&2; exit 1; }; done

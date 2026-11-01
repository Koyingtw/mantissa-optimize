#!/bin/bash
# 兩節點（n01+n02，16 張 GPU）的 HPL-MxP × PPT0 上限 × 保留位元（審稿第五輪 M6）。以 sbatch -N2 送出：
#   sbatch -N2 -w p01-r01-n01,p01-r01-n02 --export=ALL,SCRIPT=hpl2_tdp.sh mi350/slurm/run.sbatch
# 兩個節點都設上限、都跑 smi_sampler（之後合併成 <ID>_smi.csv，n02 的 GPU 編號 +8）；MPI/RCCL 設定見 env_multinode.sh。
# 環境變數：CAPS、VARIANTS（native|3|…）、N、NB、P、Q、REPEAT、START_TEMP、COOL_MAX、TAG
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$ROOT/.." && pwd)
cd "$ROOT"
source env.sh
CAPS=${CAPS:-"1000 500"}
VARIANTS=${VARIANTS:-"native 3"}
N=${N:-1458560}; NB=${NB:-5504}; P=${P:-4}; Q=${Q:-4}
REPEAT=${REPEAT:-2}
START_TEMP=${START_TEMP:-63}
COOL_MAX=${COOL_MAX:-120}
ID=${TAG:-mxp2n}_$(date +%m%d_%H%M)
exec >"logs/$ID.out" 2>&1
on_both() { srun -N2 --ntasks-per-node=1 --overlap "$@"; }
restore() {
  on_both tools/pcap.sh 1000 >/dev/null 2>&1 || true
  on_both pkill -f "smi_sampler.py logs/${ID}_" 2>/dev/null || true
  # 合併兩個節點的取樣（n02 的 gpu 欄 +8）
  .venv/bin/python - "$ID" <<'PY' || true
import sys, pandas as pd
i = sys.argv[1]
a = pd.read_csv(f"logs/{i}_smi_p01-r01-n01.csv"); b = pd.read_csv(f"logs/{i}_smi_p01-r01-n02.csv"); b["gpu"] += 8
pd.concat([a, b]).sort_values("t").to_csv(f"logs/{i}_smi.csv", index=False)
PY
}
trap restore EXIT
srun -N2 --ntasks-per-node=1 --overlap bash -c "cd $ROOT && exec .venv/bin/python tools/smi_sampler.py logs/${ID}_smi_\$(hostname).csv 0.1" &
hot_max() { on_both bash -c 'm=0; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp2_input; do t=$(( $(cat $f) / 1000 )); (( t > m )) && m=$t; done; echo $m' | sort -n | tail -1; }
run_variant() {
  srun -N2 --ntasks-per-node=8 --mpi=pmix --overlap bash -c "
    source $ROOT/env.sh >/dev/null 2>&1; source $ROOT/env_multinode.sh
    export HPLMXP_TRAILING_HIPBLAS=1 HPLMXP_IR_RICHARDSON=1 HPLMXP_PRECOND_FP32=1 HPLMXP_PROF=0 HPLMXP_PROGRESS=0
    export HPLMXP_ROCTX=0 HPLMXP_TRAILING_TN=1 HPLMXP_TN_DIRECT_PANEL=1
    M=23106754; export ROCR_VISIBLE_DEVICES=\${M:\$SLURM_LOCALID:1}
    exec $REPO/rocHPL-MxP/build_mi350_bf16_mant$1/xhplmxp-mi350 --n $N --nb $NB --p $P --q $Q --fp8 0 --itcheck 1 --approx-diag"
}
echo "id=$ID bench=mxp nodes=2 N=$N NB=$NB P=$P Q=$Q caps=[$CAPS] variants=[$VARIANTS] repeat=$REPEAT"
read -ra var <<<"$VARIANTS"
rev=(); for ((i = ${#var[@]} - 1; i >= 0; i--)); do rev+=("${var[i]}"); done
sleep 5
for cap in $CAPS; do
  on_both tools/pcap.sh "$cap" | head -2
  for ((r = 0; r < REPEAT; r++)); do
    for pass in "A$r" "B$r"; do
      list=("${var[@]}"); [[ $pass == B* ]] && list=("${rev[@]}")
      for v in "${list[@]}"; do
        waited=0
        while (( $(hot_max) > START_TEMP && waited < COOL_MAX )); do sleep 5; waited=$((waited + 5)); done
        echo "=== RUN bench=mxp cap=$cap blk=0 pass=$pass variant=$v t=$(date +%s.%N) cool=${waited}s hot=$(hot_max)"
        rc=0; run_variant "$v" || rc=$?
        echo "=== END bench=mxp cap=$cap blk=0 pass=$pass variant=$v t=$(date +%s.%N) rc=$rc"
      done
    done
  done
done
echo "mxp2n done"

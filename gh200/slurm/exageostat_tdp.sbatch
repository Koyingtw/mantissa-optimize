#!/bin/bash
# Real ExaGeoStatCPP (StarPU + Chameleon + cuBLAS) MLE under module power caps, with the mantissa shim on its
# FP64 tile GEMMs. One native data-generation run writes the synthetic data once; every measured run then reads
# that same CSV (--datapath), so truncation never touches the data, only the likelihood evaluations.
#   sbatch --export=ALL,TDPS="1000 600 450",N=60000,DTS=1920,ITERS=6 slurm/exageostat_tdp.sbatch
# Variants (VARIANTS, run in the given order per cap; repeat a name for ABBA):
#   native = no shim; copy = shim copying operands without truncation (MANT_FORCE_COPY=1, the fair baseline);
#   t<k><n|t> = shim keeping k mantissa bits, round-to-nearest (n) or truncation (t)
#SBATCH -J exa-tdp
#SBATCH -p compute
#SBATCH -N 1 -c 72
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -o logs/exageostat/%x_%j.out
#SBATCH -e logs/exageostat/%x_%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
HOST=$(hostname)
D=$(cd "$ROOT/../apps/exageostat-deps" && pwd)
I=$D/install
BIN=$D/build/exageostatcpp/examples/end-to-end/Example_Data_Generation_and_Modeling
SHIM=$ROOT/shim/libmantshim.span.so
export LD_LIBRARY_PATH=$I/OPENBLAS/lib:$I/HWLOC/lib:$I/GSL/lib:$I/NLOPT/lib:$I/STARPU/lib:$I/CHAMELEON/lib:/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}
export STARPU_HOME=$D/starpu-home STARPU_HOSTNAME=gh200 STARPU_SCHED=dmdas OPENBLAS_NUM_THREADS=1 STARPU_SILENT=1
TDPS=${TDPS:-"1000"}
VARIANTS=${VARIANTS:-"native copy t36n t28n t28n t36n copy native"}
N=${N:-60000}; DTS=${DTS:-1920}; ITERS=${ITERS:-6}; CORES=${CORES:-64}
ARGS="--N=$N --dts=$DTS --kernel=univariate_matern_stationary --computation=exact --precision=double --cores=$CORES --gpus=1 \
 --initial_theta=1:0.1:0.5 --lb=0.1:0.1:0.1 --ub=5:5:5 --tolerance=9 --seed=0 --verbose=detailed"
OUT="$ROOT/logs/exageostat/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit \
  --format=csv,noheader,nounits -lms 250 >"$OUT/smi.csv" &
SMI_PID=$!
wait_cool() { local t=0; while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -gt 45 ] && [ $t -lt 180 ]; do sleep 5; t=$((t+5)); done; }
echo "job=$SLURM_JOB_ID host=$HOST N=$N dts=$DTS iters=$ITERS variants=[$VARIANTS] tdps=[$TDPS]"

# 1) the data, once, natively (1000 W)
mkdir -p "$OUT/gen" && (cd "$OUT/gen" && "$BIN" $ARGS --max_mle_iterations=1 >gen.out 2>gen.err)
cp "$OUT/gen/configurations/synthetic_ds/SYN_${N}_1" "$OUT/data.csv" || { echo "=== no data file"; exit 1; }
echo "data: $(wc -l <"$OUT/data.csv") lines"

# 2) the measured runs
declare -A seen
for tdp in $TDPS; do
  nv_power --hosts "$HOST" "$tdp" >/dev/null
  for v in $VARIANTS; do
    seen[$tdp$v]=$(( ${seen[$tdp$v]:-0} + 1 )); pas=$(echo ABCDEFGH | cut -c${seen[$tdp$v]})
    wait_cool
    lim=$(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader,nounits | cut -d. -f1)
    if [ "$lim" != "$tdp" ]; then echo "CAP RESET detected ($lim) -> re-applying $tdp W"; nv_power --hosts "$HOST" "$tdp" >/dev/null; fi
    unset LD_PRELOAD MANT_BITS MANT_ROUND MANT_FORCE_COPY
    case $v in
      native) env_pre="" ;;
      copy) env_pre="LD_PRELOAD=$SHIM MANT_FORCE_COPY=1" ;;
      t*) k=${v:1:${#v}-2}; r=${v: -1}; env_pre="LD_PRELOAD=$SHIM MANT_BITS=$k MANT_ROUND=$([ "$r" = n ] && echo nearest || echo trunc)" ;;
    esac
    f="$OUT/tdp${tdp}_${v}_${pas}"
    mkdir -p "$f.d"
    t0=$(date +%s.%N)
    (cd "$f.d" && env $env_pre "$BIN" $ARGS --max_mle_iterations=$ITERS --datapath="$OUT/data.csv" >"$f.out" 2>"$f.err")
    rc=$?
    t1=$(date +%s.%N)
    facto=$(grep -o "Facto Time: [0-9.]*" "$f.out" | awk '{print $3}' | paste -sd' ')
    echo "RESULT_EXA tdp=$tdp variant=$v pass=$pas rc=$rc facto=[$facto] t_begin=$t0 t_end=$t1"
  done
done
echo "=== ALL DONE"

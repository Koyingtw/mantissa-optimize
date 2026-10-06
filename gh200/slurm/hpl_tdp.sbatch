#!/bin/bash
# E2/E3: end-to-end HPL (FP64, rocHPL CUDA port) or HPL-MxP (BF16) on GH200,
# one rank per node, across module TDP caps and mantissa settings.
#   sbatch -N 8 --export=ALL,BENCH=hpl  slurm/hpl_tdp.sbatch
#   sbatch -N 8 --export=ALL,BENCH=mxp  slurm/hpl_tdp.sbatch
# REPEAT=r gives r ABBA blocks per cap (passes A/B, C/D, ...); WARMUP=1 adds one uncounted run first (pass W).
# Each TDP level runs the variant list forward then backward (ABBA).
#SBATCH -J hpl-tdp
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
REPO=$(cd "$ROOT/.." && pwd)
module purge
module load nvhpc/26.5
NH=/software/nvhpc/Linux_aarch64/26.5
export LD_LIBRARY_PATH="$NH/comm_libs/nccl/lib:$NH/math_libs/13.2/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_hwloc_base_binding_policy=none

BENCH=${BENCH:-hpl}
NNODE=$SLURM_JOB_NUM_NODES
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
TDPS=${TDPS:-"1000 700 500"}
REPEAT=${REPEAT:-1}
COOLDOWN=${COOLDOWN:-30}   # idle seconds before each run, limits thermal drift between variants

restore() { nv_power --hosts "$HOSTS" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT

case "$NNODE" in
  1) P=1; Q=1 ;; 2) P=1; Q=2 ;; 4) P=2; Q=2 ;; 8) P=2; Q=4 ;; 12) P=3; Q=4 ;; 16) P=4; Q=4 ;;
  *) echo "unsupported node count $NNODE" >&2; exit 2 ;;
esac
P=${GRID_P:-$P}; Q=${GRID_Q:-$Q}

if [[ $BENCH == hpl ]]; then
  N=${N:-$((NNODE == 1 ? 98304 : 262144))}
  NB=${NB:-1024}
  VARIANTS=${VARIANTS:-"53:53 28:28"}          # A_BITS:B_BITS, 53 = native binary64
  BIN="$REPO/rocHPL/build-gh200/bin/rochpl"
  export HPL_BCAST_RCCL=${HPL_BCAST_RCCL:-1} HPL_BCAST_SPLIT=0 HPL_BCAST_P2P=0 HPL_BCAST_OVERLAP=${HPL_BCAST_OVERLAP:-1}
  export HPL_HIGH_PRIORITY_COMM=1 HPL_NUMA_BIND=0
  export NCCL_ALGO=${NCCL_ALGO:-Ring} NCCL_PROTO=${NCCL_PROTO:-Simple}
  export HPL_QUANT_C_BITS=53 HPL_QUANT_ROUND=${HPL_QUANT_ROUND:-nearest} HPL_QUANT_MODE=scalar
  unset HPL_GEMMUL8 HPL_HIPBLASLT
  # NCCL tuning found by slurm/hpl_tune.sbatch (NCCL_CH channels, NCCL_BUF bytes); unset = NCCL defaults
  NCCL_X=()
  if [[ -n ${NCCL_CH:-} ]]; then export NCCL_MIN_NCHANNELS=$NCCL_CH NCCL_MAX_NCHANNELS=$NCCL_CH; NCCL_X+=(-x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS); fi
  if [[ -n ${NCCL_BUF:-} ]]; then export NCCL_BUFFSIZE=$NCCL_BUF; NCCL_X+=(-x NCCL_BUFFSIZE); fi
else
  N=${N:-$((NNODE == 1 ? 196608 : 491520))}
  NB=${NB:-4096}
  VARIANTS=${VARIANTS:-"7 0"}                  # BF16 fraction bits kept (compile-time builds)
  export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-16}
  export HPLMXP_HBM_RESERVE_GIB=${HPLMXP_HBM_RESERVE_GIB:-4} HPLMXP_PRECOND_FP32=1 HPLMXP_IR_RICHARDSON=1
  export HPLMXP_TRAILING_TN=${HPLMXP_TRAILING_TN:-1} HPLMXP_TRAILING_HIPBLAS=${HPLMXP_TRAILING_HIPBLAS:-1}
  export HPLMXP_PROF=${HPLMXP_PROF:-0}
fi

# Since 2026-10-06 the second IB port (mlx5_1) of most nodes is ACTIVE at only 2.5-10 Gb/s (1X/4X SDR); NCCL and
# UCX then put part of the traffic on it and 8-node HPL drops from ~395 to 157 TFLOP/s. Pin both to the HDR port.
IB_HCA=${IB_HCA:-mlx5_0}
export NCCL_IB_HCA=$IB_HCA UCX_NET_DEVICES=$IB_HCA:1

NS=${NS:-$N}   # several problem sizes in one allocation (residual distribution over N)
echo "job=$SLURM_JOB_ID bench=$BENCH nodes=$NNODE ns=[$NS] hosts=$HOSTS P=$P Q=$Q N=$N NB=$NB f=${FRAC:-0.05} nccl_ch=${NCCL_CH:-default} nccl_buf=${NCCL_BUF:-default} variants=[$VARIANTS] tdps=[$TDPS] repeat=$REPEAT cooldown=$COOLDOWN start_temp=${START_TEMP:-45} ib_hca=$IB_HCA"
echo "rocHPL=$(git -C "$REPO/rocHPL" rev-parse --short HEAD) rocHPL-MxP=$(git -C "$REPO/rocHPL-MxP" rev-parse --short HEAD)"

SMI_DIR="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_smi"
mkdir -p "$SMI_DIR"
srun --overlap --ntasks="$NNODE" --ntasks-per-node=1 bash -c \
  "nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit --format=csv,noheader,nounits -lms 500 > $SMI_DIR/\$(hostname).csv" &
SMI_PID=$!

# Wait until every node's GPU is at or below START_TEMP (read from the live
# nvidia-smi logs), at least COOLDOWN and at most COOL_MAX seconds.
START_TEMP=${START_TEMP:-45}
COOL_MAX=${COOL_MAX:-240}
cool_down() {
  local waited=0 hottest
  sleep "$COOLDOWN"; waited=$COOLDOWN
  while :; do
    hottest=$(for f in "$SMI_DIR"/*.csv; do tail -n 1 "$f" | awk -F', ' '{print $7}'; done | sort -n | tail -n 1)
    [[ -n $hottest && $hottest -le $START_TEMP ]] && break
    ((waited >= COOL_MAX)) && break
    sleep 5; waited=$((waited + 5))
  done
  echo "=== COOL waited=${waited}s hottest=${hottest}C"
}

# Cumulative clock-event counters (microseconds) per node, to catch short power-brake
# or slowdown events that 500 ms sampling misses. Enabled with COUNTERS=1.
read_counters() {
  srun --overlap --ntasks="$NNODE" --ntasks-per-node=1 bash -c \
    "echo \"=== COUNTERS $1 \$(hostname) \$(nvidia-smi --query-gpu=clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.hw_power_brake_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown --format=csv,noheader,nounits)\""
}

run_one() {
  local tdp=$1 v=$2 pass=$3
  local t0 t1
  cool_down
  t0=$(date +%s.%N)
  echo "=== RUN tdp=$tdp variant=$v pass=$pass t_begin=$t0"
  [[ ${COUNTERS:-0} != 0 ]] && read_counters before
  if [[ $BENCH == hpl ]]; then
    export HPL_QUANT_A_BITS=${v%%:*} HPL_QUANT_B_BITS=${v##*:} HPL_QUANT_B_BITS_ALT=${v##*:}
    timeout -k 20 1800 mpirun -np "$NNODE" --map-by ppr:1:node --bind-to none \
      -x NCCL_IB_HCA -x UCX_NET_DEVICES \
      -x LD_LIBRARY_PATH -x HPL_QUANT_A_BITS -x HPL_QUANT_B_BITS -x HPL_QUANT_B_BITS_ALT -x HPL_QUANT_C_BITS \
      -x HPL_QUANT_ROUND -x HPL_QUANT_MODE -x HPL_BCAST_RCCL -x HPL_BCAST_SPLIT -x HPL_BCAST_P2P \
      -x HPL_BCAST_OVERLAP -x HPL_HIGH_PRIORITY_COMM -x HPL_NUMA_BIND -x NCCL_ALGO -x NCCL_PROTO "${NCCL_X[@]}" \
      "$BIN" -P "$P" -Q "$Q" -p 1 -q 1 -N "$N" -NB "$NB" -f "${FRAC:-0.05}" -it 1 || echo "=== RUN FAILED rc=$?"
  else
    timeout -k 20 1800 mpirun -np "$NNODE" --map-by ppr:1:node --bind-to none \
      -x NCCL_IB_HCA -x UCX_NET_DEVICES \
      -x LD_LIBRARY_PATH -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS -x HPLMXP_HBM_RESERVE_GIB \
      -x HPLMXP_PRECOND_FP32 -x HPLMXP_IR_RICHARDSON -x HPLMXP_TRAILING_TN -x HPLMXP_TRAILING_HIPBLAS -x HPLMXP_PROF \
      "$REPO/rocHPL-MxP/build_gh200_mant$v/xhplmxp-cuda" --n "$N" --nb "$NB" --p "$P" --q "$Q" \
      --fp8 0 --itcheck 1 --approx-diag --verbose || echo "=== RUN FAILED rc=$?"
  fi
  [[ ${COUNTERS:-0} != 0 ]] && read_counters after
  t1=$(date +%s.%N)
  echo "=== END tdp=$tdp variant=$v pass=$pass t_begin=$t0 t_end=$t1 n=$N"
}

read -ra var <<<"$VARIANTS"
rev=()
for ((i = ${#var[@]} - 1; i >= 0; i--)); do rev+=("${var[i]}"); done
for N in $NS; do
  echo "=== N $N"
  for tdp in $TDPS; do
    nv_power --hosts "$HOSTS" "$tdp" >/dev/null
    echo "TDP $tdp set on $HOSTS"
    # WARMUP=1: one uncounted run of the first variant before the first cap's pairs (pass W, ignored by the analyses)
    if [[ ${WARMUP:-0} != 0 && -z ${WARMED:-} ]]; then run_one "$tdp" "${var[0]}" W; WARMED=1; fi
    # each repeat is one ABBA block with its own pass letters (A/B, C/D, E/F, ...)
    for ((r = 0; r < REPEAT; r++)); do
      letters=ABCDEFGHIJKL
      for v in "${var[@]}"; do run_one "$tdp" "$v" "${letters:$((2 * r)):1}"; done
      for v in "${rev[@]}"; do run_one "$tdp" "$v" "${letters:$((2 * r + 1)):1}"; done
    done
  done
done
echo "=== ALL DONE"

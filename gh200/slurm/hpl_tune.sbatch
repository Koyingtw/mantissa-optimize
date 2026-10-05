#!/bin/bash
# HPL (rocHPL CUDA port, native FP64) tuning sweep on GH200 nodes at one module cap.
#   export CONFIGS="N,NB,P,Q,F,CH,BUF ..."; sbatch -N 8 --export=ALL,TDP=1000 slurm/hpl_tune.sbatch
#   (CONFIGS must come from the environment: sbatch --export splits its argument on commas)
# Each CONFIGS entry: problem size, block size, process grid P x Q, split fraction f, NCCL_MAX_NCHANNELS
# (0 = NCCL default) and NCCL_BUFFSIZE in bytes (0 = default). Every run starts from GPUs at or below 45 C.
#SBATCH -J hpl-tune
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
REPO=$(cd "$ROOT/.." && pwd)
module purge
module load nvhpc/26.5
NH=/software/nvhpc/Linux_aarch64/26.5
export LD_LIBRARY_PATH="$NH/comm_libs/nccl/lib:$NH/math_libs/13.2/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_hwloc_base_binding_policy=none
NNODE=$SLURM_JOB_NUM_NODES
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
TDP=${TDP:-1000}
BIN="$REPO/rocHPL/build-gh200/bin/rochpl"
CONFIGS=${CONFIGS:-"262144,1024,2,4,0.05,0,0"}
export HPL_BCAST_RCCL=1 HPL_BCAST_SPLIT=0 HPL_BCAST_P2P=0 HPL_BCAST_OVERLAP=1 HPL_HIGH_PRIORITY_COMM=1
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export HPL_QUANT_A_BITS=53 HPL_QUANT_B_BITS=53 HPL_QUANT_B_BITS_ALT=53 HPL_QUANT_C_BITS=53 HPL_QUANT_ROUND=nearest HPL_QUANT_MODE=scalar
unset HPL_GEMMUL8 HPL_HIPBLASLT

restore() { nv_power --hosts "$HOSTS" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
SMI_DIR="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_smi"
mkdir -p "$SMI_DIR"
srun --overlap --ntasks="$NNODE" --ntasks-per-node=1 bash -c \
  "nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit --format=csv,noheader,nounits -lms 500 > $SMI_DIR/\$(hostname).csv" &
SMI_PID=$!
nv_power --hosts "$HOSTS" "$TDP" >/dev/null
echo "job=$SLURM_JOB_ID nodes=$NNODE hosts=$HOSTS tdp=$TDP rocHPL=$(git -C "$REPO/rocHPL" rev-parse --short HEAD)"
echo "configs=[$CONFIGS]"

cool_down() {
  local waited=30 hottest
  sleep 30
  while :; do
    hottest=$(for f in "$SMI_DIR"/*.csv; do tail -n 1 "$f" | awk -F', ' '{print $7}'; done | sort -n | tail -n 1)
    [[ -n $hottest && $hottest -le 45 ]] && break
    ((waited >= 240)) && break
    sleep 5; waited=$((waited + 5))
  done
  echo "=== COOL waited=${waited}s hottest=${hottest}C"
}

for c in $CONFIGS; do
  IFS=, read -r N NB P Q F CH BUF <<<"$c"
  unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
  [[ $CH != 0 ]] && export NCCL_MAX_NCHANNELS=$CH NCCL_MIN_NCHANNELS=$CH
  [[ $BUF != 0 ]] && export NCCL_BUFFSIZE=$BUF
  cool_down
  t0=$(date +%s.%N)
  out=$(timeout -k 20 1500 mpirun -np "$NNODE" --map-by ppr:1:node --bind-to none \
    -x LD_LIBRARY_PATH -x HPL_QUANT_A_BITS -x HPL_QUANT_B_BITS -x HPL_QUANT_B_BITS_ALT -x HPL_QUANT_C_BITS \
    -x HPL_QUANT_ROUND -x HPL_QUANT_MODE -x HPL_BCAST_RCCL -x HPL_BCAST_SPLIT -x HPL_BCAST_P2P \
    -x HPL_BCAST_OVERLAP -x HPL_HIGH_PRIORITY_COMM -x NCCL_ALGO -x NCCL_PROTO \
    ${NCCL_MAX_NCHANNELS:+-x NCCL_MAX_NCHANNELS -x NCCL_MIN_NCHANNELS} ${NCCL_BUFFSIZE:+-x NCCL_BUFFSIZE} \
    "$BIN" -P "$P" -Q "$Q" -p 1 -q 1 -N "$N" -NB "$NB" -f "$F" -it 1 2>&1)
  rc=$?
  t1=$(date +%s.%N)
  gf=$(grep -o "Final Score: *[0-9.e+]*" <<<"$out" | awk '{print $3}')
  res=$(grep -o "Residual value = [0-9.e+-]*" <<<"$out" | awk '{print $4}')
  chk=$(grep -o "Residual Check: [A-Z]*" <<<"$out" | awk '{print $3}')
  echo "$out" | tail -n 6 | sed 's/^/    /'
  echo "RESULT_TUNE N=$N NB=$NB P=$P Q=$Q f=$F ch=$CH buf=$BUF rc=$rc gflops=${gf:-nan} residual=${res:-nan} check=${chk:-none} t_begin=$t0 t_end=$t1"
done
echo "=== ALL DONE"

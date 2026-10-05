#!/bin/bash
# Single-GPU HPL (rocHPL CUDA port) through the cuBLAS shim: per-row exponent span of the GEMM operands
# (MANT_SPAN) and HPL's residual under cuBLAS FP64 emulation, for comparison with in-HPL truncation.
#   sbatch --export=ALL,N=81920,VARIANTS="native span emu36 emud79 t36" slurm/hpl_shim.sbatch
# Variants: native (no shim) | span (shim statistics only) | emu<k> (fixed k-bit emulation) |
#           emud<k> (dynamic mantissa control, at most k bits) | emudl<k> (dynamic + bit-count log) |
#           t<k> (HPL's own operand rounding to k significand bits, no shim)
#SBATCH -J hpl-shim
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:45:00
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
HOST=$(hostname)
N=${N:-81920}; NB=${NB:-1024}
VARIANTS=${VARIANTS:-"native span emu36 emud79 t36"}
SHIM=${SHIM:-$ROOT/shim/libmantshim.dyn.so}
BIN="$REPO/rocHPL/build-gh200/bin/rochpl"
OUT="$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
nv_power --hosts "$HOST" 1000 >/dev/null
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,power.draw.instant,clocks.sm,temperature.gpu,module.power.limit \
  --format=csv,noheader,nounits -lms 500 >"$OUT/smi.csv" &
SMI_PID=$!
echo "job=$SLURM_JOB_ID host=$HOST N=$N NB=$NB variants=[$VARIANTS] rocHPL=$(git -C "$REPO/rocHPL" rev-parse --short HEAD)"
# One GPU needs no panel broadcast. HPL_BCAST_RCCL=0 also matters: on the RCCL path rocHPL leaves through
# _exit(0), which skips the shim's exit-time output (MANT_SPAN files, MANT_EMU_LOG histogram).
export HPL_BCAST_RCCL=0 HPL_BCAST_SPLIT=0 HPL_BCAST_P2P=0 HPL_BCAST_OVERLAP=1 HPL_HIGH_PRIORITY_COMM=1
export HPL_QUANT_ROUND=nearest HPL_QUANT_MODE=scalar
for v in $VARIANTS; do
  bits=53 preload= extra=()
  case $v in
    native) ;;
    span)    preload=$SHIM; extra=(MANT_SPAN="$OUT/span" MANT_SPAN_LINES=256) ;;
    emudl*)  preload=$SHIM; extra=(MANT_EMU_BITS=${v#emudl} MANT_EMU_DYN=1 MANT_EMU_LOG=1) ;;
    emud*)   preload=$SHIM; extra=(MANT_EMU_BITS=${v#emud} MANT_EMU_DYN=1) ;;
    emu*)    preload=$SHIM; extra=(MANT_EMU_BITS=${v#emu}) ;;
    t*)      bits=${v#t} ;;
    *) echo "unknown variant $v"; continue ;;
  esac
  waited=0
  while (( $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) > 45 && waited < 180 )); do
    sleep 5; waited=$((waited + 5)); done
  t0=$(date +%s.%N)
  env ${preload:+LD_PRELOAD=$preload} HPL_QUANT_A_BITS=$bits HPL_QUANT_B_BITS=$bits HPL_QUANT_B_BITS_ALT=$bits \
    HPL_QUANT_C_BITS=53 "${extra[@]}" \
    mpirun -np 1 --bind-to none ${preload:+-x LD_PRELOAD} -x LD_LIBRARY_PATH -x HPL_QUANT_A_BITS -x HPL_QUANT_B_BITS \
      -x HPL_QUANT_B_BITS_ALT -x HPL_QUANT_C_BITS -x HPL_QUANT_ROUND -x HPL_QUANT_MODE -x HPL_BCAST_RCCL \
      -x HPL_BCAST_SPLIT -x HPL_BCAST_P2P -x HPL_BCAST_OVERLAP -x HPL_HIGH_PRIORITY_COMM ${extra[@]/#/-x } \
      "$BIN" -P 1 -Q 1 -p 1 -q 1 -N "$N" -NB "$NB" -f 0.05 -it 1 >"$OUT/$v.out" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  gf=$(grep -o "Final Score: *[0-9.e+]*" "$OUT/$v.out" | awk '{print $3}')
  res=$(grep -o "Residual value = [0-9.e+-]*" "$OUT/$v.out" | awk '{print $4}')
  chk=$(grep -o "Residual Check: [A-Z]*" "$OUT/$v.out" | awk '{print $3}')
  echo "RESULT_HPLSHIM variant=$v rc=$rc gflops=${gf:-nan} residual=${res:-nan} check=${chk:-none} t_begin=$t0 t_end=$t1"
  grep "\[mantshim\] emu bits" "$OUT/$v.out" | head -3 | sed 's/^/    /' || true
done
echo "=== ALL DONE"

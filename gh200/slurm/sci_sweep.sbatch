#!/bin/bash
# S2: scientific applications under mantissa reduction of their FP64 GEMMs (mantshim).
#   sbatch -w work1 --export=ALL,APP=qe,CASE=si1000,INP=si1000.in slurm/sci_sweep.sbatch
#   sbatch -w work3 --export=ALL,APP=pyscf,MOL=168_Valinomycin slurm/sci_sweep.sbatch
#   sbatch -w work3 --export=ALL,APP=pyscf,MOL=113_Taxol,SCRIPT=mp2_test.py,ARGS=def2-tzvp slurm/sci_sweep.sbatch
# VARIANTS: "<bits>:<round>" list; "stats" = unmodified run with operand statistics;
# "copy" = 52 bits through the shim's copy path (the overhead baseline).
# TDPS: module power caps (nv_power on this node only, restored to 1000 W at exit).
#SBATCH -J sci-sweep
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -o logs/sci/%x_%j.out
#SBATCH -e logs/sci/%x_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
A=$(cd "$ROOT/../apps" && pwd)
HOST=$(hostname)
module purge; module load nvhpc/26.5
APP=${APP:?qe or pyscf}
VARIANTS=${VARIANTS:-"stats copy 44:trunc 36:trunc 28:trunc 20:trunc 12:trunc 28:nearest 20:nearest"}
TDPS=${TDPS:-1000}
SHIM="${SHIM:-$ROOT/shim/libmantshim.so}"
OUT="$ROOT/logs/sci/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit \
  --format=csv,noheader,nounits -lms 250 >"$OUT/smi.csv" &
SMI_PID=$!

if [[ $APP == qe ]]; then
  export OMP_NUM_THREADS=16
  CASE=${CASE:-si1000}; INP=${INP:-si1000.in}
  WORK="$OUT/$CASE"; mkdir -p "$WORK"; cp "$A/qe-inputs/$CASE/"*.in "$WORK/"
  cp "$A/qe-inputs/$CASE/"*.UPF "$A/qe-inputs/pseudo/"* "$WORK/" 2>/dev/null || true
  sed -i "s#pseudo_dir *= *'[^']*'#pseudo_dir='./'#; s#outdir *= *'[^']*'#outdir='./tmp'#" "$WORK/$INP"
  grep -E "pseudo_dir|outdir" "$WORK/$INP"
  LABEL=$CASE
else
  source "$A/pyscf-venv/bin/activate"
  export PYTHONPATH=$A/gpu4pyscf:${PYTHONPATH:-} OMP_NUM_THREADS=72 CONTRACT_ENGINE=cupy
  MOL=${MOL:-168_Valinomycin}; SCRIPT=${SCRIPT:-dft_test.py}
  XYZ=$A/gpu4pyscf/benchmarks/molecules/organic/$MOL.xyz
  WORK="$OUT/$MOL"; mkdir -p "$WORK"
  LABEL=$MOL
fi
echo "job=$SLURM_JOB_ID host=$HOST app=$APP case=$LABEL variants=[$VARIANTS] tdps=[$TDPS]"

# Other users' tools can reset the module caps of every node (seen twice); check before each run
ensure_cap() {
  local want=$1 cur
  cur=$(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader,nounits | cut -d. -f1)
  if [[ $cur != "$want" ]]; then
    echo "CAP RESET detected: limit ${cur} W, expected ${want} W -> re-applying"
    nv_power --hosts "$HOST" "$want" >/dev/null
  fi
}
declare -A SEEN
run_one() {
  ensure_cap "$1"
  local tdp=$1 v=$2 bits round env_extra=()
  local preload="$SHIM"
  case $v in
    stats) bits=52; round=trunc; env_extra=(MANT_STATS="$WORK/stats_tdp$tdp" MANT_STATS_SAMPLE=65536) ;;
    copy)  bits=52; round=trunc; env_extra=(MANT_FORCE_COPY=1) ;;
    shapes) bits=52; round=trunc; env_extra=(MANT_VERBOSE=1) ;;   # pass-through, log every GEMM's m n k
    # per-row exponent span per GEMM shape (needs a shim built with MANT_SPAN, e.g. libmantshim.span.so)
    span)  bits=52; round=trunc; env_extra=(MANT_SPAN="$WORK/span_tdp$tdp" MANT_SPAN_LINES=${SPAN_LINES:-256}) ;;
    dump)  bits=52; round=trunc; mkdir -p "$WORK/dump"; env_extra=(MANT_DUMP="$WORK/dump" MANT_DUMP_N=4 MANT_DUMP_MIN_MNK=1e10) ;;
    native) bits=52; round=trunc; preload= ;;
    # unmodified run that only warms the GPU up before the paired (ABBA) runs; analyses ignore it
    warmup) bits=52; round=trunc; preload= ;;
    # reduction built into the application (QE: psi/hpsi/spsi in Davidson), no shim
    src*)  bits=52; round=trunc; preload=; local sb=${v#src}; env_extra=(QE_MANT_BITS=${sb%%:*} QE_MANT_ROUND=${sb##*:}) ;;
    # tolerance-aware: bits follow the Davidson threshold (QE_MANT_ADAPT = margin in bits)
    adapt*) bits=52; round=trunc; preload=; local am=${v#adapt}; env_extra=(QE_MANT_ADAPT=${am%%:*} QE_MANT_ROUND=${am##*:} QE_MANT_MIN=${QE_MANT_MIN:-16}) ;;
    # single-precision replacement of every FP64 GEMM (needs a shim with MANT_F32): f32:fp32|tf32|tf32x3
    f32:*) bits=52; round=trunc; env_extra=(MANT_F32=${v#f32:}) ;;
    # cuBLAS emulation with dynamic mantissa control (cuBLAS picks the bits, native FP64 above k): emud<k>;
    # emudl<k> also logs the bit count of every call (synchronizes, so not for timing)
    emudl*) bits=52; round=trunc; env_extra=(MANT_EMU_BITS=${v#emudl} MANT_EMU_DYN=1 MANT_EMU_LOG=1) ;;
    emud*) bits=52; round=trunc; env_extra=(MANT_EMU_BITS=${v#emud} MANT_EMU_DYN=1) ;;
    # every FP64 GEMM on cuBLAS fixed-point emulation with k mantissa bits (shim, no operand copy): emu<k>
    emu*)  bits=52; round=trunc; env_extra=(MANT_EMU_BITS=${v#emu}) ;;
    # every FP64 GEMM through GEMMul8 (Ozaki scheme II, INT8, fast mode) with m moduli: g8:<m>
    g8:*)  bits=52; round=trunc; preload="$A/GEMMul8/lib/libgemmul8.so"
           env_extra=(GEMMUL8_BACKEND_GEMM=INT8 GEMMUL8_NUM_MOD_D_GEMM=${v#g8:} GEMMUL8_FASTMODE_D_GEMM=1) ;;
    *)     bits=${v%%:*}; round=${v##*:} ;;
  esac
  # a variant listed again at the same cap is its next pass (A, B, C, ...: ABBA orders); keep every output
  local tag="tdp${tdp}_${v/:/_}" letters=ABCDEFGH
  local seen=${SEEN[$tag]:-0}
  SEEN[$tag]=$((seen + 1))
  local pass=${letters:$seen:1}
  if (( seen > 0 )); then tag="${tag}_${pass,,}"; fi
  # start every run from a cool GPU
  local waited=0
  while (( $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) > 45 && waited < 180 )); do
    sleep 5; waited=$((waited + 5)); done
  local t0 t1; t0=$(date +%s.%N)
  echo "=== RUN tdp=$tdp variant=$v pass=$pass t_begin=$t0"
  if [[ $APP == qe ]]; then
    (cd "$WORK" && rm -rf tmp && env ${preload:+LD_PRELOAD=$preload} MANT_BITS=$bits MANT_ROUND=$round "${env_extra[@]}" \
      mpirun -np 1 --bind-to none ${preload:+-x LD_PRELOAD} -x MANT_BITS -x MANT_ROUND ${env_extra[@]/#/-x } \
      "$A/q-e/bin/pw.x" -nk 1 -in "$INP" > "$tag.out" 2> "$tag.err") || echo "=== RUN FAILED rc=$?"
    grep -E "^!|convergence has been achieved|convergence NOT achieved|PWSCF.*WALL" "$WORK/$tag.out" | sed 's/^/    /' || true
  else
    (cd "$WORK" && env ${preload:+LD_PRELOAD=$preload} MANT_BITS=$bits MANT_ROUND=$round "${env_extra[@]}" \
      python "$A/pyscf-inputs/$SCRIPT" "$XYZ" ${ARGS:-gpu b3lyp def2-tzvp df} > "$tag.out" 2>&1) || echo "=== RUN FAILED rc=$?"
    grep -E "^RESULT" "$WORK/$tag.out" | sed 's/^/    /' || true
  fi
  t1=$(date +%s.%N)
  echo "=== END tdp=$tdp variant=$v pass=$pass t_begin=$t0 t_end=$t1"
}

for tdp in $TDPS; do
  nv_power --hosts "$HOST" "$tdp" >/dev/null
  echo "TDP $tdp set: $(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader)"
  for v in $VARIANTS; do run_one "$tdp" "$v"; done
done
echo "=== ALL DONE"

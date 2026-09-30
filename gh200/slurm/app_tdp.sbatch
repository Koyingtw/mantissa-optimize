#!/bin/bash
# M4b: third-party non-GEMM FP64 applications (MFEM PA diffusion, SW4, LAMMPS KOKKOS) under module
# power caps with their stored FP64 coefficient arrays mantissa-reduced once at setup (MANT_BITS hooks
# built in apps/; see apps/建置日誌.md sections 5-8).
#   sbatch -w work10 --export=ALL,APP=mfem slurm/app_tdp.sbatch
#   sbatch -w work13 --export=ALL,APP=sw4,TDPS="1000 450 350" slurm/app_tdp.sbatch
#   sbatch -w work11 --export=ALL,APP=lammps,CASE=eam slurm/app_tdp.sbatch
# VARIANTS: "native" or "<bits>:<round>"; a variant listed twice at the same cap is its B pass (ABBA).
# native runs the same binary with MANT_BITS=52 (the hook does nothing at >= 52).
# TDPS: module power caps (nv_power on this node only, restored to 1000 W at exit).
#SBATCH -J app-tdp
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -o logs/apps/%x_%j.out
#SBATCH -e logs/apps/%x_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
APPS=$(cd "$ROOT/../apps" && pwd)
HOST=$(hostname)
module purge; module load nvhpc/26.5
APP=${APP:?mfem, sw4 or lammps}
VARIANTS=${VARIANTS:-"native 36:nearest 20:nearest 20:nearest 36:nearest native"}
TDPS=${TDPS:-"1000 600 500 450"}
OUT="$ROOT/logs/apps/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit \
  --format=csv,noheader,nounits -lms 250 >"$OUT/smi.csv" &
SMI_PID=$!

case $APP in
  mfem)   ARGS=${ARGS:-"-n 112 -o 3 -rep 4"}; LABEL="mfem ${ARGS}" ;;
  sw4)    INP=${INP:-loh1gradf-h50.in}; LABEL="sw4 ${INP}" ;;
  lammps) CASE=${CASE:-eam}; X=${X:-8}; NSTEPS=${NSTEPS:-2000}; LABEL="lammps ${CASE} x=${X} nsteps=${NSTEPS}"
          export OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads ;;
  *) echo "unknown APP=$APP"; exit 1 ;;
esac
echo "job=$SLURM_JOB_ID host=$HOST app=$APP case=[$LABEL] variants=[$VARIANTS] tdps=[$TDPS]"

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
  local tdp=$1 v=$2 bits round
  if [[ $v == native ]]; then bits=52; round=trunc; else bits=${v%%:*}; round=${v##*:}; fi
  local pass=A tag="tdp${tdp}_${v/:/_}"
  if [[ -n ${SEEN[$tag]:-} ]]; then pass=B; tag="${tag}_b"; fi
  SEEN[$tag]=1
  local waited=0   # start every run from a cool GPU
  while (( $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) > 45 && waited < 180 )); do
    sleep 5; waited=$((waited + 5)); done
  local t0 t1; t0=$(date +%s.%N)
  echo "=== RUN tdp=$tdp variant=$v pass=$pass t_begin=$t0"
  export MANT_BITS=$bits MANT_ROUND=$round
  case $APP in
    mfem)
      (cd "$APPS/mfem-bench" && ./ex1_bench.build-cuda $ARGS > "$OUT/$tag.out" 2>&1) || echo "=== RUN FAILED rc=$?"
      grep -E "^\[TIME\] solves|^\[RESULT\]|^\[CHECK\]" "$OUT/$tag.out" | sed 's/^/    /' || true ;;
    sw4)
      local rd="$OUT/sw4_$tag"; mkdir -p "$rd"; cp "$APPS/sw4-inputs/$INP" "$rd/"
      (cd "$rd" && mpirun -np 1 --bind-to none -x MANT_BITS -x MANT_ROUND "$APPS/sw4/optimize/sw4" "$INP" \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err") || echo "=== RUN FAILED rc=$?"
      grep -E "\[MANT\]|Execution time|solver phase|Total seconds" "$OUT/$tag.out" | head -6 | sed 's/^/    /' || true ;;
    lammps)
      (cd "$APPS/lammps-inputs" && mpirun -np 1 --bind-to none -x MANT_BITS -x MANT_ROUND \
        "$APPS/lammps/build-kokkos-cuda/lmp" -k on g 1 t 1 -sf kk -pk kokkos newton off neigh full -in "in.$CASE.kk" \
        -var x "$X" -var y "$X" -var z "$X" -var nsteps "$NSTEPS" -var nthermo 100 -log "$OUT/$tag.out" -screen none) \
        || echo "=== RUN FAILED rc=$?"
      grep -E "Loop time|Performance" "$OUT/$tag.out" | sed 's/^/    /' || true ;;
  esac
  t1=$(date +%s.%N)
  echo "=== END tdp=$tdp variant=$v pass=$pass t_begin=$t0 t_end=$t1"
}

for tdp in $TDPS; do
  nv_power --hosts "$HOST" "$tdp" >/dev/null
  echo "TDP $tdp set: $(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader)"
  for v in $VARIANTS; do run_one "$tdp" "$v"; done
done
echo "=== ALL DONE"

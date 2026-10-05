#!/bin/bash
# Geostatistics GP log-likelihood (blocked Cholesky, FP64 DGEMM trailing updates) under module power caps.
#   sbatch --export=ALL,TDPS="1000 700 500",GP_N=60000,VARIANTS="native t36n ..." slurm/gp_tdp.sbatch
#SBATCH -J gp-tdp
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -o logs/gp/%x_%j.out
#SBATCH -e logs/gp/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
A=$(cd "$ROOT/../apps" && pwd)
HOST=$(hostname)
module purge; module load nvhpc/26.5
TDPS=${TDPS:-"1000"}
VARIANTS=${VARIANTS:-"native t36n emu36 fp32"}
OUT="$ROOT/logs/gp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit \
  --format=csv,noheader,nounits -lms 250 >"$OUT/smi.csv" &
SMI_PID=$!
source "$A/pyscf-venv/bin/activate"
echo "job=$SLURM_JOB_ID host=$HOST n=${GP_N:-60000} variants=[$VARIANTS] tdps=[$TDPS]"
for tdp in $TDPS; do
  nv_power --hosts "$HOST" "$tdp" >/dev/null
  GP_CAP=$tdp GP_VARIANTS="$VARIANTS" python "$A/geostat/gp_mle.py" > "$OUT/tdp$tdp.out" 2>&1 || echo "=== RUN FAILED tdp=$tdp rc=$?"
  grep -E "^(RESULT|RANGE|MP_GP|CAP RESET|n=)" "$OUT/tdp$tdp.out" | sed 's/^/    /' || true
done
echo "=== ALL DONE"

#!/bin/bash
# Power-bound real application: GPU4PySCF DF-MP2 (~80% of GPU time in one FP64 DGEMM, GPU ~800 W at 1000 W).
# One process per cap runs DF-RHF once, then the DF-MP2 variants in MP2_VARIANTS order (apps/pyscf-inputs/
# mp2_variants.py): native, fp32 (the library's own FP32 path), t<k><n|t> (DF tensor mantissa reduced in
# place, FP64 GEMM), emu<k> / emud (cuBLAS FP64 emulation, fixed / dynamic bits).
#   sbatch -w work1 --export=ALL,TDPS="1000 700 500" slurm/mp2_tdp.sbatch
#   sbatch -w work2 --export=ALL,TDPS=1000,VARIANTS="native t44n t36n ...",RANGE_STATS=1 slurm/mp2_tdp.sbatch
#SBATCH -J mp2-tdp
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -o logs/mp2/%x_%j.out
#SBATCH -e logs/mp2/%x_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
A=$(cd "$ROOT/../apps" && pwd)
HOST=$(hostname)
module purge; module load nvhpc/26.5
MOL=${MOL:-113_Taxol}; BASIS=${BASIS:-def2-tzvp}
VARIANTS=${VARIANTS:-"native fp32 t36n t28n emu32 emud emud emu32 t28n t36n fp32 native"}
TDPS=${TDPS:-"1000 700 500"}
OUT="$ROOT/logs/mp2/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit \
  --format=csv,noheader,nounits -lms 250 >"$OUT/smi.csv" &
SMI_PID=$!

source "$A/pyscf-venv/bin/activate"
export PYTHONPATH=$A/gpu4pyscf:${PYTHONPATH:-} OMP_NUM_THREADS=72 CONTRACT_ENGINE=cupy
XYZ=$A/gpu4pyscf/benchmarks/molecules/organic/$MOL.xyz
echo "job=$SLURM_JOB_ID host=$HOST mol=$MOL basis=$BASIS variants=[$VARIANTS] tdps=[$TDPS]"
for tdp in $TDPS; do
  nv_power --hosts "$HOST" "$tdp" >/dev/null
  echo "TDP $tdp set: $(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader)"
  MP2_CAP=$tdp MP2_VARIANTS="$VARIANTS" MP2_RANGE_STATS=${RANGE_STATS:-} \
    python "$A/pyscf-inputs/mp2_variants.py" "$XYZ" "$BASIS" > "$OUT/tdp$tdp.out" 2>&1 || echo "=== RUN FAILED tdp=$tdp rc=$?"
  grep -E "^(RESULT|RANGE|CAP RESET)" "$OUT/tdp$tdp.out" | sed 's/^/    /' || true
done
echo "=== ALL DONE"

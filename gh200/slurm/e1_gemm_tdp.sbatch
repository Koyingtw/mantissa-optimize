#!/bin/bash
# E1: GEMM throughput and power across module TDP caps and operand mantissa widths.
# Each TDP level runs the config list forward then backward (ABBA) on the same node.
#   sbatch -w work1 slurm/e1_gemm_tdp.sbatch
#SBATCH -J e1-gemm-tdp
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 02:30:00
#SBATCH -o logs/e1_%j.out
#SBATCH -e logs/e1_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
HOST=$(hostname)
TDPS=${TDPS:-"1000 900 800 700 600 500 400 300"}
WARMUP=${WARMUP:-8}
SECONDS_PER=${SECONDS_PER:-20}
# dtype:frac  (fraction bits kept; FP64 native 52, BF16 native 7)
CONFIGS=${CONFIGS:-"fp64:52 fp64:35 fp64:27 fp64:12 fp64:0 bf16:7 bf16:3 bf16:0"}
START_TEMP=${START_TEMP:-45}   # each measurement starts only once the GPU has cooled to this
COOL_MAX=${COOL_MAX:-180}

restore() { nv_power --hosts "$HOST" 1000 >/dev/null 2>&1 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT

echo "job=$SLURM_JOB_ID host=$HOST tdps=[$TDPS] configs=[$CONFIGS] warmup=$WARMUP seconds=$SECONDS_PER start_temp=$START_TEMP"
nvidia-smi --query-gpu=name,uuid,driver_version,module.power.limit,power.limit,clocks.max.sm --format=csv

SMI="logs/e1_${SLURM_JOB_ID}_smi.csv"
nvidia-smi --query-gpu=timestamp,module.power.draw.instant,module.power.draw.average,power.draw.instant,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,module.power.limit \
  --format=csv,noheader,nounits -lms 250 >"$SMI" &
SMI_PID=$!

# Other users' tools can reset the module caps of every node (seen twice); check before each run
ensure_cap() {
  local want=$1 cur
  cur=$(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader,nounits | cut -d. -f1)
  if [[ $cur != "$want" ]]; then
    echo "CAP RESET detected: limit ${cur} W, expected ${want} W -> re-applying"
    nv_power --hosts "$HOST" "$want" >/dev/null
  fi
}
read -ra cfg <<<"$CONFIGS"
# SHAPES: optional list of "mxnxk" (x, not comma: sbatch --export splits on commas) (e.g. the truncation-vs-emulation crossover sweep); every config runs at every shape
read -ra shapes <<<"${SHAPES:-default}"
rev=()
for ((i = ${#cfg[@]} - 1; i >= 0; i--)); do rev+=("${cfg[i]}"); done

for tdp in $TDPS; do
  nv_power --hosts "$HOST" "$tdp" >/dev/null
  echo "TDP $tdp set: $(nvidia-smi --query-gpu=module.power.limit --format=csv,noheader)"
  for pass in A B; do
    list=("${cfg[@]}")
    [[ $pass == B ]] && list=("${rev[@]}")
    for shape in "${shapes[@]}"; do
    sx=()
    if [[ $shape != default ]]; then IFS=x read -r sm sn sk <<<"$shape"; sx=(--m "$sm" --n "$sn" --k "$sk"); fi
    for c in "${list[@]}"; do
      IFS=: read -r dtype frac low inner <<<"$c"   # dtype:frac[:low[:inner]], low = zero|ones|const
      low=${low:-zero}
      extra=()
      bench=./bin/gemm_mant_bench
      [[ $dtype == bf16 || $dtype == fp8 ]] && extra=(--k 6144)
      # non-GEMM FP64 kernels on the DFMA pipe; inner = bits kept inside the n-body loop
      [[ $dtype == dfma || $dtype == nbody || $dtype == copy || $dtype == triad ]] && bench=./bin/fp64_kernel_bench
      [[ $dtype == attn ]] && bench=./bin/attn_bench   # BF16 FlashAttention (PyTorch container)
      [[ $dtype == rbf16 || $dtype == rfp8 ]] && bench=./bin/torch_gemm_bench   # real Llama2-70B operands
      [[ $dtype == cg ]] && bench=./bin/cg_bench   # memory-bound FP64 CG / SpMV
      [[ $dtype == tf32x3 || $dtype == tf32x1 || $dtype == tfp32 ]] && bench=./bin/tf32x3_bench   # 3xTF32 (PyTorch)
      [[ $dtype == gemmul8 ]] && bench=./bin/gemmul8_bench   # Ozaki scheme II via the GEMMul8 cuBLAS hook; frac = moduli
      [[ -n ${inner:-} ]] && extra+=(--inner "$inner")
      waited=0
      while (( $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) > START_TEMP && waited < COOL_MAX )); do
        sleep 5; waited=$((waited + 5))
      done
      echo "COOL waited=${waited}s temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)C"
      ensure_cap "$tdp"
      "$bench" --dtype "$dtype" --frac "$frac" --warmup "$WARMUP" --seconds "$SECONDS_PER" \
        --low "$low" --tag "host=$HOST,tdp=$tdp,pass=$pass" "${extra[@]}" "${sx[@]}" ${EXTRA_ARGS:-}
    done
    done
  done
done
echo "E1 done"

#!/bin/bash
# Generic 1-GPU runner for the FP8 attention study.
#   sbatch --export=ALL,CMD="python bench_te.py" slurm/run.sbatch
# Runs CMD (in gh200/fp8attn) inside the NGC 25.04 container, logs nvidia-smi clocks every 200 ms.
#SBATCH -J fp8attn
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=15
#SBATCH --exclude=head
#SBATCH -o /nfs/home/koying/mantissa-optimize/gh200/fp8attn/logs/%x_%j.log
set -u
SIF=/nfs/home/koying/sc26-mlperf-build/containers/llama2_70b_lora_ngc25.04.sif
D=/nfs/home/koying/mantissa-optimize/gh200/fp8attn
cd $D
SMI=$D/logs/smi_${SLURM_JOB_ID}.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 200 > $SMI &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null" EXIT
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -u +%FT%TZ)"
echo "CMD=$CMD"
PYP=${EXTRA_PYTHONPATH:-/nfs/home/koying/fp8attn-build/fa3src/hopper}:$D
export TRITON_CACHE_DIR=/nfs/home/koying/fp8attn-build/triton-cache-$(hostname)
apptainer exec --nv --bind /nfs/home/koying $SIF bash -c "export PYTHONPATH=$PYP:\$PYTHONPATH; $CMD"
echo "end=$(date -u +%FT%TZ)"
python3 - $SMI <<'PY'
import sys, statistics as st
rows=[l.split(', ') for l in open(sys.argv[1]) if l.count(',')>=5]
clk=[int(r[1]) for r in rows if r[1].strip().isdigit()]
pw=[float(r[4]) for r in rows if r[4].replace('.','',1).strip().isdigit()]
busy=[c for c,p in zip(clk,pw) if p>150]
if clk: print(f"SMI samples={len(clk)} sm_clock median(all)={st.median(clk)} median(power>150W)={st.median(busy) if busy else 'n/a'} min={min(clk)} maxtemp={max(int(r[3]) for r in rows)}")
PY

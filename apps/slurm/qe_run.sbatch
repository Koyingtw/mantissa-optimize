#!/bin/bash
#SBATCH -J qe-run
#SBATCH -p compute
#SBATCH -w work2
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/qe_run_%j.log
# 用法: sbatch --export=ALL,CASE=si216|ausurf112,INP=si216.in qe_run.sbatch
set -e
module load nvhpc/26.5
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
CASE=${CASE:-si216}; INP=${INP:-si216.in}
PW=/nfs/home/koying/mantissa-optimize/apps/q-e/bin/pw.x
cd /nfs/home/koying/mantissa-optimize/apps/qe-inputs/$CASE
OUT=${INP%.in}.${SLURM_JOB_ID}.out
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,memory.used --format=csv -lms 500 > ${INP%.in}.${SLURM_JOB_ID}.gpu.csv &
SMI=$!
/usr/bin/time -v mpirun -np 1 --bind-to none ${PRELOAD:+-x LD_PRELOAD=$PRELOAD} $PW -nk 1 -in $INP > $OUT 2> ${OUT%.out}.err || echo "pw.x exit $?"
kill $SMI
grep -E "GPU acceleration|^!|convergence has been achieved|PWSCF.*WALL|total energy  " $OUT | tail -8
grep -E "Elapsed|Maximum resident" ${OUT%.out}.err
python3 - <<PY
import csv
r=list(csv.reader(open("${INP%.in}.${SLURM_JOB_ID}.gpu.csv")))[1:]
p=[float(x[1].split()[0]) for x in r if 'W' in x[1]]; u=[float(x[3].split()[0]) for x in r]
print(f"GPU samples={len(p)} meanPower={sum(p)/len(p):.1f}W maxPower={max(p):.1f}W meanUtil={sum(u)/len(u):.1f}%")
PY

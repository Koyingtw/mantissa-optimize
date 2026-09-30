#!/bin/bash
#SBATCH -J sw4-run
#SBATCH -p compute
#SBATCH -w work13
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 0:40:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/sw4_run_%j.log
# 用法:
#   sbatch --export=ALL,INP=loh1grad-h50.in apps/slurm/sw4_run.sbatch
#   sbatch --export=ALL,INP=loh1grad-h50.in,MANT_BITS=36,MANT_ROUND=nearest apps/slurm/sw4_run.sbatch
#   可選: BIN=/path/to/sw4（預設 apps/sw4/optimize/sw4），NSYS=1（nsys 看 kernel 時間佔比），TAG=自訂標籤
# 注意: 不改功耗上限；只記錄 nvidia-smi。
set -e
module load nvhpc/26.5
APPS=/nfs/home/koying/mantissa-optimize/apps
BIN=${BIN:-$APPS/sw4/optimize/sw4}
INP=${INP:-loh1grad-h100.in}
TAG=${TAG:-${INP%.in}_mb${MANT_BITS:-native}${MANT_ROUND:+_$MANT_ROUND}}
RUN=$APPS/sw4-inputs/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p $RUN && cd $RUN && cp $APPS/sw4-inputs/$INP .
echo "node=$(hostname) BIN=$BIN INP=$INP MANT_BITS=${MANT_BITS:-unset} MANT_ROUND=${MANT_ROUND:-unset} RUN=$RUN"
nvidia-smi --query-gpu=power.limit,enforced.power.limit,temperature.gpu --format=csv
MPIENV="-x MANT_BITS -x MANT_ROUND"
[[ -z ${MANT_BITS+x} ]] && MPIENV=""
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,memory.used,temperature.gpu --format=csv -lms 500 > gpu.csv &
SMI=$!
T0=$(date +%s.%N)
if [[ -n $NSYS ]]; then
  NSYSBIN=/software/nvhpc/Linux_aarch64/26.5/profilers/13.2/Nsight_Systems/bin/nsys
  $NSYSBIN profile -t cuda,nvtx -o sw4prof --force-overwrite true --stats=false \
    mpirun -np 1 --bind-to none $MPIENV $BIN $INP > sw4.out 2> sw4.err || echo "sw4 exit $?"
  $NSYSBIN stats -r cuda_gpu_kern_sum --format csv -o kern sw4prof.nsys-rep > /dev/null 2>&1 || true
  python3 $APPS/sw4-inputs/sw4_kern.py kern_cuda_gpu_kern_sum.csv || head -30 kern_cuda_gpu_kern_sum.csv
else
  /usr/bin/time -v mpirun -np 1 --bind-to none $MPIENV $BIN $INP > sw4.out 2> sw4.err || echo "sw4 exit $?"
fi
T1=$(date +%s.%N)
kill $SMI
grep -E "\[MANT\]|Number of time steps|Execution time|solver phase|Total seconds|ERROR|error" sw4.out | head -20
grep -E "Elapsed|Maximum resident" sw4.err || true
python3 $APPS/sw4-inputs/sw4_summary.py $RUN $T0 $T1

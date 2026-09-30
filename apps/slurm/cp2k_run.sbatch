#!/bin/bash
#SBATCH -J cp2k-run
#SBATCH -p compute
#SBATCH -w work1
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/cp2k_run_%j.log
# 用法: sbatch --export=ALL,INP=RI-MP2_ammonia,NP=4,NSYS=1 cp2k_run.sbatch
set -e
A=/nfs/home/koying/mantissa-optimize/apps
SIF=$A/cp2k/cp2k_v2024.3.sif
INP=${INP:-RI-MP2_ammonia}; NP=${NP:-4}; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$((72/NP))}
W=$A/cp2k/runs/${INP}_${SLURM_JOB_ID}; mkdir -p $W; cd $W
apptainer exec $SIF cp /opt/cp2k/benchmarks/QS_single_node/$INP.inp .
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,memory.used --format=csv -lms 500 > gpu.csv &
SMI=$!
PROF=""; [ -n "$NSYS" ] && PROF="nsys profile -t cuda,cublas -o nsys_$INP --force-overwrite true"
# LD_PRELOAD 用 APPTAINERENV_LD_PRELOAD 傳入容器，shim.so 需放在 bind 得到的路徑 (/nfs 預設有 bind)
[ -n "$PRELOAD" ] && export APPTAINERENV_LD_PRELOAD=$PRELOAD
T0=$(date +%s.%N)
apptainer exec --nv -B /nfs -B /software $SIF bash -c "cd $W && ${HOSTFFT:+export LD_LIBRARY_PATH=$A/cp2k/hostlibs:\$LD_LIBRARY_PATH &&} $PROF mpirun -np $NP --bind-to none -x OMP_NUM_THREADS -x LD_LIBRARY_PATH ${PRELOAD:+-x LD_PRELOAD} /opt/cp2k/exe/local_cuda/cp2k.psmp -i $INP.inp -o $INP.out" || echo "exit $?"
T1=$(date +%s.%N)
kill $SMI
echo "wall=$(echo "$T1-$T0"|bc) s NP=$NP OMP=$OMP_NUM_THREADS"
grep -E "Total energy:|ENERGY\| Total FORCE_EVAL|MP2 correlation energy|Total MP2 energy|SCF run converged|outer SCF loop converged|CP2K   *1 *1.0" $INP.out | tail -8
grep -A25 "T I M I N G" $INP.out | head -30
python3 - <<PY
import csv
r=list(csv.reader(open("gpu.csv")))[1:]
p=[float(x[1].split()[0]) for x in r if 'W' in x[1]]; u=[float(x[3].split()[0]) for x in r]
print(f"GPU samples={len(p)} meanPower={sum(p)/len(p):.1f}W maxPower={max(p):.1f}W meanUtil={sum(u)/len(u):.1f}%")
PY
if [ -n "$NSYS" ]; then
  apptainer exec $SIF bash -c "cd $W && ls" 
  module load nvhpc/26.5
  for f in nsys_$INP*.nsys-rep; do nsys stats -r cuda_gpu_kern_sum,nvtx_sum -f csv -o ${f%.nsys-rep} $f >/dev/null; python3 $A/slurm/kern_frac.py ${f%.nsys-rep}_cuda_gpu_kern_sum.csv; grep -i cublas ${f%.nsys-rep}_nvtx_sum.csv|head; done
fi

#!/bin/bash
#SBATCH -J qe-nsys
#SBATCH -p compute
#SBATCH -w work2
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/qe_nsys_%j.log
# 用法: sbatch --export=ALL,CASE=si512,INP=si512.in qe_nsys.sbatch
set -e
module load nvhpc/26.5
export OMP_NUM_THREADS=16
CASE=${CASE:-si512}; INP=${INP:-si512.in}
cd /nfs/home/koying/mantissa-optimize/apps/qe-inputs/$CASE
R=nsys_${INP%.in}_${SLURM_JOB_ID}
nsys profile -t cuda,cublas,openacc -o $R --force-overwrite true \
  /nfs/home/koying/mantissa-optimize/apps/q-e/bin/pw.x -nk 1 -in $INP > $R.out
grep -E "^!|convergence has been|PWSCF.*WALL" $R.out
nsys stats -r cuda_gpu_kern_sum,nvtx_sum -f csv -o $R $R.nsys-rep >/dev/null
python3 /nfs/home/koying/mantissa-optimize/apps/slurm/kern_frac.py ${R}_cuda_gpu_kern_sum.csv $R.out
head -15 ${R}_nvtx_sum.csv

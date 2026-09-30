#!/bin/bash
#SBATCH -J pyscf-run
#SBATCH -p compute
#SBATCH -w work3
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/pyscf_run_%j.log
# 用法: sbatch --export=ALL,DEV=gpu,MOL=057_Tamoxifen pyscf_run.sbatch
set -e
module load nvhpc/26.5          # 提供 libcublas.so.13 / libcusolver / nvrtc 等 (math_libs/lib64, cuda/lib64)
A=/nfs/home/koying/mantissa-optimize/apps
source $A/pyscf-venv/bin/activate
export PYTHONPATH=$A/gpu4pyscf:$PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export CONTRACT_ENGINE=${CONTRACT_ENGINE:-cupy}   # 預設關掉 cuTENSOR，讓張量縮併走 cupy.einsum → cuBLAS
[ "$CONTRACT_ENGINE" = cutensor ] && unset CONTRACT_ENGINE   # CONTRACT_ENGINE=cutensor → 用 gpu4pyscf 預設 cuTENSOR
DEV=${DEV:-gpu}; MOL=${MOL:-057_Tamoxifen}
XYZ=$A/gpu4pyscf/benchmarks/molecules/organic/$MOL.xyz
cd $A/pyscf-inputs
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,memory.used --format=csv -lms 500 > ${MOL}_${DEV}_${SLURM_JOB_ID}.gpu.csv &
SMI=$!
${NSYS:+nsys profile -t cuda,cublas -o nsys_${MOL}_${SLURM_JOB_ID} --force-overwrite true} ${PRELOAD:+env LD_PRELOAD=$PRELOAD} python ${SCRIPT:-dft_test.py} $XYZ ${ARGS:-$DEV ${XC:-b3lyp} ${BASIS:-def2-tzvp} ${MODE:-df}} > ${MOL}_${DEV}_${SLURM_JOB_ID}.out 2>&1 || echo "exit $?"
kill $SMI
grep -E "natm=|gpu4pyscf |RESULT" ${MOL}_${DEV}_${SLURM_JOB_ID}.out
tail -3 ${MOL}_${DEV}_${SLURM_JOB_ID}.out
if [ -n "$NSYS" ]; then
  R=nsys_${MOL}_${SLURM_JOB_ID}
  nsys stats -r cuda_gpu_kern_sum,nvtx_sum -f csv -o $R $R.nsys-rep >/dev/null
  python3 $A/slurm/kern_frac.py ${R}_cuda_gpu_kern_sum.csv
  grep -i cublas ${R}_nvtx_sum.csv | head -20
fi

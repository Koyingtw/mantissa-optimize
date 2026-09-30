#!/bin/bash
#SBATCH -J qe-build
#SBATCH -p compute
#SBATCH -w work2
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 2:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/qe_build_%j.log
set -ex
module load nvhpc/26.5
cd /nfs/home/koying/mantissa-optimize/apps/q-e
CUDA=/software/nvhpc/Linux_aarch64/26.5/cuda/13.2
./configure CC=nvc F90=nvfortran F77=nvfortran MPIF90=mpif90 \
  --with-cuda=$CUDA --with-cuda-cc=90 --with-cuda-runtime=13.2 \
  --enable-openmp --with-scalapack=no \
  BLAS_LIBS="-lblas" LAPACK_LIBS="-llapack" 2>&1 | tail -40
grep -E "^(DFLAGS|CUDA_|BLAS_LIBS|LAPACK_LIBS|FFT_LIBS|LD_LIBS|F90FLAGS|LDFLAGS)" make.inc
time make -j 72 pw
ls -l bin/

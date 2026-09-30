#!/bin/bash
#SBATCH -J mfem-build
#SBATCH -p compute
#SBATCH -w work10
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/mfem_build_%j.log
# 用法: sbatch --export=ALL,BDIR=build-cuda[,DRIVER_ONLY=1] mfem_build.sbatch
#   BDIR          MFEM out-of-source 建置目錄（相對於 apps/mfem），預設 build-cuda
#   DRIVER_ONLY=1 只重編 apps/mfem-bench 的 driver（需先有 $BDIR/libmfem.a）
set -ex
module load nvhpc/26.5
CUDA=/software/nvhpc/Linux_aarch64/26.5/cuda/13.2
SRC=/nfs/home/koying/mantissa-optimize/apps/mfem
BENCH=/nfs/home/koying/mantissa-optimize/apps/mfem-bench
BDIR=${BDIR:-build-cuda}
cd $SRC
nvcc --version | tail -2
g++ --version | head -1
if [ -z "$DRIVER_ONLY" ]; then
  make BUILD_DIR=$SRC/$BDIR config MFEM_USE_CUDA=YES MFEM_USE_MPI=NO MFEM_DEBUG=NO CUDA_ARCH=sm_90 CUDA_DIR=$CUDA
  time make -C $SRC/$BDIR -j 72 lib
  ls -l $SRC/$BDIR/libmfem.a
fi
# driver（用 MFEM 產生的 config.mk 取得編譯旗標）
cd $BENCH
make -f $BENCH/Makefile MFEM_BUILD=$SRC/$BDIR OUT=ex1_bench.$BDIR
ls -l $BENCH/ex1_bench.$BDIR

#!/bin/bash
#SBATCH -J g4p-build
#SBATCH -p compute
#SBATCH -w work1
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 2:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/gpu4pyscf_build_%j.log
set -ex
module load nvhpc/26.5 cmake/3.31.12
export CC=gcc CXX=g++ FC=gfortran CUDAHOSTCXX=g++
export CUDA_HOME=/software/nvhpc/Linux_aarch64/26.5/cuda/13.2
cd /nfs/home/koying/mantissa-optimize/apps/gpu4pyscf
cmake -S gpu4pyscf/lib -B build/g4p -DCUDA_ARCHITECTURES=90-real -DBUILD_LIBXC=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=$CUDA_HOME -DCMAKE_CUDA_FLAGS=-Xcompiler=-Wno-template-body -DCMAKE_MESSAGE_LOG_LEVEL=WARNING
time cmake --build build/g4p -j 72
ls gpu4pyscf/lib/*.so gpu4pyscf/lib/deps/lib

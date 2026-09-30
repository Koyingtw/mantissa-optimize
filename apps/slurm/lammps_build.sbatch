#!/bin/bash
#SBATCH -J lammps-build
#SBATCH -p compute
#SBATCH -w work11
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:30:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/lammps_build_%j.log
# LAMMPS (stable 22Jul2025 update6) KOKKOS CUDA FP64 建置，sm_90 (GH200)
# 用法: sbatch apps/slurm/lammps_build.sbatch            （第一次: configure + build）
#       sbatch --export=ALL,RECONF=0 apps/slurm/lammps_build.sbatch  （只 rebuild）
set -ex
module load nvhpc/26.5
export CUDA_HOME=/software/nvhpc/Linux_aarch64/26.5/cuda/13.2
export PATH=$CUDA_HOME/bin:$PATH
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export OMPI_CXX=g++ OMPI_CC=gcc
cd /nfs/home/koying/mantissa-optimize/apps/lammps
B=${B:-build-kokkos-cuda}
if [ "${RECONF:-1}" = "1" ]; then
cmake -S cmake -B $B -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$PWD/lib/kokkos/bin/nvcc_wrapper \
  -DPKG_KOKKOS=on -DKokkos_ENABLE_CUDA=yes -DKokkos_ARCH_HOPPER90=yes -DKokkos_ENABLE_OPENMP=yes \
  -DPKG_MANYBODY=on -DPKG_MOLECULE=off -DBUILD_MPI=yes -DBUILD_OMP=yes \
  -DCUDAToolkit_ROOT=$CUDA_HOME 2>&1 | tail -60
fi
time cmake --build $B -j 72 2>&1 | grep -E "error|Error|warning: .*deprecated|Linking|Built target lmp" | head -80
ls -l $B/lmp
ldd $B/lmp | grep -E "cuda|mpi|gomp"
$B/lmp -h | sed -n '1,60p' | grep -iE "kokkos|version|precision|MPI|OpenMP|Installed packages" -A2 | head -40
